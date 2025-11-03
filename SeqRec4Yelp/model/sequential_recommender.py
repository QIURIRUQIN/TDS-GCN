import copy
import pickle

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import logging

from torch.distributed.tensor.parallel import loss_parallel

from utils.metrics import ndcg_k, hit_ratio_k
from model.basemodel import GRU4Rec, SASRec, GatedGRU4Rec, STMP
from tqdm import tqdm
from loader.BPRData import SeqDataset

class RecNet:

    def __init__(self,
                 model_use = 'GRU4Rec',
                 weighted_method='rating_avg',
                 num_items = 1000,
                 num_users = 1000,
                 item_embd_dim = 128,
                 user_embd_dim = 64,
                 hidden_dim = 128,
                 n_layers = 4,
                 dropout = 0.1,
                 lr = 1e-3,
                 weight_decay = 1e-4,
                 K = 20,
                 num_neg = 49,
                 optimizer_use = "adamw",
                 loss = "bpr",
                 device = 'cuda', **kwargs):

        self.model_use = model_use
        self.weighted_method = weighted_method
        self.num_items = num_items
        self.num_users = num_users
        self.item_embd_dim = item_embd_dim
        self.user_embd_dim = user_embd_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.lr = lr
        self.weight_decay = weight_decay
        self.K = K
        self.num_neg = num_neg

        self.optimizer_use = optimizer_use
        self.loss = loss
        self.model = self.get_model().to(self.device)

        if self.optimizer_use == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_model(self):
        if self.model_use == 'GRU4Rec':
            return GRU4Rec(num_items=self.num_items, embd_dim=self.item_embd_dim, n_layers=self.n_layers)
        elif self.model_use == 'GatedGRU4Rec':
            return GatedGRU4Rec(num_items=self.num_items, num_users=self.num_users,embd_dim=self.item_embd_dim,user_embd_dim=self.user_embd_dim,
                                hidden_dim=self.hidden_dim,n_layers=self.n_layers,dropout=self.dropout, user_stats_dim=None, time_decay=None, window=None,
                                weighted_method=self.weighted_method)
        elif self.model_use == 'SASRec':
            return SASRec(num_items=self.num_items,embd_dim=self.item_embd_dim,n_layers=self.n_layers,dropout=self.dropout)
        elif self.model_use == 'STMP':
            return STMP(num_items=self.num_items, embd_dim=self.item_embd_dim)

    def loss_fn(self, h, pos_idxs, neg_idxs, c_u=None, v=None):

        if c_u is not None and v is not None:
            pos_scores = self.model.score_partial(h, pos_idxs, c_u, v)
            neg_scores = self.model.score_partial(h, neg_idxs, c_u, v)
        elif self.model_use == 'STMP':
            hs, ht = h
            pos_scores = self.model.score_partial(hs, ht, pos_idxs)
            neg_scores = self.model.score_partial(hs, ht, neg_idxs)
        else:
            pos_scores = self.model.score_partial(h, pos_idxs)
            neg_scores = self.model.score_partial(h, neg_idxs)

        if neg_scores.dim() == 1:
            neg_scores = neg_scores.unsqueeze(1)


        losses =  []

        if 'bpr' in self.loss:
            diff = pos_scores.unsqueeze(1) - neg_scores
            loss_bpr = -F.logsigmoid(diff).mean()
            losses.append(loss_bpr)

        if 'ce' in self.loss:

            if self.model_use == 'STMP':
                pos_labels = torch.ones_like(pos_scores, device=pos_scores.device)
                neg_labels = torch.zeros_like(neg_scores, device=neg_scores.device)

                all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # [B, 1+N_neg]
                all_labels = torch.cat([pos_labels.unsqueeze(1), neg_labels], dim=1)  # [B, 1+N_neg]

                loss_ce = F.binary_cross_entropy(all_scores, all_labels).mean()

            else:
                logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
                labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

                loss_ce = F.cross_entropy(logits, labels).mean()

            losses.append(loss_ce)


        loss = sum(losses)

        return loss
    
    def BPRLoss(self, h, pos_idxs, neg_idxs):
        pass

    def train_epoch(self, dataloader):

        user_idx = None
        prefix_rating = None
        self.model.train()

        for batch in tqdm(dataloader):

            if len(batch) == 3:
                prefix_seq, pos_idxs, neg_idxs = batch
            elif len(batch) == 5:
                user_idx, prefix_seq, prefix_rating, pos_idxs, neg_idxs = batch

            prefix_seq = prefix_seq.long().to(self.device)
            pos_idxs = pos_idxs.long().to(self.device)
            neg_idxs = neg_idxs.long().to(self.device)

            if user_idx is not None:
                user_idx = user_idx.long().to(self.device)
            if prefix_rating is not None:
                prefix_rating = prefix_rating.long().to(self.device)

            # B = prefix_seq.size(0)

            # forward model
            if self.model_use == 'GatedGRU4Rec':
                h, c_u, v = self.model(prefix_seq, prefix_rating, user_idx)
                loss = self.loss_fn(h, pos_idxs, neg_idxs, c_u=c_u, v=v)
            elif self.model_use == 'RatingGRU4Rec':
                h = self.model(prefix_seq, prefix_rating)
                loss = self.loss_fn(h, pos_idxs, neg_idxs)
            else:
                h = self.model(prefix_seq)
                loss = self.loss_fn(h, pos_idxs, neg_idxs)

            # loss backward
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),1, 2)
            self.optimizer.step()

    def test_epoch(self, dataloader):
        self.model.eval()
        losses = []
        ranks_list = []
        # true_list = []
        # pred_list = []
        c_u = None
        v = None

        with torch.no_grad():
            for batch in tqdm(dataloader):

                if len(batch) == 3:
                    prefix_seq, pos_idxs, neg_idxs = batch
                elif len(batch) == 5:
                    user_idx, prefix_seq, prefix_rating, pos_idxs, neg_idxs = batch


                prefix_seq = prefix_seq.long().to(self.device)
                pos_idxs = pos_idxs.long().to(self.device)  # (B,)
                neg_idxs = neg_idxs.long().to(self.device)  # (B, N)

                if user_idx is not None:
                    user_idx = user_idx.long().to(self.device)
                if prefix_rating is not None:
                    prefix_rating = prefix_rating.long().to(self.device)

                # forward model
                if self.model_use == 'GatedGRU4Rec':
                    h, c_u, v = self.model(prefix_seq, prefix_rating, user_idx)
                    loss = self.loss_fn(h, pos_idxs, neg_idxs, c_u=c_u, v=v)
                elif self.model_use == 'RatingGRU4Rec':
                    h = self.model(prefix_seq, prefix_rating)
                    loss = self.loss_fn(h, pos_idxs, neg_idxs)
                else:
                    h = self.model(prefix_seq)
                    loss = self.loss_fn(h, pos_idxs, neg_idxs)

                losses.append(loss.item())

                candidates = torch.cat([pos_idxs.unsqueeze(1), neg_idxs], dim=1)  # (B, 1+N)
                if self.model_use == 'GatedGRU4Rec':
                    candidate_scores = self.model.score_partial(h, candidates, c_u, v)
                elif self.model_use == 'STMP':
                    hs, ht = h
                    candidate_scores = self.model.score_partial(hs,ht,candidates)
                else:
                    candidate_scores = self.model.score_partial(h, candidates)

                _, sorted_indices = torch.sort(candidate_scores, dim=1, descending=True)
                ranks = (sorted_indices == 0).nonzero(as_tuple=False)[:, 1] + 1  # (B,) positive sample ranks
                ranks_list.append(ranks)

                # actual = pos_idxs
                # predicted = [candidates[i, sorted_indices[i, :self.K]] for i in range(candidates.shape[0])]
                # true_list.append(actual)
                # pred_list.append(predicted)


        ranks = torch.concat(ranks_list, dim=0).cpu().numpy()
        # true = torch.concat(true_list, dim=0).cpu().numpy()
        # pred = torch.concat(pred_list, dim=0).cpu().numpy()

        ndcg_list = []
        hit_ratio_list = []
        k_list = [5,10,15,20]
        for k in k_list:
            ndcg = ndcg_k(ranks,k)
            hit_ratio = hit_ratio_k(ranks,k)

            ndcg_list.append(ndcg)
            hit_ratio_list.append(hit_ratio)

        return {
            'loss': round(np.nanmean(losses).item(), 6),
            **{f'nDCG@{k}': round(ndcg,6) for k, ndcg in zip(k_list, ndcg_list)},
            **{f'HitRatio@{k}': round(hr,6) for k, hr in zip(k_list, hit_ratio_list)}
        }

    def fit(self, train_loader, valid_loader, n_epochs=10, early_stop=20, evals_result=dict()):

        best_score = -np.inf
        stop_steps = 0
        best_param = None
        best_epoch = -1
        evals_result["train"] = []
        evals_result["valid"] = []
        evals_result["time"] = []

        for step in range(n_epochs):

            logging.info(f"# {'-' * 96} #")
            logging.info(f"Epoch: {step}")
            st = time.time()

            self.train_epoch(train_loader)

            train_metrics = self.test_epoch(train_loader)
            val_metrics   = self.test_epoch(valid_loader)
            ed = time.time()

            logging.info(f"train: {train_metrics}")
            logging.info(f"valid: {val_metrics}")
            logging.info(f"time: {ed - st:.2f} s")

            evals_result["train"].append(train_metrics)
            evals_result["valid"].append(val_metrics)
            evals_result["time"].append(ed - st)

            curr_param = copy.deepcopy(self.model.state_dict())

            if val_metrics[f'nDCG@{self.K}'] > best_score:
                best_score = val_metrics[f'nDCG@{self.K}']
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(curr_param)
            else:
                stop_steps += 1
                if stop_steps >= early_stop:
                    logging.info("Early stopping.")
                    break

        logging.info(f"best score: {best_score:6f} @ {best_epoch}")
        self.model.load_state_dict(best_param)

        evals_result['best_epoch'] = best_epoch
        self.evals_result = evals_result

        if self.device != 'cpu':
            torch.cuda.empty_cache()

    def predict(self, test_loader):

        self.model.eval()

        eval_test = self.test_epoch(test_loader)
        logging.info(f"test: {eval_test}")

        return eval_test

    def to_pickle(self, save_path):
        with open(save_path + 'rec_model.pkl', 'wb') as f:
            pickle.dump(self, f)







