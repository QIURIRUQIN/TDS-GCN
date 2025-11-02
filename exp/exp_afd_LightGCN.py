import torch
import random
import numpy as np
import pandas as pd
import os
import pickle
import gc
from torch.utils.data import dataloader
import logging

from model.afd_lightgcn import AFD_LightGCN
from data_loader.BPRData import MyDataset
from CONST.path_const import root_path
from utils.metrics import hit, ndcg
from utils.get_data_tools import get_raw_data_from_file

class Exp_Afd_LightGCN():
    def __init__(self, args):
        # load dataset
        trainMat = get_raw_data_from_file('train_iter_class', suffix='.pkl')
        val_data = get_raw_data_from_file('val_data', suffix='.pkl')

        self.args = args
        self.n_users, self.n_items = trainMat.shape
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AFD_LightGCN(args=self.args, dataset=trainMat).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=0
        )

        self.train_loss = []
        self.his_hr = []
        self.his_ndcg = []

        # create dataloader
        self.trainMat = trainMat

        train_u, train_i = self.trainMat.nonzero()
        train_data = np.hstack((train_u.reshape(-1, 1), train_i.reshape(-1, 1))).tolist()
        train_dataset = MyDataset(train_data, self.n_items, self.trainMat, self.args.n_NegSamples, True)
        self.train_loader = dataloader.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=0)

        valid_dataset = MyDataset(val_data, self.n_items, self.trainMat, 0, False)
        self.val_loader = dataloader.DataLoader(valid_dataset, batch_size=self.args.batch_size * (self.args.n_NegSamples+1), shuffle=False, num_workers=0)

        self.setRandomSeed()
        gc.collect()

    def setRandomSeed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)

    def train(self):
        train_loader = self.train_loader
        logging.info('start negative sample...')
        train_loader.dataset.sample_ng()
        logging.info('finish sampling negative samples!')
        epoch_loss = 0

        for user, item_p, item_n in train_loader:
            user = user.long().cuda()
            item_p = item_p.long().cuda()
            item_n = item_n.long().cuda()

            bpr_loss, reg_loss, cor_loss_u, cor_loss_i = self.model.calculate_loss(user, item_p, item_n)

            loss = bpr_loss + self.args.coef_reg * reg_loss + 0.1 * cor_loss_u + 0.1 * cor_loss_i
            epoch_loss += bpr_loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return epoch_loss
    
    def val(self, dataloader):
        hit_ratio, NDCG = [], []
        user_embedding, item_embedding, _ = self.model()

        for user, item_p in dataloader:
            user = user.long().cuda()
            item_p = item_p.long().cuda()

            selected_user = user_embedding[user]
            selected_item_p = item_embedding[item_p]
            pred_p = torch.sum(torch.mul(selected_user, selected_item_p), dim=1)

            n = int(user.cpu().numpy().size / (self.args.n_NegSamples + 1))
            for i in range(n):
                a_user_scores = pred_p[i*(self.args.n_NegSamples + 1): (i+1)*(self.args.n_NegSamples + 1)]
                _, indices = torch.topk(a_user_scores, self.args.top_k)
                temp_item_p = item_p[i*(self.args.n_NegSamples + 1): (i+1)*(self.args.n_NegSamples + 1)]
                recommends = torch.take(temp_item_p, indices).cpu().numpy().tolist()
                target_item = temp_item_p[0].item()
                hit_ratio.append(hit(target_item, recommends))
                NDCG.append(ndcg(target_item, recommends))

        return hit_ratio, NDCG
    
    def test(self):
        testData = pd.read_pickle(os.path.join(root_path, 'data','test_data.pkl'))
        test_dataset = MyDataset(testData, self.n_items, self.trainMat, 0, False)
        self.test_dataloader = dataloader.DataLoader(test_dataset, batch_size=self.args.batch_size * (self.args.n_NegSamples + 1), shuffle=False, num_workers=0)

        hit_ratio, NCDG = self.val(self.test_dataloader)

        return hit_ratio, NCDG
    
    def run(self):
        cvWait = 0
        best_ndcg = 0.0
        best_hr = 0.0

        final_test_hr = 0
        final_test_ndcg = 0
        logging.info('**************************************')
        logging.info('start to train')
        for e in range(self.args.epochs):
            logging.info('**************************************')
            epoch_loss = self.train()
            self.train_loss.append(epoch_loss)
            logging.info(f'Train: epoch {(e+1)}/{self.args.epochs}, epoch_loss={epoch_loss:.4f}')
            
            hit_ratio_list, ndcg_list = self.val(self.val_loader)
            new_hr, new_ndcg = np.mean(hit_ratio_list), np.mean(ndcg_list)
            logging.info(f'Val: epoch {(e+1)}/{self.args.epochs}, hit_ration={new_hr:.4f},' 
                         f'ndcg={new_ndcg:.4f}')
            self.his_hr.append(new_hr)
            self.his_ndcg.append(new_ndcg)

            test_hr_list, test_ndcg_list = self.test()
            test_hr, test_ndcg = np.mean(test_hr_list), np.mean(test_ndcg_list)
            logging.info(f'Test: epoch {(e+1)}/{self.args.epochs}, hit_ration={test_hr:.4f},' 
                         f'ndcg={test_ndcg:.4f}')

            self.adjusting_learning_rate()

            if (new_ndcg > best_ndcg):
                best_ndcg = new_ndcg
                best_hr = new_hr
                cvWait = 0
                final_test_hr = test_hr
                final_test_ndcg = test_ndcg
                self.saveModel()
                logging.info(f'epoch {(e+1)}/{self.args.epochs}, saving model!')
            else:
                cvWait += 1
                logging.info(f'{cvWait} / {self.args.patience}')
            
            if cvWait >= self.args.patience:
                break

        logging.info('Train Ending!')
        logging.info('**************************************')
        logging.info(f'best performence of model: hr={final_test_hr:.4f}, ndcg={final_test_ndcg:.4f}')
    
    def saveModel(self):
        savePath = os.path.join(root_path, 'data', 'model', f'best_model_{self.args.model_name}.pth')
        params = {
            'model': self.model,
            }
        torch.save(params, savePath)
        logging.info(f'best model saved in {savePath}')

    def adjusting_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'] * self.args.decay, self.args.min_lr)
