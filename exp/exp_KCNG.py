import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import dgl
import numpy as np
from typing import Literal
import pandas as pd
import pickle
import scipy.sparse as sp
from torch.utils.data import dataloader
import gc

from exp.exp_basic import Exp_Basic
from utils.get_data_tools import get_raw_data_from_file
from utils.subgraph_tools import buildSubGraph
from data_loader.BPRData import MyDataset
from utils.metrics import hit, ndcg
from CONST.path_const import root_path

root_path = os.path.join(root_path, 'data')
class Exp_KGCN(Exp_Basic):
    def __init__(self, args, isLoad=True):
        super(Exp_KGCN, self).__init__(args)
        self.args = args
        self.learning_rate = self.args.learning_rate
        self.curEpoch = 0
        self.isLoadModel = isLoad

        self.train_loss = []
        self.his_hr = []
        self.his_ndcg = []
        
        self.datasetDir = self.args.datasetPath
        
        trainMat = get_raw_data_from_file('train_iter_class', suffix='.pkl')
        if self.args.use_multi_label:
            trainMat_p_n = get_raw_data_from_file('multi_label_graph_A_KGCN', suffix='.pkl')
            train_adj_time = get_raw_data_from_file('multi_label_graph_B_KGCN', suffix='.pkl')
        else:
            trainMat_p_n = get_raw_data_from_file('multi_graph_A', suffix='.pkl')
            train_adj_time = get_raw_data_from_file('multi_graph_B', suffix='.pkl')

        validData = get_raw_data_from_file('val_data', suffix='.pkl')
        uuMat = get_raw_data_from_file('uu_graph', suffix='.pkl')
        iiMat = get_raw_data_from_file('ii_graph', suffix='.pkl')

        self.n_users, self.n_items = trainMat.shape

        self.trainMat = trainMat
        self.uu_graph, self.ii_graph = self.create_dgi(uuMat, iiMat)

        self.uu_node_subGraph, self.uu_subGraph_adj, self.uu_dgi_node = self.create_sub_graph(fileClass='uu', Mat=uuMat)
        self.ii_node_subGraph, self.ii_subGraph_adj, self.ii_dgi_node = self.create_sub_graph(fileClass='ii', Mat=iiMat)

        self.uu_subGraph_adj_tensor = self.sparse_mx_to_torch_sparse_tensor(self.uu_subGraph_adj).cuda()
        self.uu_subGraph_adj_norm = torch.from_numpy(np.sum(self.uu_subGraph_adj, axis=1)).float().cuda()
        self.ii_subGraph_adj_tensor = self.sparse_mx_to_torch_sparse_tensor(self.ii_subGraph_adj).cuda()
        self.ii_subGraph_adj_norm = torch.from_numpy(np.sum(self.ii_subGraph_adj, axis=1)).float().cuda()

        self.uu_dgi_node = list(self.uu_dgi_node[0])
        self.ii_dgi_node = list(self.ii_dgi_node[0])

        self.uu_dgi_node_mask = np.zeros(self.n_users)
        self.uu_dgi_node_mask[self.uu_dgi_node] = 1
        self.uu_dgi_node_mask = torch.from_numpy(self.uu_dgi_node_mask).float().cuda()

        self.ii_dgi_node_mask = np.zeros(self.n_items)
        self.ii_dgi_node_mask[self.ii_dgi_node] = 1
        self.ii_dgi_node_mask = torch.from_numpy(self.ii_dgi_node_mask).float().cuda()

        self.rating_class = self.args.rating_class

        train_u, train_i = self.trainMat.nonzero()
        # assert np.sum(self.trainMat.data == 0) == 0

        train_data = np.hstack((train_u.reshape(-1, 1), train_i.reshape(-1, 1))).tolist()
        train_dataset = MyDataset(train_data, self.n_items, self.trainMat, self.args.n_NegSamples, True)
        self.train_loader = dataloader.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=0)

        valid_dataset = MyDataset(validData, self.n_items, self.trainMat, 0, False)
        self.val_loader = dataloader.DataLoader(valid_dataset, batch_size=self.args.batch_size * (self.args.n_NegSamples+1), shuffle=False, num_workers=0)

        self.process_timestep(train_adj_time, trainMat_p_n)
        self.initialize_model()
        gc.collect()

    @staticmethod
    def create_dgi(uuMat, iiMat):
        uuMat_edge_src, uuMat_edge_dst = uuMat.nonzero()
        uu_graph = dgl.graph(
            data=(uuMat_edge_src, uuMat_edge_dst),
            idtype=torch.int32,
            num_nodes=uuMat.shape[0],
            device=torch.device("cuda")
        )

        iiMat_edge_src, iiMat_edge_dst = iiMat.nonzero()
        ii_graph = dgl.graph(
            data=(iiMat_edge_src, iiMat_edge_dst),
            idtype=torch.int32,
            num_nodes=iiMat.shape[0],
            device=torch.device("cuda")
        )

        return uu_graph, ii_graph
    
    @staticmethod
    def sparse_mx_to_torch_sparse_tensor(sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)

        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)

        return torch.sparse_coo_tensor(indices, values, shape)

    def create_sub_graph(self, fileClass: Literal['uu', 'ii'], Mat):
        dataPath = os.path.join(root_path, f'{fileClass}Mat_subGraph_data.pkl')

        if self.args.clear == 1:
            if os.path.exists(dataPath):
                os.remove(dataPath)
                logging.info(f"Successfully removed the file: {dataPath}")

        if os.path.exists(dataPath):
            data = pd.read_pickle(dataPath)
            node_subGraph, subGraph_adj, dgi_node = data
        else:
            logging.info(f"Trying to create new subGraph")
            _, node_subGraph, subGraph_adj, dgi_node = buildSubGraph(Mat, self.args.subNode)
            data = (node_subGraph, subGraph_adj, dgi_node)
            with open(dataPath, 'wb') as f:
                pickle.dump(data, f)
        return node_subGraph, subGraph_adj, dgi_node

    def process_timestep(self, train_adj_time, train_p_n_class):
        logging.info('process timestep')
        self.time_step = self.args.time_step * 3600 * 24
        logging.info(f"timestep = {self.time_step:.1f} days")

        train_adj_time = train_adj_time.tocoo()
        row, col, data = train_adj_time.row, train_adj_time.col, train_adj_time.data
        assert np.sum(row == col) == 0

        minUTC = data.min()
        data = ((data - minUTC) / self.time_step).astype(np.int32) + 2
        train_adj_time_norm = sp.coo_matrix((data, (row, col)),
                                            dtype=np.int32,
                                            shape=train_adj_time.shape).tocsr()
        self.maxTime = train_adj_time_norm.max() + 1
        logging.info(f"maxTime = {self.maxTime:d}")

        num = train_adj_time_norm.shape[0]
        train_adj_time_norm = train_adj_time_norm + sp.eye(num, dtype=np.int32)

        train_p_n_class = train_p_n_class.tocsr()
        train_p_n_class = train_p_n_class + sp.eye(num, dtype=np.int32)

        coo_time = train_adj_time_norm.tocoo()
        edge_src, edge_dst, time_seq = coo_time.row, coo_time.col, coo_time.data

        weight_seq = train_p_n_class[edge_src, edge_dst].A1

        self.time_seq_tensor = torch.from_numpy(time_seq.astype(np.int32)).to('cuda')
        self.weight_tensor = torch.from_numpy(weight_seq.astype(np.float32)).to('cuda')
        self.uv_g = dgl.graph(
            (edge_src, edge_dst),
            num_nodes=num,
            idtype=torch.int64,
            device='cuda'
        )
        
        self.uv_g.edata['weight'] = self.weight_tensor

    def train(self):
        train_loader = self.train_loader
        logging.info('start negative sample...')
        train_loader.dataset.sample_ng()
        logging.info('finish sampling negative samples!')
        epoch_loss = 0
        epoch_uu_dgi_loss = 0
        epoch_ii_dgi_loss = 0
        for user, item_p, item_n in train_loader:
            user = user.long().cuda()
            item_p = item_p.long().cuda()
            item_n = item_n.long().cuda()

            if self.args.use_multi_label:
                item_embedding, user_embedding, _, all_user_embedding, all_item_embedding = self.model(
                    self.uv_g,
                    self.time_seq_tensor
                )
            else:
                _, user_embedding, item_embedding, all_user_embedding, all_item_embedding = self.model(
                    self.uv_g,
                    self.time_seq_tensor
                )
            # TODO: 这里可能需要使用 original embedding
            selected_user = user_embedding[user]
            selected_item_p = item_embedding[item_p]
            selected_item_n = item_embedding[item_n]

            pred_p = torch.sum(selected_user * selected_item_p, dim=1)
            pred_n = torch.sum(selected_user * selected_item_n, dim=1)

            bprloss = - F.logsigmoid(pred_p - pred_n).mean()
            regloss = (torch.norm(selected_user) ** 2 
                       + torch.norm(selected_item_p) ** 2 
                       + torch.norm(selected_item_n) ** 2) / self.args.batch_size

            loss = self.args.coef_bpr * bprloss + self.args.coef_reg * regloss

            uu_dgi_loss = 0
            ii_dgi_loss = 0
            if self.args.coef_uu != 0:
                uu_dgi_pos_loss, uu_dgi_neg_loss = self.uu_dgi(user_embedding, self.uu_subGraph_adj_tensor,
                                                               self.uu_subGraph_adj_norm, self.uu_node_subGraph)
                userMask = torch.zeros(self.n_users).cuda()
                userMask[user] = 1
                # user who are selected and in subGraph will be 1 otherwise 0
                userMask = userMask * self.uu_dgi_node_mask
                uu_dgi_loss = ((uu_dgi_pos_loss * userMask).sum() +
                               (uu_dgi_neg_loss * userMask).sum()) / torch.sum(userMask)
                epoch_uu_dgi_loss += uu_dgi_loss.item()
            
            if self.args.coef_ii != 0:
                ii_dgi_pos_loss, ii_dgi_neg_loss = self.ii_dgi(item_embedding, self.ii_subGraph_adj_tensor,
                                                               self.ii_subGraph_adj_norm, self.ii_node_subGraph)
                itemMask = torch.zeros(self.n_items).cuda()
                itemMask[item_p] = 1
                itemMask[item_n] = 1
                itemMask = itemMask * self.ii_dgi_node_mask
                ii_dgi_loss = ((ii_dgi_pos_loss * itemMask).sum() +
                               (ii_dgi_neg_loss * itemMask).sum()) / torch.sum(itemMask)
                epoch_ii_dgi_loss += ii_dgi_loss.item()

            loss += (self.args.coef_uu * uu_dgi_loss + self.args.coef_ii * ii_dgi_loss)

            if self.args.handle_over_corr:
                corr_loss_u, corr_loss_i = self.model.cal_corr_loss(all_user_embedding=all_user_embedding, all_item_embedding=all_item_embedding)
                
                loss += (0.005 * corr_loss_u + 0.005 * corr_loss_i)

            epoch_loss += bprloss.item()
            self.optimizers.zero_grad()
            loss.backward()
            self.optimizers.step()

        return epoch_loss, epoch_uu_dgi_loss, epoch_ii_dgi_loss

    def val(self, dataloader, is_save=True):
        hit_ratio, NDCG = [], []
        if self.args.use_multi_label:
                item_embedding, user_embedding, _, _, _ = self.model(
                    self.uv_g,
                    self.time_seq_tensor
                )
        else:
            _, user_embedding, item_embedding, _, _ = self.model(
                self.uv_g,
                self.time_seq_tensor
            )

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
        testData = pd.read_pickle(os.path.join(root_path, 'test_data.pkl'))
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
            epoch_loss, epoch_uu_dgi_loss, epoch_ii_dgi_loss = self.train()
            self.train_loss.append(epoch_loss)
            logging.info(f'Train: epoch {(e+1)}/{self.args.epochs}, epoch_loss={epoch_loss:.4f},' 
                         f'epoch_uu_dgi_loss={epoch_uu_dgi_loss:.4f}, epoch_ii_dgi_loss={epoch_ii_dgi_loss:.4f}')
            
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
        savePath = os.path.join(root_path, 'model', 'best_model.pth')
        params = {
            'model': self.model,
            }
        torch.save(params, savePath)
        logging.info(f'best model saved in {savePath}')

    def adjusting_learning_rate(self):
        for param_group in self.optimizers.param_groups:
            param_group['lr'] = max(param_group['lr'] * self.args.decay, self.args.min_lr)
