import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np
from scipy.sparse import csr_matrix
import dgl
import scipy.sparse as sp

from model.layers import GCNLayer
from model.Embed import TimeEmbedding
from utils.loss import calculate_correlation

class TDSGCN(nn.Module):
    def __init__(self, args: dict, n_users: int, n_items: int, dims: List[int], maxTime: int):
        super(TDSGCN, self).__init__()
        self.args = args

        self.n_users = n_users
        self.n_items = n_items
        self.hidden_dim = args.hidden_dim
        self.dims = [self.hidden_dim] + dims

        if self.args.handle_over_corr:
            # initialize loss
            self.corr_loss_u, self.corr_loss_i = torch.zeros((1, ), dtype=torch.float32), torch.zeros((1, ), dtype=torch.float32)
        
        assert len(dims) == self.args.n_layers, "n_layers should be equal to len(dims)"
        self.activation = nn.LeakyReLU(negative_slope=args.slope)

        self.layers = nn.ModuleList(
            [
                GCNLayer(in_feats=self.dims[i], out_feats=self.dims[i+1], weight=self.args.weight, activation=self.activation) 
                for i in range(self.args.n_layers)
            ]
        )
        self.time_encoding = TimeEmbedding(self.hidden_dim, maxTime)
        self.init_embedding()
        self.init_interaction_weight()

    def init_embedding(self):
        self.user_embd = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.hidden_dim)))
        self.item_embd = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.hidden_dim)))

    def init_interaction_weight(self):
        self.w_p = nn.Parameter(torch.ones(self.n_users + self.n_items, self.n_users + self.n_items, dtype=torch.float32))
        self.w_n = nn.Parameter(torch.ones(self.n_users + self.n_items, self.n_users + self.n_items, dtype=torch.float32))

    def process_time(self, timestamp):
        self.time_step = self.args.time_step * 3600 * 24

        n_rows = timestamp.shape[0]
        row_max = np.zeros(n_rows)

        for i in range(n_rows):
            start, end = timestamp.indptr[i], timestamp.indptr[i+1]
            data_i = timestamp.data[start:end]
            if len(data_i) > 0:
                row_max[i] = np.max(data_i)
            else:
                row_max[i] = 0

        for i in range(n_rows):
            start, end = timestamp.indptr[i], timestamp.indptr[i+1]
            timestamp.data[start:end] = np.exp(
                -((row_max[i] - timestamp.data[start:end]) / self.time_step) * self.args.scaling_factor
            )

        timestamp = timestamp + sp.eye(timestamp.shape[0], dtype=timestamp.dtype)
        return timestamp
    
    @staticmethod
    def scipy_to_torch_sparse(csr_mat):
        coo = csr_mat.tocoo()
        indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
        values = torch.tensor(coo.data, dtype=torch.float32)
        shape = torch.Size(coo.shape)

        return torch.sparse_coo_tensor(indices, values, shape)

    def weighted_pos_neg_matix(self, pos_inter, neg_inter, pos_time, neg_time):
        # process interaction time
        A_p = pos_inter
        A_n = neg_inter
        T_p = self.process_time(pos_time)
        T_n = self.process_time(neg_time)

        A_p_time = A_p.multiply(T_p)
        A_n_time = A_n.multiply(T_n)
        # A_p_time = A_p
        # A_n_time = A_n

        A_p_torch = self.scipy_to_torch_sparse(A_p_time).to(self.w_p.device).coalesce()
        A_n_torch = self.scipy_to_torch_sparse(A_n_time).to(self.w_n.device).coalesce()

        idx_p = A_p_torch.indices()
        weighted_values_p = A_p_torch.values() * self.w_p[idx_p[0], idx_p[1]]
        time_adjusted_A_p_weighted = torch.sparse_coo_tensor(
            idx_p,
            weighted_values_p,
            size=A_p_torch.shape,
            dtype=torch.float32,
            device=self.w_p.device
        )

        idx_n = A_n_torch.indices()
        weighted_values_n = A_n_torch.values() * self.w_n[idx_n[0], idx_n[1]]
        time_adjusted_A_n_weighted = torch.sparse_coo_tensor(
            idx_n,
            weighted_values_n,
            size=A_n_torch.shape,
            dtype=torch.float32,
            device=self.w_n.device
        )

        return time_adjusted_A_p_weighted - time_adjusted_A_n_weighted
    
    def record_data(self, pos_inter, neg_inter, pos_time, neg_time):
        self.pos_inter = pos_inter
        self.pos_time = pos_time
        self.neg_inter = neg_inter
        self.neg_time = neg_time

    def create_ui_dgi(self):
        weight_edge = self.weighted_pos_neg_matix(pos_inter=self.pos_inter,
                                                  pos_time=self.pos_time,
                                                  neg_inter=self.neg_inter,
                                                  neg_time=self.neg_time)
        we = weight_edge.coalesce()

        src = we.indices()[0].to(self.w_p.device)
        dst = we.indices()[1].to(self.w_p.device)
        vals = we.values().to(self.w_p.device)

        self.uv_g = dgl.graph((src, dst), num_nodes=we.size(0), device=self.w_p.device)
        self.uv_g.edata['weight'] = vals

    def forward(self, time_seq):
        self.create_ui_dgi()
        all_user_embedding = [self.user_embd]
        all_item_embedding = [self.item_embd]
        if len(self.layers) == 0:
            return self.user_embd, self.item_embd
        
        edge_feat = self.time_encoding(time_seq)
        for i, layer in enumerate(self.layers):
            if i == 0:
                embeddings = layer(self.uv_g, self.user_embd, self.item_embd, edge_feat)
            else:
                embeddings = layer(self.uv_g, embeddings[:self.n_users], embeddings[self.n_users:], edge_feat)
            
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            all_user_embedding += [norm_embeddings[:self.n_users]]
            all_item_embedding += [norm_embeddings[self.n_users:]]
        
        user_embedding = torch.cat(all_user_embedding, dim=1)
        item_embedding = torch.cat(all_item_embedding, dim=1)
        # TODO: 可以加一个线性映射层，并结合 MoE，（可以随着时间的变化来进行 router，以加入动态时间信息）
        return user_embedding, item_embedding, all_user_embedding, all_item_embedding
    
    def cal_corr_loss(self, all_user_embedding, all_item_embedding):
        # deal with over-smoothing/over-correlation problem
        if self.args.handle_over_corr:
            user_layer_correlations = []
            item_layer_correlations = []
            for single_layer_user_embedding, single_layer_item_embedding in zip(all_user_embedding[1:], all_item_embedding[1:]):
                user_layer_correlations.append(calculate_correlation(single_layer_user_embedding))
                item_layer_correlations.append(calculate_correlation(single_layer_item_embedding))
            
            user_layer_correlations, item_layer_correlations = torch.tensor(user_layer_correlations), torch.tensor(item_layer_correlations)

            if self.args.loss_weight_method == 'HM': # 调和平均
                user_layer_correlations_coef = (1 / user_layer_correlations) / torch.sum(
                    1 / user_layer_correlations)
                item_layer_correlations_coef = (1 / item_layer_correlations) / torch.sum(
                    1 / item_layer_correlations)
                for i in range(len(user_layer_correlations)):
                    self.corr_loss_u += user_layer_correlations_coef[i] * user_layer_correlations[i]
                    self.corr_loss_i += item_layer_correlations_coef[i] * item_layer_correlations[i]
            
            elif self.args.loss_weight_method == 'SM': # 简单平均
                self.corr_loss_u += torch.mean(user_layer_correlations)
                self.corr_loss_i += torch.mean(item_layer_correlations)

            elif self.args.loss_weight_method == 'MS': # 均方
                for i in range(len(user_layer_correlations)):
                    self.corr_loss_u += user_layer_correlations[i] * user_layer_correlations[i]
                    self.corr_loss_i += item_layer_correlations[i] * item_layer_correlations[i]

                self.corr_loss_u /= len(user_layer_correlations)
                self.corr_loss_i /= len(item_layer_correlations)

        self.corr_loss_u = self.corr_loss_u.mean()
        self.corr_loss_i = self.corr_loss_i.mean()
        return self.corr_loss_u, self.corr_loss_i
