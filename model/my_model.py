import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from model.layers import GCNLayer
from model.Embed import TimeEmbedding
from utils.loss import calculate_correlation

class Model(nn.Module):
    def __init__(self, args: dict, n_users: int, n_items: int, dims: List[int], maxTime: int):
        super(Model, self).__init__()
        self.args = args

        self.n_users = n_users
        self.n_items = n_items
        self.hidden_dim = args.hidden_dim
        self.dims = [self.hidden_dim] + dims
        self.use_multi_label = True if self.args.use_multi_label else False

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

    def init_embedding(self):
        self.user_embd = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.hidden_dim)))
        if self.use_multi_label:
            self.item_embd = nn.Parameter(nn.init.xavier_uniform_(torch.empty(3 * self.n_items, self.hidden_dim)))
        else:
            self.item_embd = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.hidden_dim)))

    def forward(self, graph, time_seq):
        all_user_embedding = [self.user_embd]
        all_item_embedding = [self.item_embd]
        if len(self.layers) == 0:
            return self.user_embd, self.item_embd
        
        edge_feat = self.time_encoding(time_seq)
        for i, layer in enumerate(self.layers):
            if i == 0:
                embeddings = layer(graph, self.user_embd, self.item_embd, edge_feat)
            else:
                embeddings = layer(graph, embeddings[:self.n_users], embeddings[self.n_users:], edge_feat)
            
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            all_user_embedding += [norm_embeddings[:self.n_users]]
            all_item_embedding += [norm_embeddings[self.n_users:]]
        
        user_embedding = torch.cat(all_user_embedding, dim=1)
        item_embedding = torch.cat(all_item_embedding, dim=1)
        # TODO: 可以加一个线性映射层，并结合 MoE，（可以随着时间的变化来进行 router，以加入动态时间信息）
        if self.use_multi_label:
            item_embedding_mean = item_embedding.contiguous().view(self.n_items, 3, self.hidden_dim * (len(self.layers) + 1))
            item_embedding_mean = torch.mean(item_embedding_mean, dim=1)
            return item_embedding_mean, user_embedding, item_embedding, all_user_embedding, all_item_embedding
        else:
            return [], user_embedding, item_embedding, all_user_embedding, all_item_embedding
    
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
