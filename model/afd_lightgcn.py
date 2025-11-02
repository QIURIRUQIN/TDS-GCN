import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.loss import calculate_correlation

class AFD_LightGCN(nn.Module):
    def __init__(self, args, dataset):
        super(AFD_LightGCN, self).__init__()
        # np.array sparse matrix
        self.interaction_matrix = dataset
        self.args = args

        self.n_users = self.interaction_matrix.shape[0]
        self.n_items = self.interaction_matrix.shape[1]

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.reg_coef = self.args.coef_reg

        self.norm_adj_matrix = self.get_norm_adj_mat().to('cuda')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.init_embedding()

    def init_embedding(self):
        self.user_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.hidden_dim)))
        self.item_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.hidden_dim)))

    def get_norm_adj_mat(self):
        zero_u = sp.csr_matrix((self.interaction_matrix.shape[0], self.interaction_matrix.shape[0]))
        zero_i = sp.csr_matrix((self.interaction_matrix.shape[1], self.interaction_matrix.shape[1]))
        A = sp.bmat([
            [zero_u, self.interaction_matrix],
            [self.interaction_matrix.T, zero_i]
        ], format='csr')

        degree = np.array(A.sum(axis=1)).flatten()  # shape: (num_users + num_items,)

        with np.errstate(divide='ignore'):  # 防止除以0
            d_inv_sqrt = np.power(degree, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0  # 无连接节点度为0 → 归一化设为0

        D_inv_sqrt = sp.diags(d_inv_sqrt)

        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
        # np.array -> torch.tensor
        A_coo = A_norm.tocoo().astype(np.float32)
        row = torch.from_numpy(A_coo.row).long()
        col = torch.from_numpy(A_coo.col).long()
        indices = torch.stack([row, col], dim=0)

        values = torch.from_numpy(A_coo.data).float()
        A_norm_torch = torch.sparse_coo_tensor(
            indices, values, torch.Size(A_coo.shape)
        )

        return A_norm_torch

    @staticmethod
    def BPRLoss(pos_score, neg_score):

        diff = pos_score - neg_score

        return -torch.mean(F.logsigmoid(diff))
    
    @staticmethod
    def RegLoss(user_embedding, p_item_embedding, n_item_embedding, batch_size):
        regloss = (torch.norm(user_embedding) ** 2 
                    + torch.norm(p_item_embedding) ** 2 
                    + torch.norm(n_item_embedding) ** 2) / batch_size
        return regloss
    
    def forward(self):
        all_embedding = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        all_embedding_list = [all_embedding]

        for _ in range(self.n_layers):
            all_embedding = torch.sparse.mm(self.norm_adj_matrix, all_embedding)
            all_embedding_list.append(all_embedding)
        
        lightgcn_all_embeddings = torch.stack(all_embedding_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )

        return user_all_embeddings, item_all_embeddings, all_embedding_list
    
    def calculate_loss(self, user_ids, pos_item_ids, neg_item_ids):

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()
        u_embeddings = user_all_embeddings[user_ids]
        pos_embeddings = item_all_embeddings[pos_item_ids]
        neg_embeddings = item_all_embeddings[neg_item_ids]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        bpr_loss = self.BPRLoss(pos_scores, neg_scores)

        # calculate Reg Loss
        u_ego_embeddings = self.user_embedding[user_ids]
        pos_ego_embeddings = self.item_embedding[pos_item_ids]
        neg_ego_embeddings = self.item_embedding[neg_item_ids]

        reg_loss = self.RegLoss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            batch_size=self.args.batch_size
        )

        cor_loss_u, cor_loss_i = torch.zeros((1,)).to(self.device), torch.zeros((1,)).to(self.device)

        user_layer_correlations = []
        item_layer_correlations = []
        for i in range(1, self.n_layers + 1):
            user_embeddings, item_embeddings = torch.split(embeddings_list[i], [self.n_users, self.n_items])
            user_layer_correlations.append(calculate_correlation(user_embeddings))
            item_layer_correlations.append(calculate_correlation(item_embeddings))

        user_layer_correlations_coef = (1 / torch.tensor(user_layer_correlations)) / torch.sum(
            1 / torch.tensor(user_layer_correlations))
        item_layer_correlations_coef = (1 / torch.tensor(item_layer_correlations)) / torch.sum(
            1 / torch.tensor(item_layer_correlations))

        for i in range(1, self.n_layers + 1):
            cor_loss_u += user_layer_correlations_coef[i - 1] * user_layer_correlations[i - 1]
            cor_loss_i += item_layer_correlations_coef[i - 1] * item_layer_correlations[i - 1]

        return bpr_loss, reg_loss, cor_loss_u, cor_loss_i
    
    def predict(self, user_ids, item_ids):
        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user_ids]
        i_embeddings = item_all_embeddings[item_ids]

        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

