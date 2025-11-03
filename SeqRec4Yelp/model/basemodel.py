import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from model.layers import *


class GRU4Rec(nn.Module):
    def __init__(self, num_items, embd_dim=128, hidden_dim=128, n_layers=4, dropout=0.5):

        super().__init__()
        self.padding_idx = 0
        self.n_items = num_items
        self.embd_dim = embd_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.item_embedding = nn.Embedding(num_items+1, embd_dim, padding_idx=num_items)
        self.gru = nn.GRU(embd_dim, hidden_dim, num_layers=n_layers, batch_first=True,
                           dropout=dropout if n_layers>1 else 0)

        self.fc_out = nn.Linear(hidden_dim, embd_dim)

        # bias required
        # self.item_bias = nn.Parameter(torch.zeros(num_items), requires_grad=True)

    def forward(self, seq):
        length = (seq != self.padding_idx).long().sum(dim=1)
        embd = self.item_embedding(seq)
        packed = nn.utils.rnn.pack_padded_sequence(embd, length.cpu(), batch_first=True, enforce_sorted=False)
        x, h_n = self.gru(packed)

        hidden_last = h_n[-1] # (B, hidden_dim)
        out = self.fc_out(hidden_last) # (B, embd_dim)

        return out

    def score_full(self, x):
        """
        Score all items, which is time-consuming
        :param x: (B, embd_dim)
        :return:  (B, num_items)
        """
        W = self.item_embedding.weight # (num_items, embd_dim)
        scores =  torch.matmul(x, W.T) # (B, num_items)

        return scores

    def score_partial(self, x, candidates_idx):
        """
        Score a given set of samples
        :param x: (B, embd_dim)
        :param candidates_idx: （B,）or (B,N)
        :return: （B,）or (B, N)
        """
        W = self.item_embedding.weight  # (num_items, embd_dim)

        if candidates_idx.dim() == 1:
            candidate_embd = W[candidates_idx] # (B, embed_dim)
            scores = torch.sum(x * candidate_embd, dim=-1)
            return scores
        elif candidates_idx.dim() == 2:
            candidate_embd = W[candidates_idx]  # (B, m, embd_dim)
            scores = torch.sum(x.unsqueeze(1) * candidate_embd, dim=-1)
            return scores
          
          
          
class SASRec(nn.Module):
    def __init__(self,
                 num_items,
                 embd_dim=128,
                 n_layers=2,
                 n_heads=4,
                 max_len=20,
                 dropout=0.1,
                 padding_idx=100000):
         
        super().__init__()
        self.padding_idx = padding_idx
        self.n_items = num_items
        self.embd_dim = embd_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_len = max_len
        
        self.item_embedding = nn.Embedding(num_items+1, embd_dim, padding_idx=num_items)
        self.pos_embedding = nn.Embedding(max_len, embd_dim)
        
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.pos_embedding.weight)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embd_dim, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embd_dim, embd_dim)
    
    def _generate_causal_mask(self, L, device):
        mask = torch.triu(torch.ones((L, L), device=device), diagonal=1)  # 1 on future
        attn_mask = mask.masked_fill(mask == 1, float('-inf'))
        
        return attn_mask
        
    def forward(self, seq):
        device = seq.device
        B, L = seq.size()
        
        lengths = (seq != self.padding_idx).long().sum(dim=1)  # (B,)
        
        # embed items
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # (B, L)
        x = self.item_embedding(seq) + self.pos_embedding(pos_ids)  # (B, L, embd_dim)
        x = self.dropout(x)
        
        # masks
        attn_mask = self._generate_causal_mask(L, device)
        key_padding_mask = (seq == self.padding_idx) 
        
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        
        last_pos = (lengths - 1).long()
        out = x[torch.arange(B, device=device), last_pos]

        out = self.fc_out(out)
        return out
    
    def score_full(self, x):
        W = self.item_embedding.weight[1:]  # (num_items, embd_dim)
        scores = torch.matmul(x, W.T)
        return scores

    def score_partial(self, x, candidates_idx):
        W = self.item_embedding.weight  
        if candidates_idx.dim() == 1:
            # (B,)
            candidate_embd = W[candidates_idx]  # (B, embd_dim)
            scores = torch.sum(x * candidate_embd, dim=-1)
            return scores
        elif candidates_idx.dim() == 2:
            # (B, N)
            candidate_embd = W[candidates_idx]  # (B, N, embd_dim)
            scores = torch.sum(x.unsqueeze(1) * candidate_embd, dim=-1)  # (B, N)
            return scores
        else:
            raise ValueError("candidates_idx must be 1D or 2D tensor")
        
class GateMLP(nn.Module):
    def __init__(self, gate_in_dim, dropout=0.1):
        super().__init__()
        hidden_dim = gate_in_dim // 2
        self.ln = nn.LayerNorm(gate_in_dim)
        self.net = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU()
        )
        self.out = nn.Linear(hidden_dim//2,1)
    
    def forward(self, x):
        x = self.ln(x)
        logits = self.out(self.net(x))
        v = torch.sigmoid(logits)
        
        return v
        

class GatedGRU4Rec(nn.Module):
    def __init__(self,
                 num_items,
                 num_users=None,
                 embd_dim=128,
                 hidden_dim=128,
                 n_layers=4,
                 dropout=0.1,
                 user_embd_dim=64,
                 user_stats_dim=None,
                 time_decay = None,
                 window = None,
                 weighted_method = 'rating_avg'):
       
        super().__init__()
        self.padding_idx = num_items
        self.n_items = num_items
        self.embd_dim = embd_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.time_decay = time_decay
        self.window = window
        self.weighted_method = weighted_method


        self.item_embedding = nn.Embedding(num_items + 1, embd_dim, padding_idx=self.padding_idx)

        # rating -> small vector (assume rating_seq provided as scalar float)
        # self.rating_mlp = nn.Sequential(
        #     nn.Linear(1, rating_emb_dim),
        #     nn.ReLU(),
        #     nn.Linear(rating_emb_dim, rating_emb_dim)
        # )

        # GRU accepts item_emb，rating infos are used for average only
        self.gru = nn.GRU(embd_dim , hidden_dim,
                          num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0)

        # project last hidden to item space (same dim as item embedding)
        self.fc_out = nn.Linear(hidden_dim, embd_dim)

        # user embedding (optional but helpful for gate)
        self.num_users = num_users
        if num_users is not None:
            self.user_embedding = nn.Embedding(num_users, user_embd_dim)
        else:
            user_emb_dim = 0
            self.user_embedding = None

        # gate MLP: input = [user_emb, user_stats]
        gate_in_dim = (user_embd_dim if self.user_embedding is not None else 0) + (user_stats_dim if user_stats_dim is not None else 0)
        self.gate_mlp = GateMLP(gate_in_dim,dropout=0.2)

        # explore scale parameter (learnable)
        self.gamma = nn.Parameter(torch.tensor(1.0))

        # small epsilon for numerical stability
        self.eps = 1e-8

    def forward(self, seq, rating_seq, user_ids=None, user_stats=None, time_seq=None):
      
        assert rating_seq is not None, "rating_seq must be provided (shape B,T,1)"
        B,T = seq.size()
        length = (seq != self.padding_idx).long().sum(dim=1) #(B,T)

        # --------------
        # GRU Part
        # --------------
        item_embd = self.item_embedding(seq)  # (B, T, embd_dim)
        # pack for variable lengths
        packed = nn.utils.rnn.pack_padded_sequence(item_embd, length.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        # last hidden layer (take last GRU layer)
        hidden_last = h_n[-1]  # (num_layers? h_n shape: (num_layers, B, hidden_dim)) -> pick last layer
        exploit_vec = self.fc_out(hidden_last)  # (B, embd_dim)

        # -----------------compute rating-weighted centroid c_u----------------#
        # assume rating_seq shape (B,T,1); squeeze to (B,T)
        # ---------------------------------------------------------------------#
        rating_weights = rating_seq.squeeze(-1).clone()  # (B, T)
        # mask padding positions to zero weight
        mask = (seq != self.padding_idx).float()  # (B, T)
        rating_weights = rating_weights * mask
        simple_weights = torch.ones_like(rating_weights) * mask

        # apply window
        if self.window is not None and self.window > 0:
            pass

        if self.time_decay is not None and float(self.time_decay) > 0.0:
            # prepare delta: (B,T)
            if time_seq is not None:
                t  = time_seq.squeeze(-1) if time_seq.dim()==3 else time_seq
                t = t * mask
                # last time per sample
                last_idx = (length - 1).clamp(min=0)
                last_t = t.gather(1, last_idx.unsqueeze(1)).squeeze(1) # (B,)
                delta = (last_t.unsqueeze(1)-t) * mask

            else:
                # positional distance from last valid position
                positions = torch.arange(T, device=seq.device).unsqueeze(0).expand(B, T)
                last_pos = (length - 1).clamp(min=0).unsqueeze(1).expand(B, T)
                pos_dist = (last_pos - positions).float() * mask  # (B, T)
                delta = pos_dist

            decay = torch.exp(- float(self.time_decay) * delta)  # (B, T)
            rating_weights = rating_weights * decay

        if self.weighted_method == 'rating_avg':
            denorm = rating_weights.sum(dim=1, keepdim=True) + self.eps  # (B,1)
            # weighted sum of item embeddings
            weighted_item = item_embd * rating_weights.unsqueeze(-1)  # (B, T, embd_dim)
            c_u = weighted_item.sum(dim=1) / denorm  # (B, embd_dim)
        
        elif self.weighted_method == 'simple_avg':
            denorm = simple_weights.sum(dim=1, keepdim=True) + self.eps
            weighted_item = item_embd * simple_weights.unsqueeze(-1)
            c_u = weighted_item.sum(dim=1) / denorm
        
        # normalize centroid for cosine similarity use
        c_u = F.normalize(c_u, p=2, dim=-1)
        

        # --------------
        # User Gate Part
        # --------------
        gate_components = []

        if user_stats is not None:
            gate_components.append(user_stats)

        if self.user_embedding is not None:
            assert user_ids is not None, "user_ids must be provided when model was created with num_users"
            user_embd = self.user_embedding(user_ids)  # (B, user_emb_dim)
            gate_components.append(user_embd)
        assert len(gate_components) > 0

        gate_in = torch.cat(gate_components,dim=-1)
        v = self.gate_mlp(gate_in).squeeze(-1)  # (B,) in (0,1)

        return exploit_vec, c_u, v

    def score_partial(self, exploit_vec, candidates_idx, c_u, v):
        
        W = self.item_embedding.weight  # (num_items+1, embd_dim)
        
        if candidates_idx.dim() == 1:
            candidate_embd = W[candidates_idx]  # (B, embd_dim)
            # exploit scores
            exploit_norm = F.normalize(exploit_vec, p=2, dim=-1)
            cand_norm = F.normalize(candidate_embd, p=2, dim=-1)
            exploit_scores = torch.sum(exploit_norm * cand_norm, dim=-1)
           
            # cosine sim between cand and c_u: need normalize cand
            cos_sim = torch.sum(cand_norm * c_u, dim=-1)  # (B,)
            explore_scores = - self.gamma * cos_sim  # (B,)
            v = v.view(-1)
            final = (1 - v) * exploit_scores + v * explore_scores

            return final

        elif candidates_idx.dim() == 2:
            # (B, N)
            candidate_embd = W[candidates_idx]  # (B, N, embd_dim)
            exploit_norm = F.normalize(exploit_vec,p=2,dim=-1).unsqueeze(1)
            cand_norm = F.normalize(candidate_embd, p=2, dim=-1)
            exploit_scores = torch.sum(exploit_norm * cand_norm, dim=-1)
            
            # compute cos_sim via bmm: (B, N, embd) dot (B, embd, 1) -> (B, N, 1)
            cos_sim = torch.bmm(cand_norm, c_u.unsqueeze(-1)).squeeze(-1)  # (B, N)
            explore_scores = - self.gamma * cos_sim  # (B, N)
            v = v.unsqueeze(-1)  # (B, 1)
            final = (1 - v) * exploit_scores + v * explore_scores

            return final


class RatingGRU4Rec(nn.Module):

    def __init__(self,
                 num_items,
                 embd_dim=128,
                 hidden_dim=128,
                 n_layers=1,
                 dropout=0.0,
                 time_decay=None,
                 window=None):
        super().__init__()
        self.padding_idx = 10000000
        self.num_items = num_items
        self.embd_dim = embd_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.time_decay = time_decay
        self.c_window = window
        self.eps = 1e-8

        # item embedding: index 0 reserved for padding
        self.item_embedding = nn.Embedding(num_items + 1, embd_dim, padding_idx=num_items)

        # GRU over item embeddings only
        self.gru = nn.GRU(embd_dim, hidden_dim, num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0)

        # MLP: concat(last_hidden, c_u) -> proj (same dim as embd_dim for dot scoring)
        hidden = max(128, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + embd_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embd_dim)   # project into item embedding space
        )

        # optional scaling on final dot product (learnable)
        # self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, seq, rating_seq, time_seq=None):

        assert rating_seq is not None, "rating_seq must be provided (B, T, 1)"
        B, T = seq.size()
        device = seq.device

        # seq lengths
        lengths = (seq != self.padding_idx).long().sum(dim=1)  # (B,)

        # 1) GRU path (item embeddings only)
        item_e = self.item_embedding(seq)  # (B, T, embd_dim)
        packed = nn.utils.rnn.pack_padded_sequence(item_e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        last_h = h_n[-1]  # (B, hidden_dim)

        # 2) compute rating-weighted centroid c_u
        rating_w = rating_seq.squeeze(-1).clone()   # (B, T)
        mask = (seq != self.padding_idx).float()    # (B, T)
        rating_w = rating_w * mask

        # optional window: keep only last W valid interactions
        if self.c_window is not None and self.c_window > 0:
            positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            thresholds = (lengths - self.c_window).clamp(min=0).unsqueeze(1).expand(B, T)
            window_mask = (positions >= thresholds).float() * mask
            rating_w = rating_w * window_mask

        # optional time decay
        if self.time_decay is not None and float(self.time_decay) > 0.0:
            if time_seq is not None:
                t = time_seq.squeeze(-1) if time_seq.dim() == 3 else time_seq
                t = t * mask
                last_idx = (lengths - 1).clamp(min=0)
                last_t = t.gather(1, last_idx.unsqueeze(1)).squeeze(1)
                delta = (last_t.unsqueeze(1) - t) * mask
            else:
                positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
                last_pos = (lengths - 1).clamp(min=0).unsqueeze(1).expand(B, T)
                delta = (last_pos - positions).float() * mask
            decay = torch.exp(- float(self.time_decay) * delta)
            rating_w = rating_w * decay

        denom = rating_w.sum(dim=1, keepdim=True) + self.eps   # (B,1)
        weighted_item = item_e * rating_w.unsqueeze(-1)       # (B, T, embd)
        c_u = weighted_item.sum(dim=1) / denom                # (B, embd)

        # fallback for users with almost zero denom
        zero_mask = (denom.squeeze(-1) <= self.eps * 10).float().unsqueeze(-1)
        if zero_mask.sum() > 0:
            with torch.no_grad():
                global_mean = self.item_embedding.weight[1:].mean(dim=0, keepdim=True)  # (1, embd)
            c_u = c_u * (1 - zero_mask) + global_mean * zero_mask

        # (optional) normalize c_u if you later want cosine use
        c_u = F.normalize(c_u, p=2, dim=-1)

        # 3) concat and project
        cat = torch.cat([last_h, c_u], dim=-1)  # (B, hidden_dim + embd)
        proj = self.mlp(cat)                    # (B, embd_dim)

        return proj

    def score_partial(self, proj, candidates_idx):

        W = self.item_embedding.weight  # (num_items+1, embd)
        if candidates_idx.dim() == 1:
            cand_emb = W[candidates_idx]  # (B, embd)
            scores = torch.sum(proj * cand_emb, dim=-1)
            return scores
        else:
            # (B, N, embd)
            cand_emb = W[candidates_idx]
            scores = torch.sum(cand_emb * proj.unsqueeze(1), dim=-1)
            return scores

class STMP(nn.Module):
    
    def __init__(self, num_items, embd_dim):
        super().__init__()
        self.num_items = num_items
        self.embd_dim = embd_dim
        self.padding_idx = num_items
        
        self.item_embedding = nn.Embedding(num_items+1, embd_dim, padding_idx=num_items)
        
        self.mlp_1 = nn.Sequential(
            nn.Linear(embd_dim, embd_dim//2, bias=True),
            nn.GELU(),
            nn.Linear(embd_dim//2, embd_dim, bias=True),
            nn.Dropout(0.1))
        
        self.mlp_2 = nn.Sequential(
            nn.Linear(embd_dim, embd_dim//2, bias=True),
            nn.GELU(),
            nn.Linear(embd_dim//2, embd_dim, bias=True),
            nn.Dropout(0.1))
        
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
    
    def forward(self, seq, rating=None,alpha=0.1):
    
        item_embd = self.item_embedding(seq)
        lengths = (seq!= self.padding_idx).sum(dim=1)
        mask = seq!=self.padding_idx
        mask_3 = mask.unsqueeze(-1)
        
        # 取每个batch的最后一步
        item_embd_last = item_embd[torch.arange(item_embd.shape[0], device=seq.device), lengths-1, :]
        
        # 等权平均
        item_embd_masked = item_embd.masked_fill(~mask_3,0)
        item_embd_avg = torch.sum(item_embd_masked, dim=1) / torch.sum(mask.float(), dim=1, keepdim=True)
        
        if alpha != 0:
            B, L, D = item_embd.shape
            positions = torch.arange(L, device=item_embd.device)
            weights = torch.exp(alpha * positions)
            
            batch_weights = weights.unsqueeze(0).repeat(B, 1)
            batch_weights = batch_weights * mask.float()
            
            weights_sum = batch_weights.sum(dim=1, keepdim=True) + 1e-9
            normalized_weights = (batch_weights / weights_sum).unsqueeze(-1)
            
            item_embd_ema = torch.sum(item_embd * normalized_weights, dim=1)
        
        if rating is not None:
            rating = rating.masked_fill(~mask, 0) # (B, L)
            item_embd_rtavg = torch.sum(item_embd_masked * rating.unsqueeze(-1), dim=1) / (torch.sum(rating, dim=1,keepdim=True)+1e-9)
        
        hs = self.tanh1(self.mlp_1(item_embd_avg))
        ht = self.tanh2(self.mlp_2(item_embd_last))
        
        return hs, ht
    
    def score_partial(self, hs, ht, candidates_idx):
        W = self.item_embedding.weight
        if candidates_idx.dim() == 1:
            candidate_embd = W[candidates_idx]
            ht_hadamard_candidate = ht * candidate_embd
            trilinear = (hs * ht_hadamard_candidate).sum(dim=-1)
            z = torch.sigmoid(trilinear)

            return z

        elif candidates_idx.dim() == 2:
            candidate_embd = W[candidates_idx]
            hs_expanded = hs.unsqueeze(1)
            ht_expanded = ht.unsqueeze(1)
            ht_hadamard_candidate = ht_expanded * candidate_embd
            trilinear = (hs_expanded * ht_hadamard_candidate).sum(dim=-1)
            z = torch.sigmoid(trilinear)
            y = torch.softmax(z, dim=1)
            
            return y
    
    # def score_partial(self, hs, ht, candidates_idx):
    #     W = self.item_embedding.weight
    #     candidate_embd = W[candidates_idx]
    #     B, D = hs.shape
    #     if candidate_embd.dim() == 2:
    #         candidate_embd = candidate_embd.unsqueeze(0).expand(B, -1, -1)
    #     elif candidate_embd.dim() == 3:
    #         if candidate_embd.shape[0] != B:
    #             if candidate_embd.shape[1] == B:
    #                 candidate_embd = candidate_embd.permute(1, 0, 2)
    #             else:
    #                 raise ValueError(f"candidate_embd batch dim mismatch {candidate_embd.shape} vs hs {hs.shape}")
    #     else:
    #         raise ValueError(f"unexpected candidate_embd dim {candidate_embd.dim()}")
    #     hs_exp = hs.unsqueeze(1)
    #     ht_exp = ht.unsqueeze(1)
    #     z = (hs_exp * ht_exp * candidate_embd).sum(dim=-1)
    #     z_sig = torch.sigmoid(z)
    #     if candidates_idx.dim() == 2:
    #         return torch.softmax(z_sig, dim=1)
    #     else:
    #         return z_sig.ravel()

        
            
            
            
        
        
                

