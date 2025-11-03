import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalAttention(nn.Module):

    def __init__(self,embd_dim, n_head, dropout, block_size):
        super().__init__()
        assert embd_dim // n_head == 0
        self.attn = nn.Linear(embd_dim, 3 * embd_dim)
        self.out_proj == nn.Linear(embd_dim, embd_dim)

        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        self.n_head = n_head
        self.embd_dim = embd_dim
        self.dropout = dropout

        self.register_buffer("bias", torch.tril(torch.ones(block_size,block_size))
                              .view(1,1,block_size,block_size))
        
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.attn(x).split(self.n_head, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        y = self.resid_dropout(self.c_proj(y))
        
        return y 

class FFN(nn.Module):
     
     def __init__(self, embd_dim, dropout):
         super().__init__()
         self.fc_in = nn.Linear(embd_dim, 4*embd_dim)
         self.gelu = nn.GELU()
         self.fc_proj = nn.Linear(4 * embd_dim, embd_dim)
         self.dropout = nn.Dropout(dropout)
        
     def forward(self, x):
         x = self.fc_in(x)
         x = self.gelu(x)
         x = self.fc_proj(x)
         x = self.dropout(x)

         return x


class Block(nn.Module):
      
      def __init__(self, embd_dim,n_head,block_size,dropout):
          super().__init__()
          self.ln_1 = nn.LayerNorm(embd_dim)
          self.attn = CausalAttention(embd_dim, n_head, dropout, block_size)
          self.ln_2 = nn.LayerNorm(embd_dim)
          self.ffn = FFN(embd_dim, dropout)
          
      def forward(self, x):
          x = x + self.attn(self.ln_1(x))
          x = x + self.ffn(self.ln_2(x))
          
          return x
      

class TransformerBlock(nn.Module):
    def __init__(self, embd_dim, n_heads=2, dropout=0.1, ff_hidden_mult=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embd_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(embd_dim)
        self.ff = nn.Sequential(
            nn.Linear(embd_dim, embd_dim * ff_hidden_mult),
            nn.GELU(),
            nn.Linear(embd_dim * ff_hidden_mult, embd_dim),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(embd_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # x: (B, L, embd_dim)
        # attn_mask: (L, L) with True/1 indicating masked positions (causal mask)
        # key_padding_mask: (B, L) with True for positions that should be masked (padding)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.ln1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x
          