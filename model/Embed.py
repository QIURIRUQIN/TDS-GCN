import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int):
        super(TimeEmbedding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = 1 / (10000 ** (torch.arange(0., hidden_dim * 2, 2.)) / hidden_dim / 2)

        self.embedding = nn.Embedding(max_len, hidden_dim * 2)
        self.embedding.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(hidden_dim)
        self.embedding.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(hidden_dim)
        self.embedding.requires_grad_ = False

        # 0 is useless, 1 is self->self
        self.embedding.weight.data[0] = torch.zeros_like(self.embedding.weight.data[-1]) 
        self.embedding.weight.data[1] = torch.zeros_like(self.embedding.weight.data[-1])
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, time):
        return self.proj(self.embedding(time))
