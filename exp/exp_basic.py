import torch
import torch.nn as nn
import random
import numpy as np

from model.my_model import Model
from model.KCGN import KCGN
from model.DGI import DGI
from model.TDSGCN import TDSGCN

model_name = {
    "my_model": Model,
    "KGCN": KCGN,
    "TDSGCN": TDSGCN
}

class Exp_Basic():
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.setRandomSeed()

    def setRandomSeed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
    
    def initialize_model(self):
        self.model = model_name[self.args.model_name](
            self.args, self.n_users, self.n_items, eval(self.args.dims), self.maxTime
        ).to(self.device)

        if self.args.dgi_graph_act == 'sigmoid':
            dgiGraphAct = nn.Sigmoid()
        elif self.args.dgi_graph_act == 'tanh':
            dgiGraphAct = nn.Tanh()

        self.uu_dgi = DGI(self.uu_graph, self.args.hidden_dim, self.args.hidden_dim, nn.PReLU(), dgiGraphAct).cuda()
        self.ii_dgi = DGI(self.ii_graph, self.args.hidden_dim, self.args.hidden_dim, nn.PReLU(), dgiGraphAct).cuda()

        self.optimizers = torch.optim.Adam([
            {'params': self.model.parameters(), 'weight_decay': 0},
            {'params': self.uu_dgi.parameters(), 'weight_decay': 0},
            {'params': self.ii_dgi.parameters(), 'weight_decay': 0}
        ], lr=self.learning_rate)
        