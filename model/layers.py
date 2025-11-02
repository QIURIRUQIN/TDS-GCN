import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as df

def message_func(edges):
    w = edges.data['weight'].unsqueeze(-1)
    return {'m' : w * (edges.src['n_f'])}

class GCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 weight,
                 activation=None):
        super(GCNLayer, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.weight = weight
        
        if self.weight:
            self.u_w = nn.Linear(self.in_feats, self.out_feats)
            self.i_w = nn.Linear(self.in_feats, self.out_feats)
            self.reset_parameters()

        self.activation = activation

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.u_w.weight)
        nn.init.xavier_uniform_(self.i_w.weight)

    def forward(self, graph, u_f, i_f, e_f):
        with graph.local_scope():
            if self.weight:
                u_f = self.u_w(u_f)
                i_f = self.i_w(i_f)
            node_f = torch.cat([u_f, i_f], dim=0)

            degs = graph.out_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)

            node_f = node_f * norm

            graph.ndata['n_f'] = node_f
            # graph.edata['e_f'] = e_f
            graph.update_all(message_func=message_func, reduce_func=df.sum(msg='m', out='n_f'))

            rst = graph.ndata['n_f']

            degs = graph.in_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)

            rst = rst * norm

            if self.activation is not None:
                rst = self.activation(rst)

            return rst
