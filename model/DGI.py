import torch
import torch.nn as nn

import dgl.function as df
from dgl.base import DGLError

# model the relationship of user-user or item-item
class GraphConv(nn.Module):
    def __init__(self, in_feats, out_feats, weight=False, activation=None):
        super(GraphConv, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.weight = weight
        if weight:
            self.linear = nn.Linear(self.in_feats, self.out_feats)
            nn.init.xavier_uniform_(self.linear.weight)

        self.activation = activation

    def forward(self, graph, feat):
        graph = graph.local_var()

        degs = graph.out_degrees().to().float().clamp(min=1)
        norm = torch.pow(degs, -0.5).view(-1, 1)
        if self.weight:
            feat = self.linear(feat)

        feat = feat * norm
        graph.srcdata['h'] = feat
        graph.update_all(
            df.copy_src(src='h', out='m'),
            df.sum(msg='m', out='h')
        )
        rst = graph.dstdata['h']

        degs = graph.in_degrees().to(feat.device).float().clamp(min=1)
        norm = torch.pow(degs, -0.5).view(-1, 1)
        rst = norm * rst
        if self.activation is not None:
            rst = self.activation(rst)

        return rst

# class GCN(nn.Module):
#     def __init__(self, g, in_feats, out_feats, activation=None):
#         super(GCN, self).__init__()
#         self.g = g
#         self.conv_layer = GraphConv(in_feats=in_feats, out_feats=out_feats, weight=False, activation=activation)

#     def forward(self, features):
#         return self.conv_layer(self.g, features)
    
class Encoder(nn.Module):
    def __init__(self, g, in_feats, out_feats, activation=None):
        super(Encoder, self).__init__()
        self.g = g
        self.conv_layer = GraphConv(in_feats=in_feats, out_feats=out_feats, weight=False, activation=activation)

    def forward(self, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(self.g.number_of_nodes())
            features = features[perm]
        features = self.conv_layer(self.g, features)

        return features
    
class DGI(nn.Module):
    def __init__(self, g, in_feats, out_feats, gcnAct, graphAct):
        super(DGI, self).__init__()
        self.encoder = Encoder(g=g, in_feats=in_feats, out_feats=out_feats, activation=gcnAct)
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.graphAct = graphAct

    @staticmethod
    def calScore(node_embedding, graph_embedding):
        return torch.sum(node_embedding * graph_embedding, dim=1)
    
    def forward(self, features, subGraphAdj, subGraphNorm, nodeSubGraph):
        positive = self.encoder(features, corrupt=False)
        negative = self.encoder(features, corrupt=True)

        graphEmbeddings = torch.sparse.mm(subGraphAdj, positive) / subGraphNorm
        graphEmbeddings = self.graphAct(graphEmbeddings)

        summary = graphEmbeddings[nodeSubGraph]

        positive_score = self.calScore(positive, summary)
        negative_score = self.calScore(negative, summary)

        pos_loss = self.loss(positive_score, torch.ones_like(positive_score))
        neg_loss = self.loss(negative_score, torch.ones_like(negative_score))

        return pos_loss, neg_loss
