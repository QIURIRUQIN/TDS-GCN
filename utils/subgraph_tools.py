import networkx as nx
import scipy.sparse as sp
import torch
import numpy as np

def buildSubGraph(mat, subNode):
    if isinstance(mat, torch.Tensor):
        mat = mat.numpy()
    try:
        graph = nx.from_scipy_sparse_array(mat)
    except AttributeError:
        graph = nx.from_scipy_sparse_matrix(mat)

    
    subGraphList = list(nx.connected_components(graph))
    subGraphCount = len(subGraphList)
    node_num = mat.shape[0]

    nodeSubGraph = [-1 for _ in range(node_num)]
    adjMat = sp.dok_matrix((subGraphCount, node_num), dtype=np.int32)

    node_list = []
    for subGraphID in range(subGraphCount):
        subGraph = subGraphList[subGraphID]
        if len(subGraph) > subNode:
            node_list += list(subGraph)
        
        for node_id in subGraph:
            assert nodeSubGraph[node_id] == -1
            nodeSubGraph[node_id] = subGraphID
            adjMat[subGraphID, node_id] = 1

    nodeSubGraph = np.array(nodeSubGraph)
    assert np.sum(nodeSubGraph == -1) == 0
    adjMat = adjMat.tocsr()

    return subGraphList, nodeSubGraph, adjMat, node_list
