import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.data import Data

features = torch.randn((4, 256))
print(features)
edge_index = [[], []]
print(edge_index)
edge_index[0].append(1)
edge_index[1].append(4)
edge_index[0].append(2)
edge_index[1].append(4)
edge_index[0].append(1)
edge_index[1].append(3)
print(edge_index)
edge_index = torch.LongTensor(edge_index)
print(edge_index)

"""
        x (torch.Tensor, optional): Node feature matrix with shape
            :obj:`[num_nodes, num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (torch.Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (torch.Tensor, optional): Graph-level or node-level ground-truth
            labels with arbitrary shape. (default: :obj:`None`)
        pos (torch.Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        **kwargs (optional): Additional attributes.
"""
graph = Data(x = features, edge_index = edge_index)
print(graph)
print(graph.x)
print(graph.y)
print(graph.num_features)

print(graph.num_edge_features)
print(graph.edge_index)