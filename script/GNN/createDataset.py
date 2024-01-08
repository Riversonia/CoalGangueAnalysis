import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from torch.cuda import device
from torch_geometric.data import Data, DataLoader

def main():
    strReadPath = "D:\CoalGangueCode\ReadDataset\\"
    strFileName = "sliceC000_1.csv"
    df = pd.read_csv(strReadPath + strFileName, header=None)

 #  构建sample
    x = torch.zeros(8, 128, dtype=torch.float)
    for num_nodes in range(0, 8):
        x[num_nodes, :] = torch.tensor(df.values[0, (num_nodes*127 + 3 + num_nodes): ((num_nodes+1)*127 + 3 + num_nodes + 1)], dtype=torch.float)
        # print((num_nodes*127 + 3 + num_nodes) ,((num_nodes+1)*127 + 3 + num_nodes + 1))
    # print(x)

    y = torch.tensor(df.values[0, 1], dtype=torch.float)
    print(y)

    edge_index = [[], []]
    edge_attr = []
    for i in range(0, 8):
        for j in range(i+1, 8):
            v1, v2 = x[:, i], x[:, j]
            corr = np.corrcoef(v1, v2)
            pCorrCoef = corr[0, 1] # pearson correlation coefficient matrix [[C(v1, v1) C(v1, v2)], [C(v2, v1) C(v2, v2)]]
            # print(corr)
            if pCorrCoef >= 0.4:
                edge_index[0].append(i)
                edge_index[1].append(j)
                edge_attr.append(pCorrCoef)

    graph = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

    # print(len(edge_attr))
    print(edge_index)
    # print(edge_attr)
    # print(graph)

 # sample 画图展示
 #    G = nx.graph
 #
 #    for i in edge_index:
 #        G.add_edge(edge_index[0, i], edge_index[1, i], weight=edge_attr[i])
 #    nx.draw(G, with_labels=True)
 #    plt.show()




def test():
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([7, 8, 9])

    print(torch.cat([a,b], dim=0))
    print(torch.cat([a,b], dim=-1))


main()
# test()


