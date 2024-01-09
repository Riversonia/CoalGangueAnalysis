import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data

def CreateDataset(dataset, readFile, CCfactor):   # 将文件readFile构建成为数据集

    # 读取指定文件file 第一行不做列名header=None (不添加header=None会默认第一行为行名称数据)
    df = pd.read_csv(readFile, header=None)

    # 初始化一个空数据集，接受生成的数据
    graph = dataset

    # 每行的数据格式为 content channel groupNumber feature*1024 (index from 0 ~ 1026)
    # 利用循环从文件中每行的第3个数据开始
 #  # 这里记得要改range(0, train_len)
    for row in range(0, 6):

        # 特征矩阵建立为x 8*128 8个节点，每个节点有128维的特征向量
        x = torch.zeros(8, 128, dtype=torch.float)

        for num_nodes in range(0, 8):
            x[num_nodes, :] = torch.tensor(df.values[row, (num_nodes*127 + 3 + num_nodes): ((num_nodes+1)*127 + 3 + num_nodes + 1)], dtype=torch.float)

        # 建立标签列y content含量列
        y = torch.tensor(df.values[row, 1], dtype=torch.float)


        edge_index = []
        edge_attr = []
        for i in range(0, 8):
            for j in range(i+1, 8):
                v1, v2 = x[:, i], x[:, j]

                # pearson correlation coefficient matrix [[C(v1, v1) C(v1, v2)], [C(v2, v1) C(v2, v2)]]
                corr = np.corrcoef(v1, v2)

                # 皮尔逊相关系数，np.corrcoef 生成了一个v1和v2的相关系数矩阵，形状如上，对角相等，取C(v1, v2)或C(V2, V1)
                pCorrCoef = corr[0, 1]
                # print(corr)

                # 如果某两个节点的特征向量v1, v2的相关系数大于相关系数CCfactor
                # 则会在两个节点建立边索引，并将相关系数pCorrCoef的值作为边权重赋值给edge_attr
                if pCorrCoef >= CCfactor:
                    edge_index.append([i, j])
                    edge_attr.append(pCorrCoef)

        # 将计算后的x, y, edge_index, edge_attr整理成为Data数据集，并附在graph后
        graph.append(Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr))

        # 画图函数
        # print(graph)

        # 进度显示
        print(readFile, row, "/2400 finished")

    # 最后函数将返回整理好的graph数据集
    # len(graph) = 2400
    # graph[i] = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
    # print(graph)
    return graph

    # print(len(edge_attr))
    # print(graph[0])
    # print(len(graph))
    # print(edge_index)
    # print(edge_attr)
    # print(graph)

def Draw(edge_index, edge_attr):
    import matplotlib.pyplot as plt
    import networkx as nx

    # 使用NetworkX创建图对象
    G = nx.Graph()

    # 根据边索引添加边
    for [i, j] in edge_index:
        G.add_edge(i, j)

        # 使用nx.draw()函数绘制图结构
    nx.draw(G, with_labels=True)
    plt.show()

def main():
    strReadPath = "D:\CoalGangueCode\ReadDataset\\"
    strFileName = "sliceC000_1.csv"
    strSavePath = "D:\CoalGangueCode\SaveDataset\\"

    content = ["C000", "C025", "C050", "C075", "C100"]
    channel = ["_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8"]

    print(content, channel)

    print(strFileName.replace(content[0],content[1]))
    print(strFileName)

    df = pd.read_csv(strReadPath + strFileName, header=None)
    print(len(df))
    graph = []

    # 这里记得启动制作数据集
    graph = CreateDataset(graph, strReadPath + strFileName, 0.4)
    # print(graph[0].x)
    # print(graph[1].edge_index)
    # print(graph[2].edge_attr)
    # print(graph[3].y)
    # print(graph[4])

# 构架模型
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from torch_geometric.nn import global_mean_pool

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

in_channel = 128
hidden_channel = 16
out_channel = 1

class GNN(nn.Module):
    # 初始化GNN模型
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(in_channel, hidden_channel)
        self.conv2 = GCNConv(hidden_channel, hidden_channel)
        self.out = nn.Linear(hidden_channel, out_channel)

        # 创建损失函数
        self.lossFunction = nn.MSELoss()

        # 创建优化器
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)

        # 模型训练次数计数器
        self.counter = 0

        # 模型训练损失值记录
        self.progress = []

    def forward(self, x, edge_index, edge_attr, y, batch):
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)

        x = self.conv1(x, edge_index, edge_attr, y)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr, y)
        x = x.relu()

        out = self.out(x)
        return out



# 程序入口
main()
# Draw(graph)


