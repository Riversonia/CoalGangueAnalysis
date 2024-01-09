import csv

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda import device
from torch_geometric.data import Data, DataLoader

def CreateDataset(readFile, CCfactor):   # 将文件readFile构建成为数据集

    # 读取指定文件file 第一行不做列名header=None (不添加header=None会默认第一行为行名称数据)
    df = pd.read_csv(readFile, header=None)

    # 初始化一个空数据集，接受生成的数据
    graph = []

    # 每行的数据格式为 content channel groupNumber feature*1024 (index from 0 ~ 1026)
    # 利用循环从文件中每行的第3个数据开始
    for row in range(0, 5):

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
        # 进度显示
        print(graph)
        print(readFile, row, "/2400 finished")

    # 最后函数将返回整理好的graph数据集
    # len(graph) = 2400
    # graph[i] = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
    return graph

    # print(len(edge_attr))
    # print(graph[0])
    # print(len(graph))
    # print(edge_index)
    # print(edge_attr)
    # print(graph)
def CreateDataset2(readFile, CCfactor):   # 将文件readFile构建成为数据集

    # 读取指定文件file 第一行不做列名header=None (不添加header=None会默认第一行为行名称数据)
    df = pd.read_csv(readFile, header=None)

    # 初始化一个空数据集，接受生成的数据
    graphSet = pd.DataFrame()
    header = ["x", "edge_index", "edge_attr", "y"]
    graphSet['header'] = header

    # 每行的数据格式为 content channel groupNumber feature*1024 (index from 0 ~ 1026)
    # 利用循环从文件中每行的第3个数据开始
    for row in range(0, 5):
        # 特征矩阵建立为x 8*128 8个节点，每个节点有128维的特征向量
        x = torch.zeros(8, 128, dtype=torch.float)

        for num_nodes in range(0, 8):
            x[num_nodes, :] = torch.tensor(df.values[row, (num_nodes*127 + 3 + num_nodes): ((num_nodes+1)*127 + 3 + num_nodes + 1)], dtype=torch.float)

        # 建立标签列y content含量列
        y = torch.tensor(df.values[row, 1], dtype=torch.float)
        # print(y)

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
        frame = pd.DataFrame[{ 'x': x, 'edge_index': edge_index,'edge_attr':edge_attr,'y': y}]
        graphSet.append(frame, ignore_index=True)
        # 进度显示
        print(frame)
        print(readFile, row, "/2400 finished")

    # 最后函数将返回整理好的graph数据集
    return graphSet

def main():
    strReadPath = "D:\CoalGangueCode\ReadDataset\\"
    strFileName = "sliceC000_1.csv"
    strSavePath = "D:\CoalGangueCode\SaveDataset\\"

    content = ["C000", "C025", "C050", "C075", "C100"]
    channel = ["_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8"]

    # print(content, channel)
    #
    # print(strFileName.replace(content[0],content[1]))
    # print(strFileName)

    df = pd.read_csv(strReadPath + strFileName, header=None)
    print(len(df))
    graph = CreateDataset(strReadPath + strFileName, 0.4)
    print(graph)

    graph = pd.DataFrame(graph)
    graph.to_csv(strSavePath + strFileName, index=False)

def test():

    strFileName = "sliceC000_1.csv"
    strSavePath = "D:\CoalGangueCode\SaveDataset\\"

    dataset = pd.read_csv(strSavePath + strFileName)
    print(len(dataset))
    print(dataset.iloc[0, 1])
    print(dataset.iloc[1, 1])
    print(dataset.iloc[2, 1])
    print(dataset.iloc[3, 1])
    print(dataset.iloc[4, 1])
    print(dataset.iloc[0, 2])
    print(dataset.iloc[1, 2])
    print(dataset.iloc[2, 2])
    print(dataset.iloc[3, 2])
    print(dataset.iloc[4, 2])

def Draw():
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    # strFileName = "sliceC000_1.csv"
    # strSavePath = "D:\CoalGangueCode\SaveDataset\\"
    # df = pd.read_csv(strSavePath + strFileName)

    # 假设这是你的数据
    import matplotlib.pyplot as plt
    import networkx as nx

    # 假设你已经有了一个图结构数据
    x = 8
    y = 10

    """
    edge_index = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [5, 6], [5, 7], [6, 7]] 
    edge_attr = [0.9419372762426698, 0.9130600864989608, 0.7930209913771828, 0.9878126395065165, 0.9049846923627836, 
                    0.9596334207687368, 0.4958566758702956, 0.7157760429816086, 0.9303637477061465, 0.889580827350303, 
                    0.9810356351800538]
                    
    edge_index = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [5, 6], [5, 7], [6, 7]]
    edge_attr = [0.9464319424492182, 0.8988417989433795, 0.7934420566202282, 0.9887073022024779, 0.9134521228918719, 
                    0.9543821883100962, 0.40427910673828304, 0.6506142436176181, 0.7961850895785598, 0.6434660785539529, 
                    0.9604100587504114]
                    
    edge_index = [[0, 1], [0, 3], [0, 5], [1, 2], [1, 3], [1, 6], [1, 7], [2, 3], [2, 6], [2, 7], [3, 4], [4, 5], [6, 7]]
    edge_attr = [0.6167566124110816, 0.42041026354669336, 0.6241909410196855, 0.9113056501925529, 0.4021688078785785, 
                    0.708059587361508, 0.6097398182782034, 0.593691480200076, 0.46931235450431186, 0.5257240727772813, 
                    0.5496126720378597, 0.6033790478782229, 0.8661806537935717]
                    
    edge_index = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [4, 5], [5, 6], [5, 7], [6, 7]]
    edge_attr = [0.9471492563620968, 0.6635347715765662, 0.8515851349680006, 0.44735887949189856, 0.835825573268858, 
                    0.49435549298494874, 0.8523183074316125, 0.5445930805755963, 0.9038580992555579, 0.8181364194051296, 
                    0.9342467670501787]
                    
    edge_index =  [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [3, 4], [4, 5], [5, 6], [5, 7], [6, 7]]
    edge_attr = [0.9970061706991251, 0.9599379689920652, 0.40206402700249577, 0.9646680230998222, 0.4085608429917729, 
                    0.6263495060578955, 0.7904729322963444, 0.7633856280888704, 0.8261956092156516, 0.6591123277203357, 
                    0.9477752907957924]

    
    
   
   
    """
    edge_index = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [3, 4], [4, 5], [5, 6], [5, 7], [6, 7]]
    edge_attr = [0.9970061706991251, 0.9599379689920652, 0.40206402700249577, 0.9646680230998222, 0.4085608429917729,
                 0.6263495060578955, 0.7904729322963444, 0.7633856280888704, 0.8261956092156516, 0.6591123277203357,
                 0.9477752907957924]


    # 使用NetworkX创建图对象
    G = nx.Graph()

    # 根据边索引添加边
    for i, j in edge_index:
        G.add_edge(i, j, weight=edge_attr[i])

    # 设置节点颜色和形状
    # pos = nx.spring_layout(G)  # 获取节点位置
    # nx.draw_networkx_nodes(G, pos, node_size=270, node_color='skyblue')  # 设置节点颜色为天蓝色
    # nx.draw_networkx_labels(G, pos, font_size=10)  # 设置节点标签字体大小为14
    # nx.draw_networkx_edges(G, pos, alpha=0.5)  # 设置边线透明度为50%
    #
    # # 显示图中的边权重
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=dict(zip(G.edges, edge_attr)), font_size=10)  # 设置边标签字体大小为10

    # 使用nx.draw()函数绘制图结构
    nx.draw(G, with_labels=True)
    plt.show()

# Draw()
main()
# test()


