import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np

from dgl.dataloading import DataLoader
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import GATConv
from tqdm import tqdm
from script.GNN import device

## 1 Create Graph
"""
    1
"""
def create_graph(num_nodes, data):
    features = torch.randn((num_nodes, 256))
    edge_index = [[], []]
    # data (x, num_nodes)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            x, y = data[:, i], data[:, j]
            corr = np.corrcoef(x, y)
            if corr >= 0.4:
                edge_index[0].append(i)
                edge_index[1].append(j)

    edge_index = torch.LongTensor(edge_index)
    graph = nx.Data(x = features, edge_index = edge_index)
    graph.edge_index = nx.to_undirected(graph.edge_index, num_nodes = num_nodes)

    return graph

## 2 Create Dataset
def nn_seq(num_nodes, seq_len, B, pred_step_size):
    # num_nodes 可能表示神经网络的节点数
    # seq_len   表示序列长度
    # B
    # pred_step_size

    data = pd.read_csv('data/data.csv')     # 读取csv数据集文件
    data.drop([data.columns[0]], axis=1, inplace=True)  # 删除第一列

    # split 将数据集拆分训练 验证 测试
    train = data[:int(len(data) * 0.6)]                     # 60%作为训练集
    val = data[int(len(data) * 0.6):int(len(data) * 0.8)]   # 20%作为验证集
    test = data[int(len(data) * 0.8):len(data)]             # 20%作为测试集

    # normalization
    scaler = MinMaxScaler()     # 创建了一个MinMaxScaler对象，用于将数据集标准化或归一化
    train = scaler.fit_transform(data[:int(len(data) * 0.8)].values)
    val = scaler.transform(val.values)
    test = scaler.transform(test.values)    # train val test 都是将数据集标准化处理

    graph = create_graph(num_nodes, data[:int(len(data) * 0.8)].values)     # 调用create_graph函数构建图

    def process(dataset, batch_size, step_size, shuffle):
        dataset = dataset.tolist()  # 将数据集转换为python列表list
        seq = []    # 创建一个空的序列
        for i in tqdm(range(0, len(dataset) - seq_len - pred_step_size, step_size)):

            # 前24个时刻的13个变量
            # 提取一系列的时间序列数据，并将数据存储在train_seq列表中
            train_seq = []                          # 初始化一个新的train_seq列表，存储训练序列
            for j in range(i, i + seq_len):         # 在循环为seq列表长度内遍历第i ~ i+seqlen组的数据
                x = []                              # 初始化一个空列表x
                for c in range(len(dataset[0])):    # 遍历dataset数据集中的dim1数据
                    x.append(dataset[j][c])         # 将dataset数据集中第j行的数据存储到列表x中
                train_seq.append(x)                 # 将遍历得到的第j行数据存入训练序列train_seq中

            # 下一时刻的13个变量
            train_labels = []                       # 创建一个train_labels的空列表，存储训练标签

            # 这是一个循环，它遍历数据集 dataset中每一列（或者说每一维）的长度。
            for j in range(len(dataset[0])):
                train_label = []     # 在这个循环中，它创建了一个新的列表train_label，用于存储当前列中的训练标签
                for k in range(i + seq_len, i + seq_len + pred_step_size):
                # pred_step_size设置预测步长大小，控制模型在预测过程中的步长大小，以优化模型的性能或稳定性。
                    train_label.append(dataset[k][j])
                train_labels.append(train_label)

            # tensor
            train_seq = torch.FloatTensor(train_seq)        # 将train_seq转化为torch.FloatTensor的类
            train_labels = torch.FloatTensor(train_labels)  # 将train_labels转化为torch.FloatTensor的类
            seq.append((train_seq, train_labels))           # 将训练数据集train_seq和训练标签train_labels添加到seq表中，
                                                            # 后续模型训练能够更加方便地访问和操作训练数据

        # seq = MyDataset(seq) # 自定义一个数据集类MyDataset
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)
        """
        # 使用DataLoader加载数据集，其中参数包括：
        # detaset = seq 指定要加载的数据集
        # batch_size = batch_size 指定每个批次（或者小批量） 包含的样本数量
        # shuffle = shuffle 是否在每次迭代的时候打乱数据
        # num_workers 数据加载的线程的数量，设置为0表示不需要使用额外的线程
        # drop_lasr = False 若为True 如果数据集大小不能够被批次大小整除，则会丢弃最后一批数据
        #                   若为False 如果设置为False，意味着会保留所有的数据
        """

        return seq  # 返回seq数据集，用于训练和测试

    Dtr = process(train, B, step_size=1, shuffle=True)
    # 训练集数据，step_size步长为1，shuffle=True随机打乱混洗
    Val = process(val, B, step_size=1, shuffle=True)
    # 验证集数据，step_size步长为1，shuffle=True随机打乱混洗
    Dte = process(test, B, step_size=pred_step_size, shuffle=False)
    # 测试集数据，pred_step_size预测步长为1，shuffle=False不需要随机打乱混洗

    return graph, Dtr, Val, Dte, scaler
    #* 返回图 训练集数据 验证集数据 测试集数据 *scaler对象*

## 3 construct GAT model (Graph Attention Neural Network)
# 3.1 搭建GAT模型
class GAT(torch.nn.Module): # 创建图注意力网络GAT模型

    def __init__(self, in_feats, h_feats, out_feats):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, heads=4, concat=False)
        self.conv2 = GATConv(h_feats, out_feats, heads=4, concat=False)
    """
    in_feats 输入特征
    h_feats  隐藏层特征
    out_feat 输出特征
    
    定义了两个卷积层conv1和conv2，每个卷积层有4个注意力头，不使用concat操作
    如果concat参数为False，意味着它们不执行任何操作，仅用于卷积和注意力计算。
    如果concat参数为True，则将输出注意力权重的通道与原始输入连接起来。
    这个特性可以帮助更准确地捕获输入的特征，并将其组合起来。
    """
    # concat operation
    """
    神经网络的concat操作通常指的是在神经网络架构中，
    将多个输入特征（或神经网络层的输出）按照特定的顺序进行拼接（concatenation）。
    
    在深度学习中，concatenation操作常用于处理数据大小不匹配的情况，
    通过将不同的输入维度连接起来，使得模型可以学习到更多的信息。
    
    这种操作对于扩展网络的宽度或加深网络（通过添加更多的隐藏层）非常有用。
    
    在某些情况下，例如在卷积神经网络（CNN）中，
    concatenation还可以用于合并不同空间位置的卷积响应，
    以提高模型的感知能力。
    
    需要注意的是，concatenation操作可能会增加模型的复杂性，
    并可能需要额外的计算资源来执行。
    """

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index)) # ELU 指数线性单元(Expon)激活函数
        x = self.conv2(x, edge_index)
        return x
    """
    forward
    模型的前向传播方法，用于处理输入数据和执行模型操作。
    在每次前向传播中，模型将根据其自己的内部状态以及它所处的图的属性来调整其表示。
    这里使用了激活函数 elu 用于非线性变换，这是卷积过程的一部分。
    """

# 3.2 搭建时间序列预测模型
class GAT_MLP(nn.Module):
    def __init__(self, args, graph): # 构造函数
        super(GAT_MLP, self).__init__() # 调用基类的构造函数，初始化
        self.args = args        # 保存传递来给模型的参数
        self.out_feats = 128    # 定义模型输出特征为128
        self.edge_index = graph.edge_index  # 保存传递来的图的边索引信息
        self.gat = GAT(in_feats=args.seq_len, h_feats=64, out_feats=self.out_feats)
            # 通过之前搭建好的GAT模型来捕获传递来的图的特征，传入相关参数
        self.fcs = nn.ModuleList()  # 创建了一个nn.ModuleList() 的对象，用于存储多个神经网络层
        self.graph = graph  # 创建一个图对象，用于存储传递来的图对象
        for k in range(args.input_size):    # 遍历传参中input_size大小的次数
            self.fcs.append(nn.Sequential(  # 每次添加(append)一个nn.Sequential对象到名称为fcs的nn.Modulelist中
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, args.output_size)
            ))
        # nn.Sequential模块
        """
        nn.Sequential是PyTorch中的一个模块，它允许你按照顺序堆叠多个模块，形成一个更复杂的模块。
        这个模块中包含了一个线性层（nn.Linear），一个ReLU激活函数（nn.ReLU），然后再是一个线性层（nn.Linear）。
        输入大小为128，输出大小为传入参数中的output_size大小。
        """
    def forward(self, x): # GCN的前向传播过程(forward pass)，使用了GAT(Attention)机制和FC(Fully Connected)层
        # x(batch_size, seq_len, input_size)
        x = x.permute(0, 2, 1) # 改变了输入数据的维度顺序，方便卷积操作 第一维保持不变，第二维放到第三位，第三维放到第二位

        # 1.gat
        # x(batch_size, input_size, out_feats)
        out = torch.zeros(x.shape[0], x.shape[1], self.out_feats).to(device)
        # 创建一个新的输出张量out，其形状与输入x相同，但所有元素都是零。这个张量用于存储经过网络处理后的结果。
        # x.shape0和x.shape1定义了新张量out的形状，且全部为0
        # self.out_feats 代表每个张量的特征的数量
        # .to(device) 代表将这个张量转移到指定的设备（device）上

        for k in range(x.shape[0]):
            self.graph.x = x[k, :, :]
            out[k, :, :] = self.gat(x[k, :, :], self.edge_index)

        """
        使用循环对每个输入x[k]执行GAT层和FC层的操作。
        self.graph.x保存当前输入x[k]，self.edge_index表示图的边索引，这两个参数是用于GAT层的关键。
        """
        preds = []
        # print(out.shape)  # 256 13 128
        for k in range(out.shape[1]): # 使用循环对每个输出通道执行FC层，并将结果添加到preds列表中。
            preds.append(self.fcs[k](out[:, k, :]))

        pred = torch.stack(preds, dim=0) # 最后，使用torch.stack(preds, dim=0)将所有通道的结果堆叠在一起，得到最终的预测结果。
        # print(pred.shape)

        return pred #返回预测结果





