## 1 data feature output
from torch_geometric.datasets import KarateClub

dataset = KarateClub()
print(f'Dataset:{dataset}:')
print('=' * 30)
print(f'Number of graphs:{len(dataset)}')
print(f'Number of features:{dataset.num_features}')
print(f'Number of classes:{dataset.num_classes}')

print('=' * 30)
data = dataset[0]
# train_mask = [True,False,...] ：代表第1个点是有标签的，第2个点是没标签的，方便后面LOSS的计算
print(data)  # Data(x=[节点数, 特征数], edge_index=[2, 边的条数], y=[节点数], train_mask=[节点数])
print(data.edge_index)
print(data.train_mask)

## 2 Network virtual
import os
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt


# 画图函数
def visualize_graph(G, color):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()


# 画点函数
def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch:{epoch},Loss:{loss.item():.4f}', fontsize=16)
    plt.show()


if __name__ == '__main__':
	# 不加这个可能会报错
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    dataset = KarateClub()
    print(f'Dataset:{dataset}:')
    print('=' * 30)
    print(f'Number of graphs:{len(dataset)}')
    print(f'Number of features:{dataset.num_features}')
    print(f'Number of classes:{dataset.num_classes}')

    print('=' * 30)
    data = dataset[0]
    # train_mask = [True,False,...] ：代表第1个点是有标签的，第2个点是没标签的，方便后面LOSS的计算
    print(data)  # Data(x=[节点数, 特征数], edge_index=[2, 边的条数], y=[节点数], train_mask=[节点数])

    G = to_networkx(data, to_undirected=True)
    visualize_graph(G, color=data.y)

## 3 GCN Module Construction
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(520)
        self.num_features = num_features
        self.num_classes = num_classes
        self.conv1 = GCNConv(self.num_features, 4)  # 只定义子输入特证和输出特证即可
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, self.num_classes)

    def forward(self, x, edge_index):
        # 3层GCN
        h = self.convl(x, edge_index)  # 给入特征与邻接矩阵（注意格式，上面那种）
        h = h.tanh()
        h = self.conv2(h.edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()
        # 分类层
        out = self.classifier(h)
        return out, h

## 4 Using GCN module
import os
import time

from torch_geometric.datasets import KarateClub
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv


# 画图函数
def visualize_graph(G, color):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()


# 画点函数
def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch:{epoch},Loss:{loss.item():.4f}', fontsize=16)
    plt.show()


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(520)
        self.num_features = num_features
        self.num_classes = num_classes
        self.conv1 = GCNConv(self.num_features, 4)  # 只定义子输入特证和输出特证即可
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, self.num_classes)

    def forward(self, x, edge_index):
        # 3层GCN
        h = self.conv1(x, edge_index)  # 给入特征与邻接矩阵（注意格式，上面那种）
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()
        # 分类层
        out = self.classifier(h)
        return out, h


# 训练函数
def train(data):
    optimizer.zero_grad()
    out, h = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, h


if __name__ == '__main__':
    # 不加这个可能会报错
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # 数据集准备
    dataset = KarateClub()
    data = dataset[0]

    # 声明GCN模型
    model = GCN(dataset.num_features, dataset.num_classes)

    # 损失函数 交叉熵损失
    criterion = torch.nn.CrossEntropyLoss()
    # 优化器 Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 训练
    for epoch in range(401):
        loss, h = train(data)
        if epoch % 100 == 0:
            visualize_embedding(h, color=data.y, epoch=epoch, loss=loss)
            time.sleep(0.3)
