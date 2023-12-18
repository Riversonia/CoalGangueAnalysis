from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

# 1 load the dataset
dataset = MoleculeNet(root="data", name="ESOL")

print('num_features:',dataset.num_features)
print('num_classes:',dataset.num_classes)
print('num_node_features',dataset.num_node_features)
print("size:", len(dataset))

d=dataset[10]
print("Sample:", d)
print("Sample y:", d.y)
print("Sample num_nodes:",d.num_nodes)
print("Sample num_edges:",d.num_edges)

# 2 train data and test data
data_size = len(dataset)
batch_size = 128
train_data=dataset[:int(data_size*0.8)]
test_data=dataset[int(data_size*0.8):]

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=len(test_data))

# 3 contribution GCN model
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from torch_geometric.nn import global_mean_pool

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hidden_channels = 64


class GNN(nn.Module):

    def __init__(self):
        # 初始化Pytorch父类
        super().__init__()

        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.out = nn.Linear(hidden_channels, 1)

        # 创建损失函数，使用均方误差
        self.loss_function = nn.MSELoss()

        # 创建优化器，使用Adam梯度下降
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)

        # 训练次数计数器
        self.counter = 0
        # 训练过程中损失值记录
        self.progress = []

    # 前向传播函数
    def forward(self, x, edge_index, batch):

        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        x = x.relu()

        # 全局池化
        x = global_mean_pool(x, batch)  # [x, batch]

        out = self.out(x)
        return out

    # 训练函数
    def train(self, data):

        # 前向传播计算，获得网络输出
        outputs = self.forward(data.x.float(), data.edge_index, data.batch)

        # 计算损失值
        y = data.y.to(device)
        loss = self.loss_function(outputs, y)

        # 累加训练次数
        self.counter += 1

        # 每10次训练记录损失值
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())

        # 每1000次输出训练次数
        if (self.counter % 1000 == 0):
            print(f"counter={self.counter}, loss={loss.item()}")

        # 梯度清零, 反向传播, 更新权重
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    # 测试函数
    def test(self, data):
        # 前向传播计算，获得网络输出
        outputs = self.forward(data.x.float(), data.edge_index, data.batch)

        # 把绝对值误差小于1的视为正确，计算准确度
        y = data.y.to(device)
        acc = sum(torch.abs(y - outputs) < 1) / len(data.y)
        return acc

    # 绘制损失变化图
    def plot_progress(self):
        plt.plot(range(len(self.progress)), self.progress)

# 4 training the model
model = GNN()
model.to(device)

for i in range(1001):
    for data in train_loader:
        # print(data,'num_graphs:',data.num_graphs)
        model.train(data)

# 5 vision
model.plot_progress()

# 6 test model verify
torch.set_printoptions(precision=4,sci_mode=False) #pytorch不使用科学计数法显示
for data in test_loader:
    acc=model.test(data)
    print(acc)


