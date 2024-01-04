import csv

import networkx as nx
import numpy as np
import random
import pandas as pd

# 数据可视化
import matplotlib.pyplot as plt
import matplotlib as mpl

'''
# plt.rcParams['font.sans-serif'] = ['Times New Roman']    # 正常显示英文TimesNewRoman字体标签
plt.rcParams['font.sans-serif'] = ['SimHei']    # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False      # 正常显示正负号

plt.plot([1, 2, 3], [100, 500, 300])
plt.title('matplotlib中文测试', fontsize = 25)
plt.xlabel('X-axis', fontsize = 15)
plt.ylabel('Y-axis', fontsize = 15)
plt.show()
'''

# 导入三元组连接表
strPath = "D:\研0\组会\\20231220 煤矸数据裁剪处理完成 GNN模型确立\Coal gangue datset cropped and integrated\sliceData8_Graph\C000\Channel1\\"
strFileName = "sliceGraphTableC000_1_0001.csv"

print(pd.read_csv(strPath + strFileName))   # fileName.csv需要自己写

data_csv = pd.read_csv(strPath+strFileName, header = 0, names=['head', 'tail', 'weight', 'label', 'channel', 'group'])
data_csv.to_csv(strPath + strFileName)
# print('---------------------------------')
print(pd.read_csv(strPath + strFileName))


'''
Num     Head    Tail    Relation
0       Value1  Value2  Weight12
1       Value2  Value3  Weight23
2       Value3  Value4  Weight34
...     ...     ...     ...
n       ValueN  ValueM  WeightNM
'''

# 通过连接表Edge List创建图
df = pd.read_csv(strPath + strFileName)
G = nx.Graph()
edges = [edge for edge in zip(df['head'], df['tail'])]
G.add_edges_from(edges)
print(G)

# 可视化

pos = nx.spring_layout(G, seed = 123)
plt.figure(figsize=(15, 15))
nx.draw(G, pos=pos, with_labels=True)
plt.show()
