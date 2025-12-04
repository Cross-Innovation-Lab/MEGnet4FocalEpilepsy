# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:25:19 2024
compute average group matrix
@author: Yafei
"""
import os
import scipy.io
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.utils import dense_to_sparse
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering
#from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, calinski_harabasz_score,davies_bouldin_score
from gensim.models import Word2Vec
#from node2vec import Node2Vec
import math
mat_folder_path='D:\epan\EP\\try1to9'
file_path = 'D:\\epan\\EP\BNT\\output_labels\\all_cluster_k5_try.txt' 
labels = np.loadtxt(file_path)
print(labels)
# 读取第一列作为标签
# 创建数据集
    
mat_file_paths = [os.path.join(mat_folder_path, f) for f in os.listdir(mat_folder_path) if f.endswith('.mat')]
graph_data_list = []
for mat_file_path in mat_file_paths:
    data = scipy.io.loadmat(mat_file_path)
    if 'R' in data:
        graph_data = data['R']
    else:
        print(f"Warning: 'R' not found in {mat_file_path}. Skipping...")
        continue
    graph_data = np.nan_to_num(graph_data, nan=0.0)
    graph_data_list.append(graph_data)
graph_data = np.stack(graph_data_list, axis=0)


train_dataset, val_test_dataset = train_test_split(graph_data, test_size=0.3, random_state=42)
val_dataset, test_dataset = train_test_split(val_test_dataset, test_size=0.5, random_state=42)
scipy.io.savemat("D:\\epan\\EP\\BNT\\output_matrix\\newtry_sub5.mat", {'test_dataset': test_dataset})
cluster_num = 5

for cluster_id in range(cluster_num):
    # 找到当前聚类的所有图的索引
    cluster_indices = np.where(labels == cluster_id)[0]
    
    # 确保当前聚类中有多于一个图
    if len(cluster_indices) > 1:
        # 初始化一个用于累加邻接矩阵的数组
        sum_adjacency = np.zeros((test_dataset[0].shape[0], test_dataset[0].shape[1]))
        
        # 累加当前聚类中所有图的邻接矩阵
        for idx in cluster_indices:
            sum_adjacency += test_dataset[idx]
        
        # 计算平均邻接矩阵
        representative_adjacency = sum_adjacency / len(cluster_indices)
        np.savetxt(f"D:\\epan\\EP\\BNT\\output_matrix\\newtry_sub{cluster_id}.txt", representative_adjacency, fmt='%.4f')
        # 绘制平均邻接矩阵
        plt.figure(figsize=(8, 6))
        plt.imshow(representative_adjacency, cmap='hot', interpolation='nearest')
        plt.title(f'Cluster {cluster_id} Average Adjacency Matrix')
        plt.colorbar()
        plt.show()
    else:
        print(f"Cluster {cluster_id} contains only one graph, using it as representative.")
        # 如果当前聚类中只有一个图，则直接使用该图的邻接矩阵
        representative_adjacency = test_dataset[cluster_indices[0]]
        
        # 绘制邻接矩阵
        plt.figure(figsize=(8, 6))
        plt.imshow(representative_adjacency, cmap='hot', interpolation='nearest')
        plt.title(f'Cluster {cluster_id} Representative Adjacency Matrix')
        plt.colorbar()
        plt.show()