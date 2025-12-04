"""
Created on Fri Sep 20 16:20:18 2024
gcnae
gatae
node2vec
deepwalk
kmeans
spectral
pca 
tsne
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
# 数据集类
class MyGraphDataset(InMemoryDataset):
    def __init__(self, root, graph_data, feature, transform=None, pre_transform=None):
        self.graph_data = graph_data
        self.feature = feature
        super(MyGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = self.process()

    def process(self):
        data_list = []
        for i in range(self.graph_data.shape[0]):
            edge_index = dense_to_sparse(torch.tensor(self.graph_data[i], dtype=torch.float))[0]
            x = torch.tensor(self.feature[i], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)

        data, slices = self.collate(data_list)
        return data, slices

    def processed_file_names(self):
        return ['data.pt']

# 模型定义
class GCNAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNAE, self).__init__()
        self.encoder  = nn.ModuleList([
            GCNConv(in_channels, hidden_channels),
            nn.ReLU(),
            GCNConv(hidden_channels, out_channels)
        ])
        self.decoder = nn.ModuleList([
            GCNConv(out_channels, hidden_channels),
            nn.ReLU(),
            GCNConv(hidden_channels, in_channels)
        ])

    def encode(self, x, edge_index):
        for layer in self.encoder:
            if isinstance(layer, GCNConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x
    def decode(self, z, edge_index):
        for layer in self.decoder:
            if isinstance(layer, GCNConv):
                z = layer(z, edge_index)
            else:
                z = layer(z)
        return z
    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        recon_x = self.decode(z, edge_index)
        return recon_x

class GATAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super(GATAE, self).__init__()
        self.encoder = nn.ModuleList([
            GATConv(in_channels, hidden_channels, heads=heads),
            nn.ReLU(),
            GATConv(hidden_channels * heads, out_channels, heads=1)
        ])
        self.decoder = nn.ModuleList([
            GATConv(out_channels, hidden_channels, heads=heads),
            nn.ReLU(),
            GATConv(hidden_channels * heads, in_channels, heads=1)
        ])

    def encode(self, x, edge_index):
        for layer in self.encoder:
            if isinstance(layer, GATConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x

    def decode(self, z, edge_index):
        for layer in self.decoder:
            if isinstance(layer, GATConv):
                z = layer(z, edge_index)
            else:
                z = layer(z)
        return z

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        recon_x = self.decode(z, edge_index)
        return recon_x

def init_weights(m):
    if isinstance(m, (GCNConv, GATConv)) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

# 损失函数
def reconstruction_loss(recon_x, x):
    return torch.mean((recon_x - x) ** 2)

# 训练和测试
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon_x = model(data.x, data.edge_index)
        loss = reconstruction_loss(recon_x, data.x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            recon_x = model(data.x, data.edge_index)
            loss = reconstruction_loss(recon_x, data.x)
            total_loss += loss.item()
    return total_loss / len(loader)

def get_embeddings(model, loader, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            z = model.encode(data.x, data.edge_index).reshape(-1,148,64)
            embeddings.append(z.cpu().numpy())
    return np.concatenate(embeddings, axis=0)

# 降维和聚类方法
def reduce_dimensionality(embeddings, method='pca', n_components=2):
    """
    对形状为 (B, N, F) 的嵌入数据进行降维操作，每个批次分别进行降维，最后返回 (B, N, D) 的结果。

    参数:
    - embeddings: 形状为 (B, N, F) 的嵌入数据
    - method: 降维方法，'pca' 或 'tsne'
    - n_components: 降维后的维度 D

    返回:
    - 降维后的嵌入数据，形状为 (B, N, D)
    """
    B, N, D = embeddings.shape

    # 重塑数据，将每个批次的数据展平为一个向量
    flattened_embeddings = embeddings.reshape(B, -1)
    # 根据所选方法初始化降维器
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
    else:
        raise ValueError("Unsupported dimensionality reduction method")
    reduced_embeddings = reducer.fit_transform(flattened_embeddings)
    return reduced_embeddings

def cluster_embeddings(embeddings, method='kmeans', n_clusters=5):


    # B, N, D = embeddings.shape
    #
    # # 重塑数据，将每个批次的数据展平为一个向量
    # flattened_embeddings = embeddings.reshape(B, -1)

    # 选择聚类方法
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == 'kmedoids':
        clusterer = KMedoids(n_clusters=n_clusters, random_state=42)
    elif method == 'spectral':
        clusterer = SpectralClustering(n_clusters=n_clusters, random_state=42)
    else:
        raise ValueError("Unsupported clustering method")

    # 进行聚类
    cluster_labels = clusterer.fit_predict(embeddings)

    return cluster_labels


# 使用 PCA 降维并可视化
def plot_reduced_embeddings(embeddings, labels, title, ax):
    embeddings = embeddings
    labels = labels
    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1], label=f'Cluster {label}')

    ax.set_title(title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.legend()

# 计算聚类评价指标
def evaluate_clustering(embeddings, labels):
    """
    评估聚类结果，计算每个批次的 silhouette score 和 calinski harabasz score，
    并返回 silhouette score 的平均值。

    参数:
    - embeddings: 形状为 (B, N, F) 的嵌入数据
    - labels: 形状为 (B, N) 的标签数据

    返回:
    - silhouette_scores_mean: 所有批次的 silhouette score 平均值
    - calinski_harabasz_scores: 每个批次的 calinski harabasz score
    """
    B, N, F = embeddings.shape


    # 对每个批次分别计算评估指标

    # 获取当前批次的嵌入数据和标签
    batch_embeddings = embeddings.reshape(B, N*F)  # 形状 (N, F)
    batch_labels = labels  # 形状 (N, )

    # 计算 silhouette score
    silhouette = silhouette_score(batch_embeddings, batch_labels)
    #计算DBI scores
    davies_bouldin = davies_bouldin_score(batch_embeddings, batch_labels)
    # 计算 calinski_harabasz score
    calinski_harabasz = calinski_harabasz_score(batch_embeddings, batch_labels)



    return silhouette, calinski_harabasz,davies_bouldin


def deepwalk(edge_index, num_nodes, dimensions=64, walk_length=80, num_walks=10, workers=1):
    # Step 1: Convert edge_index to NetworkX graph
    from torch_geometric.utils import to_networkx
    from tqdm import tqdm
    G = to_networkx(Data(edge_index=edge_index, num_nodes=num_nodes))

    # Step 2: Perform random walks for DeepWalk with progress bar
    walks = []
    for _ in tqdm(range(num_walks), desc="Random Walks"):
        for node in range(num_nodes):
            walk = [node]
            for _ in range(walk_length - 1):
                cur = walk[-1]
                # Randomly select the next node
                neighbors = list(G.neighbors(cur))
                if neighbors:
                    walk.append(np.random.choice(neighbors))
            walks.append(list(map(str, walk)))  # Convert node ids to string for Word2Vec

    # Step 3: Train Word2Vec on the walks
    model = Word2Vec(walks, vector_size=dimensions, window=5, min_count=0, sg=1, workers=workers)

    # Step 4: Return the learned node embeddings
    return np.array([model.wv[str(i)] for i in range(num_nodes)])
# DeepWalk 和 Node2Vec 方法
def deepwalk_embedding(edge_index, num_nodes, dimensions=64, walk_length=80, num_walks=10, workers=1):
    from torch_geometric.utils import to_networkx
    import networkx as nx

    model = deepwalk(edge_index=edge_index, num_nodes=num_nodes)
    return model

def node2vec_embedding(edge_index, num_nodes, dimensions=64, walk_length=80, num_walks=10, p=1, q=1, workers=1):
    from torch_geometric.utils import to_networkx
    import networkx as nx
    G = to_networkx(Data(edge_index=edge_index, num_nodes=num_nodes))
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, p=p, q=q, workers=workers)
    model = node2vec.fit(window=5, min_count=0, batch_words=4)
    return np.array([model.wv[str(i)] for i in range(num_nodes)])

# 主函数
def main(model_type='gatae', dim_reduction_method='pca', clustering_method='kmeans', mat_folder_path='D:\epan\EP\\try1to9', feature_file_path='D:\\epan\\EP\\output\\1delta\\DegreeCentrality\\DCtry.mat'):
    # 加载图数据
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

    # 加载特征数据
    feature = scipy.io.loadmat(feature_file_path)['aDc'][:1066,]
    feature = np.nan_to_num(feature, nan=0.0)
    scaler = MinMaxScaler()
    feature = scaler.fit_transform(feature.reshape(-1, 1)).reshape(feature.shape)
    feature = np.expand_dims(feature, axis=-1)

    # 创建数据集
    dataset = MyGraphDataset(root='.', graph_data=graph_data, feature=feature)
    train_dataset, val_test_dataset = train_test_split(dataset, test_size=0.3, random_state=42)
    val_dataset, test_dataset = train_test_split(val_test_dataset, test_size=0.5, random_state=42)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    if model_type == 'gatae':
        model = GATAE(in_channels=1, hidden_channels=32, out_channels=64).to(device)
    elif model_type == 'gcnae':
        model = GCNAE(in_channels=1, hidden_channels=32, out_channels=64).to(device)
    elif model_type == 'deepwalk':
        model = None
    elif model_type == 'node2vec':
        model = None
    else:
        raise ValueError("Unsupported model type")

    if model_type in ['gatae', 'gcnae']:
        #model.apply(init_weights)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)

        best_val_loss = float('inf')
        patience = 20
        epochs_without_improvement = 0

        for epoch in range(1, 50):
            loss = train(model, train_loader, optimizer)
            val_loss = test(model, val_loader)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')

        test_embeddings = get_embeddings(model, test_loader, device)
    elif model_type == 'deepwalk':
        test_embeddings = [deepwalk_embedding(data.edge_index, data.num_nodes) for data in test_dataset]
        test_embeddings = np.stack(test_embeddings, axis=0)
    elif model_type == 'node2vec':
        test_embeddings = [node2vec_embedding(data.edge_index, data.num_nodes) for data in test_dataset]
        test_embeddings = np.stack(test_embeddings, axis=0)

    # 计算并打印不同聚类数量的结果
    # 假设 n_clusters_range 是从 2 到 14
    n_clusters_range = range(2, 14)  # 2 到 11，共10个值
    fig, axes = plt.subplots(4, 3, figsize=(15, 10))  # 4行3列的子图布局
    ss_scores = []
    db_scores = []
    ch_scores = []
    for i, n_clusters in enumerate(n_clusters_range):
        row, col = divmod(i, 3)  # 计算当前子图的行和列位置
        ax = axes[row, col]
        reduced_test_embeddings = reduce_dimensionality(test_embeddings, method=dim_reduction_method)

        test_labels = cluster_embeddings(reduced_test_embeddings, method=clustering_method, n_clusters=n_clusters)


        plot_reduced_embeddings(reduced_test_embeddings, test_labels, f'Test, n_clusters={n_clusters}', ax)

        test_silhouette, test_calinski_harabasz, test_DBI = evaluate_clustering(test_embeddings, test_labels)
        np.savetxt(f"D:\\epan\\EP\\BNT\\output_labels\\all_cluster_k{n_clusters}.txt", test_labels, fmt='%d')
        db_scores.append(test_DBI)
        ss_scores.append(test_silhouette)
        ch_scores.append(test_calinski_harabasz)
        np.savetxt("D:\\epan\\EP\\BNT\\output_index\\silhouette_scores.txt", ss_scores, delimiter=',', header='Silhouette Scores', comments='')
        np.savetxt("D:\\epan\\EP\\BNT\\output_index\\db_scores.txt", db_scores, delimiter=',', header='Davies-Bouldin Scores', comments='')
        np.savetxt("D:\\epan\\EP\\BNT\\output_index\\ch_scores.txt", ch_scores, delimiter=',', header='Calinski_Harabasz Scores', comments='')

        print(f'n_clusters={n_clusters}, model={model_type}, dim_method={dim_reduction_method}, clustering_method={clustering_method}')
        print(f'Test Set Silhouette Score: {test_silhouette:.4f}')
        print(f'Test Set Calinski-Harabasz Index: {test_calinski_harabasz:.4f}')
        print(f'Test Set DBI scores: {test_DBI:.4f}')
        print('-' * 50)

    plt.tight_layout()
    plt.show()

# 运行主函数
if __name__ == "__main__":
    # 方法有：gcnae gatae node2vec deepwalk   降维方法有：pca  t-sne   聚类方法有：kmeans kmedoids  spectral
    main(model_type='deepwalk', dim_reduction_method='pca', clustering_method='kmeans', mat_folder_path='D:\epan\py_GNN_proj\无监督聚类\无监督聚类\Try1to9', feature_file_path='D:\epan\py_GNN_proj\无监督聚类\无监督聚类\DCtry.mat')