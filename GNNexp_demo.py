# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 21:21:14 2025

@author: Yafei
"""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool
from torch_geometric.explain import Explainer, GNNExplainer
import networkx as nx
import numpy as np

# Build the model
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.linear = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear(x)
        return x

# Creating the function to train the model
def Train(data_loader, loss_func):
    model.train()
    for data in data_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = loss_func(out, data.y)
        loss.backward()
        optimizer.step()

# Function to test the model
def Test(data_loader):
    model.eval()
    correct = 0
    for data in data_loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(data_loader.dataset)

# Function to visualize the explanation
def visualize_explanation(data, edge_mask, threshold=0.5):
    edge_mask = edge_mask.detach().cpu().numpy()
    edge_index = data.edge_index.cpu().numpy()
    G = nx.from_edgelist(edge_index.T)
    
    # Filter important edges
    important_edges = edge_index[:, edge_mask > threshold]
    important_edges = [(u, v) for u, v in zip(important_edges[0], important_edges[1])]
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    nx.draw_networkx_edges(G, pos, edgelist=important_edges, edge_color='red', width=2)
    plt.show()

if __name__ == '__main__':
    # Load the dataset
    dataset = TUDataset(root='./data/TUDataset', name='MUTAG')
    print(f'Dataset: {dataset}')
    print("Number of Graphs: ", len(dataset))
    print("Number of Features: ", dataset.num_features)
    print("Number of Classes: ", dataset.num_classes)

    # Shuffle and split the dataset
    torch.manual_seed(12345)
    dataset = dataset.shuffle()
    train_dataset = dataset[:50]
    test_dataset = dataset[50:]

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize the model
    model = GNN(hidden_channels=64, num_node_features=dataset.num_features, num_classes=dataset.num_classes)
    print(model)

    # Set the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    for epoch in range(1, 101):
        Train(train_loader, loss_func=criterion)
        train_acc = Test(train_loader)
        test_acc = Test(test_loader)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    # Explain the graph using GNNExplainer
    model.eval()
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100, lr=0.01),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='log_probs',
        ),
    )

    # Select a graph to explain
    data = dataset[5]
    explanation = explainer(data.x, data.edge_index, batch=data.batch)
    edge_mask = explanation.edge_mask

    # Visualize the explanation
    visualize_explanation(data, edge_mask)