# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:14:15 2024

@author: Yafei
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# File paths for the adjacency matrices
file_paths = [
    r"D:\epan\EP\BNT\from_finalmodels\output_matrix\percent99\sub0_matrix.xlsx",
    r"D:\epan\EP\BNT\from_finalmodels\output_matrix\percent99\sub1_matrix.xlsx",
    r"D:\epan\EP\BNT\from_finalmodels\output_matrix\percent99\sub2_matrix.xlsx",
    r"D:\epan\EP\BNT\from_finalmodels\output_matrix\percent99\sub3_matrix.xlsx"
]

# Load adjacency matrices and node names
matrices = []
node_names = []

for file_path in file_paths:
    data = pd.read_excel(file_path, header=None)
    node_names.append(data.iloc[:, 0].tolist())  # First column as node names
    adj_matrix = data.iloc[:, 1:].values
    np.fill_diagonal(adj_matrix, 0)  # Remove diagonal values
    matrices.append(adj_matrix)

# Ensure all matrices have the same node order
if not all(node_names[0] == names for names in node_names):
    raise ValueError("Node names do not match across matrices!")

nodes = node_names[0]  # Use the first set of node names

# Extract edges for each matrix
edge_sets = []
for adj_matrix in matrices:
    edges = {(nodes[i], nodes[j]) for i in range(len(adj_matrix)) for j in range(len(adj_matrix)) if adj_matrix[i, j] != 0}
    edge_sets.append(edges)

# Compare edges
unique_edges = []
for i, edges in enumerate(edge_sets):
    other_edges = set.union(*(edge_sets[:i] + edge_sets[i + 1:]))
    unique_edges.append(edges - other_edges)

# Visualize only unique edges
plt.figure(figsize=(16, 12))
for i, edges in enumerate(unique_edges):
    G = nx.Graph()
    G.add_edges_from(edges)
    
    pos = nx.spring_layout(G, seed=42, k=2)  # Layout for better spacing
    plt.subplot(2, 2, i + 1)
    nx.draw(
        G, pos, with_labels=True, node_color='skyblue', node_size=1000, edge_color='red', width=2
    )
    plt.title(f"Unique Edges in Matrix {i + 1}", fontsize=14)

plt.tight_layout()
plt.show()
