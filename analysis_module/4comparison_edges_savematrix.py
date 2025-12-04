# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:31:01 2024

@author: Yafei
"""

import numpy as np
import pandas as pd

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
    edges = {(nodes[i], nodes[j]): adj_matrix[i, j] for i in range(len(adj_matrix)) for j in range(len(adj_matrix)) if adj_matrix[i, j] != 0}
    edge_sets.append(edges)

# Compare edges and create unique edge matrices
unique_edge_matrices = []
for i, edges in enumerate(edge_sets):
    # Combine edges from all other matrices
    other_edges = set.union(*(set(edge_sets[j].keys()) for j in range(len(edge_sets)) if j != i))
    
    # Find unique edges
    unique_edges = {edge: weight for edge, weight in edges.items() if edge not in other_edges}
    
    # Create a new adjacency matrix for unique edges
    unique_adj_matrix = np.zeros_like(matrices[0])
    for (src, tgt), weight in unique_edges.items():
        src_idx = nodes.index(src)
        tgt_idx = nodes.index(tgt)
        unique_adj_matrix[src_idx, tgt_idx] = weight
    
    unique_edge_matrices.append(unique_adj_matrix)

    # Save the unique adjacency matrix
    output_path = f"unique_adj_matrix_{i + 1}.xlsx"
    pd.DataFrame(unique_adj_matrix, index=nodes, columns=nodes).to_excel(output_path, header=True, index=True)
    print(f"Unique adjacency matrix for Matrix {i + 1} saved to {output_path}")
