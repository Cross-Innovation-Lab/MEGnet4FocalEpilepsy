# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:25:02 2024

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

# Compare edges and extract unique nodes and edges
unique_edges_per_matrix = []
unique_nodes_per_matrix = []
for i, edges in enumerate(edge_sets):
    # Combine edges from all other matrices
    other_edges = set.union(*(set(edge_sets[j].keys()) for j in range(len(edge_sets)) if j != i))
    
    # Find unique edges
    unique_edges = {edge: weight for edge, weight in edges.items() if edge not in other_edges}
    unique_edges_per_matrix.append(unique_edges)
    
    # Extract unique nodes only if their edges are fully unique
    unique_nodes = set(node for edge in unique_edges.keys() for node in edge)
    unique_nodes_per_matrix.append(unique_nodes)

# Recheck if any node is marked as unique in multiple matrices
for i, unique_nodes in enumerate(unique_nodes_per_matrix):
    for j, other_nodes in enumerate(unique_nodes_per_matrix):
        if i != j:
            overlapping_nodes = unique_nodes.intersection(other_nodes)
            unique_nodes_per_matrix[i] -= overlapping_nodes

# Display results
results = {}
for i, (unique_nodes, unique_edges) in enumerate(zip(unique_nodes_per_matrix, unique_edges_per_matrix)):
    results[f"Matrix {i + 1}"] = {
        "Unique Nodes": unique_nodes,
        "Unique Edges": unique_edges
    }

results
