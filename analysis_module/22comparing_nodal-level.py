import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# File paths for the two adjacency matrices
file_path_1 = r'D:\epan\EP\BNT\from_finalmodels\output_matrix\percent99\sub2_matrix.xlsx'
file_path_2 = r'D:\epan\EP\BNT\from_finalmodels\output_matrix\percent99\sub3_matrix.xlsx'

# Load adjacency matrices and node names
data1 = pd.read_excel(file_path_1, header=None)
data2 = pd.read_excel(file_path_2, header=None)

# Extract node names and adjacency matrices
nodes = data1.iloc[:, 0].tolist()  # Assume both have the same node names
adj_matrix1 = data1.iloc[:, 1:].values
adj_matrix2 = data2.iloc[:, 1:].values

# Remove diagonal entries
np.fill_diagonal(adj_matrix1, 0)
np.fill_diagonal(adj_matrix2, 0)

# Create edge sets for comparison
edges1 = {(nodes[i], nodes[j]) for i in range(len(adj_matrix1)) for j in range(len(adj_matrix1)) if adj_matrix1[i, j] != 0}
edges2 = {(nodes[i], nodes[j]) for i in range(len(adj_matrix2)) for j in range(len(adj_matrix2)) if adj_matrix2[i, j] != 0}

# Find unique edges
unique_to_matrix1 = edges1 - edges2
unique_to_matrix2 = edges2 - edges1

# Display results
print("Edges unique to the first matrix:", unique_to_matrix1)
print("Edges unique to the second matrix:", unique_to_matrix2)

# Create graphs for visualization
G1 = nx.Graph()
G2 = nx.Graph()

# Add all edges to the first graph, highlighting unique ones
G1.add_edges_from(edges1)
G2.add_edges_from(edges2)

# Plot the graphs with unique edges highlighted
plt.figure(figsize=(14, 7))

# Plot for the first graph
plt.subplot(121)
pos1 = nx.spring_layout(G1, seed=42, k=2)  # Adjust layout for spacing
nx.draw(
    G1, pos1, with_labels=True, node_color='skyblue', node_size=1000, edge_color='gray'
)
nx.draw_networkx_edges(G1, pos1, edgelist=list(unique_to_matrix1), edge_color="red", width=2)
plt.title("First Matrix Graph (Unique Edges in Red)", fontsize=14)

# Plot for the second graph
plt.subplot(122)
pos2 = nx.spring_layout(G2, seed=42, k=2)  # Adjust layout for spacing
nx.draw(
    G2, pos2, with_labels=True, node_color='skyblue', node_size=1000, edge_color='gray'
)
nx.draw_networkx_edges(G2, pos2, edgelist=list(unique_to_matrix2), edge_color="red", width=2)
plt.title("Second Matrix Graph (Unique Edges in Red)", fontsize=14)

plt.tight_layout()
plt.show()
