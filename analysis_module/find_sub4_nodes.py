import numpy as np
import pandas as pd

# File path for the adjacency matrix
file_path = r"D:\epan\EP\BNT\from_finalmodels\output_matrix\percent99\sub3_matrix.xlsx"

# Load adjacency matrix and node names
data = pd.read_excel(file_path, header=None)

# Extract node names from the first column
node_names = data.iloc[:, 0].tolist()  # First column as node names

# Extract adjacency matrix (excluding the first column)
adj_matrix = data.iloc[:, 1:].values

# Ensure no self-loops by zeroing out the diagonal
np.fill_diagonal(adj_matrix, 0)

def find_connected_nodes_only(adj_matrix, node_names):
    """
    Finds all nodes that are connected to at least one other node.
    
    Parameters:
    - adj_matrix: 2D numpy array, adjacency matrix.
    - node_names: List of node names corresponding to the indices in adj_matrix.
    
    Returns:
    - connected_nodes: List of node names that have at least one connection.
    """
    connected_nodes = []
    for i in range(len(adj_matrix)):
        if np.any(adj_matrix[i]):  # Check if the node has any connection
            connected_nodes.append(node_names[i])
    return connected_nodes

# Get connected nodes
connected_nodes = find_connected_nodes_only(adj_matrix, node_names)

# Display the connected nodes
print("Nodes that are connected to at least one other node:")
print(sorted(connected_nodes))
