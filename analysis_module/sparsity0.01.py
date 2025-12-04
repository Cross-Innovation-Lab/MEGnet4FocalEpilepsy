# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:35:31 2024

Author: Yafei
Description: This script processes adjacency matrices by applying a percentile threshold 
to generate binary graphs, then saves the results in .mat format.
"""

import os
import numpy as np


# Define paths
DATA_FOLDER_PATH = r'D:\epan\EP\BNT\from_finalmodels\results\\correlation\\matrix_mvgrl\\'
OUTPUT_FOLDER_PATH = r'D:\\epan\\BNT\\from_finalmodels\\results\\correlation\\percent99_mvgrl'

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)

# Gather all .txt files from the data folder
data_file_paths = [
    os.path.join(DATA_FOLDER_PATH, f)
    for f in os.listdir(DATA_FOLDER_PATH)
    if f.endswith('.txt')
]

def apply_percentile_threshold(adj_matrix, percentile):
    """
    Apply a percentile-based threshold to an adjacency matrix to create a binary graph.

    Parameters:
        adj_matrix (np.ndarray): The input adjacency matrix.
        percentile (float): The percentile threshold (0-100).

    Returns:
        binary_adj_matrix (np.ndarray): The binary adjacency matrix.
        threshold (float): The threshold value used.
    """
    # Extract nonzero weights and calculate threshold
    nonzero_weights = adj_matrix[np.nonzero(adj_matrix)]
    threshold = np.percentile(nonzero_weights, percentile)
    
    # Apply threshold to create a binary matrix
    binary_adj_matrix = (adj_matrix >= threshold).astype(int)
    formal_adj_matrix = np.where(adj_matrix >= threshold, adj_matrix, 0)
    return binary_adj_matrix, formal_adj_matrix, threshold

# Process each file
for i, data_file_path in enumerate(data_file_paths):
    # Load the adjacency matrix
  
    adj_matrix = np.loadtxt(data_file_path)


    # Apply a 95th percentile threshold to the adjacency matrix
    percentile = 95
    binary_adj_matrix, formal_adj_matrix, threshold = apply_percentile_threshold(adj_matrix, percentile)
    

    
    np.savetxt(f'D:\\epan\\BNT\\from_finalmodels\\results\\correlation\\percent99_mvgrl\\binary_adj_matrix95_{i}.txt', binary_adj_matrix, fmt='%d', delimiter=' ')
    print(f"Processed and saved binary matrix as .txt for file {data_file_path}")


'''   # Display information
    print(f"Original Adjacency Matrix (File {i}):")
    print(adj_matrix)
    print(f"Threshold at {100 - percentile}% sparsity: {threshold}")
    print(f"Binary Adjacency Matrix after Applying Threshold (File {i}):")
    print(formal_adj_matrix)
    print(binary_adj_matrix)'''