# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:42:23 2024

@author: Yafei
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

# File paths
file_path = 'D:\\epan\\EP\\BNT\\from_finalmodels\\results\\gat_kmeans_k4_anova.xlsx'

# Load the Excel file
data = pd.read_excel(file_path)

# Extract labels and values
labels = data.iloc[:, 2].values  # Assuming labels are in column 3
#aCp = data.iloc[:, 3].values    # Assuming metric values are in column 4
#aLp = data.iloc[:, 4].values
#aEg = data.iloc[:, 5].values
aEloc = data.iloc[:, 6].values
# Group data by labels
group1 = aEloc[labels == 0]
group2 = aEloc[labels == 1]
group3 = aEloc[labels == 2]
group4 = aEloc[labels == 3]

# Ensure groups are not empty
if any(len(group) == 0 for group in [group1, group2, group3, group4]):
    raise ValueError("One or more groups are empty. Check the data or labels.")

# Perform one-way ANOVA
fvalue, pvalue = stats.f_oneway(group1, group2, group3, group4)
print("F-statistic:", fvalue)
print("p-value:", pvalue)

# If the ANOVA is significant, perform post-hoc analysis
if pvalue < 0.05:
    # Group data for comparisons
    groups = [group1, group2, group3, group4]
    groupnames = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
    
    # Calculate means and standard deviations for each group
    means = [np.mean(group) for group in groups]
    stds = [np.std(group, ddof=1) for group in groups]
    sizes = [len(group) for group in groups]
    
    # Plot means with error bars
    plt.bar(groupnames, means, yerr=stds, capsize=5, alpha=0.7, ecolor='black')
    plt.ylabel('Metric Mean')
    plt.title('Metric Means with Error Bars')
    plt.show()

    # Pairwise comparisons
    comparisons = [
        (i, j, means[i] - means[j], 
         np.sqrt((stds[i]**2 + stds[j]**2) / (sizes[i] + sizes[j])))
        for i in range(len(groups)) for j in range(i + 1, len(groups))
    ]
    
    # Compute t-statistics and p-values for pairwise comparisons
    results = []
    for i, j, diff, std_error in comparisons:
        t_stat = diff / std_error
        df = sizes[i] + sizes[j] - 2
        p_value = stats.t.sf(np.abs(t_stat), df) * 2
        results.append((i, j, t_stat, p_value))
    
    # Multiple testing correction
    p_values = [result[3] for result in results]
    corrected_p_values = multipletests(p_values, method='holm')[1]
    
    # Print pairwise comparison results
    print("\nPairwise Comparisons:")
    for idx, (i, j, t_stat, p_value) in enumerate(results):
        corrected_p = corrected_p_values[idx]
        print(f"Group {i+1} vs Group {j+1}: t={t_stat:.3f}, p={p_value:.3e}, corrected p={corrected_p:.3e}")
else:
    print("No significant differences among groups based on ANOVA.")
