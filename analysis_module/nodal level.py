# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:10:54 2024

@author: Yafei
"""

"""
Created on Tue Nov 19 13:43:35 2024

@author: Yafei
"""


import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os

# File paths
file_path = 'D:\\PD\\paper\\2EP\\Step2_forANOVA.xlsx'
output_dir = 'D:\\PD\\paper\\2EP\\TLE_results\\'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read Excel file
data = pd.read_excel(file_path)

# Extract parameters and labels
parameters = data.iloc[:, 7:].values  # Parameters start from the 4th column
labels = data.iloc[:, 1].values       # Group labels are in the 2nd column

# Store ANOVA results
anova_results = []
posthoc_results = []

# Iterate over each parameter
for param_index in range(parameters.shape[1]):  # Loop through each column (parameter)
    groups = [parameters[labels == label, param_index] for label in np.unique(labels)]
    
    # Ensure each group has data
    if all(len(group) > 0 for group in groups):
        # Perform one-way ANOVA
        fvalue, pvalue = f_oneway(*groups)
        anova_results.append((param_index + 1, fvalue, pvalue))
        
        # If ANOVA is significant, perform post-hoc analysis
        if pvalue < 0.001:
            # Create a flattened data array and corresponding group labels
            flattened_data = []
            flattened_labels = []
            for label, group in zip(np.unique(labels), groups):
                flattened_data.extend(group)
                flattened_labels.extend([label] * len(group))
            
            # Perform Tukey's HSD test
            tukey = pairwise_tukeyhsd(flattened_data, flattened_labels, alpha=0.001)
            posthoc_results.append((param_index + 1, tukey.summary()))

# Save ANOVA results to a text file
anova_output_file = os.path.join(output_dir, 'anova_results_degree.txt')
with open(anova_output_file, 'w') as f:
    f.write("ANOVA Results:\n")
    for param_index, fvalue, pvalue in anova_results:
        f.write(f"Parameter {param_index}: F value = {fvalue:.4f}, p value = {pvalue:.4e}\n")
print(f"ANOVA results saved to: {anova_output_file}")

# Save significant ANOVA results to a separate text file
anova_sig_output_file = os.path.join(output_dir, 'anova_results_degree_sig.txt')
with open(anova_sig_output_file, 'w') as f:
    f.write("Significant ANOVA Results (p < 0.001):\n")
    for param_index, fvalue, pvalue in anova_results:
        if pvalue < 0.001:
            f.write(f"Parameter {param_index}: F value = {fvalue:.4f}, p value = {pvalue:.4e}\n")
print(f"Significant ANOVA results saved to: {anova_sig_output_file}")

# Save post-hoc results to a text file
posthoc_output_file = os.path.join(output_dir, 'posthoc_results_degree.txt')
with open(posthoc_output_file, 'w') as f:
    f.write("Post-hoc Analysis Results (Tukey HSD):\n")
    for param_index, summary in posthoc_results:
        f.write(f"\nParameter {param_index}:\n")
        f.write(summary.as_text())
print(f"Post-hoc results saved to: {posthoc_output_file}")
