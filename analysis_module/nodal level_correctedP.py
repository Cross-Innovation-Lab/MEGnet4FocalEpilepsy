# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:40:08 2024

@author: Yafei
"""

import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os

# File paths
file_path = 'D:\\epan\\EP\\BNT\\from_finalmodels\\results\\nodal_level_aNCp.xlsx'
output_dir = 'D:\\epan\\EP\\BNT\\from_finalmodels\\results\\Results_Nodal_bonf\\'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read Excel file
data = pd.read_excel(file_path)

# Extract parameters and labels
parameters = data.iloc[:, 2:].values  # Parameters start from the 3rd column
labels = data.iloc[:, 1].values       # Group labels are in the 2nd column

# Store ANOVA and post-hoc results
anova_results = []
posthoc_results = []

# Collect p-values for FDR correction
all_p_values = []

# First pass: Perform ANOVA and collect p-values
for param_index in range(parameters.shape[1]):  # Loop through each column (parameter)
    groups = [parameters[labels == label, param_index] for label in np.unique(labels)]
    
    # Ensure groups are valid (no NaNs or all-zero groups)
    valid_groups = [group[~np.isnan(group) & (group != 0)] for group in groups]
    if all(len(group) > 0 for group in valid_groups):
        # Perform one-way ANOVA
        fvalue, pvalue = f_oneway(*valid_groups)
        anova_results.append((param_index + 1, fvalue, pvalue))
        all_p_values.append(pvalue)

# Apply FDR correction
_, corrected_p_values, _, _ = multipletests(all_p_values, method='bonferroni')

# Second pass: Process significant results and perform post-hoc analysis
for i, (param_index, fvalue, pvalue) in enumerate(anova_results):
    corrected_p = corrected_p_values[i]
    if corrected_p < 0.001:  # Threshold for significance after FDR correction
        # Save significant ANOVA results
        valid_groups = [parameters[labels == label, param_index - 1] for label in np.unique(labels)]
        valid_groups = [group[~np.isnan(group) & (group != 0)] for group in valid_groups]

        # Flatten data and labels for Tukey's HSD test
        flattened_data = []
        flattened_labels = []
        for label, group in zip(np.unique(labels), valid_groups):
            flattened_data.extend(group)
            flattened_labels.extend([label] * len(group))
        
        # Perform Tukey's HSD test
        tukey = pairwise_tukeyhsd(flattened_data, flattened_labels, alpha=0.001)
        posthoc_results.append((param_index, tukey.summary()))

# Save ANOVA results
anova_output_file = os.path.join(output_dir, 'anova_results_aNCp.txt')
with open(anova_output_file, 'w') as f:
    f.write("ANOVA Results with FDR Correction:\n")
    for i, (param_index, fvalue, pvalue) in enumerate(anova_results):
        corrected_p = corrected_p_values[i]
        f.write(f"Parameter {param_index}: F value = {fvalue:.4f}, p value = {pvalue:.4e}, corrected p = {corrected_p:.4e}\n")
print(f"ANOVA results saved to: {anova_output_file}")

# Save significant results to a separate file
anova_sig_output_file = os.path.join(output_dir, 'anova_results_aNCp_sig.txt')
with open(anova_sig_output_file, 'w') as f:
    f.write("Significant ANOVA Results (FDR Corrected, p < 0.001):\n")
    for i, (param_index, fvalue, pvalue) in enumerate(anova_results):
        corrected_p = corrected_p_values[i]
        if corrected_p < 0.001:
            f.write(f"Parameter {param_index}: F value = {fvalue:.4f}, p value = {pvalue:.4e}, corrected p = {corrected_p:.4e}\n")
print(f"Significant ANOVA results saved to: {anova_sig_output_file}")

# Save post-hoc results
posthoc_output_file = os.path.join(output_dir, 'posthoc_results_aNCp.txt')
with open(posthoc_output_file, 'w') as f:
    f.write("Post-hoc Analysis Results (Tukey HSD, Significant Parameters Only):\n")
    for param_index, summary in posthoc_results:
        f.write(f"\nParameter {param_index}:\n")
        f.write(summary.as_text())
print(f"Post-hoc results saved to: {posthoc_output_file}")
