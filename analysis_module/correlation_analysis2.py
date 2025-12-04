# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 09:23:37 2024

@author: Yafei
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 15:31:14 2024

@author: Yafei
"""

import pandas as pd
import numpy as np
import pingouin as pg
from scipy.stats import pearsonr, spearmanr

# Replace with your actual dataset file path
file_path = "D:\\PD\\paper\\2EP\\TLE_mainly2.csv"
data = pd.read_csv(file_path)

# Display the first few rows
print(data.head())

# Get the list of ROIs starting from the 10th column (index 9)
ROIs = data.columns[6:]
Age = data.columns[1]
# Open a file to write the results
with open('D:\\PD\\paper\\2EP\\TLE_results\\correlation_results_Age2.txt', 'w') as file:
    # Calculate partial correlation for each ROI with each subtype, controlling for 'Age'
    
    for subtype in ['Subtype1_Freq', 'Subtype2_Freq', 'Subtype3_Freq', 'Subtype4_Freq']:
            # Calculate partial correlation and p-value, controlling for 'Age'
            #result = pg.partial_corr(data=data, x=ROI, y=subtype, covar='Age')
        corr, p_value = pearsonr(data[Age], data[subtype])
            # Extract the correlation coefficient and p-value from the DataFrame
            #corr = result['r'].values[0]
            #p_value = result['p-val'].values[0]
            
            # Format the output string
        output_str = f"Correlation between {Age} and {subtype}: partial r = {corr:.2f}, p-value = {p_value:.3f}\n"
            
            # Write the result to the file
        file.write(output_str)
            
            # Also print the result to the console
        print(output_str, end='')

# Note: The 'end=' parameter in the print function is used to avoid adding extra newlines.