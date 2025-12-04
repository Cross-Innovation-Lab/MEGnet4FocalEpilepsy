# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:49:06 2025

@author: Yafei
"""

import pandas as pd
from scipy.stats import ttest_ind

# 读取Excel文件
df = pd.read_excel('D:\\PD\\paper\\2EP\\Step2_forANOVA.xlsx')

# 检查数据框的列名
print(df.columns)

# 过滤出每个subtype的数据
subtypes = [0, 1, 2, 3]
results = {}
ROIs = df.columns[7:]

for subtype in subtypes:
    # 过滤出当前subtype的数据
    subtype_data = df[df['subtype_gat'] == subtype]
    subtype_results = {}
    for ROI in ROIs:
        
        # 获取Region 1和Region 2的数据
        region1_data = subtype_data[subtype_data['Region'] == 1][ROI]
        region2_data = subtype_data[subtype_data['Region'] == 2][ROI]
        
        # 计算t-test
        t_stat, p_value = ttest_ind(region1_data, region2_data, equal_var=False)
        if p_value < 0.00001:
        # 保存结果
            subtype_results[ROI] = {'t_stat': t_stat, 'p_value': p_value}
    
        results[subtype] = subtype_results

# 输出结果
for subtype, subtype_results in results.items():
    print(f"Subtype {subtype}:")
    for ROI, result in subtype_results.items():
        print(f"  ROI {ROI}: t-statistic = {result['t_stat']}, p-value = {result['p_value']}")