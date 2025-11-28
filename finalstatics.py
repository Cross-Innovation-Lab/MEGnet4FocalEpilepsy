# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:24:18 2024
For finalmodels statics

@author: Yafei
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

# 3 ANOVA
# 读取Excel文件
file_path = 'D:\\epan\\EP\BNT\\from_finalmodels\\results\\gat_kmeans_k4_anova.xlsx'  
data = pd.read_excel(file_path)

# 读取第一列作为标签
labels = data.iloc[:, 2].values

# 读取第二列到第四列作为比较的参数
aCp = data.iloc[:, 3].values
group1 = aCp[labels == 0]
group2 = aCp[labels == 1]
group3 = aCp[labels == 2]
group4 = aCp[labels == 3]
print(len(labels))
#labels = np.concatenate(labels)
print(group1)
print(group2)
print(group3)
print(group4)

data = np.concatenate((group1, group2, group3))
print(len(data))
# 进行单因素方差分析(ANOVA)
fvalue, pvalue = stats.f_oneway(group1, group2, group3, group4)
print("F 统计量:", fvalue)
print("p 值:", pvalue)

# 如果p值小于显著性水平，则进行事后多重比较检验

if pvalue < 0.05:
    # 获取每组数据
    groups = [group1, group2, group3, group4]
    groupnames = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
    # 计算组间均值差异的统计量
    means = [group.mean() for group in groups]
    print(means)
    stds = [group.std(ddof=1) for group in groups]
    sizes = [len(group) for group in groups]
    # 绘制直方图
    #categories = ['Cluster 1', 'Cluster 2', 'Cluster 3']
    plt.bar(groupnames, means, yerr=stds, capsize=5, alpha=0.7, ecolor='black')
    plt.ylabel('Metric Mean')
    plt.ylim(0.120, 0.190)
   
    plt.title('Metric Means with Error Bars')
    plt.show()
    
# 计算组间差异的统计量
    comparisons = [means[0] - means[1], means[0] - means[2], means[1] - means[2]]
    #std_errors = np.sqrt([stds[0]**2 + stds[1]**2, stds[0]**2 + stds[2]**2, stds[1]**2 + stds[2]**2]) / np.sqrt(2 * (sizes[0] + sizes[1] + sizes[2]))
    std_errors = np.sqrt([
    (stds[0]**2 + stds[1]**2) / (2 * (sizes[0] + sizes[1])),
    (stds[0]**2 + stds[2]**2) / (2 * (sizes[0] + sizes[2])),
    (stds[1]**2 + stds[2]**2) / (2 * (sizes[1] + sizes[2]))
])
    # 计算t统计量和p值
    t_stats = np.array(comparisons) / std_errors
    p_values = stats.t.sf(np.abs(t_stats), sizes[0] + sizes[1] + sizes[2] - 3) * 2
    
    # 多重比较校正
    corrected_p_values = multipletests(p_values, method='holm')[1]
    
    print("组间比较:")
    print(f"Sample 1 vs Sample 2: t={t_stats[0]:.3f}, p={p_values[0]:.3e}, corrected p={corrected_p_values[0]:.3e}")
    print(f"Sample 1 vs Sample 3: t={t_stats[1]:.3f}, p={p_values[1]:.3e}, corrected p={corrected_p_values[1]:.3e}")
    print(f"Sample 2 vs Sample 3: t={t_stats[2]:.3f}, p={p_values[2]:.3e}, corrected p={corrected_p_values[2]:.3e}")
else:
    print("没有足够的证据表明组间存在显著差异")