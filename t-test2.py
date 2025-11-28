# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 10:12:54 2025

@author: Yafei
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.io import loadmat
import os

def load_data(folder_path):
    """
    从文件夹加载所有.mat文件并堆叠数据
    
    参数:
    folder_path: 文件夹路径
    
    返回:
    data: 堆叠后的数据
    """
    data = []
    # 获取文件夹下所有的.mat文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
    for file in files:
        file_path = os.path.join(folder_path, file)
        mat_data = loadmat(file_path)
        # 假设.mat文件中存储的数据变量名为'matrix'
        data.append(mat_data['a'])
    return np.array(data)

def group_pointwise_t_test(group1, group2, num_regions, num_time_points, num_frequency_bands):
    """
    对两组输入矩阵进行逐点t检验
    
    参数:
    group1, group2: 输入的两组三维矩阵，每组维度为(20, 148, 3001, 6)
    num_regions: 脑区数量
    num_time_points: 时间序列点数量
    num_frequency_bands: 频率段数量
    
    返回:
    differences: 保存每个时间点的差异值的矩阵
    p_values: 保存每个时间点的p值的矩阵
    t_values: 保存每个时间点的t值的矩阵
    """
    differences = np.zeros((num_regions, num_time_points, num_frequency_bands))
    p_values = np.zeros((num_regions, num_time_points, num_frequency_bands))
    t_values = np.zeros((num_regions, num_time_points, num_frequency_bands))

    for i in range(num_regions):
        for j in range(num_frequency_bands):
            for k in range(num_time_points):
                # 将每组的对应脑区和频率段的数据堆叠起来
                values1 = group1[:, i, k, j].flatten()
                values2 = group2[:, i, k, j].flatten()
                t_stat, p_val = ttest_ind(values1, values2)
                differences[i, k, j] = np.mean(values1) - np.mean(values2)
                p_values[i, k, j] = p_val
                t_values[i, k, j] = t_stat

    return differences, p_values, t_values

# 加载数据
folder_path1 = 'D:\\PD\\paper\\TJMEG\\ZY\\try\\group1'
folder_path2 = 'D:\\PD\\paper\\TJMEG\\ZY\\try\\group2'

group1 = load_data(folder_path1)
group2 = load_data(folder_path2)


num_regions = 2
num_time_points = 3001
num_frequency_bands = 2

differences, p_values, t_values = group_pointwise_t_test(group1, group2, num_regions, num_time_points, num_frequency_bands)

# 绘制结果
for i in range(num_regions):
    for j in range(num_frequency_bands):
        plt.figure(figsize=(12, 6))
        
        # 绘制两个矩阵的原始曲线
        plt.plot(np.arange(num_time_points), group1[0, i, :, j], label='Matrix 1', color='blue', alpha=0.3)
        plt.plot(np.arange(num_time_points), group2[0, i, :, j], label='Matrix 2', color='orange', alpha=0.3)
        
        # 绘制t值曲线
        plt.plot(np.arange(num_time_points), t_values[i, :, j], label='T-value', color='green')
        
        # 标注p<0.05的地方
        significant_indices = p_values[i, :, j] < 0.05
        plt.scatter(np.arange(num_time_points)[significant_indices],
                    t_values[i, significant_indices, j], color='red', label='p < 0.05')
        
        plt.axhline(y=0, color='k', linestyle='--')
        plt.xlabel('Time (ms)')
        plt.ylabel('Value')
        plt.title(f'Region {i+1}, Frequency Band {j+1}')
        plt.legend()
        plt.show()