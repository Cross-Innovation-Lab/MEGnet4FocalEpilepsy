# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:53:04 2024

@author: Yafei
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
file_path = 'D:\\epan\\EP\\BNT\\from_finalmodels\\results\\nodal_level_degree_clean.xlsx'
# 读取Excel文件
data = pd.read_excel(file_path)

# 假设你想要绘制第一个subtype的数据
subtype_id = 3
# 选择subtype为1的数据
subtype_data = data[data['subtype_ID'] == subtype_id]

# 选择你想要绘制的148个参数

parameters = [col for col in subtype_data.columns if 'L' in col or 'R' in col]
n_parameters = len(parameters)

# 创建一个条形图
x = np.arange(n_parameters)
fig, ax = plt.subplots()
ax.bar(x, subtype_data[parameters].mean(), width=0.4, label='Mean')

# 设置x轴标签
ax.set_xticks(x)
ax.set_xticklabels(parameters, rotation=90)

# 添加标题和标签
ax.set_title(f'Mean of 148 Parameters for Subtype {subtype_id}')
ax.set_xlabel('Parameters')
ax.set_ylabel('Mean Value')

# 显示图例
ax.legend()

# 显示图表
plt.tight_layout()
plt.show()