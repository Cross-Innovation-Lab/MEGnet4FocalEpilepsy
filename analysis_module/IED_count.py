# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:59:56 2024

@author: Yafei
"""

import pandas as pd

# 读取Excel文件
df = pd.read_excel('D:\epan\EP\BNT\\from_finalmodels\\results\\correlation\IED_count.xlsx')

# 统计每个subject中0，1，2，3的数量
result = df.groupby('name')['mvgrl'].value_counts().unstack(fill_value=0)

print(result)