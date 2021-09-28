# -*- coding: utf-8 -*-
"""
Formulate point source at monthly scale

@author: ShaobinLi
"""

import pandas as pd
import numpy as np
from data import *

def point_source(name, sw):
    '''function to formulate point source into appropriate format (year, month, subwatershed)'''
    df_point = df_point_SDD
    if name == 'nitrate':
        df_point = pd.DataFrame(df_point.iloc[:,0])
    elif name == 'phosphorus':
        df_point = pd.DataFrame(df_point.iloc[:,1])  
    df_point['month'] = df_point.index.month
    df_point['year'] = df_point.index.year
    df2_point = np.zeros((16,12)) # need to customize based on study time period
    for i in range(16):
        for j in range(12):
            df2_point[i,j] = df_point.loc[(df_point.year==2003+i) & (df_point.month==1+j)].iloc[:,0].astype('float').sum()
    df3_point = np.zeros((16,12,45))  # need to customize based on study time period and subwatershed
    df3_point[:,:,sw] = df2_point     # define where point source located；can be 
    return df2_point, df3_point

# df2_point, df3_point = point_source('phosphorus', 30)
