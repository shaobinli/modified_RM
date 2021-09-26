# -*- coding: utf-8 -*-
"""
Author: Shaobin Li (shaobin@illinois.edu)
Project: NSF INFEWS/T1 (award number: 1739788)

Purpose: Constructing the traditional response matrix method to approximate SWAT for computationally intense applications
"""

import pandas as pd
import numpy as np
import time
from calendar import monthrange
from data import *


'''Step 1: enerate a set of response matrices (Y_(m,t)), gathered from SWAT simulation outputs'''
def response_mat(name):
    '''
    return as a tuple
    unit: kg/ha for nitrate, phosphorus, soy, corn, corn silage; ton/ha for sediment; mm for water yield
    '''
    if name == 'nitrate':
        df = df_nitrate
    elif name == 'phosphorus':
        df = df_TP
    elif name == 'sediment':
        df = df_sediment
    elif name == 'streamflow':
        df = df_streamflow
    else:
        raise ValueError('please enter the correct names, e.g., nitrate, phosphorus, sediment')
    
    subwatershed = df.iloc[:,0].unique()
    year = df.iloc[:,1].unique()
    month = df.iloc[:,2].unique()
    area_sw = df.iloc[:,3].unique()
    df = df.drop(df.columns[[0,1,2,3]], axis=1)
    
    if name == 'streamflow':
        df = 1/df
        
    df_to_np = np.zeros((year.size, month.size, subwatershed.size, df.shape[1]))
    for i in range(year.size):
        for j in range(month.size):
            df2 = df.iloc[month.size*subwatershed.size*(i):month.size*subwatershed.size*(i+1),:]
            df_to_np[i,j,:,:] = df2.iloc[45*(j):45*(j+1),:]
    return df_to_np, subwatershed, year, month, df.shape[1], area_sw

'''Steps 2&3: create a decision land use matrix and get land use area'''
def basic_landuse():
    '''basic case of land use'''
    landuse = pd.read_excel(r'./support_data/landuse.xlsx').fillna(0)
    land_agri = landuse.iloc[:,1] + landuse.iloc[:,2]
    land_agri = np.mat(land_agri).T
    return landuse, land_agri

def landuse_mat():
    '''
    Return a decison matrix (# of subwatershed, # of BMPs) to decide land use fractions
    of each BMP application in each subwatershed
    '''
    linkage = pd.read_excel(r'.\support_data\Watershed_linkage.xlsx').fillna(0)
    df = pd.read_csv(r'.\response_matrix_csv\yield_nitrate.csv')
    row_sw = linkage.shape[0]
    '''minus 4 to subtract first two columns of subwatershed and area'''
    col_BMP = df.shape[1] - 4
    landuse_matrix = np.zeros((row_sw,col_BMP))
    return landuse_matrix

'''Step 4: develop a connectivity matrix (W) describing the upstream-downstream relationships of all subwatersheds'''
'''This step does not require coding, but rather prepare an excel to represent connectivity matrix'''
'''off-diagonal elements w_(i,j|i≠j ) is equal to one if subwatershed j is upstream of subwatershed i and zero otherwise'''
linkage = pd.read_excel(r'./support_data/Watershed_linkage.xlsx', index_col=0)


'''Step 5.1: estimate landscape yield during month t across all subwatersheds.'''
def get_yield(name, landuse_matrix):
    '''
    return a tuple containing two numpy array: 
        1) yield_per_BMP: (year, month, subwatershed, BMP)
        2) yield_sum: (year, month, subwatershed)
    unit: kg/ha for nitrate, phosphorus; ton/ha for sediment; mm for water yield
    '''    
    response = response_mat(name)
    response_matrix = response[0]
    subwatershed = response[1]
    year = response[2]
    month = response[3]
    BMP_num = response[4]
    
    '''landuse_matrix is expressed as %, changed as land decision changes'''
    yield_per_BMP = np.zeros((year.size, month.size, subwatershed.size, BMP_num))
    for i in range(year.size):
        for j in range(month.size):
            yield_per_BMP[i,j,:,:] = np.multiply(response_matrix[i,j,:,:], landuse_matrix)

    yield_sum = np.sum(yield_per_BMP, axis=3)
    yield_sum[:,:,30] = response_matrix[:,:,30,0]
    return yield_per_BMP, yield_sum

'''Step 5.2: estimate landscape yield during month t across all subwatersheds.'''
def loading_landscape(name, landuse_matrix):
    '''
    return
    loading: calculate the sum of landscape loss at each subwatershed: (year, month, subwatershed)
    outlet: outlet at each subwatershed: (year, month, subwatershed)
    unit of loading and outlet: kg/month for nitrate, phosphorus; ton/month for sediment; m3/month for streamflow
    '''
    response = response_mat(name)
    subwatershed = response[1]
    year = response[2]
    month = response[3]

    '''landuse for agri, expressed in ha'''
    land_agri = np.mat(basic_landuse()[1])
    landuse  = basic_landuse()[0]
    total_land = np.mat(landuse.iloc[:,-1]).T
    '''total landuse for agri, expressed in ha'''
    total_land_agri = np.multiply(landuse_matrix, land_agri)
    loading = np.zeros((year.size, month.size, subwatershed.size))
    '''get yield data'''
    yield_data = get_yield(name, landuse_matrix)[1]
    
    '''get loading'''
    for i in range(year.size):
        for j in range(month.size):
            loading[i,j,:] = np.multiply(yield_data[i,j,:], total_land.T)
    return loading

'''Step 6: Estimate in-stream loads at the outlet of each subwatershed'''
def loading_outlet_originalRM(name, landuse_matrix):
    '''
    return a numpy array: (year, month,subwatershed)
    reservoir watershed: 33; downstream of res: 32
    outlet: 34
    '''
    linkage = pd.read_excel(r'./support_data/Watershed_linkage.xlsx', index_col=0)
    loading_BMP_sum = loading_landscape(name, landuse_matrix)
   
    outlet = np.zeros((loading_BMP_sum.shape[0], loading_BMP_sum.shape[2], loading_BMP_sum.shape[1]))
    for i in range(loading_BMP_sum.shape[0]):
        outlet[i,:,:] = np.dot(linkage, loading_BMP_sum[i,:,:].T)
    outlet = np.swapaxes(outlet, 1, 2)
    return outlet

# landuse_matrix= landuse_mat(); landuse_matrix[:,1] = 1
# load_P_linkage = loading_outlet_originalRM('phosphorus', landuse_matrix)
# load_P_linkage1 - load_P_linkage3
