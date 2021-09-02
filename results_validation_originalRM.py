# -*- coding: utf-8 -*-
"""
Author: Shaobin Li (shaobin@illinois.edu)
Project: INFEWS - ITEEM (NSF award number: 1739788)

Purpose:
Prepare three key components of response matrix method:
    1) connectivity matrix
    2) response matrix
    3) landuse matrix
"""

# import basic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from calendar import monthrange

#-----Function for connectivity matrix-----
def watershed_linkage(**kwargs):
    linkage = pd.read_excel(r'C:\ITEEM\Submodel_SWAT\Watershed_linkage.xlsx').fillna(0)
    nodes = linkage.shape[0]
    linkage_W = np.zeros((nodes,nodes))
    
    for j in range(1,5):
        for i in range (0,nodes):
            if linkage.iloc[i,j] != 0:
                col = int(linkage.iloc[i,j]) - 1
                linkage_W[i,col] = 1
    np.fill_diagonal(linkage_W,-1)
    linkage_W_inv = np.linalg.inv(linkage_W)
    if kwargs:
        print('Outlet is at subbasin', *kwargs.values())
    return linkage_W, linkage_W_inv      
            
# linkage_W = watershed_linkage(outlet=34)
# linkage_W = watershed_linkage()[0]
# linkage_W_inv = watershed_linkage()[1]


#-----Function for response matrix-----
def response_mat(name):
    '''
    return as a tuple
    unit: kg/ha for nitrate, phosphorus, soy, corn, corn silage; ton/ha for sediment; mm for water yield
    '''
    if name == 'nitrate':
        # df = pd.read_excel(r'C:\ITEEM\Submodel_SWAT\Response_matrix_BMPs.xlsx',sheet_name=0)
        df = pd.read_csv(r'C:\ITEEM\Submodel_SWAT\response_matrix_csv\yield_nitrate.csv')
    elif name == 'phosphorus':
        # df = pd.read_excel(r'C:\ITEEM\Submodel_SWAT\Response_matrix_BMPs.xlsx',sheet_name=1)
        df = pd.read_csv(r'C:\ITEEM\Submodel_SWAT\response_matrix_csv\yield_phosphorus.csv')
    elif name == 'sediment':
        # df = pd.read_excel(r'C:\ITEEM\Submodel_SWAT\Response_matrix_BMPs.xlsx',sheet_name=2)
        df = pd.read_csv(r'C:\ITEEM\Submodel_SWAT\response_matrix_csv\yield_sediment.csv')
    elif name == 'streamflow':
        # df = pd.read_excel(r'C:\ITEEM\Submodel_SWAT\Response_matrix_BMPs.xlsx',sheet_name=3)
        df = pd.read_csv(r'C:\ITEEM\Submodel_SWAT\response_matrix_csv\yield_streamflow.csv')        
    elif name == 'soybean':
        df = pd.read_csv(r'C:\ITEEM\Submodel_SWAT\response_matrix_csv\yield_soybean.csv')
    elif name == 'corn':
        df = pd.read_csv(r'C:\ITEEM\Submodel_SWAT\response_matrix_csv\yield_corn.csv')
    elif name == 'corn sillage':
        df = pd.read_csv(r'C:\ITEEM\Submodel_SWAT\response_matrix_csv\yield_corn_silage.csv')
    else:
        raise ValueError('please enter the correct names, e.g., nitrate, phosphorus, sediment')
    
    subwatershed = df.iloc[:,0].unique()
    year = df.iloc[:,1].unique()
    month = df.iloc[:,2].unique()
    area_sw = df.iloc[:,3].unique()
#    response_matrix = df.set_index(['Year','Month'])
    df = df.drop(df.columns[[0,1,2,3]], axis=1)
    
    if name == 'streamflow':
        df = 1/df
        
    df_to_np = np.zeros((year.size, month.size, subwatershed.size, df.shape[1]))
    for i in range(year.size):
        for j in range(month.size):
            df2 = df.iloc[month.size*subwatershed.size*(i):month.size*subwatershed.size*(i+1),:]
#            df = df.reset_index(inplace=False, drop= True)
            df_to_np[i,j,:,:] = df2.iloc[45*(j):45*(j+1),:]
    return df_to_np, subwatershed, year, month, df.shape[1], area_sw


# response_mat_all = response_mat('streamflow')
#response_nitrate = response_mat_all[0]
#reseponse_nitrate_yr1_month1 = response_nitrate[0,0,:,:]


#-----Functions for land use fraction of each BMP at each subwatershed-----
def basic_landuse():
    '''basic case of land use'''
    landuse = pd.read_excel(r'C:\ITEEM\Submodel_SWAT\landuse.xlsx').fillna(0)
    land_agri = landuse.iloc[:,1] + landuse.iloc[:,2]
    land_agri = np.mat(land_agri).T
    ##return as pandas dataframe##
    return landuse, land_agri

# landuse, land_agri = basic_landuse()

def landuse_mat(scenario_name):
    '''
    Return a decison matrix (# of subwatershed, # of BMPs) to decide land use fractions
    of each BMP application in each subwatershed
    '''
    linkage = pd.read_excel(r'C:\ITEEM\Submodel_SWAT\Watershed_linkage.xlsx').fillna(0)
    df = pd.read_csv(r'C:\ITEEM\Submodel_SWAT\response_matrix_csv\yield_nitrate.csv')
    row_sw = linkage.shape[0]
    '''minus 4 to subtract first two columns of subwatershed and area'''
    col_BMP = df.shape[1] - 4
    landuse_matrix = np.zeros((row_sw,col_BMP))
    # scenario_name = 'BMP01' 
    n = int(scenario_name[-2:])
    landuse_matrix[:,n] = 1.0
#     '''Creating a matrix of arbitrary size where rows sum to 1'''
#     if args:
#         if random == 0:
#             landuse_matrix = np.random.rand(row_sw,col_BMP)
#             landuse_matrix = landuse_matrix/landuse_matrix.sum(axis=1)[:,None]
# #            np.sum(landuse_matrix, axis=1)
    return landuse_matrix

# landuse_matrix = landuse_mat('BMP00')
# landuse_matrix[:,0:5] = 0.2


#-----Function for calculating yield of N, P, sediment, streamflow for each subwatershed-----
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
    # landuse_matrix = landuse_mat(scenario_name)
    
    yield_per_BMP = np.zeros((year.size, month.size, subwatershed.size, BMP_num))
    for i in range(year.size):
        for j in range(month.size):
            yield_per_BMP[i,j,:,:] = np.multiply(response_matrix[i,j,:,:], landuse_matrix)

    yield_sum = np.sum(yield_per_BMP, axis=3)
    yield_sum[:,:,30] = response_matrix[:,:,30,0]
    return yield_per_BMP, yield_sum

# yield_sw = get_yield('streamflow', 'BMP50')[1]
# yield_sw_flat = yield_sw.flatten()
# yield_sw_yr1 = yield_sw[0,:,:,:][0]
# yield_sw_yr2 = yield_sw[1,:,:,:][0]

#-----Function for calculating crop yield for each subwatershed-----
def get_yield_crop(name, landuse_matrix):
    '''
    return a tuple: (crop yield per unit (kg/ha) [subwatershed, year], 
    total crop yield per subwatershed (kg) [subwatershed, year] ) 
    calculate crop yield for each subwatershed
    '''
    crop = loading_per_sw(name, landuse_matrix)
    crop[np.isnan(crop)] = 0
    crop_total = np.zeros((crop.shape[2], crop.shape[0]))
    for i in range(crop.shape[2]):
        for j in range(crop.shape[0]):
            crop_total[i,j] = np.sum(crop[j,:,i,:])
    crop_unit = crop_total/basic_landuse()[1]
    crop_unit[np.isnan(crop_unit)] = 0
    return crop_total, crop_unit
    
#crop_corn = get_yield_crop('corn')
    
#-----Function for calculating loadings of N, P, sediment, streamflow for each subwatershed-----
def loading_per_sw(name, landuse_matrix):
    '''
    return a numpy array (year, month, subwatershed)
    calculate the background loading from the yield at each subwatershe
    unit: kg for nitrate, phosphorus; ton for sediment; mm for water 
    '''
    response = response_mat(name)
    response_matrix = response[0]
    subwatershed = response[1]
    year = response[2]
    month = response[3]
    BMP_num = response[4]
    '''landuse_matrix is expressed as %, changed as land decision changes'''
    # landuse_matrix = landuse_mat(scenario_name)
    
    '''landuse for agri, expressed in ha'''
    land_agri = np.mat(basic_landuse()[1])
    landuse  = basic_landuse()[0]
    total_land = np.mat(landuse.iloc[:,-1]).T
    '''total landuse for agri, expressed in ha'''
    total_land_agri = np.multiply(landuse_matrix, land_agri)
    loading = np.zeros((year.size, month.size, subwatershed.size))
    '''get yield data'''
    yield_data = get_yield(name, landuse_matrix)[1]
    # yield_data = get_yield('nitrate', 'Sheet1')
    # test = np.multiply(np_yield_s1[0,0,:],total_land.T)
    '''get loading'''
    for i in range(year.size):
        for j in range(month.size):
            loading[i,j,:] = np.multiply(yield_data[i,j,:], total_land.T)
    # '''add nutrient contribution from urban'''
    # loading[:,:,30] = response_matrix[:,:,30,0]*total_land[30,0]
    return loading

# loading_per_sw_test = loading_per_sw('streamflow')
# loading_per_sw_yr1_month1 = loading_per_sw_test[0,0,:,:]
# loading_per_sw_yr1_month2 = loading_per_sw_test[0,1,:,:]
# loading_per_sw_yr2 = loading_per_sw_test[1,:,:,:]

#-----Function for calculating outlet loading of N, P, sediment, streamflow for each subwatershed-----
def loading_outlet_originalRM(name, landuse_matrix):
    '''
    return a numpy (year, month, watershed)
    reservoir watershed: 33; downstream of res: 32; outlet: 34
    '''
    # name = 'nitrate'
    # scenario_name = 'BMP00'
    # df = pd.read_excel(r'C:\ITEEM\Submodel_SWAT\results_validation\NitrateAndStreamflowAtSub32.xlsx', sheet_name=2)
    # df[np.isnan(df)] = 0
        
    linkage_W_inv = watershed_linkage()[1]
    loading_BMP_sum = loading_per_sw(name, landuse_matrix)
    outlet = np.zeros((loading_BMP_sum.shape[0], loading_BMP_sum.shape[2], loading_BMP_sum.shape[1]))
    for i in range(loading_BMP_sum.shape[0]):
        loading_BMP_sum_minus = np.mat(loading_BMP_sum[i,:,:] * -1).T
        outlet[i,:,:]= np.dot(linkage_W_inv, loading_BMP_sum_minus)
    
    outlet = np.swapaxes(outlet,1,2)
    if name == 'streamflow':
        outlet = outlet*10   # convert mm*ha to m3 by 10
        
    return outlet

# load_P = loading_outlet_originalRM('phosphorus', landuse_matrix)[:,:,33].sum()
