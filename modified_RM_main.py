# -*- coding: utf-8 -*-
"""
Author: Shaobin Li (shaobin@illinois.edu)
Project: NSF INFEWS/T1 (award number: 1739788)

Purpose: Constructing the modified response matrix method to approximate SWAT for computationally intense applications
"""

# Import required packages for data processing
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from data import *  # load data
from point_source import point_source


'''Step 1: enerate a set of response matrices (Y_(m,t)), gathered from SWAT simulation outputs'''
def response_mat(name):
    '''
    Process yield data in csv file into the required data format
    In total 62 BMP combination is simulated, with each BMP representing one column.
    unit: kg/ha for nitrate, phosphorus; ton/ha for sediment; mm for water yield
    '''
    if name == 'nitrate':
        df = df_nitrate   # df_nitrate is stored in "response_matrix_csv" folder and called out from data.py file.
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
    df_to_np = np.zeros((year.size, month.size, subwatershed.size, df.shape[1]))
    for i in range(year.size):
        for j in range(month.size):
            df2 = df.iloc[month.size*subwatershed.size*(i):month.size*subwatershed.size*(i+1),:]
            df_to_np[i,j,:,:] = df2.iloc[45*(j):45*(j+1),:]
    return df_to_np, subwatershed, year, month, df.shape[1], area_sw

'''Steps 2&3: create a decision land use matrix and get land use area'''
def basic_landuse():
    '''
    Purpose: get the total land use area and agricultural land use area for each sbuwatershed;
    Return: total landuse and agricutlural land use.
    '''
    landuse = df_landuse # df_landuse called out from data.py
    land_agri = landuse.iloc[:,1] + landuse.iloc[:,2]
    land_agri = np.mat(land_agri).T
    return landuse, land_agri

def landuse_mat():
    '''
    Return: a decison matrix (# of subwatershed, # of BMPs) to decide land use fractions
    of each BMP application in each subwatershed
    '''
    linkage = df_linkage
    df = df_nitrate
    row_sw = linkage.shape[0]
    '''minus 4 to subtract first two columns of subwatershed and area'''
    col_BMP = df.shape[1] - 4
    landuse_matrix = np.zeros((row_sw,col_BMP))
    # n = int(scenario_name[-2:])
    # landuse_matrix[:,n] = 1.0
    return landuse_matrix

'''auxiliary function'''
def get_area_prcnt(sheet_name):
    '''    
    This function is only used for getting the land use fraction for randomized BMP allocations.
    Each run is a randomized BMP alloations
    this function is not necessary for constructing modified RM.
    Return: area percentage of agricultural land for each BMP
    '''
    # sheet_name = 'Sheet01'
    df = pd.read_excel(xls, sheet_name)
    df2 = df.iloc[:,6:10].reindex()
    BMPs = df2.iloc[0:45,2].unique()  
    BMPs = np.sort(BMPs)
    BMP_list = [int(i) for i in list(BMPs)]
    df_BMP = pd.DataFrame()
    df_temp = pd.DataFrame()
    for i in range(45):      # 45 is the # of subwatershed; needs to be updated as needed.
        df_temp = pd.DataFrame()
        for j in range(len(BMP_list)):
            df3 = df2[(df2.SUBBASIN==i+1) & (df2.BMPsAdopted==BMP_list[j])]
            df4 = pd.DataFrame([df3.iloc[:,-1].sum()])
            df_temp = df_temp.append(df4, ignore_index=True) 
            df_temp_T = df_temp.T
        df_BMP = df_BMP.append(df_temp_T, ignore_index=True)
    
    landuse, land_agri = basic_landuse()
    total_land = np.mat(landuse.iloc[:,-1]).T
    df_BMP_Prcnt = df_BMP/land_agri
    df_BMP_Prcnt.columns = BMP_list
    np_BMP_Prcnt = np.array(df_BMP_Prcnt)
    return df_BMP_Prcnt, BMP_list

# scenario_01, BMP_list = get_area_prcnt('Sheet01')

'''Step 4: develop a connectivity matrix (W) describing the upstream-downstream relationships of all subwatersheds'''
'''This step does not require coding, but rather prepare an excel to represent connectivity matrix'''
'''off-diagonal elements w_(i,j|i≠j ) is equal to one if subwatershed j is upstream of subwatershed i and zero otherwise'''
linkage = pd.read_excel(r'./support_data/Watershed_linkage.xlsx', index_col=0)


'''Step 5.1: estimate landscape yield during month t across all subwatersheds.'''
def get_yield(name, scenario_name):
    '''
    return a tuple containing two numpy array: 
        1) yield_per_BMP: (year, month, subwatershed, BMP)
        2) yield_sum: (year, month, subwatershed)
    unit: kg/ha for nitrate, phosphorus; ton/ha for sediment; mm/ha for water yield
    '''    
    # name = 'phosphorus'; scenario_name = 'Sheet01'
    response = response_mat(name)
    response_matrix = response[0]
    subwatershed = response[1]
    year = response[2]
    month = response[3]
    BMP_num = response[4]
    landuse_matrix = landuse_mat()
    '''landuse_matrix is expressed as %, changed as land decision changes'''
    scenario, BMP_list= get_area_prcnt(scenario_name)
    for i in BMP_list:
        landuse_matrix[:,i] = scenario.loc[:,i]
        
    yield_per_BMP = np.zeros((year.size, month.size, subwatershed.size, BMP_num))
    for i in range(year.size):
        for j in range(month.size):
            yield_per_BMP[i,j,:,:] = np.multiply(response_matrix[i,j,:,:], landuse_matrix)
            
    yield_sum = np.sum(yield_per_BMP, axis=3)
    yield_sum[:,:,30] = response_matrix[:,:,30,0]
    return yield_per_BMP, yield_sum


'''Step 5.2: estimate landscape yield during month t across all subwatersheds.'''
def loading_landscape(name, scenario_name):
    '''
    name = {'nitrate', 'phosphorus', 'sediment', 'streamflow'}
    return a numpy array (year, month, subwatershed)
    calculate the landscape loading from the yield at each subwatershe
    unit: kg for nitrate, phosphorus; ton for sediment; mm for water 
    '''
    response = response_mat(name)
    response_matrix = response[0]
    subwatershed = response[1]
    year = response[2]
    month = response[3]
    BMP_num = response[4]
    landuse_matrix = landuse_mat()
    '''landuse_matrix is expressed as %, changed as land decision changes'''
    scenario, BMP_list= get_area_prcnt(scenario_name)
    for i in BMP_list:
        landuse_matrix[:,i] = scenario.loc[:,i] 
    '''landuse for agri, expressed in ha'''
    land_agri = np.mat(basic_landuse()[1])
    landuse  = basic_landuse()[0]
    total_land = np.mat(landuse.iloc[:,-1]).T
    '''total landuse for agri, expressed in ha'''
    total_land_agri = np.multiply(landuse_matrix, land_agri)
    loading = np.zeros((year.size, month.size, subwatershed.size))
    '''get yield data'''
    yield_data = get_yield(name, scenario_name)[1]
    
    '''get non-point source loading'''
    for i in range(year.size):
        for j in range(month.size):
            loading[i,j,:] = np.multiply(yield_data[i,j,:], total_land.T)
    
    return loading

# landscape_loading_nitrate = loading_landscape('phosphorus', 'Sheet01')

'''Step 6a: traditional RM method to estimate in-stream loads at the outlet of each subwatershed'''
def loading_outlet_traditionalRM(name, scenario_name):
    '''
    return a numpy array: (year, month,subwatershed)
    reservoir watershed: 33; downstream of res: 32
    outlet: 34
    '''
    linkage = pd.read_excel(r'./support_data/Watershed_linkage.xlsx', index_col=0)
    loading_BMP_sum = loading_landscape(name, scenario_name)
    
    '''add point source'''
    loading_BMP_sum = loading_BMP_sum + point_source(name, 30)[1] # 30 is sw index where point souce located

    outlet = np.zeros((loading_BMP_sum.shape[0], loading_BMP_sum.shape[2], loading_BMP_sum.shape[1]))
    for i in range(loading_BMP_sum.shape[0]):
        outlet[i,:,:] = np.dot(linkage, loading_BMP_sum[i,:,:].T) 
    outlet = np.swapaxes(outlet, 1, 2)
    return outlet

# df = loading_outlet_traditionalRM('phosphorus', 'Sheet01')

'''Step 6b: modified RM method to estimate in-stream loads at the outlet of each subwatershed with modifications:'''
def loading_outlet_modifiedRM(name, scenario_name):
    '''
    function used to estimate subwatershed outlet loading of nitrate, phosphus and streamflow
    return a numpy (year, month, watershed)
    reservoir watershed: 32; downstream of res: 31; outlet: 33
    '''
    linkage = pd.read_excel(r'./support_data/Watershed_linkage.xlsx', index_col=0)
    loading_BMP_sum = loading_landscape(name, scenario_name)

    '''******************Step 6.1: Start of reservior trapping effect*******************'''    
    res_downstream = [30, 31, 29, 36, 28, 34, 41, 33, 35, 38, 40, 42, 44, 23, 27]  # subwatersheds located downstream reservoir
    res_upstream = list(set([i for i in range(45)]) - set(res_downstream)) # subwatersheds located upstream reservoir
    linkage_W = linkage
    np_min_trap = np.zeros((loading_BMP_sum.shape[2],1))
    if name == 'phosphorus':
        coef = 1-0.8811; min_trap = 2128.1
    elif name == 'nitrate':
        coef = 1-0.8694; min_trap = 46680.0
    elif name == 'streamflow':
        coef = 1-1.0075; min_trap = 1.9574
    elif name == 'sediment':
        coef = 1-0.294; min_trap = 100.0
    
    for i in range(loading_BMP_sum.shape[2]):
        if i in res_downstream:
            np_min_trap[i] = min_trap
        for j in range(loading_BMP_sum.shape[2]):
            if j in res_upstream and i in res_downstream:
                linkage_W.iloc[i,j] = 1-coef
                
    '''***********************Step 6.2: Start of point source*************************'''
    loading_BMP_sum = loading_BMP_sum + point_source(name, 30)[1] # 30 is subwatershed where point souce located
 
    outlet = np.zeros((loading_BMP_sum.shape[0], loading_BMP_sum.shape[2], loading_BMP_sum.shape[1]))
    for i in range(loading_BMP_sum.shape[0]):
        outlet[i,:,:] = np.dot(linkage_W, loading_BMP_sum[i,:,:].T) - np_min_trap
        outlet[i,:,:] = np.where(outlet[i,:,:]<0, 0, outlet[i,:,:])
    outlet = np.swapaxes(outlet, 1, 2)
    if name == 'streamflow':
        outlet = outlet*10   # convert mm*ha to m3 by 10
    return outlet

# df  = loading_outlet_modifiedRM('phosphorus', 'Sheet01')


'''Step 6.3: P modification'''
def streamflow_inv(sw, scenario_name):
    '''original RM'''
    landuse_matrix = landuse_mat()
    scenario, BMP_list= get_area_prcnt(scenario_name)
    for i in BMP_list:
        landuse_matrix[:,i] = scenario.loc[:,i]
    streamflow = loading_outlet_traditionalRM('streamflow', landuse_matrix)
    return streamflow[:,:,sw]

def phosphorus_instream(sw, scenario_name):
    '''method 1: 1/Q '''
    streamflow = loading_outlet_modifiedRM('streamflow', scenario_name)
    streamflow_sw = streamflow[:,:,sw]
    x2 = 1/streamflow_sw 
    p_loss = loading_outlet_modifiedRM('phosphorus', scenario_name) # use original RM to predict 
    x1 = p_loss[:,:,sw]
    pd_coef = pd.read_excel(r'.\support_data\phosphorus_streamflow_regression_coefs.xlsx', usecols='B:D', sheet_name='invert q')
    p_instream = pd_coef.iloc[sw,0]*x1 + pd_coef.iloc[sw,1]*x2
    p_instream = np.where(p_instream<0, 0, p_instream)
    return p_instream
'''End Section: P modification'''



'''Step 6.4: sediment modification'''
def sediment_instream(sw, scenario_name, mode='poly'):
    '''apply linear or polynomial regression to estiamte instream sediment'''
    streamflow = loading_outlet_modifiedRM('streamflow', scenario_name)
    streamflow = streamflow[:,:,sw]
    if mode == 'linear':
        pd_coef = pd.read_excel(r'.\support_data\sediment_streamflow_regression_coefs.xlsx', sheet_name='linear', usecols='B:C')
        sediment = pd_coef.iloc[sw,1]*streamflow + pd_coef.iloc[sw,0]
    if mode == 'poly':
        pd_coef = pd.read_excel(r'.\support_data\sediment_streamflow_regression_coefs.xlsx', sheet_name='poly', usecols='B:D')
        sediment = pd_coef.iloc[sw,0]*streamflow**2 + pd_coef.iloc[sw,1]*streamflow + pd_coef.iloc[sw,2]
    sediment = np.where(sediment<0, 0, sediment)
    return sediment
'''End Section: Sediment modification'''

