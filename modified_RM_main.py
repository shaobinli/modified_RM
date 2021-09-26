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
    df = pd.read_excel(xls, sheet_name)
    df = pd.read_excel(xls, 'Sheet01')
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
# scenario_01.sum(axis=1)

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
    name = 'phosphorus'; scenario_name = 'Sheet01'
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
    '''get loading'''
    for i in range(year.size):
        for j in range(month.size):
            loading[i,j,:] = np.multiply(yield_data[i,j,:], total_land.T)
    return loading

# landscape_loading_nitrate = loading_landscape('nitrate', 'Sheet01')

'''Step 6a: traditional RM method to estimate in-stream loads at the outlet of each subwatershed'''
def loading_outlet_traditionalRM(name, landuse_matrix):
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

'''Step 6b: modified RM method to estimate in-stream loads at the outlet of each subwatershed with modifications:'''
def loading_outlet_modifiedRM(name, scenario_name):
    '''
    return a numpy (year, month, watershed)
    reservoir watershed: 33; downstream of res: 32; outlet: 34
    '''
    df = pd.read_excel(r'.\support_data\Watershed_linkage_v2.xlsx')
    df[np.isnan(df)] = 0
    loading_BMP_sum = loading_landscape(name, scenario_name)
    outlet = np.zeros((loading_BMP_sum.shape[0], loading_BMP_sum.shape[1], loading_BMP_sum.shape[2]))
    
    sw_upstream = [i for i in range(33)]
    
    return 


def loading_outlet_modifiedRM(name, scenario_name):
    '''
    function used to estimate loading of nitrate, phosphus and streamflow
    return a numpy (year, month, watershed)
    reservoir watershed: 33; downstream of res: 32; outlet: 34
    '''
    df = pd.read_excel(r'.\support_data\Watershed_linkage_v2.xlsx')
    df[np.isnan(df)] = 0
    loading_BMP_sum = loading_landscape(name, scenario_name)
    outlet = np.zeros((loading_BMP_sum.shape[0], loading_BMP_sum.shape[1], loading_BMP_sum.shape[2]))
    for i in range(33):
        a = df.loc[i].unique().astype('int')
        a = a[a!=0]
        for j in a:
            outlet[:,:,i] += loading_BMP_sum[:,:,j-1]     
    # Total loading in sw32 = res_out + background loading
    
    '''******************Start of reservior trapping effect*******************'''
    res_in = outlet[:,:,32]
    if name == 'nitrate':
        res_out = res_in * 0.8694 - 46680.0 # equationd derived from reservoir file in SWAT using nitrate_in and nitrate_out
    elif name =='phosphorus':
        res_out = res_in * 0.8811 - 2128.1  # equationd derived from reservoir file in SWAT using P_in and P_out
    elif name =='streamflow':
        res_out = res_in * 1.0075 - 1.9574  # equationd derived from reservoir file in SWAT using flow_in and flow_out
    res_out = np.where(res_out<0, 0, res_out)
        
    # sw32 is the downstream of reservoir
    outlet[:,:,31] = loading_BMP_sum[:,:,31] + res_out
    '''******************End of reservior trapping effect*******************'''
    # update loading in SDD (sw31)
    outlet[:,:,30] = loading_BMP_sum[:,:,30] + outlet[:,:,31]
    
    '''***********************Start of point source*************************'''
    if name == 'nitrate' or name == 'phosphorus':
        df_point = df_point_SDD
        if name == 'nitrate':
            df_point = pd.DataFrame(df_point.iloc[:,0])
        elif name == 'phosphorus':
            df_point = pd.DataFrame(df_point.iloc[:,1])  
        df_point['month'] = df_point.index.month
        df_point['year'] = df_point.index.year
        df2_point = np.zeros((16,12))
        for i in range(16):
            for j in range(12):
                df2_point[i,j] = df_point.loc[(df_point.year==2003+i) & (df_point.month==1+j)].iloc[:,0].astype('float').sum()
        if name =='nitrate':
            outlet[:,:,30] = loading_BMP_sum[:,:,30] + outlet[:,:,31] + df2_point
        elif name == 'phosphorus':
            outlet[:,:,30] = loading_BMP_sum[:,:,30] + outlet[:,:,31] + df2_point
    '''***********************End of point source*************************'''

    # b contains all upstream subwatersheds for sw31
    b = df.loc[30].unique().astype('int')
    b = b[b!=0]
    # get unique subwatersheds that do not contribute to reservoir
    for i in range(33,45):
        c = df.loc[i].unique().astype('int')
        c = c[c!=0]
        d = list(set(c) - set(b))
        # Key step: the following equation takes the trapping efficiency into account. 
        # All upstream contributions of sw32 is reconsidered with trapping efficiency 
        if 31 in list(c):
            # print ('true, i=',i)
            outlet[:,:,i] = outlet[:,:,30]
        for j in d:
            outlet[:,:,i] += loading_BMP_sum[:,:,j-1]
    # update the loadings for upperstream that has higher values
    e = b[b>33] 
    for i in e:
        f = df.loc[i-1].unique().astype('int')
        f = f[f!=0]
        for j in f:
            outlet[:,:,i-1] += loading_BMP_sum[:,:,j-1]
    if name == 'streamflow':
        outlet = outlet*10   # convert mm*ha to m3 by 10

    return outlet


'''Start Section: sediment modification'''
def sediment_instream(sw, scenario_name, mode='poly'):
    '''apply '''
    streamflow = loading_outlet_USRW('streamflow', scenario_name)
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


'''Start Section: P modification'''
def streamflow_inv(sw, scenario_name):
    '''original RM'''
    landuse_matrix = landuse_mat()  # (45,56)
    scenario, BMP_list= get_area_prcnt(scenario_name)
    for i in BMP_list:
        landuse_matrix[:,i] = scenario.loc[:,i]
    streamflow = loading_outlet_traditionalRM('streamflow', landuse_matrix)
    return streamflow[:,:,sw]

def phosphorus_instream(sw, scenario_name, reg):
    '''method 1: 1/Q '''
    streamflow = loading_outlet_USRW('streamflow', scenario_name)
    streamflow_sw = streamflow[:,:,sw]
    x2 = 1/streamflow_sw 
    
    # '''method 2: 1/yield '''
    # x2 = streamflow_inv(sw, scenario_name)

    p_loss = loading_outlet_USRW('phosphorus', scenario_name) # use original RM to predict 
    x1 = p_loss[:,:,sw]
    if reg == 'linear':
        pd_coef = pd.read_excel(r'.\support_data\phosphorus_streamflow_regression_coefs.xlsx', usecols='B:D', sheet_name='invert q')
        # pd_coef = pd.read_excel(r'.\support_data\phosphorus_streamflow_regression_coefs.xlsx', usecols='B:D', sheet_name='invert_yield')
        p_instream = pd_coef.iloc[sw,0]*x1 + pd_coef.iloc[sw,1]*x2
    if reg =='interaction':
        pd_coef = pd.read_excel(r'.\support_data\phosphorus_streamflow_regression_coefs.xlsx', usecols='B:D',sheet_name=1)
        x3 = x1*x2
        p_instream = pd_coef.iloc[sw,0]*x1 + pd_coef.iloc[sw,1]*x2 + pd_coef.iloc[sw,2]*x3
    p_instream = np.where(p_instream<0, 0, p_instream)
    return p_instream

# df_iteem = loading_outlet_USRW('phosphorus', 'Sheet01')
# df_iteem_sw = df_iteem[:,:,33].flatten()
# a = phosphorus_instream(33, 'Sheet01', reg='interaction').flatten()
'''End Section: P modification'''

# test_N_1D = test_N.flatten()
# start = time.time()
# trial_list1 = ['Sheet0' + str(i) for i in range(1,10)]
# trial_list2 = ['Sheet' + str(i) for i in range(10,101)]
# trial_list = trial_list1 + trial_list2
# for i in trial_list:
#     test_TP = loading_outlet_USRW('phosphorus', i)
#     test_nitrate = loading_outlet_USRW('nitrate', i)
#     test_streamflow = loading_outlet_USRW('streamflow', i)
#     test_sediment = loading_outlet_USRW('streamflow', i)
# end = time.time()
# print('simulation time is {:.1f} miniutes'.format((end-start)/60))