# -*- coding: utf-8 -*-
"""
Author: Shaobin Li (shaobin@illinois.edu)
Project: INFEWS - ITEEM (NSF award number: 1739788)

Purpose: validation test of response matrix method for SWAT
"""

# Import required packages for data processing
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from metrics import pbias, nse
from results_validation_originalRM import loading_outlet_originalRM
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path
from data import *


def response_mat(name):
    '''
    Process yield data in csv file into the required data format
    return as a tuple
    unit: kg/ha for nitrate, phosphorus, soy, corn, corn silage; ton/ha for sediment; mm for water yield
    '''
    if name == 'nitrate':
        # df = pd.read_excel(r'C:\ITEEM\Submodel_SWAT\Response_matrix_BMPs.xlsx',sheet_name=0)
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
    df_to_np = np.zeros((year.size, month.size, subwatershed.size, df.shape[1]))
    for i in range(year.size):
        for j in range(month.size):
            df2 = df.iloc[month.size*subwatershed.size*(i):month.size*subwatershed.size*(i+1),:]
            df_to_np[i,j,:,:] = df2.iloc[45*(j):45*(j+1),:]
    return df_to_np, subwatershed, year, month, df.shape[1], area_sw


def basic_landuse():
    '''
    Purpose: get the total land use and agricultural land use for each sbuwatershed
    Return: total landuse and agricutlural land use.
    '''
    landuse = df_landuse
    land_agri = landuse.iloc[:,1] + landuse.iloc[:,2]
    land_agri = np.mat(land_agri).T
    return landuse, land_agri

# a = basic_landuse

def landuse_mat(scenario_name):
    '''
    Return a decison matrix (# of subwatershed, # of BMPs) to decide land use fractions
    of each BMP application in each subwatershed
    '''
    linkage = df_linkage
    df = df_nitrate
    row_sw = linkage.shape[0]
    '''minus 4 to subtract first two columns of subwatershed and area'''
    col_BMP = df.shape[1] - 4
    landuse_matrix = np.zeros((row_sw,col_BMP))
    n = int(scenario_name[-2:])
    landuse_matrix[:,n] = 1.0
    return landuse_matrix


def get_area_prcnt(sheet_name):
    '''    
    return area percentage of agricultural land for each BMP
    '''
    df = pd.read_excel(xls, sheet_name)
    # pd.read_excel(xls, 'Sheet01')
    df2 = df.iloc[:,6:10].reindex()
    BMPs = df2.iloc[0:45,2].unique()
    BMPs = np.sort(BMPs)
    BMP_list = [int(i) for i in list(BMPs)]
    df_BMP = pd.DataFrame()
    df_temp = pd.DataFrame()
    for i in range(45):
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
# scenario_02, BMP_list = get_area_prcnt('Sheet02')
# scenario_02.sum(axis=1)

def get_yield(name, scenario_name):
    '''
    return a tuple containing two numpy array: 
        1) yield_per_BMP: (year, month, subwatershed, BMP)
        2) yield_sum: (year, month, subwatershed)
    unit: kg/ha for nitrate, phosphorus; ton/ha for sediment; mm/ha for water yield
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
        
    yield_per_BMP = np.zeros((year.size, month.size, subwatershed.size, BMP_num))
    for i in range(year.size):
        for j in range(month.size):
            yield_per_BMP[i,j,:,:] = np.multiply(response_matrix[i,j,:,:], landuse_matrix)
            
    yield_sum = np.sum(yield_per_BMP, axis=3)
    yield_sum[:,:,30] = response_matrix[:,:,30,0]
    return yield_per_BMP, yield_sum

# yield_sum_sheet02 = get_yield('nitrate','Sheet02')[1]
# yield_sheet01 = get_yield('nitrate','Sheet01')[0][:,:,7,:]
# sw8_sheet08 = yield_sum_sheet02[:,:,7].flatten()

def get_yield_1D(name, sheet_name):  
    ''' for yield data validations
    name represents pollutant category
    sheet_name represents scenarios 
    '''
    scenario = get_area_prcnt(sheet_name)[0]
    scenario = np.array(scenario)
    yield_s1 = get_yield(name, sheet_name)[1]
    return yield_s1.flatten()

# yield_data_s1_tot_1D = get_yield_1D('nitrate','Sheet1')
# yield_data_1D_streamflow = get_yield_1D('streamflow','Sheet1')


def loading_per_sw(name, scenario_name):
    '''
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
    # '''add nutrient contribution from urban'''
    # loading[:,:,30] = response_matrix[:,:,30,0]*total_land[30,0]
    return loading

# landscape_loading_nitrate = loading_per_sw('nitrate', 'Sheet01')

def loading_outlet_USRW(name, scenario_name):
    '''
    return a numpy (year, month, watershed)
    reservoir watershed: 33; downstream of res: 32; outlet: 34
    '''
    df = pd.read_excel(r'C:\ITEEM\Submodel_SWAT\Watershed_linkage_v2.xlsx')
    df[np.isnan(df)] = 0
    loading_BMP_sum = loading_per_sw(name, scenario_name)
    outlet = np.zeros((loading_BMP_sum.shape[0], loading_BMP_sum.shape[1], loading_BMP_sum.shape[2]))
    for i in range(33):
        a = df.loc[i].unique().astype('int')
        a = a[a!=0]
        for j in a:
            # print (j)
            outlet[:,:,i] += loading_BMP_sum[:,:,j-1]     
    # Total loading in sw32 = res_out + background loading
    '''******************Start of reservior trapping effect*******************'''
    res_in = outlet[:,:,32]
    if name == 'nitrate':
        res_out = res_in * 0.8694 - 46680.0 # equationd derived from data
    elif name =='phosphorus':
        res_out = res_in * 0.8811 - 2128.1  # equationd derived from data
    elif name =='streamflow':
        res_out = res_in * 1.0075 - 1.9574  # equationd derived from data
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
            # point_Nitrate = 1315.43*30 # kg/month, average
            outlet[:,:,30] = loading_BMP_sum[:,:,30] + outlet[:,:,31] + df2_point
        elif name == 'phosphorus':
            # point_TP = 1923.33*30# kg/month, average
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
    # add adjustment coefficient
    # if name =='phosphorus':
        # outlet[:,:,33] = outlet[:,:,33]/1.07  # 1.07% overestimates across all BMPs
    return outlet


'''Start Section: sediment modification'''
def sediment_instream(sw, scenario_name, mode='poly'):
    # scenario_name= 'Sheet01'; sw=25
    streamflow = loading_outlet_USRW('streamflow', scenario_name)
    streamflow = streamflow[:,:,sw]
    if mode == 'linear':
        pd_coef = pd.read_excel(r'C:\ITEEM\Submodel_SWAT\results_validation\sediment_streamflow_regression_coefs.xlsx', sheet_name='linear', usecols='B:C')
        sediment = pd_coef.iloc[sw,1]*streamflow + pd_coef.iloc[sw,0]
    if mode == 'poly':
        pd_coef = pd.read_excel(r'C:\ITEEM\Submodel_SWAT\results_validation\sediment_streamflow_regression_coefs.xlsx', sheet_name='poly', usecols='B:D')
        sediment = pd_coef.iloc[sw,0]*streamflow**2 + pd_coef.iloc[sw,1]*streamflow + pd_coef.iloc[sw,2]
    if mode == 'supply':
        pd_coef = pd.read_excel(r'C:\ITEEM\Submodel_SWAT\results_validation\sediment_streamflow_regression_coefs.xlsx', sheet_name='supply', usecols='B:C')
        sediment_loss = loading_outlet_USRW('sediment', scenario_name)[:,:,sw]
        sediment = sediment_loss + pd_coef.iloc[sw,1]*streamflow
    sediment = np.where(sediment<0, 0, sediment)
    return sediment
# sediment = sediment_instream(25, 'Sheet01', mode='supply')
'''End Section: Sediment modification'''


# test_N_1D = test_N.flatten()
start = time.time()
trial_list1 = ['Sheet0' + str(i) for i in range(1,10)]
trial_list2 = ['Sheet' + str(i) for i in range(10,101)]
trial_list = trial_list1 + trial_list2
for i in trial_list:
    test_TP = loading_outlet_USRW('phosphorus', i)
    test_nitrate = loading_outlet_USRW('nitrate', i)
    test_streamflow = loading_outlet_USRW('streamflow', i)
    test_sediment = loading_outlet_USRW('streamflow', i)
end = time.time()
print('simulation time is {:.1f} miniutes'.format((end-start)/60))


'''Start Section: P modification'''
def streamflow_inv(sw, scenario_name):
    '''original RM'''
    landuse_matrix = landuse_mat()  # (45,56)
    scenario, BMP_list= get_area_prcnt(scenario_name)
    for i in BMP_list:
        landuse_matrix[:,i] = scenario.loc[:,i]
    streamflow = loading_outlet_originalRM('streamflow', landuse_matrix)
    # streamflow_max = streamflow.max(axis=(0,1)) # max flow for each sw
    return streamflow[:,:,sw]
    
from sklearn.linear_model import LinearRegression
def phosphorus_instream_coefs(sw, scenario_name):
    # '''method 1: 1/Q '''
    # streamflow = loading_outlet_USRW('streamflow', scenario_name)
    # streamflow_sw = streamflow[:,:,sw]
    # x2 = 1/streamflow_sw
    '''method 2: 1/yield '''
    x2 = streamflow_inv(sw, scenario_name)
    
    phosphorus_loss = loading_outlet_USRW('phosphorus', scenario_name) # use original RM to predict 
    # phosphorus_loss_original = loading_outlet_originalRM('phosphorus', landuse_matrix) # use original RM to predict 
    # sw=33
    x1 = phosphorus_loss[:,:,sw] 
    df = pd.read_csv(r'C:\ITEEM\Submodel_SWAT\results_validation\100Randomizations\loading_phosphorus_March2021.csv')
    subwatershed = df.iloc[:,0].unique()
    year = df.iloc[:,1].unique()
    month = df.iloc[:,2].unique()
    # area_sw = df.iloc[:,3].unique()
    # response_matrix = df.set_index(['Year','Month'])
    df = df.drop(df.columns[[0,1,2,3]], axis=1)
    df_to_np = np.zeros((year.size, month.size, subwatershed.size, df.shape[1]))
    for i in range(year.size):
        for j in range(month.size):
            df2 = df.iloc[month.size*subwatershed.size*(i):month.size*subwatershed.size*(i+1),:]
            # df = df.reset_index(inplace=False, drop= True)
            df_to_np[i,j,:,:] = df2.iloc[45*(j):45*(j+1),:]
            
    n = int(scenario_name[-2:])-1
    df_swat = df_to_np[:,:,sw,n]
    y = df_swat
    # x3 = x1*x2
    reg = LinearRegression(fit_intercept=False)
    X = np.array((x1.flatten(), x2.flatten())).T
    reg.fit(X, y.flatten())
    reg.score(X,y.flatten())
    return reg.coef_, reg.intercept_, reg.score(X,y.flatten()), df_swat, x1.flatten(), x2.flatten()

# p_data = phosphorus_instream_coefs(33, 'Sheet01')
# pd_coef_kpp = pd.DataFrame(columns=['k_p,p'])
# pd_coef_kpq = pd.DataFrame(columns=['k_p,q'])
# pd_coef_r2 = pd.DataFrame(columns=['r2'])
# for sw in range(45):
#         coef, _, r2, _,_,_ = phosphorus_instream_coefs(sw, 'Sheet01')
#         pd_coef_kpp.loc[sw] = coef[0]
#         pd_coef_kpq.loc[sw] = coef[1]
#         # pd_coef_c.loc[sw] = coef[2]
#         pd_coef_r2.loc[sw] = r2
def phosphorus_instream(sw, scenario_name, reg):
    '''method 1: 1/Q '''
    streamflow = loading_outlet_USRW('streamflow', scenario_name)
    streamflow_sw = streamflow[:,:,sw]
    x2 = 1/streamflow_sw 
    
    '''method 2: 1/yield '''
    x2 = streamflow_inv(sw, scenario_name)

    p_loss = loading_outlet_USRW('phosphorus', scenario_name) # use original RM to predict 
    x1 = p_loss[:,:,sw]
    if reg == 'linear':
        # pd_coef = pd.read_excel(r'C:\ITEEM\Submodel_SWAT\results_validation\phosphorus_streamflow_regression_coefs.xlsx', usecols='B:D', sheet_name='invert q')
        pd_coef = pd.read_excel(r'C:\ITEEM\Submodel_SWAT\results_validation\phosphorus_streamflow_regression_coefs.xlsx', usecols='B:D', sheet_name='invert_yield')
        p_instream = pd_coef.iloc[sw,0]*x1 + pd_coef.iloc[sw,1]*x2
    if reg =='interaction':
        pd_coef = pd.read_excel(r'C:\ITEEM\Submodel_SWAT\results_validation\phosphorus_streamflow_regression_coefs.xlsx', usecols='B:D',sheet_name=1)
        x3 = x1*x2
        p_instream = pd_coef.iloc[sw,0]*x1 + pd_coef.iloc[sw,1]*x2 + pd_coef.iloc[sw,2]*x3
    p_instream = np.where(p_instream<0, 0, p_instream)
    return p_instream
# df_iteem = loading_outlet_USRW('phosphorus', 'Sheet01')
# df_iteem_sw = df_iteem[:,:,33].flatten()
# a = phosphorus_instream(33, 'Sheet01', reg='interaction').flatten()
'''End Section: P modification'''
