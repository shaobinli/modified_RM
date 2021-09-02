# -*- coding: utf-8 -*-
"""
Author: Shaobin Li (shaobin@illinois.edu)
Project: INFEWS - ITEEM (NSF award number: 1739788)

Purposes:
Prepare three key components of response matrix method:
    1) connectivity matrix
    2) response matrix
    3) landuse matrix
"""

# import packages
import pandas as pd
import numpy as np
from calendar import monthrange
from data import *


#-----Function for connectivity matrix-----
def watershed_linkage(**kwargs):
    linkage = df_linkage
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

#-----Function for response matrix-----
def response_mat(name):
    '''
    sa = sensitivity analysis
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
    # response_matrix = df.set_index(['Year','Month'])
    df = df.drop(df.columns[[0,1,2,3]], axis=1)
    df_to_np = np.zeros((year.size, month.size, subwatershed.size, df.shape[1]))
    for i in range(year.size):
        for j in range(month.size):
            df2 = df.iloc[month.size*subwatershed.size*(i):month.size*subwatershed.size*(i+1),:]
            # df = df.reset_index(inplace=False, drop= True)
            df_to_np[i,j,:,:] = df2.iloc[45*(j):45*(j+1),:]
    return df_to_np, subwatershed, year, month, df.shape[1], area_sw

# response_mat_all = response_mat('phosphorus')[0][:,:,7,0]
#response_nitrate = response_mat_all[0]

#-----Functions for land use fraction of each BMP at each subwatershed-----
def basic_landuse():
    '''basic case of land use'''
    landuse = df_landuse
    land_agri = landuse.iloc[:,1] + landuse.iloc[:,2]
    land_agri = np.mat(land_agri).T
    return landuse, land_agri

# landuse, land_agri = basic_landuse()

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

# landuse_matrix = landuse_mat('BMP55')
# landuse_matrix[:,0:5] = 0.2
# landuse_matrix = np.zeros((45,56))
# landuse_matrix[:, 5] = 1.0

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
    # landuse_matrix = landuse_matrix_combinedS1
    '''landuse_matrix is expressed as %, changed as land decision changes'''
    # landuse_matrix = landuse_mat(scenario_name)
    yield_per_BMP = np.zeros((year.size, month.size, subwatershed.size, BMP_num))
    for i in range(year.size):
        for j in range(month.size):
            yield_per_BMP[i,j,:,:] = np.multiply(response_matrix[i,j,:,:], landuse_matrix)

    yield_sum = np.sum(yield_per_BMP, axis=3)
    yield_sum[:,:,30] = response_matrix[:,:,30,0]
    return yield_per_BMP, yield_sum

# landuse_matrix = np.zeros((45,62)); landuse_matrix[:,1] = 1
# yield_sw = get_yield('streamflow', landuse_matrix)[1]
# yield_sum = yield_sw.sum(axis=1)
# yield_ave = yield_sum.mean(axis=0)
# yield_sw_flat = yield_sw.flatten()
# yield_sw_yr1 = yield_sw[0,:,:,:][0]
# yield_sw_yr2 = yield_sw[1,:,:,:][0]

#-----Function for calculating loadings of N, P, sediment, streamflow for each subwatershed-----
def loading_landscape(name, landuse_matrix):
    '''
    return
    loading: calculate the sum of landscape loss at each subwatershed: (year, month, subwatershed)
    outlet: outlet at each subwatershed: (year, month, subwatershed)
    unit of loading and outlet: kg/month for nitrate, phosphorus; ton/month for sediment; m3/month for streamflow
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
    '''get background loading'''
    for i in range(year.size):
        for j in range(month.size):
            loading[i,j,:] = np.multiply(yield_data[i,j,:], total_land.T)
    # '''add nutrient contribution from urban'''
    # loading[:,:,30] = response_matrix[:,:,30,0]*total_land[30,0]
    '''get landscape outlet''' 
    linkage_W_inv = watershed_linkage()[1]
    loading_BMP_sum = loading
    outlet = np.zeros((loading_BMP_sum.shape[0], loading_BMP_sum.shape[2], loading_BMP_sum.shape[1]))
    for i in range(loading_BMP_sum.shape[0]):
        loading_BMP_sum_minus = np.mat(loading_BMP_sum[i,:,:] * -1).T
        outlet[i,:,:]= np.dot(linkage_W_inv, loading_BMP_sum_minus)
    
    outlet = np.swapaxes(outlet,1,2)
    if name == 'streamflow':
        outlet = outlet*10   # convert mm*ha to m3 by 10
    return loading, outlet

# landuse_matrix = np.zeros((45,62)); landuse_matrix[:,1] = 1
# loading, outlet = loading_landscape('sediment', landuse_matrix)
# loading_sw_yr2 = loading_sw_test[1,:,:,:]

#-----Function for calculating outlet loading of N, P, sediment, streamflow for each subwatershed-----
def loading_outlet_USRW(name, landuse_matrix, tech_wwt='AS', nutrient_index=1.0, flow_index=1.0):
    '''
    return a numpy array: (year, month,subwatershed)
    reservoir watershed: 33; downstream of res: 32
    outlet: 34
    '''
    df = df_linkage2
    df[np.isnan(df)] = 0
    # name = 'nitrate'
    # scenario_name = 'BMP00'
    loading_BMP_sum = loading_landscape(name, landuse_matrix)[0]
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
    elif name =='sediment':
        res_out = 14.133*res_in**0.6105     # equationd derived from data
        # res_out = res_in
    elif name =='streamflow':
        res_out = res_in * 1.0075 - 1.9574  # equationd derived from data
    res_out = np.where(res_out<0, 0, res_out)
        
    # sw32 is the downstream of reservoir
    outlet[:,:,31] = loading_BMP_sum[:,:,31] + res_out
    '''******************End of reservior trapping effect*******************'''
    
    # update loading in SDD (sw31)
    outlet[:,:,30] = loading_BMP_sum[:,:,30] + outlet[:,:,31]
    
    '''***********************Start of point source*************************'''
    # name = 'nitrate'
    if tech_wwt == 'AS':
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
            # Calculate loading in sw31 with point source
            # loading_BMP_sum[i,j,30] = ANN...
            if name =='nitrate':
                # point_Nitrate = 1315.43*30 # kg/month, average
                outlet[:,:,30] = loading_BMP_sum[:,:,30] + outlet[:,:,31] + df2_point
            elif name == 'phosphorus':
                # point_TP = 1923.33*30# kg/month, average
                outlet[:,:,30] = loading_BMP_sum[:,:,30] + outlet[:,:,31] + df2_point
                
    # Calculate loading in sw31 with point source
    elif tech_wwt != 'AS':
        if name == 'nitrate' or name == 'phosphorus':
            instance = WWT_SDD(tech=tech_wwt, multiyear=True, start_yr=2003, end_yr=2018)
            output_scaled, output_raw, influent_tot = instance.run_model(sample_size=1000, nutrient_index=nutrient_index, flow_index=flow_index)
    
            if name == 'nitrate':
                nitrate_load = output_raw[:,:,0]*influent_tot[:,:,0]
                loading_day = nitrate_load.mean(axis=1)/1000  # loading: kg/d
                loading_day = loading_day.reshape(16,12)
                
            elif name == 'phosphorus':
                tp_load = output_raw[:,:,2]*influent_tot[:,:,0]
                loading_day = tp_load.mean(axis=1)/1000  # loading: kg/d
                loading_day = loading_day.reshape(16,12)

            loading_month = np.zeros((16,12))    #16 yr, 12 month
            for i in range(16):
                for j in range(12):
                        loading_month[i,j] = loading_day[i,j]*monthrange(2003+i,j+1)[1] # loading: kg/month
            if name =='nitrate':
                outlet[:,:,30] = loading_BMP_sum[:,:,30] + outlet[:,:,31] + loading_month
            elif name == 'phosphorus':
                outlet[:,:,30] = loading_BMP_sum[:,:,30] + outlet[:,:,31] + loading_month
    '''***********************End of point source***************************'''

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
    if name =='phosphorus':
        outlet = outlet/1.07  # 7% overestimates across all BMPs
    return outlet

# landuse_matrix = np.zeros((45,62)); landuse_matrix[:,1]=1; 
# landuse_matrix[7,1]=0;landuse_matrix[7,30]=1
# tp = loading_outlet_USRW('phosphorus', landuse_matrix, 'AS')
# tp_sw8 = tp[:,:,7].mean()

# nitrate = loading_outlet_USRW('sediment', landuse_matrix, 'ASCP')[:,:,33].sum(axis=1).mean()
# landuse_matrix = np.zeros((45,56))
# landuse_matrix[:,48]=1
# sediment = loading_outlet_USRW('sediment', landuse_matrix, 'AS')[:,:,33].sum(axis=1).mean()

def sediment_instream(sw, landuse_matrix):
    streamflow = loading_outlet_USRW('streamflow', landuse_matrix, 'AS')
    streamflow = streamflow[:,:,sw]
    pd_coef_poly = df_pd_coef_poly
    sediment = pd_coef_poly.iloc[sw,0]*streamflow**2 + pd_coef_poly.iloc[sw,1]*streamflow + pd_coef_poly.iloc[sw,2]
    sediment = np.where(sediment<0, 0, sediment)
    return sediment

# start = time.time()
# test_sed = sediment_instream(32, landuse_matrix).sum(axis=1).mean()
# BMP0_sed_outlet = sediment_instream(33, landuse_matrix).sum(axis=1).mean()
# BMP0_sed_above = sediment_instream(26, landuse_matrix).sum(axis=1).mean()
# BMP0_sed_lake = sediment_instream(32, 'BMP00').sum(axis=1).mean()
# end = time.time()
# print('simulation time is {:.1f} seconds'.format(end-start))