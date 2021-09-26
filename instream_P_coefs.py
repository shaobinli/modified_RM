# -*- coding: utf-8 -*-
"""
Derive instream P - streawmflow relationship

@author: Shaobin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from modified_RM_main import * 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


def streamflow_inv(sw, scenario_name):
    '''original RM'''
    landuse_matrix = landuse_mat()  # (45,56)
    scenario, BMP_list= get_area_prcnt(scenario_name)
    for i in BMP_list:
        landuse_matrix[:,i] = scenario.loc[:,i]
    streamflow = loading_outlet_traditionalRM('streamflow', landuse_matrix)
    return streamflow[:,:,sw]

def phosphorus_instream_coefs(sw, scenario_name):
    '''function used to estimate phosphorus instream coefficients'''
    '''method 1: 1/Q '''
    streamflow = loading_outlet_modifiedRM('streamflow', scenario_name)
    streamflow_sw = streamflow[:,:,sw]
    x2 = 1/streamflow_sw
    
    # '''method 2: 1/yield '''
    # x2 = streamflow_inv(sw, scenario_name)
    
    phosphorus_loss = loading_outlet_modifiedRM('phosphorus', scenario_name) # use original RM to predict 

    x1 = phosphorus_loss[:,:,sw] 
    df = pd.read_csv(r'.\100Randomizations\loading_phosphorus.csv')
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