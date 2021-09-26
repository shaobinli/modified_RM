# -*- coding: utf-8 -*-
"""
Author: Shaobin Li (shaobin@illinois.edu)
Project: NSF INFEWS/T1 (award number: 1739788)

Purpose: comparing traditional RM and modified RM
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from modified_RM_main import loading_outlet_traditionalRM, loading_outlet_modifiedRM
from metrics import pbias, nse
from data import *


'''Start Section: comparison of original SWAT vs RM'''
def swat_vs_RM(name, sw, ag_scenario, plot=True, p_adjust=False, mode='poly'):
    '''note: only works for one specificed sw: traditional RM and modified RM'''
    # get loading from original SWAT results...
    if name == 'nitrate':
        df = pd.read_csv('./100Randomizations/loading_nitrate.csv')
    elif name == 'phosphorus':
        df = pd.read_csv('./100Randomizations/loading_phosphorus.csv')
    elif name == 'sediment':
        df = pd.read_csv('./100Randomizations/loading_sediment.csv')
    elif name == 'streamflow':
        df = pd.read_csv('./100Randomizations/loading_streamflow.csv')
        df = df*30*60*60*24 # m3/month
    subwatershed = df.iloc[:,0].unique()
    year = df.iloc[:,1].unique()
    month = df.iloc[:,2].unique()

    df = df.drop(df.columns[[0,1,2,3]], axis=1)
    df_to_np = np.zeros((year.size, month.size, subwatershed.size, df.shape[1]))
    for i in range(year.size):
        for j in range(month.size):
            df2 = df.iloc[month.size*subwatershed.size*(i):month.size*subwatershed.size*(i+1),:]
            df_to_np[i,j,:,:] = df2.iloc[45*(j):45*(j+1),:]
            
    n = int(ag_scenario[-2:])-1
    df_swat = df_to_np[:,:,sw,n]

    df_iteem = loading_outlet_modifiedRM(name, ag_scenario)
    df_iteem_sw = df_iteem[:,:,sw]
    
    if name =='sediment':
        df_iteem_sw = sediment_instream(sw, ag_scenario, mode)
    if name == 'phosphorus' and p_adjust:
        df_iteem_sw = phosphorus_instream(sw, ag_scenario, reg='linear')
    pbias0 = pbias(obs=df_swat, sim=df_iteem_sw)
    nse0 = nse(obs=df_swat, sim=df_iteem_sw)
    
    '''original RM results'''
    landuse_matrix = landuse_mat()
    scenario, BMP_list= get_area_prcnt(ag_scenario)
    for i in BMP_list:
        landuse_matrix[:,i] = scenario.loc[:,i]
    df_iteem_originalRM = loading_outlet_traditionalRM(name, landuse_matrix)
    df_iteem_sw_originalRM = df_iteem_originalRM[:,:,sw]
    
    pbias_originalRM = pbias(obs=df_swat, sim=df_iteem_sw_originalRM)
    nse_originalRM = nse(obs=df_swat, sim=df_iteem_sw_originalRM)
    
    if name == 'nitrate' or name == 'phosphorus':
        df_point = pd.read_csv('./support_data/SDD_interpolated_2000_2018_Inputs.csv', 
                          parse_dates=['Date'],index_col='Date')
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

        df_iteem_sw_originalRM_point = df_iteem_originalRM[:,:,sw] + df2_point
        pbias_originalRM_point = pbias(obs=df_swat, sim=df_iteem_sw_originalRM_point).round(1)
        nse_originalRM_point = nse(obs=df_swat, sim=df_iteem_sw_originalRM_point).round(2)
    '''End: original RM results'''
    
    if plot == True:
        fig, ax = plt.subplots(figsize=(6.5,4))
        if name == 'phosphorus' and (sw==33 or sw==30 or sw==34):
            plt.text(x=0.1, y=1.07, s= 'Modified RM: P-bias=' + str(pbias0.round(1)) + '%',transform=ax.transAxes)
            plt.text(x=0.85, y=1.07, s= 'NSE=' + str(nse0.round(2)), transform=ax.transAxes)
            # plt.text(x=0.1, y=1.07, s= 'Traditional RM: P-bias= ' + str(pbias_originalRM) + '%',transform=ax.transAxes)
            # plt.text(x=0.85, y=1.07, s= 'NSE=' + str(nse_originalRM),transform=ax.transAxes)
            plt.text(x=0.1, y=1.02, s= 'Traditional RM + point source: P-bias=' + 
                     str(pbias_originalRM_point.round(1)) + '%', transform=ax.transAxes)
            plt.text(x=0.85, y=1.02, s= 'NSE=' + str(nse_originalRM_point.round(2)), transform=ax.transAxes)

        else:
            plt.text(x=0.1, y=1.07, s= 'Nonlinear approximation: P-bias=' + str(pbias0.round(1)) + '%',transform=ax.transAxes)
            plt.text(x=0.65, y=1.07, s= 'NSE=' + str(nse0.round(2)),transform=ax.transAxes)
            plt.text(x=0.1, y=1.02, s= 'Traditional RM: P-bias= ' + str(pbias_originalRM.round(1)) + '%',transform=ax.transAxes)
            plt.text(x=0.65, y=1.02, s= 'NSE=' + str(nse_originalRM.round(2)),transform=ax.transAxes)
            pass 
            
        plt.plot(df_swat.flatten(), color='red', label='SWAT', linewidth=1.5)
        plt.plot(df_iteem_sw.flatten(), color='blue', linestyle='solid', label='Modified RM', linewidth=1.5)
        # plt.plot(df_iteem_sw_originalRM.flatten(), color='blue', linestyle='dashed', label='Traditional RM', linewidth=1.5)
        
        if name =='phosphorus'and (sw==33 or sw==30 or sw==34):
            plt.plot(df_iteem_sw_originalRM_point.flatten(), color='g', linestyle='dashed', label='Traditional RM\n+ point source', linewidth=1.5)
        
        if name == 'streamflow':
            plt.ylabel(name.capitalize() +'\n($\mathregular{m^{3}}$/month)', fontsize=10)
        elif name =='sediment':
            plt.ylabel(name.capitalize() +' loads (ton/month)', fontsize=10)
        else:
            plt.ylabel(name.capitalize() +' loads (kg/month)', fontsize=10)
        
        plt.xlabel('Time (2003-2018)', fontsize=10)
        labels = [2003] + [str(i)[-2:] for i in range(2004,2020)]
        plt.xticks(np.arange(0, 192+1, 12), labels)
        
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))

        # plt.text(0.3, 1.07, 'Subwatershed ' + str(sw+1), transform=ax.transAxes, fontsize=10)
        # plt.text(0.3, 1.02, 'Scenario ' + ag_scenario, transform=ax.transAxes, fontsize=10)        
        # plt.legend(loc='upper left', fontsize=12 )
        if name == 'phosphorus':
            plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(0.40, 0.88),frameon=False,ncol=2,handleheight=2.4, labelspacing=0.05)
        else:
            plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(0.01, 0.88),frameon=False)
        plt.tight_layout()
        plt.show()

    return pbias0, nse0, pbias_originalRM, nse_originalRM, df_swat.flatten(), df_iteem_sw.flatten()

# test = swat_vs_iteem('nitrate', sw=7, ag_scenario='Sheet01')
# test = swat_vs_iteem('phosphorus', sw=33, ag_scenario='Sheet01', p_adjust=True)
# test = swat_vs_iteem('sediment', sw=32, ag_scenario='Sheet01', p_adjust=True, mode='poly')