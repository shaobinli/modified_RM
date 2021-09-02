# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 22:23:01 2020

@author: Shaobin
"""
from Submodel_SWAT.SWAT_functions import *
from Submodel_SWAT.results_validation_originalRM import loading_outlet_originalRM

from Submodel_SWAT.Analysis_RM_randomized import get_area_prcnt
import pandas as pd
import numpy as np
import numpy
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
# import matplotlib.ticker as mtick

import seaborn as sns
import time
from calendar import monthrange
from Submodel_WWT.SDD_analysis.wwt_model_SDD import WWT_SDD
from Submodel_WWT.SDD_analysis.influent_SDD import influent_SDD
import scipy.io

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


'''Plots for RM manuscript'''
def swat_vs_tradional_RM(name, sw, ag_scenario, landuse_matrix):
    '''compare traditional RM with SWAT'''
    # ag_scenario = 'BMP00'
    # name = 'nitrate'
    # sw = 32
    # Step 1: get loading from original SWAT results...
    if name == 'nitrate':
        # df = pd.read_excel(r'C:\ITEEM\Submodel_SWAT\Response_matrix_BMPs.xlsx',sheet_name=0)
        df = pd.read_csv(r'C:\ITEEM\Submodel_SWAT\response_matrix_csv\loading_nitrate_Feb2021.csv')
    elif name == 'phosphorus':
        # df = pd.read_excel(r'C:\ITEEM\Submodel_SWAT\Response_matrix_BMPs.xlsx',sheet_name=1)
        df = pd.read_csv(r'C:\ITEEM\Submodel_SWAT\response_matrix_csv\loading_phosphorus_Feb2021.csv')
    elif name == 'sediment':
        # df = pd.read_excel(r'C:\ITEEM\Submodel_SWAT\Response_matrix_BMPs.xlsx',sheet_name=2)
        df = pd.read_csv(r'C:\ITEEM\Submodel_SWAT\response_matrix_csv\loading_sediment_Feb2021.csv')
    elif name == 'streamflow':
        # df = pd.read_excel(r'C:\ITEEM\Submodel_SWAT\Response_matrix_BMPs.xlsx',sheet_name=3)
        df = pd.read_csv(r'C:\ITEEM\Submodel_SWAT\response_matrix_csv\loading_streamflow_Feb2021.csv')
        df = df*30*60*60*24
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
    
    n = int(ag_scenario[-2:])
    df_swat = df_to_np[:,:,sw,n]
    # df_iteem = loading_outlet_USRW(name, landuse_matrix)
    df_iteem = loading_outlet_originalRM(name, landuse_matrix) # use original RM to predict 
    df_iteem_sw = df_iteem[:,:,sw]
    
    # if name =='sediment':
    #     df_iteem_sw = sediment_instream(sw, ag_scenario)
    
    pbias0 = pbias(obs=df_swat, sim=df_iteem_sw).round(2)
    nse0 = nse(obs=df_swat, sim=df_iteem_sw).round(3)
    
    fig, ax = plt.subplots()
    plt.text(x=0.03, y=0.79, s= 'P-bias: ' + str(pbias0) + '%',transform=ax.transAxes, fontsize=10)
    plt.text(x=0.03, y=0.73, s= 'NSE: ' + str(nse0),transform=ax.transAxes, fontsize=10)
    plt.plot(df_swat.flatten(), color='red', label='SWAT', linewidth=1.5)
    plt.plot(df_iteem_sw.flatten(), color='blue', linestyle='dashed', label='Traditional RM', linewidth=1.5)
    if name == 'streamflow':
        plt.ylabel(name.capitalize() +' (m3/month)', fontsize=10)
    elif name =='sediment':
        plt.ylabel(name.capitalize() +' loading (ton/month)', fontsize=10)
    else:
        plt.ylabel(name.capitalize() +' loading (kg/month)', fontsize=10)
    
    plt.xlabel('Time (2003-2018)', fontsize=10)
    labels = [2003] + [str(i)[-2:] for i in range(2004,2020)]
    plt.xticks(np.arange(0, 192+1, 12), labels)
    
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # plt.legend(loc='upper left', fontsize=12 )
    plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(0.01, 0.9),frameon=False)
    plt.tight_layout()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))

    # plt.savefig(r'C:\ITEEM\ITEEM_figures\Jan_2021\originalRM_'+name+'_' + ag_scenario +'_sw'+str(sw+1) +'.tif', dpi=150)
    plt.show()

# landuse_matrix = np.zeros((45,56))
# landuse_matrix[:,1] = 1
# swat_vs_tradional_RM('nitrate', sw=33, ag_scenario='BMP00', landuse_matrix=landuse_matrix)
# swat_vs_tradional_RM('sediment', sw=33, ag_scenario='BMP00', landuse_matrix=landuse_matrix)
# swat_vs_iteem('streamflow', sw=33, ag_scenario='BMP00', landuse_matrix=landuse_matrix)
# swat_vs_iteem('phosphorus', sw=33, ag_scenario='BMP00', landuse_matrix=landuse_matrix)

def box_plot(name, sw):
    '''old boxplot'''
    fig = plt.figure(figsize=(6.5,5))
    df = pd.read_excel(r'C:\ITEEM\ITEEM_figures\June14\data.xlsx')
    
    if name =='phosphorus':
        g = sns.boxplot(y=df.TP, x=df.Scenario, palette="Set1", showfliers = True)
        g = sns.swarmplot(y=df.TP, x=df.Scenario, color="black")
        plt.axhline(y=324*1000*0.75, color='blue', linestyle='dashdot', label='25% Reductional Goal by 2025', alpha=0.5, linewidth=5)
        plt.axhline(y=324*1000*0.55, color='blue', linestyle=':', label='45% Reductional Goal by 2045', alpha=0.5, linewidth=5)
    elif name == 'nitrate':
        g = sns.boxplot(y=df.Nitrate, x=df.Scenario, palette="Set1", showfliers = True)
        g = sns.swarmplot(y=df.Nitrate, x=df.Scenario, color="black")
        plt.axhline(y=7240*1000*0.85, color='blue', linestyle='dashdot', label='15% Reductional Goal by 2025', alpha=0.5, linewidth=5)
        plt.axhline(y=7240*1000*0.55, color='blue', linestyle=':', label='45% Reductional Goal by 2045', alpha=0.5, linewidth=5)
    
    g.artists[0].set_facecolor('blue')
    # g.artists[0].set_alpha(0.8)
    g.artists[1].set_facecolor('purple')
    g.artists[2].set_facecolor('green')
    g.artists[3].set_facecolor('red')

    # plt.plot(test_annual*0.85, color='blue', marker='o', linestyle='dashdot', label='15% Reductional Goal by 2025', alpha=.5, linewidth=2)
    # plt.plot(test_annual*0.55, color='blue', marker='o', linestyle=':', label='45% Reductional Goal by 2045', alpha=.5, linewidth=2)

    # ax.set_xticklabels([i for i in range(2003,2019)])
    # plt.yticks(np.arange(0, 25, 2))
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    # plt.legend(loc='upper left', fontsize=12 )
    plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(0.03, 1.2))
    plt.ylabel(name +' annual loading (kg/year)', fontsize=14)
    plt.tight_layout()
    # plt.savefig(r'C:\ITEEM\ITEEM_figures\Jan_2021\\'+name+'_'+ str(sw) +'_Boxplot.tif', dpi=150)
    plt.show()
    return
# box_plot('nitrate', 33)
# box_plot('phosphorus', 33)

# load data again
# matdata = scipy.io.loadmat(r'C:\ITEEM\Submodel_SWAT\results_validation\100Randomizations\np_nitrate_Dec2020.mat')
# np_nitrate_mat = matdata['out']

# matdata = scipy.io.loadmat(r'C:\ITEEM\Submodel_SWAT\results_validation\100Randomizations\np_streamflow.mat')
# np_streamflow_mat = matdata['out']

# example of data format
# x = data_sdd[period].append(data_df[period], ignore_index=True)
# y = data_sdd[output_name].append(data_df[output_name], ignore_index=True)
# by_category = data_sdd['tech'].append(data_df['tech'], ignore_index=True)

# np_x = np.zeros(4500)
# for i in range(45):
#     np_x[100*i:100*(i+1)] = int(i+1) 

def boxplot_for_sw(name, indicator_name):
    '''for all subwatersheds'''
    np_x = numpy.zeros(4500)
    for i in range(45):
        np_x[100*i:100*(i+1)] = int(i+1) 
    if name == 'phosphorus':
        matdata = scipy.io.loadmat(r'C:\ITEEM\Submodel_SWAT\results_validation\100Randomizations\np_TP_May2021.mat')
        np = matdata['out']
        np_pbias = np[:,:,0].flatten()
        np_nse = np[:,:,1].flatten()
        
    elif name == 'sediment':
        matdata = scipy.io.loadmat(r'C:\ITEEM\Submodel_SWAT\results_validation\100Randomizations\np_sediment_March2021.mat')
        np = matdata['out']
        np_pbias = np[:,:,0].flatten()
        np_nse = np[:,:,1].flatten()
        
    elif name == 'nitrate':
        matdata = scipy.io.loadmat(r'C:\ITEEM\Submodel_SWAT\results_validation\100Randomizations\np_nitrate_March2021.mat')
        np = matdata['out']
        np_pbias = np[:,:,0].flatten()
        np_nse = np[:,:,1].flatten()
    elif name == 'streamflow':
        matdata = scipy.io.loadmat(r'C:\ITEEM\Submodel_SWAT\results_validation\100Randomizations\np_streamflow_March2021.mat')
        np = matdata['out']
        np_pbias = np[:,:,0].flatten()
        np_nse = np[:,:,1].flatten()
    
    fig, ax = plt.subplots(figsize=(7.0,3.5))
    
    if indicator_name == 'pbias':
        ax = sns.boxplot(y=np_pbias, x=np_x.astype('int'), showfliers = False, fliersize=0.5, linewidth=0.5)
        plt.ylabel('Percent bias (P-bias, %)', fontsize=11)
        plt.ylim(-15, 20)
    elif indicator_name =='nse':
        ax = sns.boxplot(y=np_nse, x=np_x.astype('int'), showfliers = False, fliersize=0.5, linewidth=0.5)
        plt.ylabel('Nash-Sutcliffe efficiency', fontsize=11)
        plt.ylim(0.95, 1.0)
    
    df_linkage = pd.read_excel(r'C:\ITEEM\Submodel_SWAT\Watershed_linkage_v2.xlsx')
    # sw_number = df_linkage.fillna(0).gt(0).sum(axis=1) - 1
    df_hru = pd.read_csv(r'C:\ITEEM\Submodel_SWAT\results_validation\RM_HRUReportAndYield.csv')
    # df_hru['hru'] = 1
    df_hru2 = df_hru.groupby(['SUB']).count()
    np_linkage = df_linkage.fillna(0).to_numpy()
    # np_linkage2 = np_linkage[np_linkage !=0]
    np_hru2 = df_hru2.iloc[:,0].to_numpy()
    
    import numpy as np
    total_hru = np.zeros((45,1))
    for i in range(45):
        np_linkage2 = np_linkage[i,:]
        np_linkage3 = np_linkage2[np_linkage2!=0]
        hru_j = 0
        for j in np_linkage3:
            # print(j)
            hru_j += np_hru2[int(j)-1]
        total_hru[i] = hru_j
    # total_hru_log = np.log(total_hru)
    # color = sns.color_palette("hls", 2)
    color = sns.color_palette("OrRd", 5) 
    # sns.palplot(sns.color_palette("OrRd", 5))

    # colors = ['blue', 'green', 'purple', 'tan', 'pink', 'red']
    for i in range(45):
        mybox = ax.artists[i]
        if total_hru[i]<=5:
            mybox.set_facecolor(color[0])
        elif 5<total_hru[i]<=25:
            mybox.set_facecolor(color[1])
        elif 25<total_hru[i]<=100:
            mybox.set_facecolor(color[2])
        elif 100<total_hru[i]<=250:
            mybox.set_facecolor(color[3])
        elif 250<total_hru[i]<=825:
            mybox.set_facecolor(color[4])
            
    # for i in range(45):
    #     mybox = ax.artists[i]
    #     mybox.set_facecolor(color[int(total_hru[i]-1)])

    # reservoir_all = [i+1 for i in range(45)]    
    # reservoir_downstream = [33, 32, 31, 37, 35, 29, 42, 39, 34, 43, 41, 45, 36]
    # reservoir_upstream = list(set(reservoir_all)^set(reservoir_downstream))
    # for i in reservoir_upstream:
    #     mybox = ax.artists[i-1]
    #     mybox.set_facecolor(color[0])
    # for i in reservoir_downstream:
    #     mybox = ax.artists[i-1]
    #     mybox.set_facecolor(color[1])
    
    # main_channel = [4, 7, 10, 13, 26, 27, 32, 31, 37, 35, 34]
    # tributary = list(set(reservoir_all)^set(main_channel))
    # for i in main_channel:
    #     mybox = ax.artists[i-1]
    #     mybox.set_facecolor(color[0])
    # for i in tributary:
    #     mybox = ax.artists[i-1]
    #     mybox.set_facecolor(color[1]) 
    
    # text1 = 'subwatershed upstream reservoir'
    # text2 = 'subwatershed downstream reservoir'
    # text1 = 'subwatersheds (main channel)'
    # text2 = 'subwatersheds (tributaries)'
    
    # if name == 'phosphorus' and indicator_name == 'pbias':
    #     fig.text(0.35, 0.34, text1,
    #          backgroundcolor=color[0], color='black', size='x-small', fontsize=11)
    #     fig.text(0.35, 0.25, text2,
    #          backgroundcolor=color[1], color='black', size='x-small', fontsize=11)
        
    # elif name == 'sediment' and indicator_name=='nse':
    #   fig.text(0.35, 0.34, text1,
    #        backgroundcolor=color[0], color='black', size='x-small', fontsize=11)
    #   fig.text(0.35, 0.25, text2,
    #        backgroundcolor=color[1], color='black', size='x-small', fontsize=11)
    
    # elif name == 'nitrate' or name == 'streamflow':
    #     fig.text(0.35, 0.34, text1,
    #          backgroundcolor=color[0], color='black', size='x-small', fontsize=11)
    #     fig.text(0.35, 0.25, text2,
    #          backgroundcolor=color[1], color='black', size='x-small', fontsize=11)
    plt.xticks(fontsize=10, rotation=90)
    plt.xlabel('Subwatershed index', fontsize=11)
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_vertical(size=0.15,pad=0.3)    
    bounds = [0, 1, 2, 3, 4, 5]
    
    # cmap = matplotlib.colors.ListedColormap(mpl.cm.get_cmap("OrRd").colors[:5])

    cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=mpl.cm.OrRd, 
                                    boundaries=bounds,
                                    spacing='uniform',
                                    orientation='horizontal')
    plt.gcf().add_axes(ax_cb)
    
    plt.text(x=0.0, y=1.3, fontsize=10, s="Total number of upstream HRUs",
                 transform=ax.transAxes)
    
    plt.tight_layout() 
    plt.savefig(r'C:\ITEEM\ITEEM_figures\RM_Jan_2021\\'+name+'_'+ indicator_name +'_Boxplot_May2021_v2.tif', dpi=300)
    plt.savefig(r'C:\ITEEM\ITEEM_figures\RM_Jan_2021\\'+name+'_'+ indicator_name +'_Boxplot_May2021_v2.pdf')
    plt.show()

# sns.palplot(sns.color_palette("hls", 2))
# sns.color_palette("hls", 2)
# boxplot_for_sw('phosphorus', 'pbias')
# boxplot_for_sw('phosphorus', 'nse')
# boxplot_for_sw('sediment', 'pbias')
# boxplot_for_sw('sediment', 'nse')
