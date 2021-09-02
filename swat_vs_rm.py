# -*- coding: utf-8 -*-


# Import required packages for data processing
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from metrics import pbias, nse
from results_validation_originalRM import loading_outlet_originalRM
from matplotlib.ticker import FormatStrFormatter
from data import *


'''Start Section: comparison of original SWAT vs RM'''
def swat_vs_RM(name, sw, ag_scenario, plot=True, p_adjust=False, mode='poly'):
    '''note: only works for one specificed sw: traditional RM and modified RM'''
    # Step 1: get loading from original SWAT results...
    if name == 'nitrate':
        # df = pd.read_excel(r'C:\ITEEM\Submodel_SWAT\Response_matrix_BMPs.xlsx',sheet_name=0)
        df = pd.read_csv(r'C:\ITEEM\Submodel_SWAT\results_validation\100Randomizations\loading_nitrate_March2021.csv')
        # df = pd.read_excel(xls_load, 'Nitrate')
    elif name == 'phosphorus':
        df = pd.read_csv(r'C:\ITEEM\Submodel_SWAT\results_validation\100Randomizations\loading_phosphorus_March2021.csv')
        # df = pd.read_excel(xls_load, 'Phosphorus')
    elif name == 'sediment':
        df = pd.read_csv(r'C:\ITEEM\Submodel_SWAT\results_validation\100Randomizations\loading_sediment_March2021.csv')
        # df = pd.read_excel(xls_load, 'Sediments')
    elif name == 'streamflow':
        df = pd.read_csv(r'C:\ITEEM\Submodel_SWAT\results_validation\100Randomizations\loading_streamflow_March2021.csv')
        # df = pd.read_excel(xls_load, 'Water')
        df = df*30*60*60*24
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
            
    n = int(ag_scenario[-2:])-1
    df_swat = df_to_np[:,:,sw,n]

    df_iteem = loading_outlet_USRW(name, ag_scenario)
    df_iteem_sw = df_iteem[:,:,sw]
    
    if name =='sediment':
        df_iteem_sw = sediment_instream(sw, ag_scenario, mode)
    if name == 'phosphorus' and p_adjust:
        df_iteem_sw = phosphorus_instream(sw, ag_scenario, reg='linear')
    pbias0 = pbias(obs=df_swat, sim=df_iteem_sw)
    nse0 = nse(obs=df_swat, sim=df_iteem_sw)
    
    '''original RM results'''
    landuse_matrix = landuse_mat()  # (45,56)
    scenario, BMP_list= get_area_prcnt(ag_scenario)
    for i in BMP_list:
        landuse_matrix[:,i] = scenario.loc[:,i]
    df_iteem_originalRM = loading_outlet_originalRM(name, landuse_matrix) # use original RM to predict 
    df_iteem_sw_originalRM = df_iteem_originalRM[:,:,sw]
    
    pbias_originalRM = pbias(obs=df_swat, sim=df_iteem_sw_originalRM)
    nse_originalRM = nse(obs=df_swat, sim=df_iteem_sw_originalRM)
    
    if name == 'nitrate' or name == 'phosphorus':
        df_point = pd.read_csv(r'C:\ITEEM\Submodel_SWAT\results_validation\SDD_interpolated_2000_2018_Inputs.csv', 
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
            # pass
            plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(0.01, 0.88),frameon=False)
        plt.tight_layout()
        # plt.savefig(r'C:\ITEEM\ITEEM_figures\RM_Jan_2021\randomized\\'+name+'_' + ag_scenario +'_sw'+str(sw+1)+'P_adjust_'+str(p_adjust)+'_invert_yield_June2021.tif', dpi=300)
        plt.show()
    # df_swat.flatten(), df_iteem_sw.flatten(), df_iteem_sw_originalRM_point.flatten()
    return pbias0, nse0, pbias_originalRM, nse_originalRM, df_swat.flatten(), df_iteem_sw.flatten()

# test = swat_vs_iteem('nitrate', sw=7, ag_scenario='Sheet01')
# test = swat_vs_iteem('phosphorus', sw=33, ag_scenario='Sheet01', p_adjust=True)
# test = swat_vs_iteem('sediment', sw=32, ag_scenario='Sheet01', p_adjust=True, mode='poly')
# df_swat = test[-2].sum()/test[-3].sum()
# df_iteem_sw = test[-2].sum()
# df_iteem_sw_originalRM = test[-1].sum()

# trial_list1 = ['Sheet0' + str(i) for i in range(1,10)]
# trial_list2 = ['Sheet' + str(i) for i in range(10,101)]
# trial_list = trial_list1 + trial_list2
# df_nitrate = pd.DataFrame(columns=['pbias_%', 'nse'])
# np_nitrate = np.zeros((45,100,4))
# np_TP = np.zeros((45,100,4))
# np_streamflow = np.zeros((45,100,4))
# np_sediment_linear = np.zeros((45,100,4))
# np_sediment_poly = np.zeros((45,100,4))

# start = time.time()
# for sw in range(45):
#     print('simulating subwatershed: ', sw)
#     for i in range(100):
#         # np_nitrate[sw,i,:] = swat_vs_iteem('nitrate', sw, ag_scenario=trial_list[i], plot=False)
#         np_sediment_linear[sw,i,:] = swat_vs_iteem('sediment', sw, ag_scenario=trial_list[i], plot=False, mode='linear')       
#         np_sediment_poly[sw,i,:] = swat_vs_iteem('sediment', sw, ag_scenario=trial_list[i], plot=False, mode='poly')     
#         # np_streamflow[sw,i,:] = swat_vs_iteem('streamflow', sw, ag_scenario=trial_list[i], plot=False)
#         # np_sediment[sw,i,:] = swat_vs_iteem('sediment', sw, ag_scenario=trial_list[i], plot=False)
#         print('simulating scenario: ', i)
# end = time.time()
# print('Run time (hrs): ', str((end-start)/3600)[:3])

# import scipy.io
# scipy.io.savemat(r'C:\ITEEM\Submodel_SWAT\results_validation\np_nitrate_Apr2021.mat', mdict={'out': np_nitrate}, oned_as='row')
# scipy.io.savemat(r'C:\ITEEM\Submodel_SWAT\results_validation\np_TP_May2021.mat', mdict={'out': np_TP}, oned_as='row')
# scipy.io.savemat(r'C:\ITEEM\Submodel_SWAT\results_validation\np_streamflow_Apr2021.mat', mdict={'out': np_streamflow}, oned_as='row')
# scipy.io.savemat(r'C:\ITEEM\Submodel_SWAT\results_validation\np_sediment_linear_May2021.mat', mdict={'out': np_sediment_linear}, oned_as='row')
# scipy.io.savemat(r'C:\ITEEM\Submodel_SWAT\results_validation\np_sediment_poly_May2021.mat', mdict={'out': np_sediment_poly}, oned_as='row')
# load data again
# RM_sens = scipy.io.loadmat(r'C:\Users\Shaobin\Box\SWATBaseline\ScenarioResults\ForITEEM\Sensitivity Analysis\YieldsAndLoads_Feb6_2021.mat')['out']
# plt.hist(np_TP[0,:,1])
