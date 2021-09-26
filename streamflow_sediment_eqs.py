# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 14:23:01 2020

Derive sediment-streawmflow relationship

@author: Shaobin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from modified_RM_main import * 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


def response_mat_load(name):
    '''
    return as a tuple
    unit: kg/month for nitrate, phosphorus, soy, corn, corn silage; ton/month for sediment; mm/month for water yield
    '''
    if name == 'nitrate':
        # df = pd.read_excel(r'C:\ITEEM\Submodel_SWAT\Response_matrix_BMPs.xlsx',sheet_name=0)
        df = pd.read_csv(r'.\response_matrix_csv\loading_nitrate.csv')
    elif name == 'phosphorus':
        df = pd.read_csv(r'.\response_matrix_csv\loading_phosphorus.csv')
    elif name == 'sediment':
        df = pd.read_csv(r'.\response_matrix_csv\loading_sediment.csv')
    elif name == 'streamflow':
        df = pd.read_csv(r'.\response_matrix_csv\loading_streamflow.csv')
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


def polynomial_function(sw, plot=True):
    '''generic format: y = a*x**2 + b*x + c
    x and y are vector data
    '''
    x = np.array(swat_load_streamflow[:,:,sw].flatten()*days*3600*24)
    y = swat_load_sediment[:,:,sw].flatten()
    a, b, c = np.polyfit(x, y, 2)
    myline = np.linspace(x.min(), x.max(), 192)
    mymodel = np.poly1d(np.polyfit(x, y, 2))
    r2 = r2_score(y, mymodel(x))
    if plot==True:
        f, ax = plt.subplots(figsize=(6,4))
        plt.scatter(x, y, c = "blue")
        plt.plot(myline, mymodel(myline), c='red')
        ax.text(0.1, 0.9, 'Subwatershed ' + str(sw+1), transform=ax.transAxes, fontsize=12)
        ax.text(0.1, 0.8, 'R-square ' + str(r2)[:5], transform=ax.transAxes, fontsize=12)
        ax.set_xlabel('Streamflow (m3/month)', fontsize=12)
        ax.set_ylabel('Sediment loading (ton/month)', fontsize=12)
    return a, b, c, r2, mymodel


def power_function(sw, plot=True):
    '''generic format: y = a*x**b'''
    x = np.array(swat_load_streamflow[:,:,sw].flatten()*days*3600*24)
    y = swat_load_sediment[:,:,sw].flatten()
    x_log = np.log(x)
    y_log = np.log(y)
    if np.isinf(x_log).any() or np.isinf(y_log).any():
        print('zero values in the inputs, power function is not appropriate')
        a = 'invalid'
        b = 'invalid'
        r2 = 'invalid'
        mymodel = 'invalid'
    else:     
        b, a_log = np.polyfit(x_log, y_log, 1)
        a = np.e**a_log
        myline = np.linspace(x.min(),x.max(), 192)
        mymodel = np.poly1d(np.polyfit(x_log, y_log, 1))
        r2 = r2_score(y, np.e**mymodel(np.log(x)))
        
    if plot==True:
        f, ax = plt.subplots(figsize=(6,4))
        plt.scatter(x, y, c = "blue")
        plt.plot(myline, np.e**mymodel(np.log(myline)), c='red')
        ax.set_xlabel('Streamflow (m3/month)', fontsize=12)
        ax.set_ylabel('Sediment loading (ton/month)', fontsize=12)
        ax.text(0.1, 0.9, 'Subwatershed ' + str(sw+1), transform=ax.transAxes, fontsize=12)
        ax.text(0.1, 0.8, 'R-square = ' + str(r2)[:5], transform=ax.transAxes, fontsize=12)
    return a, b, r2, mymodel

# power_function(26, plot=False)

def linear_function(sw, plot=True):
    x = np.array(swat_load_streamflow[:,:,sw].flatten()*days*3600*24)
    y = swat_load_sediment[:,:,sw].flatten()
    b, a = np.polyfit(x, y, 1)
    mymodel = np.poly1d(np.polyfit(x, y, 1))
    r2 = r2_score(y, mymodel(x))
    if plot==True:
        f, ax = plt.subplots(figsize=(6,4))
        plt.scatter(x, y, c = "blue")
        myline = np.linspace(x.min(),x.max(), 192)
        plt.plot(myline, mymodel(myline), c='red')
        ax.set_xlabel('Streamflow (m3/month)', fontsize=12)
        ax.set_ylabel('Sediment loading (ton/month)', fontsize=12)
        r2 = r2_score(y, mymodel(x))
        ax.text(0.1, 0.9, 'Subwatershed ' + str(sw+1), transform=ax.transAxes, fontsize=12)
        ax.text(0.1, 0.8, 'R-square = ' + str(r2)[:5], transform=ax.transAxes, fontsize=12)
    return a, b, r2, mymodel

# linear_function(33)

def streamflow_vs_sediment(sw):
    sr = pd.date_range("2003-01-01", periods=192, freq="M")
    sr = pd.to_datetime(sr) 
    days = sr.days_in_month
    swat_load_streamflow = response_mat_load('streamflow')[0][:,:,:,0]
    swat_load_sediment = response_mat_load('sediment')[0][:,:,:,0]
    x = np.array(swat_load_streamflow[:,:,sw].flatten()*days*3600*24)
    y = swat_load_sediment[:,:,sw].flatten()
    
    f, ax = plt.subplots(figsize=(6,4))
    ax.scatter(x, y, c='blue')
    ax.set_xlabel('Streamflow (m3/month)', fontsize=12)
    ax.set_ylabel('Sediment loading (ton/month)', fontsize=12)
    ax.text(0.1, 0.9, 'Subwatershed ' + str(sw+1), transform=ax.transAxes, fontsize=12)
    plt.show()
    
# streamflow_vs_sediment(sw=33)

def overall_plot(sw, plotshow=True):
    x = np.array(swat_load_streamflow[:,:,sw].flatten()*days*3600*24)
    y = swat_load_sediment[:,:,sw].flatten()
    myline = np.linspace(x.min(),x.max(), 192)
    x_log = np.log(x)
    y_log = np.log(y)
    
    linear = linear_function(sw, plot=False)
    poly = polynomial_function(sw, plot=False)
    power = power_function(sw, plot=False)
    mymodel_linear = linear[-1]
    mymodel_poly = poly[-1]
    mymodel_power = power[-1]
    
    r2_linear = r2_score(y, mymodel_linear(x))
    r2_poly = r2_score(y, mymodel_poly(x))
    a_linear, b_linear = linear[0], linear[1]
    a_poly, b_poly, c_poly = poly[0], poly[1], poly[2]
    
    if np.isinf(x_log).any() or np.isinf(y_log).any():
        r2_power = 0
        a_power, b_power = 0, 0
    else:
        r2_power =  r2_score(y, np.e**mymodel_power(np.log(x)))
        a_power, b_power = power[0], power[1]
        
    if plotshow==True:
        f, ax = plt.subplots(figsize=(6,4))
        ax.scatter(x, y, c='blue')
        ax.set_xlabel('Streamflow (m$\mathregular{^3}$/month)', fontsize=11)
        ax.set_ylabel('Sediment loading (ton/month)', fontsize=11)
        if np.isinf(x_log).any() or np.isinf(y_log).any():
            ax.plot(myline, mymodel_linear(myline), c='red', linestyle='solid', label='Linear')
            ax.plot(myline, mymodel_poly(myline), c='red', linestyle='dotted', label='Poly')
            ax.text(0.3, 0.8, 'R-square (linear) = ' + str(r2_linear)[:5], transform=ax.transAxes, fontsize=11)
            ax.text(0.3, 0.75, 'R-square (poly) = ' + str(r2_poly)[:5], transform=ax.transAxes, fontsize=11)
        else:
            ax.plot(myline, mymodel_linear(myline), c='red', linestyle='solid', label='Linear')
            ax.plot(myline, mymodel_poly(myline), c='red', linestyle='dotted', label='Poly')
            ax.text(0.3, 0.8, 'R-square (linear) = ' + str(r2_linear)[:5], transform=ax.transAxes, fontsize=11)
            ax.text(0.3, 0.75, 'R-square (poly) = ' + str(r2_poly)[:5], transform=ax.transAxes, fontsize=11)

        ax.legend(bbox_to_anchor=(0.25, 0.9))
        plt.tight_layout()
        
    linear_coef = [a_linear, b_linear]
    poly_coef = [a_poly, b_poly, c_poly]
    power_coef = [a_power, b_power]
    r2 = [r2_linear, r2_poly, r2_power]
    return linear_coef, poly_coef, power_coef, r2
