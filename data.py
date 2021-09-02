# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 02:58:05 2021

@author: Shaobin
"""

import pandas as pd
from pathlib import Path

# set up global variables, relative path start from ITEEM/...
df_nitrate_path = Path('./response_matrix_csv/yield_nitrate.csv')
df_TP_path = Path('./response_matrix_csv/yield_phosphorus.csv')
df_sediment_path = Path('./response_matrix_csv/yield_sediment.csv')
df_streamflow_path = Path('./response_matrix_csv/yield_streamflow.csv')
# df_link_path = Path('./Watershed_linkage_v2.xlsx')
df_point_SDD_path = Path('./data/SDD_interpolated_2000_2018_Inputs.csv')
df_linkage_path = Path('./data/Watershed_linkage.xlsx')
df_linkage2_path = Path('./data/Watershed_linkage_v2.xlsx')
df_landuse_path = Path('./data/landuse.xlsx')
df_pd_coef_poly_path = Path('./data/sediment_streamflow_regression_coefs.xlsx')
xls_path = Path('./results_validation/100Randomizations/100RandomizeHRUstorePercentage_4BMPs_March192021.xlsx')


df_nitrate = pd.read_csv(df_nitrate_path)
df_TP = pd.read_csv(df_TP_path)
df_sediment = pd.read_csv(df_sediment_path)
df_streamflow = pd.read_csv(df_streamflow_path)
# df_link = pd.read_excel(df_link_path)
df_point_SDD = pd.read_csv(df_point_SDD_path, parse_dates=['Date'], index_col='Date')
df_linkage = pd.read_excel(df_linkage_path).fillna(0)
df_linkage2 = pd.read_excel(df_linkage2_path)
df_landuse = pd.read_excel(df_landuse_path).fillna(0)
df_pd_coef_poly = pd.read_excel(df_pd_coef_poly_path, sheet_name='poly', usecols='B:D')
xls = pd.ExcelFile(xls_path)
