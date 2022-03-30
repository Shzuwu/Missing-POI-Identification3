# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 10:32:09 2022

@author: Administrator
"""
import numpy as np
import pandas as pd
from collections import Counter
from pylab import scatter
import pylab
import time
import seaborn as sns
import matplotlib.pyplot as plt

data = 'TKY'
#data = 'TKY'
#data = 'checkins-gowalla1'
#data = 'checkins-4sq'

if data == 'NYC' or data == 'TKY':
    path = r'data/original data/dataset_TSMC2014_{}.txt'.format(data)
    df = pd.read_csv(path, header=None, sep='\t',encoding='unicode_escape')
    df_columns = ['user', 'poi', 'categoryID', 'category_name', 'lat', 'lon', 'off_set', 'timestamp']
    df.columns = df_columns
elif data == 'checkins-gowalla1' or data == 'checkins-4sq':
    path = r'data/original data/{}.txt'.format(data)
    df = pd.read_csv(path, header=None, sep='\t',encoding='unicode_escape')
    df_columns = ['user','timestamp','lat','lon','poi']
    df.columns = df_columns

data_xx = pd.DataFrame()
data_xx['poi']=df['poi']
data_xx['user'] = df['user']
data_xx['category_name'] = df['category_name']
data_xx1=data_xx.set_index('poi')

yy1=Counter(list(df['poi']))
xx = [i for i in yy1.keys() if yy1[i]>9]

POI_count = len(xx)
checkIn_count = len(list(data_xx1.loc[xx]['user']))
user_count = len(Counter(list(data_xx1.loc[xx]['user'])).keys())
category_count = len(Counter(list(data_xx1.loc[xx]['category_name'])).keys())

print('Data:',data)
print('Number of POIs:',POI_count)
print('Number of check-ins:',checkIn_count)
print('Number of users:',user_count)
print('Number of categories:',category_count)
print('Avg_check-in for each user:', checkIn_count/user_count)