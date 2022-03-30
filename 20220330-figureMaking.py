# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:23:40 2022

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


data='TKY'
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
