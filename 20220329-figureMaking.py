# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 19:53:19 2022

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
    
list_lat=list(df['lat'])
list_lon=list(df['lon'])  
list_category_key = list(df['category_name'])

df_new = pd.DataFrame()
df_new['lat'] = list_lat
df_new['lon'] = list_lon
df_new['category'] = list_category_key
df_new['category1']= list_category_key
df_new1=df_new.set_index('category1')

category_counter = Counter(list_category_key)
top_items = list(category_counter.most_common(20))
print('Top:',top_items)

all_lat, all_lon, all_category = [],[],[]
targert_items = ['Home (private)','Office','Train Station']
for i in targert_items:
    temp = df_new1.loc[i]
    all_lat += list(temp['lat'])
    all_lon += list(temp['lon'])
    all_category += list(temp['category'])
df_new2 = pd.DataFrame()
df_new2['lat'] = all_lat
df_new2['lon'] = all_lon
df_new2['category'] = all_category   

color = ['yellow','blue','red']
color1 = color[:len(targert_items)]
colors = dict(zip(targert_items, color))
print('colors:',colors)

ax = df_new2.plot.scatter(x='lat', y='lon', s=20, c=df_new2['category'].apply(lambda x: colors[x]), alpha=0.1)
plt.xlabel('Lat', fontproperties = 'Times New Roman',fontsize=15, color='k') #x轴label的文本和字体大小
plt.ylabel('Lon', fontproperties = 'Times New Roman',fontsize=15, color='k') #y轴label的文本和字体大小
plt.savefig('123.png', dpi=1000, bbox_inches='tight')
plt.show()
    

#fig = ax.get_figure()
#fig.savefig('12345.png', bbox_inches='tight')
#
##plt.savefig('12345.pdf', bbox_inches='tight')
#plt.show()

  

#path = r'{}_category.csv'.format(data)
#df_NYC_category = pd.read_csv(path, header=None, sep='\t',encoding='unicode_escape')
#catg_dict = {}
#catg_key = list(df_NYC_category[0])
#catg_value=list(df_NYC_category[1])
#for i in range(len(catg_key)):
#    catg_dict[catg_key[i]] = catg_value[i]
#    
#list_category_value = []
#for i in range(len(list_category_key)):
#    list_category_value.append(catg_dict[list_category_key[i]])
#
#df_new = pd.DataFrame()
#df_new['lat'] = list_lat
#df_new['lon'] = list_lon
#df_new['category'] = list_category_value
#df_new['category1']= list_category_value
#category_set2list = list(set(list_category_value))
#df_new1 = df_new.set_index('category1')
#
#
#df1 = df_new1.loc['Food']
#df2 = df_new1.loc['Residence']
##df3 = df_new1.loc['Professional & Other Places'] 
#lat_all, lon_all, category_all = [],[],[]
##lat_all = list(df1['lat']) + list(df2['lat']) + list(df3['lat'])
##lon_all = list(df1['lon']) + list(df2['lon']) + list(df3['lon'])
##category_all = list(df1['category']) + list(df2['category']) + list(df3['category'])
#lat_all = list(df1['lat']) + list(df2['lat'])
#lon_all = list(df1['lon']) + list(df2['lon'])
#category_all = list(df1['category']) + list(df2['category'])
#df_new2 = pd.DataFrame()
#df_new2['lat'] = lat_all
#df_new2['lon'] = lon_all
#df_new2['category'] = category_all
#
#colors = { 'Food': 'red', 'Residence': 'blue', 'Shop & Service': 'yellow' }
#
#df_new2.plot.scatter(x='lat', y='lon', c=df_new2['category'].apply(lambda x: colors[x]))
#plt.show()



#category_set2list = list(set(list_category_value))
#color = sns.color_palette()
#color_dict={}
#for i in range(len(category_set2list)):
#    color_dict[category_set2list[i]]=color[i]
    

#data_xx = pd.DataFrame()
#data_xx['poi']=df['poi']
#data_xx['user'] = df['user']
#data_xx['category_name'] = df['category_name']
#data_xx1=data_xx.set_index('poi')