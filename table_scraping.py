#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 12:20:07 2022

@author: neelampatel
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt

path = r'/Users/neelampatel/Documents/GitHub/dataviz_final'

#Source: https://medium.com/analytics-vidhya/how-to-scrape-a-table-from-website-using-python-ce90d0cfb607

# Create an URL object
url = 'https://broadbandnow.com/research/county-broadband-statistics'
# Create object page
page = requests.get(url)

# parser-lxml = Change html to Python friendly format
# Obtain page's information
soup = BeautifulSoup(page.text, 'lxml')
table1 = soup.find('table')
print(table1)

#pull header titles
headers = []
for i in table1.find_all('th'):
 title = i.text
 headers.append(title)
 
#create dataframe 
mydata = pd.DataFrame(columns = headers)

#pull data and add to dataframe 
for j in table1.find_all('tr')[1:]:
 row_data = j.find_all('td')
 row = [i.text for i in row_data]
 length = len(mydata)
 mydata.loc[length] = row
 
#format data 
mydata.info()
mydata['LOWEST MONTHLY PRICE*'] = mydata['LOWEST MONTHLY PRICE*'].str.strip('$')
mydata['LOWEST MONTHLY PRICE*'] = pd.to_numeric(mydata['LOWEST MONTHLY PRICE*'])

mydata['# OF WIRED BB PROVIDERS'] = pd.to_numeric(mydata['# OF WIRED BB PROVIDERS'])

mydata['# OF WIRED BB PROVIDERS'] = np.clip(mydata['# OF WIRED BB PROVIDERS'], None, 10)
df_broadband_plot = mydata.groupby('# OF WIRED BB PROVIDERS')['LOWEST MONTHLY PRICE*'].agg(['mean', 'count']).reset_index()
df_broadband_plot = df_broadband_plot[df_broadband_plot['# OF WIRED BB PROVIDERS'] != 0]
df_broadband_plot.to_csv(os.path.join(path, 'summary_plot.csv'), index=False)

mydata_plot = mydata[['LOWEST MONTHLY PRICE*', '# OF WIRED BB PROVIDERS']]
mydata_plot = mydata_plot[mydata_plot['# OF WIRED BB PROVIDERS'] != 0]
mydata_plot.to_csv(os.path.join(path, 'plot.csv'), index=False)


mydata_plot = mydata_plot.pivot_table(values='# OF WIRED BB PROVIDERS', index='LOWEST MONTHLY PRICE*', columns='# OF WIRED BB PROVIDERS', aggfunc='count')
sns.heatmap(mydata_plot)

fig, ax = plt.subplots(figsize=(12, 6))
z = np.polyfit(df_broadband_plot['# OF WIRED BB PROVIDERS'], df_broadband_plot['mean'],1)
p = np.poly1d(z)
sns.scatterplot(df_broadband_plot['# OF WIRED BB PROVIDERS'], df_broadband_plot['mean'], size=df_broadband_plot['count'],
           sizes=(100,700), alpha=0.8, legend=False)
ax.set(xlim=(0.5,10.5),ylim=(64,70.5))
ax.plot(df_broadband_plot['# OF WIRED BB PROVIDERS'],p(df_broadband_plot['# OF WIRED BB PROVIDERS']))
plt.savefig(os.path.join(path, 'plot.pdf'), bbox_inches='tight')

fig, ax = plt.subplots(figsize=(12, 6))
sns.kdeplot(mydata['# OF WIRED BB PROVIDERS'], bw=0.19)
plt.savefig(os.path.join(path, 'densityplot.pdf'), bbox_inches='tight')

#ax.set_xlim([1, 10])
#ax.set_ylim([64, 70])
