#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 12:20:07 2022

@author: neelampatel
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

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
