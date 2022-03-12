# PPHA 30560
# Winter 2022
# Final Project

# Neelam Patel
# patel-neelam

import os
import requests
import pandas as pd
import us
import re
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas
import numpy as np
from zipfile import ZipFile
from io import BytesIO
import statsmodels.formula.api as smf
from functools import reduce


path = r'/Users/neelampatel/Documents/GitHub/dataviz_final'

# Source:https://stackoverflow.com/questions/28144529/how-to-check-if-file-already-exists-if-not-download-on-python


def download_csvs():
    url1 = r'https://github.com/microsoft/USBroadbandUsagePercentages/raw/master/dataset/broadband_data_2020October.csv'
    fname1 = url1.split('/')[-1]
    if not os.path.isfile(os.path.join(path, fname1)):
        df1 = pd.read_csv(url1,
                          skiprows=18,
                          na_values=['-', ' -   '])
        df1.to_csv(os.path.join(path, fname1), index=False)
    else:
        df1 = pd.read_csv(os.path.join(path, fname1))

    # Source: https://stackoverflow.com/questions/5552555/unicodedecodeerror-invalid-continuation-byte
    url2 = r'https://github.com/BroadbandNow/Open-Data/raw/master/broadband_data_opendatachallenge.csv'
    fname2 = url2.split('/')[-1]
    if not os.path.isfile(os.path.join(path, fname2)):
        df2 = pd.read_csv(url2,
                          encoding='latin-1')
        df2.to_csv(os.path.join(path, fname2), index=False)
    else:
        df2 = pd.read_csv(os.path.join(path, fname2), encoding='latin-1')

    url3 = r'https://www.ers.usda.gov/webdocs/DataFiles/48652/ERSCountyTypology2015Edition.xls?v=8780.2'
    fname3 = url3.split('/')[-1].split('?')[0]
    if not os.path.isfile(os.path.join(path, fname3)):
        df3 = pd.read_excel(url3,
                            skiprows=3)
        df3.to_excel(os.path.join(path, fname3))
    else:
        df3 = pd.read_excel(os.path.join(path, fname3))

    return df1, df2, df3

df_microsoft, df_broadbandnow, df_ers = download_csvs()

###### webscraping from Naco
# Source: https://towardsdatascience.com/web-scraping-basics-82f8b5acd45c

def naco_data_scraper(data_map, year, ind, csv_name):
    url = r'https://ce.naco.org/get/data'

    headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36'}
    payload = {data_map: {'year': year, 'inds': [ind]}}
    response = requests.post(url, headers=headers, json=payload)

    j = response.json()
    df = pd.DataFrame(j['data'], columns=j['columns'])

    if not os.path.isfile(os.path.join(path, csv_name)):
        df.to_csv(os.path.join(path, csv_name), index=False)
    else:
        df = pd.read_csv(os.path.join(path, csv_name))

    df['FIPS'] = pd.to_numeric(df['FIPS'])
    df = df[df[ind].notnull()]

    return df

df_internet_providers_2015 = naco_data_scraper('Transportation_Infrastructure', 2015, 'TRN_BC_Number_of_Internet_Providers', 'TRN_BC_Number_of_Internet_Providers_2015.csv')
df_poverty_rate_2019 = naco_data_scraper('Health_Human_Services', 2019, 'HHS_SAIPE_Poverty_Rate', 'Poverty_Rate_2019.csv')

###### Merge Naco Data into DF
# Source: https://stackoverflow.com/a/44338256

naco_dfs = [df_internet_providers_2015, df_poverty_rate_2019]
df_naco_merged = reduce(lambda left, right: pd.merge(left, right, on=['FIPS'],
                                                     how='outer'), naco_dfs)
###### format and clean data

def data_cleaner(df, column_str, old_code, new_code): 
    df.loc[(df[column_str] == old_code), column_str] = new_code

def clean_df_format():

    # format df_microsoft
    df_microsoft.columns = [c.strip() for c in df_microsoft.columns]
    df_microsoft_clean = df_microsoft.rename(columns={'COUNTY NAME': 'County_name',
                         'COUNTY ID': 'County_ID',
                         'BROADBAND AVAILABILITY PER FCC': 'Broadband_Avail_FCC',
                         'BROADBAND USAGE': 'Broadband_Usage'})

    # format df_ers
    df_ers_clean = df_ers[df_ers['FIPStxt'] != 51515]
    df_ers_clean.rename(columns={'Metro-nonmetro status, 2013 0=Nonmetro 1=Metro': 'Metro_ind',
                                   'Farming_2015_Update (allows overlap, 1=yes)': 'Farm_ind'}, inplace=True)
    df_ers_clean = df_ers_clean[['FIPStxt', 'State', 'County_name', 'Metro_ind', 'Farm_ind']]

    # Replaced FIPS codes based on convention seen in Geocorr 2018 notes: https://mcdc.missouri.edu/applications/docs/geocorr2018-notes.html
    data_cleaner(df_ers_clean, 'FIPStxt', 46113, 46102)  # Shannon County is now Oglala County
    data_cleaner(df_ers_clean, 'FIPStxt', 2270, 2158)  # Wade Hampton Census Area is now Kusilvak Census Area

    # format df_broadband
    df_broadbandnow_clean = df_broadbandnow[df_broadbandnow['Population'].notnull()]

    state_dict = us.states.mapping('name', 'abbr')
    df_broadbandnow_clean['State'] = df_broadbandnow_clean['State'].map(state_dict)

    df_broadbandnow_clean = df_broadbandnow_clean[df_broadbandnow_clean['Population'] != 0]
    df_broadbandnow_clean = df_broadbandnow_clean[~df_broadbandnow_clean.State.isin(['VI', 'PR'])]

    return df_microsoft_clean, df_ers_clean, df_broadbandnow_clean

df_microsoft_clean, df_ers_clean, df_broadbandnow_clean = clean_df_format()

df_broadbandnow_clean.info()
df_broadbandnow_clean['%Access to Terrestrial Broadband'] = df_broadbandnow_clean['%Access to Terrestrial Broadband'].str.strip('%')
df_broadbandnow_clean['%Access to Terrestrial Broadband'] = pd.to_numeric(df_broadbandnow_clean['%Access to Terrestrial Broadband'])/100


###### Merge county-level dataframes and add policy indicator

# Source: https://stackoverflow.com/questions/44826282/python-how-to-remove-words-from-a-string
# Source: https://stackoverflow.com/questions/3663450/remove-substring-only-at-the-end-of-string

def county_name_stripper(county):
    removal_list = ['county', 'city', 'census', 'area', 'census area', 'and borough',
                    'borough', 'municipality', ' area, ', ' census area, ak', 'ak', 'parish']

    edit_string_as_list = county.split()
    final_list = [word for word in edit_string_as_list if word not in removal_list]
    final_string = ' '.join(final_list)
    final_string = re.sub(' and$', '', final_string)
    return final_string


def merge_county_dfs():

    df_merge = df_microsoft_clean.merge(df_ers_clean, how='outer', left_on='County_ID', right_on='FIPStxt')
    df_merge = df_merge.merge(df_naco_merged, how='outer', left_on='County_ID', right_on='FIPS')

    # Source: https://broadbandnow.com/report/municipal-broadband-roadblocks/
    muni_broadband_policy_restrict = ['AL', 'FL', 'LA', 'MN', 'MO', 'MT', 'MI', 'NE', 'NV', 'NC',
                                      'PA', 'SC', 'TN', 'TX', 'UT', 'VA', 'WA', 'WI']

    df_merge['Policy_ind'] = df_merge['State'].apply(lambda x: 1 if x in muni_broadband_policy_restrict else 0)
    df_merge = df_merge.drop(['ST', 'FIPStxt', 'FIPS'], axis=1)

    df_merge['County_name_y'] = df_merge.County_name_y.str.lower()
    df_merge['County_name_y'] = df_merge.County_name_y.str.strip()
    df_merge['County_name_y'] = df_merge.County_name_y.map(county_name_stripper)

    data_cleaner(df_merge, 'County_name_y', 'la salle', 'lasalle')
    data_cleaner(df_merge, 'County_name_y', 'price of wales hyder', 'prince of wales-hyder')
    data_cleaner(df_merge, 'County_name_y', 'hoonah-angoon area,', 'hoonah-angoon')

    return df_merge

df_merge = merge_county_dfs()

###### Aggregate zip code level data to county data

def zip_to_county_avg(ind_column):
   
    df_pricing = df_broadbandnow_clean.groupby(['County', 'State'])[ind_column].agg(['mean', 'count']).reset_index()
    df_pricing['County'] = df_pricing['County'].str.lower()
    df_pricing['County'] = df_pricing.County.map(county_name_stripper)

    data_cleaner(df_pricing, 'County', 'doã±a ana', 'dona ana')
    data_cleaner(df_pricing, 'County', 'la salle', 'lasalle')	

    return df_pricing

df_pricing = zip_to_county_avg('Lowest Priced Terrestrial Broadband Plan')

###### Create final dataframe with all data

def create_final_df(file_name):

    df_all_data = df_merge.merge(df_pricing, how='left', left_on=['County_name_y', 'State'], right_on=['County', 'State'])

    cols = df_all_data.columns.tolist()
    cols = ['County_ID', 'State', 'County_name_x', 'Metro_ind', 'Farm_ind', 'Policy_ind', 'Broadband_Avail_FCC', 
            'Broadband_Usage', 'TRN_BC_Number_of_Internet_Providers', 'HHS_SAIPE_Poverty_Rate', 'mean', 'count']
    df_all_data = df_all_data[cols]
    df_all_data = df_all_data.rename(columns={'mean': 'mean_plan_price',
                                              'count': 'plan_count',
                                              'County_name_x': 'County_name'})
    df_all_data.sort_values(by=['County_ID'])
    df_all_data.to_csv(os.path.join(path, file_name), index=False)

    return df_all_data

df_all_data = create_final_df('df_all_data.csv')

###### Calculate Broadband Penetration/Coverage by State

df_all_data['coverage_count'] = df_all_data['Broadband_Avail_FCC'].apply(lambda x: 1 if x >= 0.95 else 0)

df_sum = df_all_data.groupby(['State'])[['coverage_count']].agg(['sum','count']).reset_index()
df_sum.columns = df_sum.columns.droplevel(0)
df_sum['percent_coverage'] = df_sum['sum']/df_sum['count']
df_sum = df_sum.rename(columns={'': 'State'})

###### Summary Tables

# Source: https://stackoverflow.com/questions/22233488/pandas-drop-a-level-from-a-multi-level-column-index

def summary_by_state(df, df2, file_name):
    df = df.groupby(['State', 'Policy_ind']).agg(['mean']).reset_index()
    df.columns = df.columns.droplevel(1)
    drops = ['County_ID', 'Metro_ind', 'Farm_ind', 'plan_count', 'coverage_count']
    df = df.drop(drops, axis=1)
    df3 = df.merge(df_sum, how='outer', on='State') 
    df = df3
    df.sort_values(by=['State'])
    df.to_csv(os.path.join(path, file_name), index=False)

    return df

df_by_state = summary_by_state(df_all_data, df_sum, 'df_by_state.csv')


### Plotting
sns.set_style('whitegrid')
df_density = df_broadbandnow_clean.groupby(['County', 'State'])['WiredCount_2020'].agg(['mean', 'count']).reset_index()
sns.displot(df_density['mean'], kde=True)
sns.displot(df_all_data['TRN_BC_Number_of_Internet_Providers'], kde=True)
sns.displot(df_broadbandnow_clean['WiredCount_2020'], kde=True)
sns.kdeplot(df_broadbandnow_clean['WiredCount_2020'], bw=0.2)

