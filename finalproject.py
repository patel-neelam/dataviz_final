# PPHA 30536
# Autumn 2021
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


path = r'/Users/neelampatel/Documents/GitHub/final-project-neelam-patel'

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
df_total_housing_2020 = naco_data_scraper('Demographics', 2020, 'DEM_2020Census_HousingUnits_Total', 'Total_Housing_2020.csv')
df_usda_rural_dev_2016 = naco_data_scraper('Federal_Funding', 2016, 'FED_USDA_Amount', 'USDA_Rural_Dev_2016.csv')
df_poverty_rate_2019 = naco_data_scraper('Health_Human_Services', 2019, 'HHS_SAIPE_Poverty_Rate', 'Poverty_Rate_2019.csv')

###### Merge Naco Data into DF
# Source: https://stackoverflow.com/a/44338256

naco_dfs = [df_internet_providers_2015, df_total_housing_2020, df_usda_rural_dev_2016, df_poverty_rate_2019]
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
            'Broadband_Usage', 'TRN_BC_Number_of_Internet_Providers', 'DEM_2020Census_HousingUnits_Total', 
            'FED_USDA_Amount', 'HHS_SAIPE_Poverty_Rate', 'mean', 'count']
    df_all_data = df_all_data[cols]
    df_all_data = df_all_data.rename(columns={'mean': 'mean_plan_price',
                                              'count': 'plan_count',
                                              'County_name_x': 'County_name'})
    df_all_data.sort_values(by=['County_ID'])
    df_all_data.to_csv(os.path.join(path, file_name), index=False)

    return df_all_data

df_all_data = create_final_df('df_all_data.csv')

###### Summary Tables

# Source: https://stackoverflow.com/questions/22233488/pandas-drop-a-level-from-a-multi-level-column-index

def summary_by_state(df, file_name):
    df = df.groupby(['State', 'Policy_ind']).agg(['mean']).reset_index()
    df.columns = df.columns.droplevel(1)
    drops = ['County_ID', 'Metro_ind', 'Farm_ind', 'plan_count']
    df = df.drop(drops, axis=1)
    df.sort_values(by=['State'])
    df.to_csv(os.path.join(path, file_name), index=False)

    return df

df_by_state = summary_by_state(df_all_data, 'df_by_state.csv')

def summary_by_policy(df, file_name):
    df = df.groupby(['Policy_ind']).agg(['mean']).reset_index()
    df.columns = df.columns.droplevel(1)
    drops = ['County_ID', 'Metro_ind', 'Farm_ind', 'plan_count']
    df = df.drop(drops, axis=1)
    df.to_csv(os.path.join(path, file_name), index=False)

    return df

df_by_policy = summary_by_policy(df_all_data, 'df_by_policy.csv')

def summary_by_farm(df, file_name):
    df = df.groupby(['State', 'Farm_ind']).agg(['mean']).reset_index()
    df.columns = df.columns.droplevel(1)
    drops = ['County_ID', 'Metro_ind', 'plan_count']
    df = df.drop(drops, axis=1)
    df.sort_values(by=['State'])
    df.to_csv(os.path.join(path, file_name), index=False)

    return df

df_by_farm = summary_by_farm(df_all_data, 'df_by_farm.csv')

###### Plotting

### Extract US shapefile

# Source: https://medium.com/@loldja/reading-shapefile-zips-from-a-url-in-python-3-93ea8d727856

def get_zip(file_url, file_name):
    url = file_url
    fname = url.split('/')[-1].split('.')[0]
    if not os.path.isfile(os.path.join(path, fname, '.shp')):
        url_zip = requests.get(url)
        zipfile = ZipFile(BytesIO(url_zip.content))
        zipfile.extractall(path=os.path.join(path, 'County_shapefiles'))
        df_shp = geopandas.read_file(os.path.join(path, 'County_shapefiles', file_name))
    else:
        df_shp = geopandas.read_file(os.path.join(path, 'County_shapefiles', file_name))

    df_shp['GEOID'] = pd.to_numeric(df_shp['GEOID'])

    return df_shp

df_us_shp = get_zip(r'https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_5m.zip',
                    'cb_2020_us_county_5m.shp')
df_state_shp = get_zip(r'https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_state_5m.zip',
                      'cb_2020_us_state_5m.shp')

### Merge data

def merge_spatial_data(df, df_shp, df_shp_merge_col, df_merge_col, file_name):

    df_merged = df_shp.merge(df, left_on=df_shp_merge_col, right_on=df_merge_col, how='left')
    df_merged = df_merged[~df_merged.STUSPS.isin(['AK', 'HI', 'AS', 'GU', 'MP', 'PR', 'VI'])].reset_index(drop=True)

    df_merged.to_csv(os.path.join(path, file_name), index=False)

    return df_merged

df_by_county_plot = merge_spatial_data(df_all_data, df_us_shp, 'GEOID', 'County_ID', 'df_by_county_plot.csv')
#df_by_state_plot = merge_spatial_data(df_by_state, df_state_shp, 'STUSPS', 'State', 'df_by_state_plot.csv')

### Generate and export plots 

def generate_scatterplot_by_policytype(x, y, title, x_label, y_label):     
    fig, ax = plt.subplots(figsize=(12, 6))

    for type, group in df_by_state.groupby('Policy_ind'):
        ax.scatter(group[x], group[y], label=type)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.legend(loc='best', fontsize=12)
    ax.legend(['No municipal broadband restrictions', 'municipal broadband restrictions'])

    fig.savefig(os.path.join(path, title))
    return fig, ax

generate_scatterplot_by_policytype('TRN_BC_Number_of_Internet_Providers', 'mean_plan_price',
                                   'United States Average Lowest Priced Plan Price by Number of Providers',
                                   'Average # of Providers (By State)', 'Average Lowest Priced Plan')

generate_scatterplot_by_policytype('TRN_BC_Number_of_Internet_Providers', 'Broadband_Usage',
                                   'United States Average Broadband Usage by Number of Providers',
                                   'Average # of Providers (By State)', 'Average Broadband Usage Rate')

# Source: https://matplotlib.org/stable/tutorials/text/annotations.html

def generate_histogram(df, var, title):
    fig, ax = plt.subplots(figsize=(12, 4))

    high_vals_collected = np.clip(df_all_data[var], None, 20)

    n, bins, patches = ax.hist(high_vals_collected,
                               bins=20, edgecolor='black', rwidth=0.8)
    ax.set_ylabel('# of Counties')
    ax.set_xlabel('# of Internet Providers')
    ax.set_title(title)

    from matplotlib.ticker import FormatStrFormatter
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.0f'))

    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    ax.set_xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w))
    ax.set_xlim(bins[0], bins[-1]);
    ax.annotate('20 or more', xy=(19, 100),  xycoords='data',
                xytext=(0.8, 0.95), textcoords='axes fraction',
                arrowprops=dict(facecolor='black', shrink=0.01),
                horizontalalignment='right', verticalalignment='top',
                )
    fig.savefig(os.path.join(path, title))

    return fig, ax

generate_histogram(df_all_data, 'TRN_BC_Number_of_Internet_Providers',
                   'Number of Internet Providers By County')

def generate_boxplot(df, x, y, x_label, y_label, hue, hue_label, title):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.boxplot(x=x, y=y, orient="h", hue=hue, data=df, palette='Paired')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.legend(title=hue_label)
    fig.savefig(os.path.join(path, title))

    return fig, ax

generate_boxplot(df_all_data, 'TRN_BC_Number_of_Internet_Providers',
                 'Farm_ind', '# of Internet Providers', 'Farm County Indicator (1)',
                 'Policy_ind', 'Policy Indicator', 'Distribution of Internet Providers By County')

###### Linear Regression Model


def run_ols_model(parameters, file_name):

    model = smf.ols(parameters, data=df_all_data)
    result = model.fit()
    rs = result.summary()

    #Source: https://stackoverflow.com/questions/51734180/converting-statsmodels-summary-object-to-pandas-dataframe
    results_as_html = rs.tables[1].as_html()
    df = pd.read_html(results_as_html, header=0, index_col=0)[0]
    df.to_csv(os.path.join(path, file_name))
    print(rs)
    return df

provider_ols = run_ols_model('TRN_BC_Number_of_Internet_Providers ~ Metro_ind + Farm_ind + Policy_ind + FED_USDA_Amount + np.log(HHS_SAIPE_Poverty_Rate) + DEM_2020Census_HousingUnits_Total',
                             'Internet Providers OLS.csv')

broadband_avail_ols = run_ols_model('Broadband_Avail_FCC ~ Metro_ind + Farm_ind + Policy_ind + FED_USDA_Amount + np.log(HHS_SAIPE_Poverty_Rate) + DEM_2020Census_HousingUnits_Total',
                                    'Broadband Availability OLS.csv')

price_ols = run_ols_model('mean_plan_price ~ Metro_ind + Farm_ind + Policy_ind + FED_USDA_Amount + np.log(HHS_SAIPE_Poverty_Rate) + DEM_2020Census_HousingUnits_Total',
                          'Lowest Priced Plan OLS.csv')
