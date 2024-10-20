# Empirical exercise for "Income Risk, Global Shocks, and International Capital Flows"
# October 2024
# Data download script

# Import libraries
import os
import pandas as pd
import numpy as np
import pandas_datareader as pdr
import plotly.express as px
import dbnomics as dn


# Define countries
countries = ['Argentina', 'Brazil', 'Canada', 'Denmark', 'France', 'Germany', 'Italy', 'Mexico', 'Norway', 'Spain', 'Sweden', 'United States']
country_codes = ['AR', 'BR', 'CA', 'DK', 'FR', 'DE', 'IT', 'MX', 'NO', 'ES', 'SE', 'US']


##############################
# Download HFI Monetary Policy Shocks
##############################

# Get data on Monetary Policy Shocks from Jarocinski and Karadi (2020)
jk_shocks_m = pd.read_csv('https://github.com/marekjarocinski/jkshocks_update_fed/blob/aae4960759326f4d6452ca882d34c1ff88b38083/shocks_fed_jk_m.csv?raw=true')
jk_shocks_m.to_csv('data/shocks_fed_jk_m.csv', index=False)

jk_shocks_t = pd.read_csv('https://github.com/marekjarocinski/jkshocks_update_fed/blob/aae4960759326f4d6452ca882d34c1ff88b38083/shocks_fed_jk_t.csv?raw=true')
jk_shocks_t.to_csv('data/shocks_fed_jk_t.csv', index=False)

jk_data = pd.read_csv('https://github.com/marekjarocinski/jkshocks_update_fed/blob/aae4960759326f4d6452ca882d34c1ff88b38083/source_data/fomc_surprises_jk.csv?raw=true')
jk_data.to_csv('data/fomc_surprises_jk.csv', index=False)


##############################
# Convert Excel files to CSV
##############################

# Converting the excel files from Bauer and Swanson (2022) and Miranda-Agrippino (2014) to csv

# Bauer and Swanson (2022) data
data_BS = pd.read_excel('data/FOMC_Bauer_Swanson.xlsx', sheet_name='Monthly SVAR Data', index_col=0)
data_BS.index = pd.to_datetime(data_BS.index)
# Extract the monetary policy shocks
data_BS_MP_shocks = data_BS.filter(like='MPS')
data_BS_MP_shocks.columns = ['BS_MPS', 'BS_MPS_ORTH']
# Resample to quarterly frequency - as usual, sum the shocks in the quarter
data_BS_MP_shocks = data_BS_MP_shocks.dropna().resample('QE').sum()
# Fix the index so it is consistent with the other data
data_BS_MP_shocks.index = data_BS_MP_shocks.index.to_period('Q').to_timestamp()
data_BS_MP_shocks.to_csv('data_clean/FOMC_Bauer_Swanson.csv')

# Agrippino-Miranda (2014) data
data_SMA = pd.read_excel('data/NarrativeRomerRomerShock.xlsx', sheet_name='data', index_col=1, header=1)
data_SMA.index = pd.to_datetime(data_SMA.index)
data_SMA.drop(columns='Unnamed: 0', inplace=True)
data_SMA['Miranda-Agrippino (2014) quarterly'] = data_SMA['Unnamed: 4']
data_SMA.drop(columns=['Unnamed: 4', 'Miranda-Agrippino (2014)'], inplace=True)
# Resample to quarterly frequency - as usual, sum the shocks in the quarter
data_SMA = data_SMA.resample('QE').sum()
# Fix the index so it is consistent with the other data
data_SMA.index = data_SMA.index.to_period('Q').to_timestamp()
data_SMA.to_csv('data_clean/NarrativeRomerRomerShock.csv')

# Fernald (2024) data
data_JF = pd.read_excel('data/quarterly_TFP.xlsx', sheet_name='quarterly', index_col=0, header=1)

# Drop the summary rows at the end 
# CAUTION: The number of rows to drop is hard-coded
data_JF = data_JF.iloc[:-10, :]
data_JF.index = pd.PeriodIndex(data_JF.index.str.replace(':Q', 'Q'), freq='Q').to_timestamp()
data_JF.columns = ['jf_' + col for col in data_JF.columns]
data_JF.to_csv('data_clean/quarterly_TFP.csv')


##############################
# Fix csv files
##############################

# Bu, Rogers & Wu
brw_shocks = pd.read_csv('data/brw-shock-series.csv', index_col=0)
brw_shocks = brw_shocks.loc[~brw_shocks.index.isna()]
brw_shocks.index = pd.to_datetime(brw_shocks.index.str.replace('m', '-'), format='%Y-%m')
brw_shocks = brw_shocks['BRW_monthly (updated)']
# Resample to quarterly frequency - as usual, sum the shocks in the quarter
brw_shocks = pd.DataFrame(brw_shocks.resample('QE').sum())
brw_shocks.columns = ['BRW_shocks']
# Fix the index so it is consistent with the other data
brw_shocks.index = brw_shocks.index.to_period('Q').to_timestamp()
brw_shocks.to_csv('data_clean/brw-shock-series_clean.csv')

# Jarocinski and Karadi (2020)
data_jk = pd.read_csv('data/shocks_fed_jk_m.csv')
# Fix the index
data_jk['date'] = pd.to_datetime(data_jk[['year', 'month']].assign(day=1))
data_jk.set_index('date', inplace=True)
data_jk.drop(columns=['year', 'month'], inplace=True)
data_jk.columns = ['jk_' + col for col in data_jk.columns]
data_jk = data_jk.resample('QE').sum()
# Fix the index so it is consistent with the other data
data_jk.index = data_jk.index.to_period('Q').to_timestamp()
data_jk.to_csv('data_clean/shocks_fed_jk_q.csv')

# GRID data about Income Dynamics
id_data = pd.read_csv('data/Income_dynamics.csv')
# Replace country names with country codes used throughout this project
countries_id_names = id_data['country'].unique()
id_data['country'] = id_data['country'].replace(dict(zip(countries_id_names, country_codes)))
id_data.drop(columns=['gender', 'age'], inplace=True)

# Make the data wide format
id_data_wide = id_data.pivot(index='year', columns='country', values=id_data.columns[2:].to_list())
id_data_wide.columns = [col[1] + '_' + col[0] for col in id_data_wide.columns]
# Fix the index so it is consistent with the other data
id_data_wide.index = pd.to_datetime(id_data_wide.index, format='%Y')
id_data_wide.index = id_data_wide.index.to_period('Q')
# Insert rows for second, third, and fourth quarter with NaN values
# Create a new index that includes all quarters
end = id_data_wide.index[-1].asfreq('Y').end_time
new_index = pd.period_range(start=id_data_wide.index.min(), end=end, freq='Q')
# Reindex the DataFrame to include the new quarters, filling with NaN
id_data_wide = id_data_wide.reindex(new_index)
# Convert the index back to datetime for consistency
id_data_wide.index = id_data_wide.index.to_timestamp()
# Fill the NaN values with the last non-NaN value but 3 times max
id_data_wide = id_data_wide.ffill(limit=3)
id_data_wide.to_csv('data_clean/Income_dynamics_wide.csv')


##############################
# Setup for working with dbnomics
##############################

# Define data series
series_dict = {
    'current_account': ['IMF/BOP/Q.', '.BCA_BP6_XDC'],
    'gdp': ['IMF/IFS/Q.', '.NGDP_NSA_XDC'],
    'cpi': ['IMF/IFS/Q.', '.PCPI_IX'],
    'policy_rate': ['IMF/IFS/Q.', '.FPOLM_PA'],
    '3m_treasury_rate': ['IMF/IFS/Q.', '.FITB_3M_PA'],
    'fx': ['IMF/IFS/Q.', '.ENDA_XDC_USD_RATE']
}


def extract_series(df):
    '''Extracts the series from the dataframe
    Input is a dataframe from a dbnomics query and the output is a single Pandas series
    
    Args:
        df: Pandas dataframe from dbnomics query
    
    Returns:
        sr: Pandas series with the data
    '''
    df = df[['period', 'value']]
    df.loc[:, 'period'] = pd.to_datetime(df['period'])
    df = df.set_index('period')
    sr = df['value']
    return pd.DataFrame(sr)

# Create list of all series and corresponding names
all_series = []
names = []
for country in country_codes:
    for key, value in series_dict.items():
        all_series.append(f'{value[0]}{country}{value[1]}')
        names.append(f'{country}_{key}')
        
# Since Euro area has a common monetary policy, ECB rates are used 
# (which are only available with the code U2)
eu_countries = ['FR', 'DE', 'IT', 'ES']
for country in eu_countries:
    all_series = [s.replace(f'IMF/IFS/Q.{country}.FPOLM_PA', 'IMF/IFS/A.U2.FIDFR_PA') for s in all_series]
    
# 3M Treasury rate is not available for all countries
# For US using Treasury bills (FITB_PA) is a good proxy, so get that for the other countries
countries_3m_proxy = ['BR', 'CA', 'DE', 'IT', 'MX', 'ES', 'SE']
for country in countries_3m_proxy:
    all_series = [s.replace(f'IMF/IFS/Q.{country}.FITB_3M_PA', f'IMF/IFS/Q.{country}.FITB_PA') for s in all_series]
        
# Also an ok proxy is the interest rate on general government debt (FIGB_PA)
countries_other_proxy = ['DK', 'NO']
for country in countries_other_proxy:
    all_series = [s.replace(f'IMF/IFS/Q.{country}.FITB_3M_PA', f'IMF/IFS/Q.{country}.FIGB_PA') for s in all_series]

# Organize series in a dataframe to check
series_array = np.array(all_series).reshape(len(countries), len(series_dict))
series_df = pd.DataFrame(series_array, columns=series_dict.keys(), index=countries)

# Replace list as an array
all_series = series_df.values.flatten()
names = np.array(names)


##############################
# Download data from dbnomics
##############################

# Download data
macro_data = pd.DataFrame()
for i in range(len(all_series)):
    series = all_series[i]
    name = names[i]
    data = dn.fetch_series(series)
    
    # Extract data and resample to yearly (because GDP data is quarterly)
    if len(data) > 0:
        data = extract_series(data)
        data.columns = [name]
        macro_data = macro_data.join(data, how='outer')

# For EU countries, exchange rate changes in 1999
EUR_fx = extract_series(dn.fetch_series('IMF/IFS/Q.U2.ENDA_XDC_USD_RATE'))

# Replace exchange rate for Euro area countries only if it is not NaN
for country in eu_countries:
    EUR_fx = EUR_fx.reindex(macro_data.index).dropna()
    macro_data.loc[EUR_fx.index, f'{country}_fx'] = EUR_fx.values.flatten() 


# Data that is not on dbnomics:
# Current account data for Argentina, Brazil, Mexico, and the United States
# CPI data for Argentina
# Treasury rate data for Argentina


##############################
# Download missing data from FRED
##############################

# Current account data
# Argentina: CUAEEFARQ052N (Dollars) - until 2008
# Argentina: ARGB6BLTT02STSAQ (Percent of GDP) - from 2006
# Brazil: CUAEEFBRQ052N (Dollars) - until 2008
# Brazil: BRAB6BLTT02STSAQ (Percent of GDP) - from 1996
# Mexico: MEXB6BLTT02STSAQ (Percent of GDP) - from 2006
# United States: NETFI (Billions of Dollars)

# CPI data
# Argentina: DDOE01ARA086NWDB - until 2014 (but only anual data)

# Treasury rate data
# Argentina: ?

# Get data from FRED
fred_series = ['CUAEEFARQ052N', 'ARGB6BLTT02STSAQ', 'CUAEEFBRQ052N', 'BRAB6BLTT02STSAQ', 'MEXB6BLTT02STSAQ', 'NETFI', 'DDOE01ARA086NWDB']
fred_names = ['AR_current_account', 'AR_current_account_gdp', 'BR_current_account', 'BR_current_account_gdp', 'MX_current_account_gdp', 'US_current_account', 'AR_cpi']

fred_data = pdr.get_data_fred(fred_series, start='1980-01-01')
fred_data.columns = fred_names

# Fix Units for current account and percentage data
fred_data[['AR_current_account_gdp', 'BR_current_account_gdp', 'MX_current_account_gdp']] = fred_data[['AR_current_account_gdp', 'BR_current_account_gdp', 'MX_current_account_gdp']] / 100
fred_data['US_current_account'] = fred_data['US_current_account'] * 1000

# Fix units and currency for Argentina and Brazil
fx_fix_countries = ['AR', 'BR']
for country in fx_fix_countries:
    fred_data[f'{country}_current_account'] = fred_data[f'{country}_current_account'] / 1e6
    fred_data[f'{country}_current_account'] = fred_data[f'{country}_current_account'] / macro_data[f'{country}_fx'].reindex_like(fred_data[f'{country}_current_account'])

# Change annual data to quarterly for Argentina CPI
end = fred_data['AR_cpi'].last_valid_index()
end = end + pd.DateOffset(months=11)
fred_data.loc[:end, 'AR_cpi'] = fred_data.loc[:end, 'AR_cpi'].ffill()

ar_cpi = pd.DataFrame(fred_data['AR_cpi'])

# Add AR CPI to macro data and remove from FRED data
macro_data = macro_data.join(ar_cpi, how='outer')

fred_data = fred_data.drop(columns='AR_cpi')


# Construct a series for inflation for each country
for country in country_codes:
    macro_data[f'{country}_inflation'] = macro_data[f'{country}_cpi'].pct_change(4, fill_method=None)


# Join all current account and GDP data into a single dataframe
fred_data = fred_data.join(macro_data.filter(like='current_account'), how='outer')
fred_data = fred_data.join(macro_data.filter(like='gdp'), how='outer')


# Calculate the current account as percentage of GDP
ca_gdp_list = []
ca_gdp_list_countries = []

for country in country_codes:
    ca_gdp_list.append(f'{country}_current_account_gdp')
    ca_gdp_list_countries.append(f'{country}')

# Check if the data is already fetched
already_fetched = list(fred_data.filter(like='current_account_gdp').columns)
for name in already_fetched:
    ca_gdp_list.remove(name)
    ca_gdp_list_countries.remove(name[:2])

# Add columns to the dataframe with NaN values
for name in ca_gdp_list:
    fred_data[name] = np.nan
    
# If the data is na, fill by the ratio of the current account to GDP
for country in country_codes:
    ca_gdp = f'{country}_current_account_gdp'
    
    # If the data is not already fetched, calculate it entirely
    if ca_gdp not in already_fetched:
        fred_data[ca_gdp] = fred_data[f'{country}_current_account'] / fred_data[f'{country}_gdp']
    
    # Mexico only has current account as percentage of GDP
    elif country in ['MX']:
        continue
    
    # If the data is already fetched, fill the missing values only
    else:
        rep_mask = fred_data[ca_gdp].isna()
        fred_data.loc[rep_mask, ca_gdp] = (
            fred_data.loc[rep_mask, f'{country}_current_account'] / 
            fred_data.loc[rep_mask, f'{country}_gdp']
        )

# Add the new data to the macro data
macro_data = macro_data.join(fred_data.filter(like='current_account_gdp'), how='outer')

# Remove current account data in macro data
# US current account data is only in the FRED data
ca_gdp_list_countries.remove('US')

for country in ca_gdp_list_countries:
    macro_data = macro_data.drop(f'{country}_current_account', axis=1)

macro_data.to_csv('data_clean/macro_data.csv')


##############################
# Create fully merged dataset
##############################

# Merge all data
data = macro_data.join(data_BS_MP_shocks, how='outer')
data = data.join(data_SMA, how='outer')
data = data.join(data_JF, how='outer')
data = data.join(brw_shocks, how='outer')
data = data.join(data_jk, how='outer')
data = data.join(id_data_wide, how='outer')

# Save the data
data.to_csv('data_clean/full_data.csv')