# Removing unnecessary columns
import os
import pandas as pd
import python.Config as Config

print('Selecting columns...')

cols = ['location_cd', 'promotion', 'time_key',
        'sku', 'change_pct', 'price_retail', 'tematico_ind',
        'folheto_ind', 'tv_ind', 'price', 'card', 'quantity_int',
        'gross_sls_amt_eur_int', 'quantity_time_key']

df = pd.read_pickle(Config.H5_PATH + '/AggregatedDataset.pkl')

if not os.path.isfile(Config.H5_PATH + '/ColumnedDataset.pkl'):
        df[cols].to_pickle(Config.H5_PATH + '/ColumnedDataset.pkl')

print(df[cols].describe())

del cols
del df
