# Removing unnecessary columns
import os

import pandas as pd

print('Selecting columns...')

cols = ['location_cd', 'promotion', 'time_key',
        'sku', 'change_pct', 'price_retail', 'tematico_ind',
        'folheto_ind', 'tv_ind', 'price', 'card', 'quantity_int',
        'gross_sls_amt_eur_int', 'quantity_time_key']

df = pd.read_pickle('h5/AggregatedDataset.pkl')

if not os.path.isfile('h5/ColumnedDataset.pkl'):
        df[cols].to_pickle('h5/ColumnedDataset.pkl')

df = pd.read_pickle('h5/ColumnedDataset.pkl')

print(df.describe())

del cols
del df
