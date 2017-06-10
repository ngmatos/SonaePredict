# Aggregates all the CSV into one PKL file

import os
import glob
import pandas as pd


def get_merged_csv(flist, **kwargs):
    return pd.concat([pd.read_csv(f, **kwargs) for f in flist], ignore_index=True)

print('Aggregating all CSV files...')

path = '/Users/mercurius/GoogleDrive/FEUP/ADES/data'
fmask = os.path.join(path, '*.csv')

df = get_merged_csv(glob.glob(fmask), index_col=None)

if not os.path.isfile('h5/AggregatedDataset.pkl'):
    df.to_pickle('h5/AggregatedDataset.pkl')

print('Aggregated all files.')
print(df.head())

del df
del path
del fmask
