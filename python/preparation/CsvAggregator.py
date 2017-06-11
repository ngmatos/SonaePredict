# Aggregates all the CSV into one PKL file

import os
import glob
import pandas as pd
import python.Config as Config


def get_merged_csv(flist, **kwargs):
    return pd.concat([pd.read_csv(f, **kwargs) for f in flist], ignore_index=True)

print('Aggregating all CSV files...')

path = Config.FILE_PATH
fmask = os.path.join(path, '*.csv')

df = get_merged_csv(glob.glob(fmask), index_col=None)

if not os.path.isfile(Config.H5_PATH + '/AggregatedDataset.pkl'):
    df.to_pickle(Config.H5_PATH + '/AggregatedDataset.pkl')

print('Aggregated all files.')
print(df.head())

del df
del path
del fmask
