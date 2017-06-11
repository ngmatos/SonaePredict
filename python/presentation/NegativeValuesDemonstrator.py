# We have negative values in currencies and in quantity so we have to take care of it

import pandas as pd
import os
import python.Config as Config

print('Demonstrating that there are some negative values that are inconsistent')

df = pd.read_pickle(Config.H5_PATH + '/ColumnedDataset.pkl')

stat = df.describe()

columns = df.columns.values.tolist()

for name in columns:
    if stat[name]['min'] < 0:
        df = df[df[name] >= 0]

if not os.path.isfile(Config.H5_PATH + '/ColumnedDatasetNonNegative.pkl'):
    df.to_pickle(Config.H5_PATH + '/ColumnedDatasetNonNegative.pkl')

print(df.describe())

del df
del stat
del columns
