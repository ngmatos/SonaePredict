import os
import pandas as pd
from sklearn.preprocessing import Imputer

df = pd.read_hdf('h5/ColumnedDatasetNonNegativeWithDateBinary.h5')


def treat_price_retail(data, column, strategy='mean'):
    imp = Imputer(missing_values='NaN', strategy=strategy, axis=1)
    data[column] = imp.fit_transform(data[column]).T

    return data


df = treat_price_retail(df, 'price_retail')
df = treat_price_retail(df, 'tematico_ind')
df = treat_price_retail(df, 'folheto_ind')
df = treat_price_retail(df, 'tv_ind')

# remove time_key because it's unnecessary since we already have date in other format
df = df.drop('time_key', 1)

if not os.path.isfile('h5/ColumnedDatasetNonNegativeWithDateBinaryImputer.h5'):
    df.to_hdf('h5/ColumnedDatasetNonNegativeWithDateBinaryImputer.h5', key='data', format='table')

print(df.head())


