import os
import pandas as pd
import numpy as np
import python.Config as Config
from sklearn.preprocessing import StandardScaler, MinMaxScaler


df = pd.read_hdf(Config.H5_PATH + '/ColumnedDatasetNonNegativeWithDateImputer.h5')
cols_to_norm = ['change_pct', 'price_retail', 'quantity_int', 'gross_sls_amt_eur_int', 'price']

df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

if not os.path.isfile(Config.H5_PATH + '/Normalized.h5'):
    df.to_hdf(Config.H5_PATH + '/Normalized.h5', key='data', format='table')

print(df.head())

''' 

colsToConcat = ['location_cd', 'week', 'week_day', 'sku', 'tematico_ind', 'folheto_ind', 'tv_ind', 'card',
                'quantity_time_key']
dfToConcat = df[colsToConcat]

min_max_scaler = MinMaxScaler()

final = min_max_scaler.fit_transform(df[cols])
fds = pd.DataFrame(final, columns=df[cols].columns, index=df.index)

print(fds.head())

toConcat = [fds, dfToConcat]

fds = pd.concat(toConcat)
print(fds.head())

'''

'''

scaler = StandardScaler(copy=False)

n = df[cols].shape[0]  # number of rows
batch_size = 1000  # number of rows in each call to partial_fit
index = 0  # helper-var

while index < n:
    partial_size = min(batch_size, n - index)  # needed because last loop is possibly incomplete
    partial_x = df[index:index + partial_size]
    scaler.partial_fit(partial_x)
    index += partial_size
    print("Got to... " + str(index))

print("Starting transform")

index = 0
scaled = pd.DataFrame(data=np.zeros(df.shape), columns=df.columns)
while index < n:
    partial_size = min(batch_size, n - index)  # needed because last loop is possibly incomplete
    partial_x = df[index:index + partial_size]
    scaled[index:index + partial_size] = scaler.transform(partial_x)
    index += partial_size
    print("Transformed... " + str(index))
# scaled = scaler.transform(df)

if not os.path.isfile(Config.H5_PATH + '/Normalized.h5'):
    df.to_hdf(Config.H5_PATH + '/Normalized', key='data', format='table')

print(scaled)

'''