import os

import python.Data as Data
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA, PCA
import python.Config as Config
from python.Timer import Timer

df = Data.read_chunks('ColumnedDatasetNonNegativeWithDateImputerBinary.h5')

target = df['quantity_time_key']

df = df.drop('quantity_time_key', 1)

rows = df.shape[0]
chunk_size = Config.CHUNK_SIZE
components = 50
ipca = IncrementalPCA(n_components=components, batch_size=32)

time = Timer()
for i in range(0, rows // chunk_size):
    time2 = Timer()
    print('Processing row: ', i)
    ipca.partial_fit(df[i * chunk_size: (i + 1) * chunk_size])
    print('Finished Processing row: ', i)
    print('TIME ELAPSED ON ROW: ', time2.get_time_hhmmss())
    time2.restart()
print('FINISHED PARTIAL FIT')
print('TIME ELAPSED: ', time.get_time_hhmmss())

if not os.path.isfile(Config.H5_PATH + '/PCAed.pkl'):
    print('SAVING TO PICKLE')
    time.restart()
    pd.to_pickle(ipca, Config.H5_PATH + '/PCAed.pkl')
    print('TIME ELAPSED: ', time.get_time_hhmmss())

print('ZEROS')
out = np.zeros((rows, components))
time.restart()
for i in range(0, rows // chunk_size):
    time2 = Timer()
    print("On: ", i)
    out[i * chunk_size: (i + 1) * chunk_size] = ipca.transform(df[i * chunk_size: (i + 1) * chunk_size])
    print('TIME ELAPSED ON ROW: ', time2.get_time_hhmmss())
    time2.restart()
print('TIME ELAPSED ON TRANSFORM: ', time.get_time_hhmmss())

df = pd.DataFrame(data=out)

if not os.path.isfile(Config.H5_PATH + '/PCAed50.h5'):
    print('SAVING TO H5')
    time.restart()
    df.to_hdf(Config.H5_PATH + '/PCAed50.h5', key='data', format='table')
    print('TIME ELAPSED: ', time.get_time_hhmmss())

print(df.head())
