# Convert nominal attributed to binary attributes
# Attributes are location_cd, sku, week, week_day
import os
import pandas as pd
import python.Config as Config


print('Transforming nominal attributes to binary')

df = pd.read_hdf(Config.H5_PATH + '/ColumnedDatasetNonNegativeWithDateImputer.h5')

cols_to_encode = ['location_cd', 'sku', 'week', 'week_day']

tmp_df = pd.get_dummies(df, columns=cols_to_encode)

if not os.path.isfile(Config.H5_PATH + '/ColumnedDatasetNonNegativeWithDateImputerBinary.h5'):
    tmp_df.to_hdf(Config.H5_PATH + '/ColumnedDatasetNonNegativeWithDateImputerBinary.h5', key='data', format='table')

del df
del cols_to_encode
del tmp_df
