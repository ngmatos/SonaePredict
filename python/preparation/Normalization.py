import pandas as pd
import numpy as np
import python.Config as Config
from sklearn.preprocessing import StandardScaler


df = pd.read_hdf(Config.H5_PATH + '/ColumnedDatasetNonNegativeWithDateImputerBinary.h5')
df = df.drop('promotion', 1)
df = df.drop('year', 1)
print(df.head())

scaler = StandardScaler(copy=False)

n = df.shape[0]  # number of rows
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

print(scaled)
