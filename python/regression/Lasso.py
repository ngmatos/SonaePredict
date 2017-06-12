# import python.sampling.KFold as Sample
from math import sqrt

import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Lasso
import numpy as np
import python.Config as Config
import python.Timer as Timer

# Lasso without K-Fold Cross Validation

# Global vars
CHUNK_SIZE = 300000
time = Timer.Timer()


print('Reading chunks from ColumnedDatasetNonNegativeWithDateImputerBinary')
iter_hdf = pd.read_hdf(Config.H5_PATH + '/ColumnedDatasetNonNegativeWithDateImputerBinary.h5', chunksize=CHUNK_SIZE)

chunks = []
count = 0

# Read by chunks and join them
for chunk in iter_hdf:
    count += 1
    print('Joining at chunk', count)
    chunks.append(chunk)

print('Concatenating chunks')
chunks = pd.concat(chunks)
print('ALL CHUNKS:\n', chunks)
print('TIME ELAPSED: ', time.get_time_hhmmss())

# Generating X and y
y = chunks['quantity_time_key']
X = chunks.drop('quantity_time_key', 1)

print('CHUNKS AFTER REMOVING:\n', X)

lasso = Lasso(alpha=0.1)

Train_set, Test_set, Target_train, Target_test = train_test_split(X.iloc[0:2500000],
                                                                  y.iloc[0:2500000],
                                                                  test_size=0.3,
                                                                  random_state=42)
time.restart()
print('Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...')
lasso.fit(Train_set, Target_train)
print('TIME ELAPSED: ', time.get_time_hhmmss())

time.restart()
print('Predicting target with X_test (TEST SET)')
y_prediction = lasso.predict(X=Test_set)
print('TIME ELAPSED: ', time.get_time_hhmmss())

# Constructing a data frame for visualisation purposes
df = pd.DataFrame(data=y_prediction, columns=['TargetPrediction(yPred)'], index=Test_set.index)
df['TargetTest(yTest)'] = pd.Series(Target_test)

print('Mean of Target: ', np.mean(y))

print('Full data frame head:\n', df.head())
print('Full data frame info:\n', df.info())
print('Full data frame description:\n', df.describe())

print('Lasso Score (R^2): ', lasso.score(Test_set, Target_test))
print('Mean Squared Error', mean_squared_error(Target_test, y_prediction))
print('Root Mean Squared Error', sqrt(mean_squared_error(Target_test, y_prediction)))
print('Mean Absolute Error', mean_absolute_error(Target_test, y_prediction))


del time
del chunks
del count
del y
del X
del Train_set
del Target_train
del Test_set
del Target_test
del y_prediction
del df
'''
kf = Sample.kf

X_train, X_test, y_train, y_test = Sample.X_train, Sample.X_test, Sample.y_train, Sample.y_test

# Run Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Performance
print('Lasso score (R^2):', lasso.score(X_test, y_test))
'''
