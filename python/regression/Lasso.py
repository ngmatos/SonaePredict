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
DATA_SET_SIZE = Config.TRIM_DATA_SET
time = Timer.Timer()


print('Reading chunks from ColumnedDatasetNonNegativeWithDateImputer')
iter_hdf = pd.read_hdf(Config.H5_PATH + '/ColumnedDatasetNonNegativeWithDateImputer.h5',
                       chunksize=Config.CHUNK_SIZE)

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

Train_set, Test_set, Target_train, Target_test = train_test_split(X,
                                                                  y,
                                                                  test_size=0.4,
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
df = pd.DataFrame(data=y_prediction, columns=['TargetPrediction(yPred)'], index=Target_test.index)
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
Full data frame description:
        TargetPrediction(yPred)  TargetTest(yTest)
count             1.363804e+06       1.363804e+06
mean              1.070951e-01       1.064061e-01
std               5.712002e-01       6.454332e-01
min              -7.745373e-03       0.000000e+00
25%               9.026660e-03       0.000000e+00
50%               1.965499e-02       0.000000e+00
75%               5.194681e-02       3.333000e-02
max               4.920873e+01       6.728900e+01
Lasso Score (R^2):  0.749612538813
Mean Squared Error 0.104307328479
Root Mean Squared Error 0.3229664510120647
Mean Absolute Error 0.0734283518468
'''
