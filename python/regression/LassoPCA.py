from math import sqrt

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

import python.Config as Config
from python.Timer import Timer

# Lasso with PCA Classes

time = Timer()
print('Reading PCA file')
df = pd.read_hdf(Config.H5_PATH + '/PCAed.h5')
print('Read PCA')
print('TIME ELAPSED: ', time.get_time_hhmmss())

# Getting target attribute
time.restart()
print('Reading target file')
target = pd.read_hdf(Config.H5_PATH + '/ColumnedDatasetNonNegativeWithDateImputer.h5')
target = target['quantity_time_key']
print('Read target file')
print('TIME ELAPSED: ', time.get_time_hhmmss())

lasso = Lasso(alpha=0.1)
# Splitting Sets
print('Splitting Sets')
Train_set, Test_set, Target_train, Target_test = train_test_split(df, target, test_size=0.3, random_state=42)

# Fitting
time.restart()
print('Fitting model with DF_train(PCAed) and Target(TrainSet)...')
lasso.fit(Train_set, Target_train)
print('TIME ELAPSED: ', time.get_time_hhmmss())

# Predicting
time.restart()
print('Predicting target with DF_test(PCAed)')
y_prediction = lasso.predict(X=Test_set)
print('TIME ELAPSED: ', time.get_time_hhmmss())

# Constructing a data frame for visualisation purposes
df = pd.DataFrame(data=y_prediction, columns=['TargetPrediction(yPred)'], index=Target_test.index)
df['TargetTest(yTest)'] = pd.Series(Target_test)

print('Full data frame:\n', df.head())
print('Full data frame info:\n', df.info())
print('Full data frame description:\n', df.describe())

print('Train_Set count: ', Train_set.shape)
print('Target_train count: ', Target_train.shape)

print('Test_Set count: ', Test_set.shape)
print('Target_test count: ', Target_test.shape)

print('Lasso Score (R^2): ', lasso.score(Test_set, Target_test))
print('Mean Squared Error', mean_squared_error(Target_test, y_prediction))
print('Root Mean Squared Error', sqrt(mean_squared_error(Target_test, y_prediction)))
print('Mean Absolute Error', mean_absolute_error(Target_test, y_prediction))

del target
del time
del df
del Train_set
del Target_train
del Test_set
del Target_test
del y_prediction
