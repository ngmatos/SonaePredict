from math import sqrt

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import python.Config as Config
import python.Timer as Timer
import python.Data as Data
import python.sampling.RandomSplit as RandomSplit
import matplotlib.pyplot as plot
import numpy as np

# Random Forest Regression

# Global vars
time = Timer.Timer()


def main():
    data, x = read_normal()

    # Run this function for each alpha
    run_rfr(data, x)


def read_normal():
    chunks = Data.read_chunks('ColumnedDatasetNonNegativeWithDateImputer.h5')

    # Generating X and y
    y = chunks['quantity_time_key']
    x = chunks.drop('quantity_time_key', 1)

    return RandomSplit.get_sample(x, y), x


def run_rfr(data, x):
    train_set, test_set, target_train, target_test = data
    time.restart()

    print('Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...')
    clf = RandomForestRegressor(verbose=1, n_jobs=-1, n_estimators=50)
    clf.fit(train_set, target_train)
    time.print()

    time.restart()
    print('Predicting target with X_test (TEST SET)')
    y_prediction = clf.predict(test_set)
    time.print()

    Data.calc_scores(target_test, y_prediction)

    print('RFR Score (R^2):', clf.score(test_set, target_test))
    # print('Mean Squared Error:', mean_squared_error(target_test, y_prediction))
    # print('Root Mean Squared Error:', sqrt(mean_squared_error(target_test, y_prediction)))
    # print('Mean Absolute Error:', mean_absolute_error(target_test, y_prediction))

# Run script
main()

'''
Using Random Split for evaluating estimator performance
Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...
[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:  3.6min finished
TIME ELAPSED:  00:03:34
Predicting target with X_test (TEST SET)
[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:    7.0s finished
TIME ELAPSED:  00:00:07
Mean Absolute Error 0.0535519379687
Root Mean Squared Error 0.24679369106447704
MSE: 0.0609
R2: 0.8538
'''

'''
Jobs = -1
Estimators = 15
Using Random Split for evaluating estimator performance
Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...
[Parallel(n_jobs=-1)]: Done  15 out of  15 | elapsed:  1.7min finished
TIME ELAPSED:  00:01:39
Predicting target with X_test (TEST SET)
[Parallel(n_jobs=8)]: Done  15 out of  15 | elapsed:    2.5s finished
TIME ELAPSED:  00:00:02
Mean Absolute Error 0.0527707313798
Root Mean Squared Error 0.24402824322289024
MSE: 0.0595
R2: 0.8571
'''

'''
Estimators = 50
Using Random Split for evaluating estimator performance
Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...
[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:  4.4min
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:  5.6min finished
TIME ELAPSED:  00:05:39
Predicting target with X_test (TEST SET)
[Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    6.4s
[Parallel(n_jobs=8)]: Done  50 out of  50 | elapsed:    8.4s finished
TIME ELAPSED:  00:00:08
Mean Absolute Error 0.0515096754874
Root Mean Squared Error 0.2384443351819688
MSE: 0.0569
R2: 0.8635
'''