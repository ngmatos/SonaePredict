from math import sqrt

import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import python.Config as Config
import python.Timer as Timer
import python.Data as Data
import python.sampling.RandomSplit as RandomSplit
import matplotlib.pyplot as plot
import numpy as np

# Bagging Regression

# Global vars
time = Timer.Timer()


def main():
    data, x = read_normal()

    # Run this function for each alpha
    run_br(data, x)


def read_normal():
    chunks = Data.read_chunks('ColumnedDatasetNonNegativeWithDateImputer.h5')

    # Generating X and y
    y = chunks['quantity_time_key']
    x = chunks.drop('quantity_time_key', 1)

    return RandomSplit.get_sample(x, y), x


def read_pca():
    df = Data.read_hdf('/PCAed50.h5')

    target = Data.read_hdf('ColumnedDatasetNonNegativeWithDateImputer.h5')
    target = target['quantity_time_key']

    return RandomSplit.get_sample(df, target), df


def run_br(data, x):
    train_set, test_set, target_train, target_test = data
    time.restart()

    print('Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...')
    clf = BaggingRegressor(n_jobs=-1, n_estimators=1000, verbose=1)
    clf.fit(train_set, target_train)
    time.print()

    time.restart()
    print('Predicting target with X_test (TEST SET)')
    y_prediction = clf.predict(test_set)
    time.print()

    Data.print_scores(target_test, y_prediction)

# Run script
main()

'''
Using Random Split for evaluating estimator performance
Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...
Time elapsed: 00:03:41 

Predicting target with X_test (TEST SET)
Time elapsed: 00:00:08 

R^2 Score: 0.852091180664
Mean Squared Error: 0.0616163993607
Root Mean Squared Error: 0.2482265081748665
Mean Absolute Error: 0.0536155326917
'''

'''
Using Random Split for evaluating estimator performance
Estimators = 20
Jobs = -1
Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...
[Parallel(n_jobs=8)]: Done   2 out of   8 | elapsed:  2.4min remaining:  7.2min
[Parallel(n_jobs=8)]: Done   8 out of   8 | elapsed:  3.3min finished
Time elapsed: 00:03:17 

Predicting target with X_test (TEST SET)
[Parallel(n_jobs=8)]: Done   2 out of   8 | elapsed:    7.3s remaining:   21.9s
[Parallel(n_jobs=8)]: Done   8 out of   8 | elapsed:   12.9s finished
Time elapsed: 00:00:13 

R^2 Score: 0.860246893495
Mean Squared Error: 0.0582188625466
Root Mean Squared Error: 0.241285852354897
Mean Absolute Error: 0.0523197127932
'''
