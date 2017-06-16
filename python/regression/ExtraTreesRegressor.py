from math import sqrt

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import python.Config as Config
import python.Timer as Timer
import python.Data as Data
import python.sampling.RandomSplit as RandomSplit
import matplotlib.pyplot as plot
import numpy as np
import _pickle as pickle

# Extra Trees Regressor

# Global vars
time = Timer.Timer()


def main():
    data = read_normal()
    # Run this function for each alpha
    run_etr(data)


def read_normal():
    chunks = Data.read_chunks('ColumnedDatasetNonNegativeWithDateImputer.h5')

    # Generating X and y
    y = chunks['quantity_time_key']
    x = chunks.drop('quantity_time_key', 1)

    return x, y


def run_etr(data):
    x, y = data
    train_set, test_set, target_train, target_test = RandomSplit.get_sample(x, y)
    time.restart()

    print('Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...')
    clf = ExtraTreesRegressor(n_estimators=100, n_jobs=-1, verbose=1, bootstrap=True)
    clf.fit(train_set, target_train)
    time.print()

    # time.restart()
    # print('Saving model')
    # filename = 'ETRModel.pkl'
    # pickle.dump(clf, open(filename, 'wb'))
    # print('TIME SPENT: ', time.get_time_hhmmss())

    time.restart()
    print('Predicting target with X_test (TEST SET)')
    y_prediction = clf.predict(test_set)
    time.print()

    Data.calc_scores(target_test, y_prediction)

    # Plotting Results
    fig, ax = plot.subplots()
    ax.scatter(target_test, y_prediction)
    ax.plot([target_test.min(), target_test.max()], [target_test.min(), target_test.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plot.show()

# Run script
main()

'''
Reading /PCAed50.h5 file
TIME ELAPSED:  00:00:04
Reading /ColumnedDatasetNonNegativeWithDateImputer.h5 file
TIME ELAPSED:  00:00:02
Using Random Split for evaluating estimator performance
Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...
TIME ELAPSED:  00:14:01
Predicting target with X_test (TEST SET)
TIME ELAPSED:  00:00:10
Mean Absolute Error 0.055160486109
Root Mean Squared Error 0.2537421547073574
MSE: 0.0644
R2: 0.8454
'''

'''
Estimators = 30
Using Random Split for evaluating estimator performance
Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...
[Parallel(n_jobs=-1)]: Done  30 out of  30 | elapsed:  2.3min finished
TIME ELAPSED:  00:02:18
Predicting target with X_test (TEST SET)
-[Parallel(n_jobs=8)]: Done  30 out of  30 | elapsed:    5.6s finished
TIME ELAPSED:  00:00:05
Mean Absolute Error 0.0534063973
Root Mean Squared Error 0.24408269319884998
MSE: 0.0596
R2: 0.8570
'''

'''
Estimators = 50
Using Random Split for evaluating estimator performance
Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...
[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:  3.0min
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:  3.9min finished
TIME ELAPSED:  00:03:53
Predicting target with X_test (TEST SET)
[Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    7.2s
[Parallel(n_jobs=8)]: Done  50 out of  50 | elapsed:    9.5s finished
TIME ELAPSED:  00:00:10
Mean Absolute Error 0.0530685872765
Root Mean Squared Error 0.24284371232589078
MSE: 0.0590
R2: 0.8584
'''

'''
Using Random Split for evaluating estimator performance
Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...
[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:  2.3min
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:  6.0min finished
Time elapsed: 00:05:57 

Predicting target with X_test (TEST SET)
[Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    8.3s
[Parallel(n_jobs=8)]: Done 100 out of 100 | elapsed:   24.2s finished
Time elapsed: 00:00:25 

R^2 Score: 0.863203416737
Mean Squared Error: 0.0569872232321
Root Mean Squared Error: 0.2387199682308713
Mean Absolute Error: 0.0530372614282
'''
