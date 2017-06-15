from math import sqrt

import pandas as pd
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import python.Config as Config
import python.Timer as Timer
import python.Data as Data
from python.sampling import RandomSplit, KFold
import matplotlib.pyplot as plot
import numpy as np
import pickle

# Random Forest Regression

# Global vars
time = Timer.Timer()
params = {'verbose': 1, 'n_jobs': -1, 'n_estimators': 50}
K_FOLD = True
K_PARTITIONS = 3


def main():
    data = read_normal()

    # Run this function for each alpha
    run_rfr(data)


def read_normal():
    chunks = Data.read_chunks('ColumnedDatasetNonNegativeWithDateImputer.h5')

    # Generating X and y
    y = chunks['quantity_time_key']
    x = chunks.drop('quantity_time_key', 1)

    return x, y


def run_rfr(data):
    clf = RandomForestRegressor(**params)
    x, y = data

    time.restart()
    cv = KFold.get(K_PARTITIONS)
    if K_FOLD:
        scores, mse, mae, y_prediction = Data.cross_val_execute(clf, x, y, cv, n_jobs=-1)
        Data.print_scores(np.mean(scores), np.mean(mse), np.mean(mae))
        time.print()

        plot_x = y
        plot_y = y_prediction
    else:
        train_set, test_set, target_train, target_test = data

        print('Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...')
        clf.fit(train_set, target_train)
        time.print()
        time.restart()
        print('Predicting target with X_test (TEST SET)')
        y_prediction = clf.predict(test_set)
        time.print()

        Data.calc_scores(target_test, y_prediction)

        print('RFR Score (R^2):', clf.score(test_set, target_test))

        plot_x = target_test
        plot_y = y_prediction

    # Plotting Results
    fig, ax = plot.subplots()
    ax.scatter(plot_x, plot_y)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plot.show()

    print('Saving model')
    filename = 'RFRModel.pkl'
    # pickle.dump(clf, open(filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    joblib.dump(clf, filename, 5, pickle.HIGHEST_PROTOCOL)
    print('TIME SPENT: ', time.get_time_hhmmss())


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

'''
Estimators = 100
Using Random Split for evaluating estimator performance
Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...
[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:  4.4min
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed: 11.7min finished
TIME ELAPSED: 00:11:40
Saving model
TIME SPENT:  00:07:25
Predicting target with X_test (TEST SET)
[Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    6.4s
[Parallel(n_jobs=8)]: Done 100 out of 100 | elapsed:   19.0s finished
TIME ELAPSED: 00:00:20
[Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    6.5s
[Parallel(n_jobs=8)]: Done 100 out of 100 | elapsed:   16.6s finished
RFR Score (R^2): 0.865608822831
Mean Squared Error: 0.0559851703244
Root Mean Squared Error: 0.2366118558407467
Mean Absolute Error: 0.0512401905142
'''

'''
Using Random Split for evaluating estimator performance
Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...
[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:  4.5min
[Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed: 22.9min
[Parallel(n_jobs=-1)]: Done 250 out of 250 | elapsed: 30.8min finished
TIME ELAPSED: 00:30:46
Saving model
TIME SPENT:  00:22:31
Predicting target with X_test (TEST SET)
[Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:   18.5s
[Parallel(n_jobs=8)]: Done 184 tasks      | elapsed:  1.1min
[Parallel(n_jobs=8)]: Done 250 out of 250 | elapsed:  2.4min finished
TIME ELAPSED: 00:02:58
[Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:   13.8s
[Parallel(n_jobs=8)]: Done 184 tasks      | elapsed:   53.4s
[Parallel(n_jobs=8)]: Done 250 out of 250 | elapsed:  3.0min finished
RFR Score (R^2): 0.866474634714
Mean Squared Error: 0.0556244872293
Root Mean Squared Error: 0.23584844122718152
Mean Absolute Error: 0.0510577645485

'''