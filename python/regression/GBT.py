from math import sqrt

import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import GradientBoostingRegressor
import python.Config as Config
import python.Timer as Timer
import python.Data as Data
import python.sampling.RandomSplit as RandomSplit
import matplotlib.pyplot as plot
import numpy as np

# Gradient Boosting Trees
# Can run with or without PCA (specify in global var)

# Global vars
time = Timer.Timer()
RUN_WITH_PCA = True


def main():
    if RUN_WITH_PCA:
        data, x = read_pca()
    else:
        data, x = read_normal(Config.TRIM_DATA_SET)

    # Run this function for each alpha
    run_gbt(data, x)


def read_normal(lines):
    chunks = Data.read_chunks('ColumnedDatasetNonNegativeWithDateImputerBinary.h5')

    # Generating X and y
    y = chunks['quantity_time_key']
    x = chunks.drop('quantity_time_key', 1)

    print('CHUNKS AFTER REMOVING:\n', x)

    return RandomSplit.get_sample(x.iloc[0:lines], y.iloc[0:lines]), x


def read_pca():
    df = Data.read_hdf('/PCAed50.h5')
    target = Data.read_hdf('/ColumnedDatasetNonNegativeWithDateImputer.h5')
    target = target['quantity_time_key']

    return RandomSplit.get_sample(df.iloc[0:500000], target.iloc[0:500000]), df


def run_gbt(data, x):
    train_set, test_set, target_train, target_test = data
    time.restart()

    print('Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...')
    params = {'n_estimators': 200, 'max_depth': 3,
              'learning_rate': 0.1, 'loss': 'huber', 'alpha': 0.95}
    clf = GradientBoostingRegressor(**params)
    clf.fit(train_set, target_train)
    print('TIME ELAPSED: ', time.get_time_hhmmss())

    time.restart()
    print('Predicting target with X_test (TEST SET)')
    y_prediction = clf.predict(test_set)
    print('TIME ELAPSED: ', time.get_time_hhmmss())

    mse = mean_squared_error(target_test, y_prediction)
    r2 = r2_score(target_test, y_prediction)

    print("MSE: %.4f" % mse)
    print("R2: %.4f" % r2)

    test_score = np.zeros((params['n_estimators'], ), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_predict(test_set)):
        test_score[i] = clf.loss_(target_test, y_pred)

    plot.figure(figsize=(100, 100))
    plot.subplot(1, 1, 1)
    plot.title('Deviance')
    plot.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-', label='Training Set Deviance')
    plot.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-', label='Test Set Deviance')
    plot.legend(loc='upper right')
    plot.xlabel('Boosting Iterations')
    plot.ylabel('Deviance')
    plot.show()


# Run script
main()
