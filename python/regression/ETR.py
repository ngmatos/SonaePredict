from math import sqrt

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error,r2_score
import python.Config as Config
import python.Timer as Timer
import python.Data as Data
import python.sampling.RandomSplit as RandomSplit
import matplotlib.pyplot as plot
import numpy as np

# Extra Trees Regressor
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
    run_etr(data, x)


def read_normal(lines):
    chunks = Data.read_chunks('ColumnedDatasetNonNegativeWithDateImputerBinary.h5')

    # Generating X and y
    y = chunks['quantity_time_key']
    x = chunks.drop('quantity_time_key', 1)

    print('CHUNKS AFTER REMOVING:\n', x)

    return RandomSplit.get_sample(x.iloc[0:lines], y.iloc[0:lines]), x


def read_pca():
    df = Data.read_hdf('/PCAed.h5')
    target = Data.read_hdf('/ColumnedDatasetNonNegativeWithDateImputer.h5')
    target = target['quantity_time_key']

    return RandomSplit.get_sample(df, target), df


def run_etr(data, x):
    train_set, test_set, target_train, target_test = data
    time.restart()

    print('Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...')
    clf = ExtraTreesRegressor()
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


# Run script
main()
