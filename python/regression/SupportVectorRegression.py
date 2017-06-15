from math import sqrt

import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
import numpy as np
import python.Config as Config
import python.Timer as Timer
import python.Data as Data
import python.sampling.RandomSplit as RandomSplit
import matplotlib.pyplot as plot
from sklearn import datasets

# Support Vector Regression
# Can run with or without PCA (specify in global var)

# Global vars
time = Timer.Timer()
RUN_WITH_PCA = False


def main():
    if RUN_WITH_PCA:
        data = read_pca()
    else:
        data = read_normal(Config.TRIM_DATA_SET)

    run_svr(data)


def read_normal(lines):
    chunks = Data.read_chunks('ColumnedDatasetNonNegativeWithDateImputer.h5')

    # Generating X and y
    y = chunks['quantity_time_key']
    x = chunks.drop('quantity_time_key', 1)

    return RandomSplit.get_sample(x.iloc[0:lines], y.iloc[0:lines])


def read_pca():
    df = Data.read_hdf('PCAed.h5')

    target = Data.read_hdf('ColumnedDatasetNonNegativeWithDateImputer.h5')
    target = target['quantity_time_key']

    return RandomSplit.get_sample(df, target)


def run_svr(data):
    svr = SVR()
    train_set, test_set, target_train, target_test = data
    time.restart()

    print('Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...')
    svr.fit(train_set, target_train)
    time.print()

    time.restart()
    print('Predicting target with X_test (TEST SET)')
    y_prediction = svr.predict(X=test_set)
    time.print()

    Data.calc_scores(target_test, y_prediction)


# Run script
main()
