from math import sqrt

import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import python.Config as Config
import python.Timer as Timer
import python.Data as Data
import python.sampling.RandomSplit as RandomSplit
import matplotlib.pyplot as plot
import numpy as np

# Ada Boosting Trees

# Global vars
time = Timer.Timer()


def main():
    data, x = read_normal()
    # Run this function for each alpha
    run_etr(data, x)


def read_normal():
    chunks = Data.read_chunks('ColumnedDatasetNonNegativeWithDateImputer.h5')

    # Generating X and y
    y = chunks['quantity_time_key']
    x = chunks.drop('quantity_time_key', 1)

    return RandomSplit.get_sample(x, y), x


def run_etr(data, x):
    train_set, test_set, target_train, target_test = data
    time.restart()

    print('Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...')
    clf = AdaBoostRegressor()
    clf.fit(train_set, target_train)
    time.print()

    time.restart()
    print('Predicting target with X_test (TEST SET)')
    y_prediction = clf.predict(test_set)
    time.print()

    Data.calc_scores(target_test, y_prediction)


# Run script
main()
