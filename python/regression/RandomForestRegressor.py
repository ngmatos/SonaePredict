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
    clf = RandomForestRegressor()
    clf.fit(train_set, target_train)
    print('TIME ELAPSED:', time.get_time_hhmmss())

    time.restart()
    print('Predicting target with X_test (TEST SET)')
    y_prediction = clf.predict(test_set)
    print('TIME ELAPSED:', time.get_time_hhmmss())

    print('RFR Score (R^2):', clf.score(test_set, target_test))
    print('Mean Squared Error:', mean_squared_error(target_test, y_prediction))
    print('Root Mean Squared Error:', sqrt(mean_squared_error(target_test, y_prediction)))
    print('Mean Absolute Error:', mean_absolute_error(target_test, y_prediction))


# Run script
main()
