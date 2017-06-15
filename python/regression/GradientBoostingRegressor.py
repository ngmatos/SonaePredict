from math import sqrt

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import python.Config as Config
import python.Timer as Timer
import python.Data as Data
import python.sampling.RandomSplit as RandomSplit
import matplotlib.pyplot as plot
import numpy as np

# Gradient Boosting Trees

# Global vars
time = Timer.Timer()
params = {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1, 'loss': 'huber', 'alpha': 0.95}


def main():
    data, x = read_normal()
    # Run this function for each alpha
    run_gbt(data, x)


def read_normal():
    chunks = Data.read_chunks('ColumnedDatasetNonNegativeWithDateImputer.h5')

    # Generating X and y
    y = chunks['quantity_time_key']
    x = chunks.drop('quantity_time_key', 1)

    return RandomSplit.get_sample(x, y), x


def run_gbt(data, x):
    train_set, test_set, target_train, target_test = data
    time.restart()

    print('Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...')
    clf = GradientBoostingRegressor(**params)
    clf.fit(train_set, target_train)
    print('TIME ELAPSED:', time.get_time_hhmmss())

    time.restart()
    print('Predicting target with X_test (TEST SET)')
    y_prediction = clf.predict(test_set)
    print('TIME ELAPSED:', time.get_time_hhmmss())

    print('GBR Score (R^2):', clf.score(test_set, target_test))
    print('Mean Squared Error:', mean_squared_error(target_test, y_prediction))
    print('Root Mean Squared Error:', sqrt(mean_squared_error(target_test, y_prediction)))
    print('Mean Absolute Error:', mean_absolute_error(target_test, y_prediction))

    # Plotting
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