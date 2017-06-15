from math import sqrt

import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Lasso
import numpy as np
import python.Config as Config
import python.Timer as Timer
import python.Data as Data
import python.sampling.RandomSplit as RandomSplit
import matplotlib.pyplot as plot
from sklearn import datasets

# Lasso Regression
# Can run with or without PCA (specify in global var)

# Global vars
time = Timer.Timer()
RUN_WITH_PCA = False


def main():
    if RUN_WITH_PCA:
        data = read_pca()
    else:
        data = read_normal(Config.TRIM_DATA_SET)

    # Run this function for each alpha
    for alpha in np.logspace(-5, 0.1, 3):
        run_lasso(alpha, data)


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


def run_lasso(alpha, data):
    lasso = Lasso(alpha=alpha)
    train_set, test_set, target_train, target_test = data
    time.restart()

    print('Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...')
    lasso.fit(train_set, target_train)
    time.print()

    time.restart()
    print('Predicting target with X_test (TEST SET)')
    y_prediction = lasso.predict(X=test_set)
    time.print()

    # Constructing a data frame for visualisation purposes
    df = pd.DataFrame(data=y_prediction, columns=['TargetPrediction(yPred)'], index=test_set.index)
    df['TargetTest(yTest)'] = pd.Series(target_test)

    Data.print_scores(target_test, y_prediction)

    # Plotting
    # target_plot = [val for val in target_test]

    # print(len(target_plot))
    # print(len(y_prediction))

    # plot.figure()
    # plot.scatter(target_plot, y_prediction, c="darkorange", label="data")
    # plot.xlabel("data")
    # plot.ylabel("target")
    # plot.title("Lasso Data")
    # plot.legend()
    # plot.show()


# Run script
main()
