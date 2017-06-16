from math import sqrt

import pandas as pd
from sklearn.model_selection import cross_val_predict, cross_val_score
import sklearn.metrics as metrics
from sklearn.linear_model import Lasso
import numpy as np
import python.Config as Config
import python.Timer as Timer
import python.Data as Data
import python.sampling.KFold as KFold
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
    # for alpha in np.logspace(-5, 0.1, 3):
    run_lasso(0.1, data)


def read_normal(lines):
    chunks = Data.read_chunks('ColumnedDatasetNonNegativeWithDateImputer.h5')

    # Generating X and y
    y = chunks['quantity_time_key']
    x = chunks.drop('quantity_time_key', 1)

    return x.iloc[0:lines], y.iloc[0:lines]


def read_pca():
    df = Data.read_hdf('PCAed50.h5')

    target = Data.read_hdf('ColumnedDatasetNonNegativeWithDateImputer.h5')
    target = target['quantity_time_key']

    return df, target


def run_lasso(alpha, data):
    lasso = Lasso(alpha=alpha)
    x, y = data

    time.restart()

    for each in y:
        print(each)

    print(y.max())

    print('Fitting model with X_train and y_train...')
    train_set, test_set, target_train, target_test = RandomSplit.get_sample(x, y)
    lasso.fit(train_set, target_train)

    # print('Saving model')
    # filename = 'LASSOModel.pkl'
    # pickle.dump(lasso, open(filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    # joblib.dump(lasso, filename, 5, pickle.HIGHEST_PROTOCOL)
    # print('TIME SPENT: ', time.get_time_hhmmss())

    y_prediction = lasso.predict(X=test_set)
    Data.calc_scores(target_test, y_prediction)

    time.print()

    # Plotting
    fig, ax = plot.subplots()
    ax.scatter(target_test, y_prediction)
    ax.plot([target_test.min(), target_test.max()], [target_test.min(), target_test.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plot.show()


# Run script
main()

'''
Full data frame description:
        TargetPrediction(yPred)  TargetTest(yTest)
count             1.363804e+06       1.363804e+06
mean              1.070951e-01       1.064061e-01
std               5.712002e-01       6.454332e-01
min              -7.745373e-03       0.000000e+00
25%               9.026660e-03       0.000000e+00
50%               1.965499e-02       0.000000e+00
75%               5.194681e-02       3.333000e-02
max               4.920873e+01       6.728900e+01
Lasso Score (R^2):  0.749612538813
Mean Squared Error 0.104307328479
Root Mean Squared Error 0.3229664510120647
Mean Absolute Error 0.0734283518468
'''

'''
No PCA 2M rows Binary
Fitting model with X_train and y_train...
Using Random Split for evaluating estimator performance
R^2 Score: 0.778697954815
Mean Squared Error: 0.168339913564
Root Mean Squared Error: 0.41029247319879236
Mean Absolute Error: 0.0901767197459
Time elapsed: 00:02:31 
'''
