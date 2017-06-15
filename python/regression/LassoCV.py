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
K_PARTITIONS = 3


def main():
    if RUN_WITH_PCA:
        data = read_pca()
    else:
        data = read_normal(Config.TRIM_DATA_SET)

    alphas = np.logspace(-5, -0.5, 5)
    scores = list()
    scores_std = list()

    # Run this function for each alpha
    for alpha in alphas:
        mean, std = run_lasso(alpha, data)
        scores.append(mean)
        scores_std.append(std)

    # Plot alphas
    # plot_alphas(alphas, scores, scores_std)


def read_normal(lines):
    chunks = Data.read_chunks('ColumnedDatasetNonNegativeWithDateImputerBinary.h5')

    # Generating X and y
    y = chunks['quantity_time_key']
    x = chunks.drop('quantity_time_key', 1)

    return x.iloc[0:lines], y.iloc[0:lines]


def read_pca():
    df = Data.read_hdf('PCAed.h5')

    target = Data.read_hdf('ColumnedDatasetNonNegativeWithDateImputer.h5')
    target = target['quantity_time_key']

    return df, target


def run_lasso(alpha, data):
    lasso = Lasso(alpha=alpha)
    x, y = data

    time.restart()
    cv = KFold.get(K_PARTITIONS)

    print('Fitting model with X_train and y_train...')
    scores, mse, mae, y_prediction = Data.cross_val_execute(lasso, x, y, cv, n_jobs=-1)
    Data.print_scores(np.mean(scores), np.mean(mse), np.mean(mae))
    r2 = np.mean(scores)
    std = np.std(scores)

    time.print()

    # Plotting
    fig, ax = plot.subplots()
    ax.scatter(y, y_prediction)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plot.show(block=False)

    # Return scores
    return r2, std


def plot_alphas(alphas, scores, scores_std):
    scores, scores_std = np.array(scores), np.array(scores_std)

    plot.figure().set_size_inches(10, 8)
    plot.semilogx(alphas, scores)
    std_error = scores_std / np.sqrt(K_PARTITIONS)
    plot.semilogx(alphas, scores + std_error, 'b--')
    plot.semilogx(alphas, scores - std_error, 'b--')

    # alpha=0.2 controls the translucency of the fill color
    plot.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

    plot.ylabel('CV score +/- std error')
    plot.xlabel('alpha')
    plot.axhline(np.max(scores), linestyle='--', color='.5')
    plot.xlim([alphas[0], alphas[-1]])

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
