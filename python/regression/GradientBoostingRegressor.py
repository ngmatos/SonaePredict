from math import sqrt

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import python.Config as Config
import python.Timer as Timer
import python.Data as Data
from python.sampling import KFold, RandomSplit
import matplotlib.pyplot as plot
import numpy as np

# Gradient Boosting Trees

# Global vars
time = Timer.Timer()
params = {'n_estimators': 1000, 'max_depth': 3, 'learning_rate': 0.1, 'loss': 'huber', 'alpha': 0.95, 'verbose': 1}
K_PARTITIONS = 3
K_FOLD = False


def main():
    data = read_normal()
    # Run this function for each alpha
    run_gbt(data)


def read_normal():
    chunks = Data.read_chunks('/ColumnedDatasetNonNegativeWithDateImputer.h5')

    # Generating X and y
    y = chunks['quantity_time_key']
    x = chunks.drop('quantity_time_key', 1)

    return x.iloc[0:100000], y.iloc[0:100000]


def run_gbt(data):
    clf = GradientBoostingRegressor(**params)
    x, y = data

    time.restart()
    cv = KFold.get(K_PARTITIONS)

    print('Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...')
    if K_FOLD:
        scores, mse, mae, y_prediction = Data.cross_val_execute(clf, x, y, cv, n_jobs=-1)
        Data.print_scores(np.mean(scores), np.mean(mse), np.mean(mae))
        time.print()
        plot_x = y
        plot_y = y_prediction
    else:
        train_set, test_set, target_train, target_test = RandomSplit.get_sample(x, y)
        clf.fit(train_set, target_train)

        # print('Saving model')
        # filename = 'GBTModel.pkl'
        # pickle.dump(clf, open(filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        # joblib.dump(clf, filename, 5, pickle.HIGHEST_PROTOCOL)
        # print('TIME SPENT: ', time.get_time_hhmmss())

        time.print()
        time.restart()
        print('Predicting target with X_test (TEST SET)')
        y_prediction = clf.predict(test_set)
        time.print()
        Data.calc_scores(target_test, y_prediction)

        # Plotting Deviance
        test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
        for i, y_pred in enumerate(clf.staged_predict(test_set)):
            test_score[i] = clf.loss_(target_test, y_pred)

        plot.figure(figsize=(8, 6))
        plot.subplot(1, 1, 1)
        plot.title('Deviance')
        plot.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-', label='Training Set Deviance')
        plot.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-', label='Test Set Deviance')
        plot.legend(loc='upper right')
        plot.xlabel('Boosting Iterations')
        plot.ylabel('Deviance')
        plot.show(block=False)

        plot_x = target_test
        plot_y = y_prediction

    # Plotting Results
    fig, ax = plot.subplots()
    ax.scatter(plot_x, plot_y)
    ax.plot([plot_x.min(), plot_x.max()], [plot_x.min(), plot_x.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plot.show()

    # Feature Importance
    feature_importance(clf, x)


def feature_importance(clf, X):
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plot.subplot(1, 2, 2)
    plot.barh(pos, feature_importance[sorted_idx], align='center')
    plot.yticks(pos, X.columns[sorted_idx])
    plot.xlabel('Relative Importance')
    plot.title('Variable Importance')
    plot.show()

# Run script
main()

'''
Reading /PCAed50.h5 file
TIME ELAPSED:  00:00:03
Reading /ColumnedDatasetNonNegativeWithDateImputer.h5 file
TIME ELAPSED:  00:00:02
Using Random Split for evaluating estimator performance
Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...
TIME ELAPSED:  01:11:54
Predicting target with X_test (TEST SET)
TIME ELAPSED:  00:00:03
MSE: 0.1002
R2: 0.7595
'''

'''
[4546011 rows x 13 columns]
350 estimators
Using Random Split for evaluating estimator performance
Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...
TIME ELAPSED:  00:31:33
Predicting target with X_test (TEST SET)
TIME ELAPSED:  00:00:06
Mean Absolute Error 0.0621326114504
Root Mean Squared Error 0.29159367020577187
MSE: 0.0850
R2: 0.7959
'''

'''
Estimators = 100
Using K-Fold Cross Validation for evaluating estimator performance
Fitting model with X_train (TRAIN SET) and y_train (TARGET TRAIN SET)...
R^2 Score: 0.735274230117
Mean Squared Error: 0.127313089077
Root Mean Squared Error: 0.35680959779238075
Mean Absolute Error: 0.0673262423782
Time elapsed: 00:09:56 
'''