import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plot
import numpy as np
import python.Data as Data


def main():
    data = read_normal()

    feature_importance(data)


def read_normal():
    chunks = Data.read_chunks('/ColumnedDatasetNonNegativeWithDateImputer.h5')

    # Generating X and y
    y = chunks['quantity_time_key']
    x = chunks.drop('quantity_time_key', 1)

    return x, y


def feature_importance(data):
    X, y = data

    # Build a forest and compute the feature importances
    forest = ExtraTreesRegressor(n_estimators=10, random_state=0, verbose=1, n_jobs=-1)

    print("Fitting ExtraTreesRegressor")
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plot.figure()
    plot.title("Feature importances")
    plot.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plot.xticks(range(X.shape[1]), indices)
    plot.xlim([-1, X.shape[1]])
    plot.show()


# Run script
main()