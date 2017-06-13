# Using K-Fold Cross Validation for evaluating estimator performance
# from IPython.display import display
from sklearn.model_selection import KFold

# Global vars
K_PARTITIONS = 2


def get_sample(data):
    print('Using K-Fold Cross Validation for evaluating estimator performance')

    # Run K-fold
    print('Applying k-fold cross validation')
    kf = KFold(n_splits=K_PARTITIONS)

    return kf.split(data)

    # y = target, X = data
    # for train, test in kf.split(data):
    #    X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
