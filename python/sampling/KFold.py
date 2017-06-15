# Using K-Fold Cross Validation for evaluating estimator performance
from sklearn.model_selection import KFold


def get(partitions):
    print('Using K-Fold Cross Validation for evaluating estimator performance')

    # Run K-fold
    kf = KFold(n_splits=partitions)

    return kf
