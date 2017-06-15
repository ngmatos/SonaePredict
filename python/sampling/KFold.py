# Using K-Fold Cross Validation for evaluating estimator performance
from sklearn.model_selection import KFold

# Global vars
K_PARTITIONS = 3


def get():
    print('Using K-Fold Cross Validation for evaluating estimator performance')

    # Run K-fold
    kf = KFold(n_splits=K_PARTITIONS)

    return kf
