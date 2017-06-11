# Using K-Fold Cross Validation for evaluating estimator performance
# from IPython.display import display
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
import numpy as np

# Global vars
CHUNK_SIZE = 2048
K_PARTITIONS = 2

print('Using K-Fold Cross Validation for evaluating estimator performance')
iter_hdf = pd.read_hdf('h5/ColumnedDatasetNonNegativeWithDateBinary.h5', chunksize=CHUNK_SIZE)

chunks = []
count = 0

# Read by chunks and join them
for chunk in iter_hdf:
    count += 1
    print('Joining at chunk', count)
    chunks.append(chunk)

# Run K-fold
print('Applying k-fold cross validation')
kf = KFold(n_splits=K_PARTITIONS)
chunks = pd.concat(chunks)

# Generating X and y
y = chunks['quantity_time_key']
X = chunks.drop('quantity_time_key', 1)

# Run Lasso
for train, test in kf.split(chunks):
    lasso = Lasso(alpha=0.1)
    print('Já fiz um lassito')
    X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
    print('Já transformei a chossa')
    lasso.fit(X_train, y_train)
    print('Já tou fitness')
    print(lasso.score(X_test, y_test))
    break
