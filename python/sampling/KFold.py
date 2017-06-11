# Using K-Fold Cross Validation for evaluating estimator performance
# from IPython.display import display
import pandas as pd
from sklearn.model_selection import KFold
import python.Config as Config
import python.Data as Data

# Global vars
K_PARTITIONS = 2

print('Using K-Fold Cross Validation for evaluating estimator performance')
chunks = Data.read_chunks('ColumnedDatasetNonNegativeWithDateBinaryImputer.h5')

# Run K-fold
print('Applying k-fold cross validation')
kf = KFold(n_splits=K_PARTITIONS)

# Generating X and y
y = chunks['quantity_time_key']
X = chunks
# X = chunks.drop('quantity_time_key', 1)

# promotion = chunks['promotion']

# X = X.drop('promotion', 1)
# X = X.drop('year', 1)

for train, test in kf.split(chunks):
    X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
