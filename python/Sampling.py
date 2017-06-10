# Using K-Fold Cross Validation for evaluating estimator performance
# from IPython.display import display
import pandas as pd
from sklearn.model_selection import KFold

'''
def to_array(array, indexes):
    result = []
    for i in indexes:
        result.append(array[i])
    return result
'''

CHUNK_SIZE = 2048
K_PARTITIONS = 2

print('Using K-Fold Cross Validation for evaluating estimator performance')
iter_hdf = pd.read_hdf('h5/ColumnedDatasetNonNegativeWithDateBinary.h5', chunksize=CHUNK_SIZE)

chunks = []
count = 0

for chunk in iter_hdf:
    count += 1
    print('Joining at chunk', count)
    chunks.append(chunk)

print('Applying k-fold cross validation')
kf = KFold(n_splits=K_PARTITIONS)
chunks = pd.concat(chunks)
for train, test in kf.split(chunks):
    print(train, test)
