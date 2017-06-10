# Using K-Fold Cross Validation for evaluating estimator performance
import time
import pandas as pd
from sklearn.model_selection import KFold

'''
def to_array(array, indexes):
    result = []
    for i in indexes:
        result.append(array[i])
    return result
'''

CHUNK_SIZE = 1024
K_PARTITIONS = 10

print('Using K-Fold Cross Validation for evaluating estimator performance')

iter_hdf = pd.read_hdf('h5/ColumnedDatasetNonNegativeWithDateBinary.h5', chunksize=CHUNK_SIZE)
count = 0

print('Each chunk size is', CHUNK_SIZE)
print('Starting partition in 3 seconds...')

time.sleep(3)

for chunk in iter_hdf:
    kf = KFold(n_splits=K_PARTITIONS)
    count += 1
    print('Partitioning chunk ', count)
    for train, test in kf.split(chunk):
        print(train, test)
