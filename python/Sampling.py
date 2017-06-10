# Using K-Fold Cross Validation for evaluating estimator performance
import pandas as pd
from sklearn.model_selection import KFold

'''
def to_array(array, indexes):
    result = []
    for i in indexes:
        result.append(array[i])
    return result
'''

print('Using K-Fold Cross Validation for evaluating estimator performance')

df = pd.read_hdf('h5/ColumnedDatasetNonNegativeWithDateBinary.h5')

print('FINISH READING FILE')

kf = KFold(n_splits=2)

for train, test in kf.split(df):
    print("%s %s" % (train, test))
    # for t1 in train:
    #    print('TRAIN SET: ', df[t1])
