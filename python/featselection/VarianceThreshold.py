import python.Data as Data
from sklearn.feature_selection import VarianceThreshold

X = Data.read_chunks('ColumnedDatasetNonNegativeWithDateBinaryImputer.h5')

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X = sel.fit_transform(X)

print(X)