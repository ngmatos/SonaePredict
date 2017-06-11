import python.Data as Data
import numpy as np
from sklearn.decomposition import IncrementalPCA

reader = Data.chunk_reader('ColumnedDatasetNonNegativeWithDateBinaryImputer.h5')

dimensions = 100

pca = IncrementalPCA(n_components=dimensions)

count = 0
for chunk in reader:
    count += 1
    print('Chunk', count)
    Data.drop_attributes(chunk)
    pca.partial_fit(chunk)

# Computed mean per feature
mean = pca.mean_
# and stddev
stddev = np.sqrt(pca.var_)

print(pca)