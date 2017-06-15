import sys
import pandas as pd
import python.Config as Config
import python.Timer as Timer
import numpy as np
from time import sleep
from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def read_chunks(file):
    time = Timer.Timer()
    print('Reading chunks from', file, '...')

    iter_hdf = pd.read_hdf(Config.H5_PATH + '/' + file, chunksize=Config.CHUNK_SIZE)
    rows = iter_hdf.nrows
    chunk_amount = int(rows / Config.CHUNK_SIZE)
    chunks = []
    percentage = 0

    # Read by chunks and join them
    for i, chunk in enumerate(iter_hdf):
        percentage += np.asscalar(100 * Config.CHUNK_SIZE / rows)
        chunks.append(chunk)
        if i % int(chunk_amount / 5) == 0:
            sys.stdout.write('%d%% ' % percentage)
            sys.stdout.flush()

    print('All chunks read. Joining results...')
    chunks = pd.concat(chunks)
    time.print()
    del time
    return chunks


def chunk_reader(file):
    iter_hdf = pd.read_hdf(Config.H5_PATH + '/' + file, chunksize=Config.CHUNK_SIZE)
    return iter_hdf


def read_hdf(file):
    time = Timer.Timer()
    print('Reading', file, 'file...')
    df = pd.read_hdf(Config.H5_PATH + file)
    time.print()
    del time
    return df


def print_scores(target_test, y_prediction):
    print('R^2 Score:', r2_score(target_test, y_prediction))
    print('Mean Squared Error:', mean_squared_error(target_test, y_prediction))
    print('Root Mean Squared Error:', sqrt(mean_squared_error(target_test, y_prediction)))
    print('Mean Absolute Error:', mean_absolute_error(target_test, y_prediction))
