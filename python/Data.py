import sys
import pandas as pd
import python.Config as Config
import python.Timer as Timer
import numpy as np
from time import sleep
from math import sqrt
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, r2_score


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
    df = pd.read_hdf(Config.H5_PATH + '/' + file)
    time.print()
    del time
    return df


def calc_scores(target_test, y_prediction):
    print_scores(r2_score(target_test, y_prediction), mean_squared_error(target_test, y_prediction),
                 mean_absolute_error(target_test, y_prediction))


def print_scores(r2, mse, mae):
    print('R^2 Score:', r2)
    print('Mean Squared Error:', mse)
    print('Root Mean Squared Error:', sqrt(mse))
    print('Mean Absolute Error:', mae)


def cross_val_execute(alg, x, y, cv, fit_params=None, n_jobs=1):
    parallel = Parallel(n_jobs=n_jobs, verbose=0, pre_dispatch='2*n_jobs')
    results = parallel(delayed(fit_predict)(alg, x, y, train, test, fit_params)
                 for train, test in list(cv.split(x, y)))

    scores = [key[0] for (key, val) in results]
    mse = [key[1] for (key, val) in results]
    mae = [key[2] for (key, val) in results]
    predictions = [val for (key, val) in results]

    return scores, mse, mae, np.concatenate(predictions)


def fit_predict(alg, x, y, train, test, fit_params):
    fit_params = fit_params if fit_params is not None else {}
    x_train, x_test, y_train, y_test = x.iloc[train], x.iloc[test], y.iloc[train], y.iloc[test]

    alg.fit(x_train, y_train, **fit_params)
    y_predict = alg.predict(X=x_test)

    return [alg.score(x_test, y_test), mean_squared_error(y_test, y_predict),
            mean_absolute_error(y_test, y_predict)], y_predict
