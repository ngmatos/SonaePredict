import pandas as pd
import python.Config as Config
import python.Timer as Timer


def read_chunks(file):
    time = Timer.Timer()
    print('Reading chunks from', file)

    iter_hdf = pd.read_hdf(Config.H5_PATH + '/' + file, chunksize=Config.CHUNK_SIZE)
    chunks = []
    count = 0

    # Read by chunks and join them
    for chunk in iter_hdf:
        count += 1
        print('Reading at chunk', count)
        chunks.append(chunk)

    print('Concatenating chunks')
    chunks = pd.concat(chunks)
    # Drop values
    # drop_attributes(chunks)
    print('ALL CHUNKS:\n', chunks)
    print('TIME ELAPSED: ', time.get_time_hhmmss())
    del time
    return chunks


def chunk_reader(file):
    iter_hdf = pd.read_hdf(Config.H5_PATH + '/' + file, chunksize=Config.CHUNK_SIZE)
    return iter_hdf


def read_hdf(file):
    time = Timer.Timer()
    print('Reading', file, 'file')
    df = pd.read_hdf(Config.H5_PATH + file)
    print('TIME ELAPSED: ', time.get_time_hhmmss())
    del time
    return df


def drop_attributes(chunk):
    chunk.drop('quantity_time_key', 1)
    chunk.drop('promotion', 1)
    chunk.drop('year', 1)
