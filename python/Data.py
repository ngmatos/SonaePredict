import pandas as pd
import python.Config as Config


def read_chunks(file):
    iter_hdf = pd.read_hdf(Config.H5_PATH + '/' + file, chunksize=Config.CHUNK_SIZE)
    chunks = []
    count = 0

    # Read by chunks and join them
    for chunk in iter_hdf:
        count += 1
        print('Reading at chunk', count)
        chunks.append(chunk)

    chunks = pd.concat(chunks)
    # Drop values
    # drop_attributes(chunks)
    return chunks


def chunk_reader(file):
    iter_hdf = pd.read_hdf(Config.H5_PATH + '/' + file, chunksize=Config.CHUNK_SIZE)
    return iter_hdf


def drop_attributes(chunk):
    chunk.drop('quantity_time_key', 1)
    chunk.drop('promotion', 1)
    chunk.drop('year', 1)
