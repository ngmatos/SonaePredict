import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = '/Users/pedro/Documents/Ficheiros/Sonae-ADES'
# FILE_PATH = '/Users/mercurius/GoogleDrive/FEUP/ADES/data'
H5_PATH = ROOT_PATH + '/h5'
# CHUNK_SIZE = 1024
TRIM_DATA_SET = 2000000
# TRIM_DATA_SET = 2500000
CHUNK_SIZE = 300000
PREDICTIVE_ATTRIBUTE = 'quantity_time_key'
