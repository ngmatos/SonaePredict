# Transforming date into week, week_day, year
import os
from datetime import date, datetime
import math
import pandas as pd
import python.Config as Config
from python.Timer import Timer


df = pd.read_pickle(Config.H5_PATH + '/ColumnedDatasetNonNegative.pkl')

print('Transforming date into week, week_day, year')


def get_date(value, keyword):
    date_value = str(value)

    year = int(date_value[0:4])
    month = int(date_value[4:6])
    day = int(date_value[6:8])

    if keyword == 'year':
        return year
    if keyword == 'month':
        return month
    if keyword == 'day':
        return day
    if keyword == 'week':
        difference = datetime(year, month, day) - datetime(year, 1, 1)
        return math.floor(difference.days / 7.0)
    if keyword == 'week_day':
        return date(year, month, day).weekday()

time = Timer()
df.insert(3, 'week_day', df['time_key'].map(lambda value: get_date(value, keyword='week_day')))
df.insert(3, 'week', df['time_key'].map(lambda value: get_date(value, keyword='week')))
df.insert(3, 'year', df['time_key'].map(lambda value: get_date(value, keyword='year')))
print('TIME ELAPSED: ', time.get_time_hhmmss())

# Dropping time_key
df = df.drop('time_key', 1)

if not os.path.isfile(Config.H5_PATH + '/ColumnedDatasetNonNegativeWithDate.pkl'):
    time.restart()
    df.to_pickle(Config.H5_PATH + '/ColumnedDatasetNonNegativeWithDate.pkl')
    print('TIME ELAPSED SAVING: ', time.get_time_hhmmss())

print(df.head())

del df
del time
