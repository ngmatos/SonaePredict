# Transforming date into week, week_day, year
import os
from datetime import date, datetime
import math
import pandas as pd

df = pd.read_pickle('h5/ColumnedDatasetNonNegative.pkl')

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


df.insert(3, 'week_day', df['time_key'].map(lambda value: get_date(value, keyword='week_day')))
df.insert(3, 'week', df['time_key'].map(lambda value: get_date(value, keyword='week')))
df.insert(3, 'year', df['time_key'].map(lambda value: get_date(value, keyword='year')))

if not os.path.isfile('h5/ColumnedDatasetNonNegativeWithDate.pkl'):
    df.to_pickle('h5/ColumnedDatasetNonNegativeWithDate.pkl')

# df = pd.read_pickle('h5/ColumnedDatasetNonNegativeWithDate.pkl')
print(df.head())

del df
