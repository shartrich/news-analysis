import pandas as pd
import os
from datetime import datetime, timedelta


def file_format_datetime(datetime_value):
    return datetime_value.isoformat(timespec='seconds').replace(':','')


days_ago_start = 14

current_datetime = datetime.now()
current_datetime_string = current_datetime.isoformat(timespec='seconds')
from_datetime = current_datetime - timedelta(days=days_ago_start)
from_datetime_string = from_datetime.isoformat(timespec='seconds')

data = pd.read_csv('./data/news.csv')

for days_ago in range(days_ago_start-1, 0, -1):
    from_datetime = current_datetime - timedelta(days=days_ago)
    from_datetime_string = from_datetime.isoformat(timespec='seconds')
    to_datetime = from_datetime + timedelta(days=1)
    to_datetime_string = to_datetime.isoformat(timespec='seconds') 

    # print(from_datetime_string, '-', to_datetime_string)

    df = data[(data['publishedAt'] >= from_datetime_string) & (data['publishedAt'] <= to_datetime_string) & (data['publishedAt'].str.slice(0, 4) == '2020')]

    save_path = './data/news/%s.csv' % file_format_datetime(to_datetime)
    df.to_csv(save_path, index=False, encoding='utf-8')

