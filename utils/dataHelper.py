import pandas as pd
import os
from langdetect import detect
from utils.configs.settings import PROJECT_DIRECTORY, NEWS_PATH

def load_news_data(from_datetime, to_datetime, languages=['en']):
    """Take in 2 UTC datetimes and return the news articles published within that time period
        Data is returned as a Pandas dataframe with languages matching the input specification
    """
    li = []
    all_files = os.listdir('%s/%s' % (PROJECT_DIRECTORY, NEWS_PATH))

    to_datetime_string = to_datetime.isoformat(timespec='seconds').replace(':','')
    from_datetime_string = from_datetime.isoformat(timespec='seconds').replace(':','')

    valid_files = list(filter(lambda file_name: file_name.split('.')[0] >= from_datetime_string and file_name.split('.')[0] < to_datetime_string, all_files))

    for file_name in valid_files:
        df = pd.read_csv('%s/%s/%s' % (PROJECT_DIRECTORY, NEWS_PATH, file_name), index_col=None, header=0)
        li.append(df)
    
    data = pd.concat(li, axis=0, ignore_index=True)
    data = data[~data['description'].isnull()]
    data = data.drop_duplicates('description')

    if len(languages) > 0:
        data = data[data['description'].map(detect).map(lambda x: x in languages) == True]
        data['language'] = data['description'].map(detect)

    return data