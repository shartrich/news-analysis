import requests
from datetime import datetime
import pandas as pd
import time
from functools import reduce
import os
from dotenv import load_dotenv
load_dotenv()

current_directory = os.path.dirname(os.path.abspath(__file__))
news_directory = '/data/news'

KEY_1 = os.getenv('KEY_1')
KEY_2 = os.getenv('KEY_2')
KEY_3 = os.getenv('KEY_3')
KEY_4 = os.getenv('KEY_4')


def file_format_datetime(datetime_value):
    return datetime_value.isoformat(timespec='seconds').replace(':','')

def get_last_recorded_timestamp():
    past_files = os.listdir('%s/%s' % (current_directory, news_directory))
    if not past_files:
        return None
    iso_datetime_strings = list(map(lambda file_name: file_name.split('.')[0], past_files))

    return max(iso_datetime_strings)


def get_sources():
    source_url  =  'https://newsapi.org/v1/sources?language=en'
    response  =  requests.get(source_url).json()
    sources  =  []
    for source in response['sources']:
             sources.append(source['id'])
    return sources

def mapping():
    d = {}
    response = requests.get('https://newsapi.org/v1/sources?language=en')
    response = response.json()
    for s in response['sources']:
        d[s['id']] = s['category']
    return d

def category(source, m):
    try:
        return m[source]
    except:
        return 'NC'

def get_daily_news(key):
    to_datetime = datetime.now()
    from_datetime = get_last_recorded_timestamp()

    url = 'https://newsapi.org/v1/articles?pageSize=100&to=%s' % to_datetime.isoformat(timespec='seconds') + '&source={0}&sortBy={1}&apiKey={2}'
    if from_datetime:
        url += '&from=%s' % from_datetime

    responses = []
    sources = get_sources()
    for i, source in enumerate(sources):

        try:
            u = url.format(source, 'top', key)
        except:
            u = url.format(source, 'latest', key)

        response = requests.get(u)
        r = response.json()
        try:
            for article in r['articles']:
                article['source'] = source
            responses.append(r)
        except:
            print('Rate limit exceeded ... please wait and retry in 6 hours')
            return None

    articles = list(map(lambda r: r['articles'], responses))
    articles = list(reduce(lambda x,y: x+y, articles))

    news = pd.DataFrame(articles)
    news = news.dropna()
    news = news.drop_duplicates()
    news.reset_index(inplace=True, drop=True)
    d = mapping()
    
    news['category'] = news['source'].map(lambda s: category(s, d))
    news['scraping_date'] = to_datetime
    

    save_path = current_directory + '/data/news/%s.csv' % file_format_datetime(to_datetime)
    news.to_csv(save_path, index=False, encoding='utf-8')

    print('Done', datetime.now())

n = datetime.now()
h = n.hour % 4

if h == 1:
    get_daily_news(KEY_1)
elif h == 2:
    get_daily_news(KEY_2)
elif h == 3:
    get_daily_news(KEY_3)
else:
    get_daily_news(KEY_4)
