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

TEST_RUN = os.getenv('TEST_RUN')
KEYS = []
KEYS.append(os.getenv('KEY_0'))
KEYS.append(os.getenv('KEY_1'))
KEYS.append(os.getenv('KEY_2'))
KEYS.append(os.getenv('KEY_3'))
KEYS.append(os.getenv('KEY_4'))
KEYS.append(os.getenv('KEY_5'))

SOURCE_CHUNK_SIZE = int(os.getenv('SOURCE_CHUNK_SIZE'), '19')

def chunkify_array(array, chunk_size):
    """Yield successive n-sized chunks from arr."""
    arr_chunks = []
    for i in range(0, len(array), chunk_size):
        arr_chunks.append(array[i:i + chunk_size]) 
    return arr_chunks


def file_format_datetime(datetime_value):
    return datetime_value.isoformat(timespec='seconds').replace(':','')

def get_last_recorded_timestamp():
    past_files = os.listdir('%s/%s' % (current_directory, news_directory))
    if not past_files:
        return None
    iso_datetime_strings = list(map(lambda file_name: file_name.split('.')[0], past_files))
    return max(iso_datetime_strings)


def get_sources():
    source_url = 'https://newsapi.org/v1/sources?language=en'
    response = requests.get(source_url).json()
    sources = []
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

def request_articles(url):
    response = requests.get(url)
    r = response.json()
    try:
        for article in r['articles']:
            article['source'] = article['source']['id']
        return r
    except:
        print('API Error...')
        print(r)
        return None


def get_daily_news(key):
    # to_datetime = datetime.now()
    to_datetime = datetime.utcnow()
    from_datetime = datetime.strptime(get_last_recorded_timestamp(), '%Y-%m-%dT%H%M%S').strftime('%Y-%m-%dT%H:%M:%S')

    sources = get_sources()
    sources_in_chunks = chunkify_array(sources, SOURCE_CHUNK_SIZE)
    sources_strings = list(map(lambda x: ','.join(x), sources_in_chunks))
    print(from_datetime, to_datetime)
    # return
    
    base_url = 'https://newsapi.org/v2/everything?pageSize=100&to=%s' % to_datetime.isoformat(timespec='seconds') + '&sortBy={0}&apiKey={1}'
    url = base_url.format('publishedAt', key)
    if from_datetime:
        url += '&from=%s' % from_datetime

    responses = []
    for source_string in sources_strings:
        sources_url = url + '&sources=%s' % source_string

        response = requests.get(sources_url)
        r = response.json()
        try:
            for article in r['articles']:
                article['source'] = article['source']['id']
            responses.append(r)

            print(source_string)
            print(r['totalResults'])

            # TODO free api version caps results to 100, no pagination allowed
            # if r['totalResults'] > 100:
            #     for page_number in range(1, 2 + r['totalResults'] // 100 ):
            #         responses.append(request_articles(sources_url + '&page=%s' % page_number))
        except:
            print('API Error...')
            print(r)
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
    
    if len(news.index) > 0:
        max_article_published_at = news['publishedAt'].max()
        max_datetime = datetime.strptime(max_article_published_at, '%Y-%m-%dT%H:%M:%SZ')
        save_path = current_directory + '/data/news/%s.csv' % file_format_datetime(max_datetime)
        news.to_csv(save_path, index=False, encoding='utf-8')

    print('Done', datetime.now(), len(news.index))


if __name__ == "__main__":
    n = datetime.now()
    h = n.hour % 5

    if TEST_RUN:
        print('KEY_5')
        get_daily_news(KEYS[5])
    else:
        print('KEY_%s' % h)
        get_daily_news(KEYS[h])
