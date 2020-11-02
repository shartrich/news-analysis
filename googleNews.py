import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
import re
import csv
from datetime import datetime, timedelta
import os
from utils.configs.constants import GOOGLE_NEWS_DIRECTORY, GOOGLE_NEWS_BASE_URL, GOOGLE_NEWS_URL, VALID_CATEGORIES, BAD_PARENTS_MAP, BAD_HOSTS, FILE_DATETIME_FORMAT, ISO_DATETIME_FORMAT


def get_last_recorded_timestamp():
    # get the datetime of the last run, or default to a week ago
    past_files = os.listdir(GOOGLE_NEWS_DIRECTORY)
    if not past_files:
        return (datetime.utcnow() - timedelta(days=7)).strftime(ISO_DATETIME_FORMAT)
    iso_datetime_strings = list(map(lambda file_name: file_name.split('.')[0], past_files))
    return datetime.strptime(max(iso_datetime_strings), FILE_DATETIME_FORMAT).strftime(ISO_DATETIME_FORMAT)


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]', 'nav', '']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def element_has_bad_parent(element, bad_parents=BAD_PARENTS_MAP):
    # ignore elements in irrelevent compnents like nav, footer if it is nested in at least n layers of bad_parent
    bad_parents_tracker = {}
    index = 0
    while element.parent:
        index += 1
        if (element.name in bad_parents.keys()):
            bad_parents_tracker[element.name] = bad_parents_tracker.setdefault(element.name, 0) + 1
            if bad_parents_tracker[element.name] >= bad_parents[element.name]:
                return False
        element = element.parent
        if index > 100:
            return True
    return True

def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    valid_texts = filter(element_has_bad_parent, visible_texts)
    # join all the text segments
    body = u" ".join(t.strip() for t in valid_texts)
    # remove access spaces
    body = re.sub(' +', ' ', body)
    # join <a/> tags to end of sentences and strip remaining spaces:
    return body.strip().replace(' .', '.')


def make_google_news_url(url_fragment):
    return GOOGLE_NEWS_BASE_URL + url_fragment[1:]


def get_category_urls():
    res = requests.get(GOOGLE_NEWS_URL)
    html_text = res.text
    soup = BeautifulSoup(html_text, 'html.parser')

    url_map = {}
    for category in VALID_CATEGORIES:
        url = soup.find('a', { 'aria-label': category})
        url_map[category] = make_google_news_url(url.get('href'))
    return url_map

def process_url(url):
    data = {
        'success': False,
        'body': None,
        'url': None
    }
    try:
        res = requests.get(url, timeout=(2, 4))
    except requests.exceptions.Timeout as e:
        print('Timeout error:', e, url)
        return data
    except requests.exceptions.ConnectionError as e:
        print('Connection refused error:', e, url)
        return data

    for bad_url in BAD_HOSTS:
        if bad_url in res.url:
            return data

    data['body'] = text_from_html(res.text)
    data['url'] = res.url
    data['success'] = True
    return data


def aggregate_headlines(category_url, filter_datetime_string=''):
    # For each news topic, scrape the page articles int a csv, we will visit each url later to get the actual content
    res = requests.get(category_url)
    html_text = res.text
    soup = BeautifulSoup(html_text, 'html.parser')
    all_articles = soup.find_all('article')

    headlines = []
    for article in all_articles:
        headers = []
        headers.extend(article.findChildren("h3" , recursive=True))
        headers.extend(article.findChildren("h4" , recursive=True))

        timing_eles = article.findChildren("time" , recursive=True)
        if not timing_eles:
            continue
        timing = timing_eles[0].get('datetime')
        
        source = 'N/A'
        url = 'XXX'
        all_links = article.findChildren("a" , recursive=True)
        for link in all_links:
            if './publications' in link.get('href', ''):
                source = link.get_text()
            if './article' in link.get('href', ''):
                url = make_google_news_url(link.get('href'))

        if source == 'N/A':
            for link in timing_eles[0].parent.findChildren("a" , recursive=True):
                if link.get_text():
                    source = link.get_text()
                    
        if timing >= filter_datetime_string:
            headlines.append({
                'category': 'X',
                'source': source, 
                'datetime': timing, 
                'headline': headers[0].get_text(),
                'url': url
            })
    return headlines

def main():
    start = datetime.utcnow()
    categories = get_category_urls()
    period_start_datetime_string = get_last_recorded_timestamp()
    data = []
    for category, category_url in categories.items():
        category_headlines = aggregate_headlines(category_url, period_start_datetime_string)
        print('%s:' % category, len(category_headlines))
        headlines = [dict(item, **{'category': category}) for item in category_headlines]

        for h in headlines:
            url_data = process_url(h['url'])
            if url_data['success']:
                h['url'] = url_data['url']
                h['story'] = url_data['body']
                data.append(h)
    
    print('Total stories loaded:', len(data))

    # make directory if not exists
    os.makedirs(GOOGLE_NEWS_DIRECTORY, exist_ok=True)
    csv_file = GOOGLE_NEWS_DIRECTORY + '/' + start.isoformat(timespec='seconds').replace(':', '') + '.csv'
    
    with open(csv_file, 'w', newline='') as write_file:
        field_names = data[0].keys()
        dict_writer = csv.DictWriter(write_file, field_names)
        dict_writer.writeheader()
        dict_writer.writerows(data)

if __name__ == '__main__':
    main()
    # test_url = 'https://www.cbsnews.com/news/walmart-returning-firearm-guns-ammo-store-displays/'
    # test_url = 'https://news.google.com/articles/CAIiEJ0ZsMvmKn3aAO3jKnT6-S8qFggEKg4IACoGCAowl6p7MN-zCTDlkko?hl=en-US&gl=US&ceid=US%3Aen'
    # test_url = 'https://www.newsobserver.com/news/state/north-carolina/article246835757.html'
    # data = get_url_text(test_url)
    # r = requests.get(test_url)
    # print(r.url)
