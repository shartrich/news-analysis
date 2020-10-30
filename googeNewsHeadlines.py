import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
import re
import csv
from datetime import datetime
import os

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
NEWS_DIRECTORY = CURRENT_DIRECTORY + '/data/googleNews'

BASE_URL = 'https://news.google.com'
NEW_URL = 'https://news.google.com/topstories?tab=wn&hl=en-US&gl=US&ceid=US:en'

VALID_CATEGORIES = ['U.S.', 'World', 'Business', 'Technology', 'Entertainment', 'Sports', 'Science', 'Health']
BAD_PARENTS={'nav': 1, 'header': 1, 'footer': 1, 'article': 2, 'figcaption': 1, 'figure': 1, 'ul': 1, 
    'h1': 1,
    'h2': 1,
    'h3': 1,
    'h4': 1,
    'h5': 1
}

# some hosts block scraping:
BAD_HOSTS = ['washingtonpost.com']


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]', 'nav', '']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def element_has_bad_parent(element, bad_parents=BAD_PARENTS):
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
    return BASE_URL + url_fragment[1:]



def get_category_urls():
    res = requests.get(NEW_URL)
    html_text = res.text
    soup = BeautifulSoup(html_text, 'html.parser')

    url_map = {}
    for category in VALID_CATEGORIES:
        url = soup.find('a', { 'aria-label': category})
        url_map[category] = make_google_news_url(url.get('href'))
    return url_map

def get_url_text(url):
    try:
        res = requests.get(url, timeout=(2, 4))
    except requests.exceptions.Timeout as e:
        print('Timeout error:', e)
        return None
    return text_from_html(res.text)


def aggregate_headlines(category_url):
    # For each news topic, scrape the page articles int oa csv
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
    data = []
    for category, category_url in categories.items():
        base_headlines = aggregate_headlines(category_url)
        print('%s:' % category, len(base_headlines))
        headlines = [dict(item, **{'category': category}) for item in base_headlines]

        for h in headlines:
            h['story'] = get_url_text(h['url'])
            if h['story']:
                data.append(h)
    
    print('Total stories loaded:', len(data))
    csv_file = NEWS_DIRECTORY + '/' + start.isoformat(timespec='seconds').replace(':', '') + '.csv'
    with open(csv_file, 'wb') as write_file:
        fieldnames = data[0].keys()
        csv_writer = csv.DictWriter(write_file, delimiter=',', fieldnames=fieldnames)
        for row in data:
            csv_writer.writerow(row)

if __name__ == '__main__':
    main()
    # test_url = 'https://www.cbsnews.com/news/walmart-returning-firearm-guns-ammo-store-displays/'
    # test_url = 'https://news.google.com/articles/CAIiEPRNUL7_efGlbCRt05BaZVsqGQgEKhAIACoHCAownob-CjCUzPYCMK_h3AU?hl=en-US&gl=US&ceid=US%3Aen'
    # test_url = 'https://www.newsobserver.com/news/state/north-carolina/article246835757.html'
    # data = get_url_text(test_url)
    # print(data)
