import requests
from bs4 import BeautifulSoup


NEW_URL = 'https://news.google.com/topstories?tab=wn&hl=en-US&gl=US&ceid=US:en'


def main():
    res = requests.get(NEW_URL)
    html_text = res.text
    soup = BeautifulSoup(html_text, 'html.parser')
    all_articles = soup.find_all('article')

    for article in all_articles:
        headers = []
        headers.extend(article.findChildren("h3" , recursive=True))
        headers.extend(article.findChildren("h4" , recursive=True))

        timing_eles = article.findChildren("time" , recursive=True)
        if not timing_eles:
            continue
        timing = timing_eles[0].get('datetime')

        source = 'N/A'
        all_links = article.findChildren("a" , recursive=True)
        for link in all_links:
            if './publications' in link.get('href', ''):
                source = link.get_text()

        if source == 'N/A':
            for link in timing_eles[0].parent.findChildren("a" , recursive=True):
                if link.get_text():
                    source = link.get_text()

        print(source, timing, headers[0].get_text())




if __name__ == '__main__':
    main()