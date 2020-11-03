import os

FILE_DATETIME_FORMAT = '%Y-%m-%dT%H%M%S'
ISO_DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%SZ'

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__)) + '/../../'
GOOGLE_NEWS_BASE_URL = 'https://news.google.com'
GOOGLE_NEWS_DIRECTORY = CURRENT_DIRECTORY + '/data/googleNews'

GOOGLE_NEWS_URL = 'https://news.google.com/topstories?tab=wn&hl=en-US&gl=US&ceid=US:en'

VALID_CATEGORIES = [
    'U.S.',
    'World',
    'Business',
    'Technology',
    'Entertainment',
    'Sports',
    'Science',
    'Health'
]

BAD_PARENTS_MAP = {
    'nav': 1,
    'header': 1,
    'footer': 1,
    'article': 2,
    'figcaption': 1,
    'figure': 1,
    'ul': 1,
    'h1': 1,
    'h2': 1,
    'h3': 1,
    'h4': 1,
    'h5': 1
}

# some hosts block scraping:
BAD_HOSTS = [
    'washingtonpost.com',
    'youtube.com'
]

BAD_SOURCES = [
    'The Washington Post',
    'Miami Herald'
]
