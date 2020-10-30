import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_DIRECTORY = os.path.dirname(os.path.abspath(__file__)) + '/..'

NEWS_PATH = 'data/news'
RESULTS_PATH = 'output'

RELATIVE_PATH_TO_FLASK_APP = os.getenv('PATH_TO_FLASK_APP')
PATH_TO_FLASK_APP = '%s/%s' % (PROJECT_DIRECTORY, RELATIVE_PATH_TO_FLASK_APP)

TEST_OUTPUT_PATH = '%s/%s' % (PROJECT_DIRECTORY, 'output/test')
INPUT_PATH = '%s/%s' % (PROJECT_DIRECTORY, 'input')