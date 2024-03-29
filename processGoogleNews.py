import pandas as pd
from datetime import datetime, timedelta
pd.options.display.max_columns = 200
pd.options.mode.chained_assignment = None
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from string import punctuation
from collections import Counter
import re
import os
import numpy as np
from matplotlib import pyplot as plt
from functools import reduce
from sklearn.manifold import TSNE

from IPython.display import display, HTML


import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook, reset_output
from bokeh.palettes import d3
from bokeh.palettes import brewer
import bokeh.models as bmo
from bokeh.io import save, output_file
from langdetect import detect


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
# from wordcloud import WordCloud
from shutil import copyfile


from dotenv import load_dotenv

load_dotenv()

# profile 1 week of news at a time
period_days = 7
current_directory = os.path.dirname(os.path.abspath(__file__))
news_directory = '/data/googleNews'
results_directory = 'output'


RELATIVE_PATH_TO_FLASK_APP = os.getenv('PATH_TO_FLASK_APP')
OPTIMIZE = os.getenv('OPTIMIZE', False)
PATH_TO_FLASK_APP = '%s/%s' % (current_directory, RELATIVE_PATH_TO_FLASK_APP)
FULL_RUN = os.getenv('FULL_RUN', True)
CLUSTER_SIZE = int(os.getenv('CLUSTER_SIZE', '8'))

num_clusters = 60
max_line = 60


def load_news_data(from_datetime, to_datetime):
    li = []
    all_files = os.listdir('%s/%s' % (current_directory, news_directory))

    to_datetime_string = to_datetime.isoformat(timespec='seconds').replace(':','')
    from_datetime_string = from_datetime.isoformat(timespec='seconds').replace(':','')

    valid_files = list(filter(lambda file_name: file_name.split('.')[0] >= from_datetime_string and file_name.split('.')[0] < to_datetime_string, all_files))

    for file_name in valid_files:
        df = pd.read_csv('%s/%s/%s' % (current_directory, news_directory, file_name), index_col=None, header=0)
        li.append(df)

    return pd.concat(li, axis=0, ignore_index=True)

def file_format_datetime(datetime_value):
    return datetime_value.isoformat(timespec='seconds').replace(':','')

def title_format_datetime(datetime_value):
    return datetime_value.isoformat(timespec='seconds').split('T')[0]

# def plot_word_cloud(terms):
#    text = terms.index
#    text = ' '.join(list(text))
#    # lower max_font_size
#    wordcloud = WordCloud(max_font_size=40).generate(text)
#    plt.figure(figsize=(25, 25))
#    plt.imshow(wordcloud, interpolation="bilinear")
#    plt.axis("off")
#    plt.show()


def test_no_overlap(word_list):
    multi_words = {}
    for phrase in word_list:
        if ' ' in phrase:
            words = phrase.split()
            for word in words:
                multi_words[word] = 1

        
def find_nearest_space(text, mid_location):
    #diff = len(text) - mid_location
    for i in range(0, mid_location - 1):
        if text[mid_location+i] == " ":
            return mid_location+i
        if text[mid_location-i] == " ":
            return mid_location-i

    #no space found
    return mid_location

    
def break_up_string(text):
    if len(text) >= max_line:

        split_1 = find_nearest_space(text, len(text) // 2)

        left = text[:split_1-1] + '\n'
        right = text[split_1:]

        
        return break_up_string(left) + break_up_string(right)
                    
    else:
        return text

def shorten_str(text):
    if type(text) != str:
        print('shorten_str error:')
        print(text)
        print(type(text))
    if len(text) >= 100:
        return text[:100] + '...'
    return text

print('Running')
def count_word(df, word):
    #for row in df[]
    pass

def remove_non_ascii(string):
    string = string.replace("“", "'")
    string = string.replace("”", "'")
    #string = string.replace(":", "'")
    if string == '':
        return string
    return ''.join([i if ord(i) < 128 else ' ' for i in string])

def _removeNonAscii(s): 
    return "".join(i for i in s if ord(i)<128)

def clean_text(text):
    text = text.lower() #see if necessary
    text = re.sub(r"what's", "what is ", text)
    text = text.replace('(ap) - ', '')
    text = text.replace('&lsquo;', "'")
    text = text.replace('&rsquo;', "'")
    text = text.replace('Skip to main content', "")
    text = text.replace('Skip to content', "")
    text = text.replace('Skip to Article', "")
    text = text.replace('Skip Navigation', "")
    text = text.replace('Skip to navigation', "")
    text = text.replace('Advertisement', "")
    text = text.replace('(ap)', '')
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)
    text = re.sub('[^a-zA-Z ?!]+', '', text)
    text = _removeNonAscii(text)
    text = text.strip()
    return text

def tokenizer(text):
    text = clean_text(text)    
    tokens = [word_tokenize(sent) for sent in sent_tokenize(text)]
    
    tokens = list(reduce(lambda x,y: x+y, tokens))
    tokens = list(filter(lambda token: token not in (stop_words + list(punctuation)) , tokens))
    return tokens


def keywords(category, cluster_size):
    tokens = data[data['category'] == category]['tokens']
    alltokens = []
    for token_list in tokens:
        alltokens += token_list
    counter = Counter(alltokens)
    return counter.most_common(cluster_size)


def detect_language(text):
    try:
        return detect(text)
    except:
        return 'Error'

# use UTC time
to_datetime = datetime.utcnow()
from_datetime = to_datetime - timedelta(days=period_days)
data = load_news_data(from_datetime, to_datetime)

data['description'] = data['story']
data = data[~data['description'].isnull()]
data = data.drop_duplicates('description')
data = data[data['description'].map(detect_language) == 'en']


stop_words = []

f = open(current_directory + '/input/stopwords.txt', 'r')
for l in f.readlines():
    stop_words.append(l.replace('\n', ''))
    
additional_stop_words = ['t', 'will']
stop_words += additional_stop_words

data = data[(data.description.map(len) > 140)]
data.reset_index(inplace=True, drop=True)
# print(len(data))
# data = data[(data.description.map(len) > 140) & (data.description.map(len) <= 300)]
# data.reset_index(inplace=True, drop=True)
# print(len(data))

data['tokens'] = data['description'].map(lambda d: tokenizer(d))


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=5, analyzer='word', ngram_range=(1, 2), stop_words='english')
vz = vectorizer.fit_transform(list(data['tokens'].map(lambda tokens: ' '.join(tokens))))


#low impact generic words
# plot_word_cloud(tfidf.sort_values(by=['tfidf'], ascending=True).head(40))
# output_file('output/generic-words.png')

#high impact words
# plot_word_cloud(tfidf.sort_values(by=['tfidf'], ascending=False).head(40))
# output_file('output/high-impact-words.png')


distorsions = []
sil_scores = []
k_max = 80

best_score = 0

if OPTIMIZE == True:
    for k in range(2, k_max):
        print('n_clusters=%s' % k)
        kmeans_model = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1, random_state=42,  
                            init_size=1000, verbose=False, max_iter=1000)
        kmeans_model.fit(vz)
        sil_score = silhouette_score(vz, kmeans_model.labels_)
        sil_scores.append(sil_score)
        distorsions.append(kmeans_model.inertia_)

        if sil_score > best_score:
            best_score = sil_score
            num_clusters = k

print('n_clusters=%s' % num_clusters)
kmeans_model = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1, random_state=42,                       
                         init_size=1000, batch_size=1000, verbose=False, max_iter=1000, )
kmeans = kmeans_model.fit(vz)
kmeans_clusters = kmeans.predict(vz)
kmeans_distances = kmeans.transform(vz)


sorted_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

multi_words = {}
for word in terms:
    if ' ' in word:
        items = word.split(' ')
        for item in items:
            multi_words[item] = 1

all_keywords = []
for i in range(num_clusters):
    topic_keywords = []
    for j in sorted_centroids[i, :CLUSTER_SIZE]:
        if j not in multi_words:
            topic_keywords.append(terms[j])
            
    all_keywords.append(topic_keywords)

keywords_df = pd.DataFrame(index=['topic_{0}'.format(i) for i in range(num_clusters)], 
                           columns=['keyword_{0}'.format(i) for i in range(CLUSTER_SIZE)],
                           data=all_keywords)


s2 = {}
abbreviated_clusters = {}
for idx, row in keywords_df.iterrows():
    abbreviated_clusters[row.name] = [row.keyword_0, row.keyword_1, row.keyword_2, row.keyword_3, row.keyword_4, row.keyword_5, row.keyword_6, row.keyword_7]


keywords_df['Snippet'] = keywords_df.index.to_series().map(abbreviated_clusters)

if FULL_RUN:
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, n_iter=500)
    tsne_kmeans = tsne_model.fit_transform(kmeans_distances)
    kmeans_df = pd.DataFrame(tsne_kmeans, columns=['x', 'y'])
    kmeans_df['cluster'] = kmeans_clusters
    kmeans_df['cluster'] = kmeans_df['cluster'].map(str)
    kmeans_df['headline'] = data['headline']
    # kmeans_df['headline'] = data['description']
    kmeans_df['description'] = kmeans_df['headline'].map(shorten_str)
    kmeans_df['category'] = data['category']
    kmeans_df.to_csv(current_directory + '/data/tsne_kmeans.csv', index=False, encoding='utf-8')
else:
    kmeans_df = pd.read_csv(current_directory + '/data/tsne_kmeans.csv')
    kmeans_df['cluster'] = kmeans_df['cluster'].map(str)


s = {}
# TODO add this a toggle for full cluster names in legend vs abbreviated names
# for idx, row in enumerate(keywords_df['Summary']):
for idx, row in enumerate(keywords_df['Snippet']):
    s[idx] = row
	

kmeans_df['int_vals'] = kmeans_df['cluster'].map(int)
kmeans_df['Desc'] = kmeans_df['int_vals'].map(s)


reset_output()
output_notebook()

plot_title = "KMeans Clustering of the News: %s - %s (%s articles)" % (title_format_datetime(from_datetime), title_format_datetime(to_datetime), len(data.index))
plot_kmeans = bp.figure(plot_width=1600, plot_height=1200, title=plot_title,
    tools="pan,wheel_zoom,box_zoom,reset,hover",
    x_axis_type=None, y_axis_type=None, min_border=1)

palette = d3['Category20'][20] + d3['Category20b'][20] + d3['Category20c'][20] + brewer['BrBG'][9] + brewer['RdYlBu'][11]
color_map = bmo.CategoricalColorMapper(factors=kmeans_df['cluster'].unique(), palette=palette)

plot_kmeans.scatter('x', 'y', source=kmeans_df, 
                    color={'field': 'cluster', 'transform': color_map})
hover = plot_kmeans.select(dict(type=HoverTool))
hover.tooltips={"Category": "@category", "Cluster": "@Desc", "Description": "@description"}



save_path = current_directory + '/output/bokeh/kmeans/%s.html' % file_format_datetime(to_datetime)
copy_path = PATH_TO_FLASK_APP + '/%s.html' % file_format_datetime(to_datetime)
save(plot_kmeans, save_path, title='News Analysis')

copyfile(save_path, copy_path)
print('Done')
