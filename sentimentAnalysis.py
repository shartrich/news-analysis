from datetime import datetime, timedelta
from utils.dataHelper import load_news_data
from utils.config import TEST_OUTPUT_PATH, INPUT_PATH
import csv


from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier

import re, string, random


def remove_noise(tweet_tokens, stop_words = ()):
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

def classify_string(text, classifier):
    custom_tokens = remove_noise(word_tokenize(text))
    return classifier.classify(dict([token, True] for token in custom_tokens))

def main():

    model_csv = INPUT_PATH + '/newsSentiment.csv'
    stop_words = stopwords.words('english')
    all_model_data = []
    model_data = {
        'positive': [],
        'neutral': [],
        'negative': []
    }
    tokenized_data_rows = []
    with open(model_csv, newline='', encoding="ISO-8859-1") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            # model_data.append((row[1], row[0]))
            all_model_data.append((row[0], row[1]))
            tokens = remove_noise(word_tokenize(row[1]), stop_words)
            model_data[row[0]].append(tokens)
            tokenized_data_rows.append(
                (tokens, row[0])
            )

    # use UTC time
    to_datetime = datetime.utcnow()
    from_datetime = to_datetime - timedelta(days=7)
    news_data = load_news_data(from_datetime, to_datetime)
    news_data = news_data[news_data['category'] == 'business']
    # remove tiny snippets
    # news_data = news_data[(news_data.description.map(len) > 140)]

    # positive_tweets = twitter_samples.strings('positive_tweets.json')
    # negative_tweets = twitter_samples.strings('negative_tweets.json')
    
    # text = twitter_samples.strings('tweets.20150430-223406.json')
    # tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]


    # positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    # negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    # positive_cleaned_tokens_list = []
    # negative_cleaned_tokens_list = []

    # for tokens in positive_tweet_tokens:
    #     positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    # for tokens in negative_tweet_tokens:
    #     negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    # all_pos_words = get_all_words(model_data['positive'])

    # freq_dist_pos = FreqDist(all_pos_words)
    # print(freq_dist_pos.most_common(10))

    # positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    # negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

    # positive_dataset = [(tweet_dict, "Positive")
    #                      for tweet_dict in positive_tokens_for_model]

    # negative_dataset = [(tweet_dict, "Negative")
    #                      for tweet_dict in negative_tokens_for_model]

    # [(tweet_dict, "Negative")
    #                      for tweet_dict in negative_tokens_for_model]

    # dataset = positive_dataset + negative_dataset

    dataset = []
    for key, token_list in model_data.items():
        tokens_for_model = get_tweets_for_model(token_list)
        # for token_row in token_list:
        dataset.extend([(tweet_dict, key)
                        for tweet_dict in tokens_for_model])

    random.shuffle(dataset)

    partition_number = len(dataset)*3//4
    train_data = dataset[:partition_number]
    test_data = dataset[partition_number:]

    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy is:", classify.accuracy(classifier, test_data))
    print(classifier.show_most_informative_features(10))

    # custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."
    # custom_tokens = remove_noise(word_tokenize(custom_tweet))
    # print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))
    news_data['sentiment'] = news_data['title'].apply(classify_string, args=(classifier,))
    save_path = TEST_OUTPUT_PATH + '/data3.csv'
    results = news_data[['title', 'sentiment']]
    results.to_csv(save_path, index=False, encoding='utf-8')


if __name__ == "__main__":
    main()
