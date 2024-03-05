import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import preprocessor as tweet_preprocessor
from ttp import ttp
import emoji
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('./data.csv')

def convert_to_emoji(text):
    return emoji.demojize(text)

tweets = data['tweet']
tweets_tokenized = []
tt = nltk.tokenize.TweetTokenizer()
for tweet in tweets:
    tweet = convert_to_emoji(tweet)
    # initializing punctuations string
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~"'?'''
 
    # Removing punctuations in string
    # Using loop + punctuation string
    for ele in tweet:
        if ele in punc:
            tweet = tweet.replace(ele, "")
 
    tweets_tokenized.append(tt.tokenize(text=tweet))

#remove stopwords
stemmer = nltk.stem.porter.PorterStemmer()
stoplist = nltk.corpus.stopwords.words('english')

stemmed = []

for tweet in tweets_tokenized:
    processed_tokens = [stemmer.stem(t) for t in tweet if t not in stoplist]
    stemmed.append(processed_tokens)

def super_simple_preprocess(text):
  # lowercase
  text = text.lower()
  # remove non alphanumeric characters
  text = re.sub('[^A-Za-z0-9 ]+','', text)
  return text

data_updated = []
for tweet in stemmed:
    data_updated.append(super_simple_preprocess(" ".join(tweet)))
pd.DataFrame(data_updated).shape

data_updated = pd.DataFrame(data_updated)
data_updated.insert(1, 'w', data['label'])
data = data_updated


X = data.iloc[:, 0]
y = data.iloc[: , 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

def to_csvs(X_train,y_train,X_test,y_test,X_val,y_val):
    X_train = pd.DataFrame(X_train)
    X_train.insert(1, 'label', y_train)
    with open('train.pickle', 'wb') as file:
        pickle.dump(X_train, file)
    X_train.to_csv('train.csv')
    X_test = pd.DataFrame(X_test)
    X_test.insert(1, 'label', y_test)
    X_test.to_csv('test.csv')
    with open('test.pickle', 'wb') as file:
        pickle.dump(X_test, file)
    X_val = pd.DataFrame(X_val)
    X_val.insert(1, 'label', y_val)
    X_val.to_csv('val.csv')
    with open('val.pickle', 'wb') as file:
        pickle.dump(X_val, file)

to_csvs(X_train,y_train,X_test,y_test,X_val,y_val)

text_transformer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=150000)
X_train_text = text_transformer.fit_transform(X_train)
X_test_text = text_transformer.transform(X_test)
X_val_text = text_transformer.transform(X_val)

with open('train_vectorized.pickle', 'wb') as file:
    pickle.dump(X_train_text, file)

with open('test_vectorized.pickle', 'wb') as file:
    pickle.dump(X_test_text, file)

with open('val_vectorized.pickle', 'wb') as file:
    pickle.dump(X_val_text, file)

data['combined'] = "__label__" + data['w'] + " " + data.iloc[:, 0]
train, test = train_test_split(data, test_size=0.2)
test, val = train_test_split(test, test_size=0.5)

train.to_csv('data.train', columns=['combined'], index=False, header=False)
train.to_csv('data.test', columns=['combined'], index=False, header=False)
train.to_csv('data.val', columns=['combined'], index=False, header=False)