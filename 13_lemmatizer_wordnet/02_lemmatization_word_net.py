import nltk
wn = nltk.WordNetLemmatizer ()

# Read raw text
import pandas as pd
import re
import string
pd.set_option('display.max_colwidth', 100)

stopwords = nltk.corpus.stopwords.words('english')
data = pd.read_csv('../data/smsspamcollection/SMSSpamCollection', \
    sep='\t', header=None)

data.columns=['label', 'msg']
print('\ndata.head():')
print(data.head())

# Cleaning
def clean_text (txt):
    # remvoe the !#$ ... special characters
    txt = "".join([c for c in txt if c not in string.punctuation])
    # put into array tokens
    tokens = re.split('\W+', txt)
    # remove stop words (popular words): if, I, am, 
    txt = [word for word in tokens if word not in stopwords]
    return txt

data['msg_nostop'] = data['msg'].apply(lambda x: clean_text(x))
print('\nNo Stop Word => data.head():')
print(data.head())

# Lemmatization
def lemmatization (token_txt):
    text = [wn.lemmatize(word) for word in token_txt]
    return text

data['msg_lemmatized'] = data['msg_nostop'].apply(lambda x : lemmatization(x))
print('\nLemmatied => data.head():')
print(data.head())