import nltk
import pandas as pd

# Remove Punctuation
# Read dataset
pd.set_option('display.max_colwidth', 100)  # We can longer display
data = pd.read_csv('../data/smsspamcollection/SMSSpamCollection', \
    sep='\t', header=None)

data.columns=['label', 'msg']
print('set columns => data.head():')
print (data.head())

import string
print ('\nstring.punctuation:', string.punctuation)

def remove_punctuation (txt):
    # for all the character in txt.
    # if is not in punctuation
    txt_nopunct = "".join([c for c in txt if c not in string.punctuation])
    return txt_nopunct

data['msg_clean'] = data['msg'].apply(lambda x: remove_punctuation (x))
print('\nremove punctuation => data.head():')
print (data.head())

# Tokenization
import re

def tokenize(txt):
    tokens = re.split('\W+', txt)
    return tokens

data['msg_clean_tokenized'] = data['msg_clean'].apply(lambda x:tokenize(x.lower()))
print('\ntokenize => data.head():')
print(data.head())

# Remove stop words
# import nltk
# We need to specify the language 
stopwords = nltk.corpus.stopwords.words('english')
print('\nstopwords:')
print(stopwords)
print('\nstopwords[0:10]:')
print(stopwords[0:10])

def remove_stopwords(txt_tokenized):
    txt_clean = [word for word in txt_tokenized if word not in stopwords]
    return txt_clean

data['msg_no_sw'] = data['msg_clean_tokenized'].apply(lambda x: remove_stopwords(x))
print('\nremove stopwords => data.head():')
print (data.head())

# Porter Stemmer
# import nltk
from nltk.stem import PorterStemmer
ps = PorterStemmer()
# See what kinds functions available.
print('\ndir(ps):')
print (dir(ps))