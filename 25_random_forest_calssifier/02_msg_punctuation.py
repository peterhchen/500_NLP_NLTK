# Read data
import pandas as pd
import nltk
import re
import string

stopwords = nltk.corpus.stopwords.words ('english')
ps = nltk.PorterStemmer()

data = pd.read_csv('../data/smsspamcollection/SMSSpamCollection', \
    sep='\t', header=None)
data.columns = ['label', 'msg']
print(data.head())

def punctuation_count(txt):
    count = sum ([1 for c in txt if c in string.punctuation])
    return 100*count/len(txt)

data['msg_len'] = data['msg'].apply(lambda x: len(x))
data['punctuation_%'] = data['msg'].apply (lambda x: punctuation_count(x))

print('punctuation_count => data.head()')
print(data.head())