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
