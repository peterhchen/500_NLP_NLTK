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

# Add Features
def punctuation_count(txt):
    count = sum ([1 for c in txt if c in string.punctuation])
    return 100*count/len(txt)

data['msg_len'] = data['msg'].apply(lambda x: len(x))
data['punctuation_%'] = data['msg'].apply (lambda x: punctuation_count(x))

print('punctuation_count => data.head()')
print(data.head())

# clean data by fit_tranform(data['msg'])
def clean_text(txt):
    #print ('clean_text:', txt)
    txt = "".join([c for c in txt if c not in string.punctuation])
    tokens = re.split('\W+', txt)
    txt = [ps.stem(word) for word in tokens if word not in stopwords]
    return txt

print('punctuation_count => data.head()')
print(data.head())

# Clean up msg and Vectorized:
# TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
# TfidfVectorizer () just setup clean_text function handler.
# There is not data passing. Nothing trigger yet.
tfidf = TfidfVectorizer(analyzer=clean_text) 
# The fit_transform trigger the clean_text and data['msg']
print("data['msg'] => data['msg'].head()")
print(data['msg'].head())
X_tfidf = tfidf.fit_transform(data['msg'])

print('IF-IDF Vectorization => X_tfidf.shape')
print(X_tfidf.shape)
print('IF-IDF Vectorization => X_tfidf.toarray()')
print(X_tfidf.toarray())
