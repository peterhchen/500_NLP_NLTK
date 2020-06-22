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
print('\ndata.columns:')
print(data.columns)
print("data['msg'] => data['msg'].head()")
print(data['msg'].head())
X_tfidf = tfidf.fit_transform(data['msg'])

print('IF-IDF Vectorization => X_tfidf.shape')
print(X_tfidf.shape)
print('IF-IDF Vectorization => X_tfidf.toarray()')
print(X_tfidf.toarray())

# Combine msg_len, punctuation_x%', and Vectorized msg  
X = pd.concat ([data['msg_len'], data['punctuation_%'], \
    pd.DataFrame(X_tfidf.toarray())], axis=1)

print('Combine msg_len, punctuation_%, msg => X.head()')
print(X.head())

# RadomForestClassifier and Cross-Validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
kf = KFold (n_splits=5)
rf = RandomForestClassifier (n_jobs=-1)
print ("\ncross_val_score (rf, X, data['label'], cv=kf, n_jobs=-1):")
print ("\ndata['label'].head(10):")
print (data['label'].head(10))
# Pandas copy
# https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
from contextlib import contextmanager
@contextmanager
def SuppressPandasWarning():
    with pd.option_context("mode.chained_assignment", None):
        yield

with SuppressPandasWarning():
    i = 0
    for x in data['label']:
        if i == 0:
            data['label'][i] = "social" 
        elif i == 1:
            data['label'][i] = "promotion"
        elif i == 2:
            data['label'][i] = "family"
        elif i == 3: 
            data['label'][i] = "friend"
        i += 1

print ("\nupdated data => data['label'].head(10)")
print (data['label'].head(10))
print (cross_val_score (rf, X, data['label'], cv=kf, n_jobs=-1)) 