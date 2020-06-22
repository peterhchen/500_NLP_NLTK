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

print('TF-IDF Vectorization => X_tfidf.shape')
print(X_tfidf.shape)
print('IF-IDF Vectorization => X_tfidf.toarray()')
print(X_tfidf.toarray())

# Combine msg_len, punctuation_x%', and Vectorized msg  
X = pd.concat ([data['msg_len'], data['punctuation_%'], \
    pd.DataFrame(X_tfidf.toarray())], axis=1)

print('Combine msg_len, punctuation_%, msg => X.head()')
print(X.head())

# Explore RandomFOrest with Holdout Test Set
from sklearn.metrics import precision_recall_fscore_support as prfs_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = \
    train_test_split (X, data['label'], test_size=0.1)

rf = RandomForestClassifier (n_estimators=100, max_depth=15, n_jobs=-1)
rfmodel = rf.fit(X_train, y_train) # Contain 90% of data
print ('\nsorted(rfmodel.feature_importances_, reverse=True)[0:5]:')
print (sorted(zip(rfmodel.feature_importances_, X_train.columns), reverse=True)[0:5])
# [(0.05670296459780532, 'msg_len'): Msg_len is the most important feature.
# feature_importances_ is very handy feature
y_pred = rfmodel.predict(X_test)
# Ths will return the value with Ham or Spam.
precision, recall, fscore , support = \
    prfs_score(y_test, y_pred, pos_label='spam', average='binary')

accuracy = (y_pred == y_test).sum()/len(y_pred)
print(f'accuracy = {accuracy}')
print(f'precision = {precision}')
print(f'recall = {recall}')
print(f'fscore = {fscore}')