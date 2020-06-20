# TF-IDF
import pandas as pd
import re
import string
import nltk

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

data = pd.read_csv('../data/smsspamcollection/SMSSpamCollection', \
    sep='\t', header=None)
data.columns = ['label', 'msg']
print(data.head())

# Text Cleaning
def clean_text(txt):
    txt = "".join([c for c in txt if c not in string.punctuation])
    tokens = re.split('\W+', txt)
    txt = [ps.stem(word) for word in tokens if word not in stopwords]
    #txt = " ".join([ps.stem(word) for word in tokens if word not in stopwords])
    return txt

data ['msg_clean'] = data['msg'].apply(lambda x: clean_text(x))
print('\nclean_text => data_head()')
print(data.head()) 

# TF-IDF Vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
# Tfidf()
tfidf_vec = TfidfVectorizer () 
corpus = ["This is a sentence is",
"This is another sentence",
"third document is here"]

X = tfidf_vec.fit(corpus)
print('\nX.vocabulary_:', X.vocabulary_)
print('tfidf_vec.get_feature_names():', tfidf_vec.get_feature_names())

X = tfidf_vec.transform(corpus)
#X = tfidf_vec.fit_transform(corpus)
print('X.shape:', X.shape)
print('X:')
print(X)
print('X.toarray():')
print(X.toarray())
df = pd.DataFrame(X.toarray(), columns = tfidf_vec.get_feature_names())
print('df:')
print(df)
# IF-IDF Vectorization on SMSSpamCollection
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_cv1 = TfidfVectorizer(analyzer=clean_text)
X = tfidf_cv1.fit_transform(data['msg'])
print('X.shape:', X.shape)
print('cv1.get_feature_names():', tfidf_cv1.get_feature_names())
df = pd.DataFrame(X.toarray(), columns=tfidf_cv1.get_feature_names())

# TF-IDF Vectorization on SMSSpamCollection for data[0:10]
data_sample = data[0:10]
tfidf_cv2 = TfidfVectorizer(analyzer=clean_text)
# X = tfidf_cv2.fit(data_sample['msg'])
# X = tfidf_cv2.transform(data_sample['msg'])
X = tfidf_cv2.fit_transform(data_sample['msg'])
print('\nX.shape:', X.shape)
# print('X:')
# print(X)
print('X.toarray():')
print(X.toarray())

df = pd.DataFrame(X.toarray(), columns=tfidf_cv2.get_feature_names())
print('\ndf.head(10):')
print(df.head(10))