# N-Grams
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
    #txt = [ps.stem(word) for word in tokens if word not in stopwords]
    txt = " ".join([ps.stem(word) for word in tokens if word not in stopwords])
    return txt

data ['msg_clean'] = data['msg'].apply(lambda x: clean_text(x))
print('\nclean_text => data_head()')
print(data.head()) 

# Count Vectorize
# from sklearn.feature_extraction.text import CountVectorizer
# # only 2 grams (from 2 to 2)
# cv = CountVectorizer (ngram_range=(2, 2)) 
# corpus = ["This is a sentence is",
# "This is another sentence",
# "third document is here"]
# X = cv.fit(corpus)
# X = cv.transform(corpus)
# X = cv.fit_transform(corpus)
# # print('X.vocabulary_:', X.vocabulary_)
# # print('cv.get_feature_names():', cv.get_feature_names())
# print('X.shape:', X.shape)
# print('X:')
# print(X)
# print('X.toarray():')
# print(X.toarray())
# df = pd.DataFrame(X.toarray(), columns = cv.get_feature_names())
# print('df:')
# print(df)

# CountVectorization on SMSSpamCollection
from sklearn.feature_extraction.text import CountVectorizer
# cv1 = CountVectorizer(analyzer=clean_text)
# cv1 = CountVectorizer(ngram_range=(2, 3))
# X = cv1.fit_transform(data['msg_clean'])
# print('X.shape:', X.shape)
# print('cv1.get_feature_names():', cv1.get_feature_names())
data_sample = data[0:10]
#cv2 = CountVectorizer(analyzer=clean_text)
cv2 = CountVectorizer(ngram_range=(2, 3))
X = cv2.fit_transform(data_sample['msg_clean'])
print('X.shape:', X.shape)
print('\ncv2.get_feature_names():')
print(cv2.get_feature_names())
df = pd.DataFrame (X.toarray(), columns=cv2.get_feature_names())
print('df.head(10)')
print(df.head(10))
