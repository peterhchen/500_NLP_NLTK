# Count Vectorization
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

corpus = ["This is a sentence is",
"This is another sentence",
"Third document is there"]

X = cv.fit(corpus)
print ('\nX.vocabulary_:')
print (X.vocabulary_)
print ('\ncv.get_feature_names():')
print (cv.get_feature_names())

X = cv.transform (corpus)
print ('\nX.shape:')
print (X.shape)

print ('\nX:')
print (X)

print ('\nX.toarray():')
print (X.toarray()) 

