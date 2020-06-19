# Porter Stemmer
import nltk
from nltk.stem import PorterStemmer
ps = PorterStemmer()
# See what kinds functions available.
print('\ndir(ps):')
print(dir(ps))

print("\nps.stem('coder'):", ps.stem('coder'))  # coder
print("ps.stem('coding'):", ps.stem('coding'))  # code
print("ps.stem('code'):", ps.stem('code'))      # code

print("\nps.stem('data'):", ps.stem('data'))    # data
print("ps.stem('datum'):",ps.stem('datum'))     # datum

print("\nps.stem('bowl'):", ps.stem('bowl'))        # bowl
print("ps.stem('bowling'):", ps.stem('bowling'))    # bowl
print("ps.stem('bowler'):", ps.stem('bowler'))      # bowler

# SMSSPamCollection - Cleaning
import pandas as pd
import re
import string
pd.set_option('display.max_colwidth', 100)

stopwords = nltk.corpus.stopwords.words('english')
data = pd.read_csv('../data/smsspamcollection/SMSSpamCollection', \
    sep='\t', header=None)

data.columns=['label', 'msg']
def clean_text(text):
    # Remove punctuation: single quote, double quote, dollar sign, percentage, 
    # comma, period, exclamation, and etc.
    text = ''.join([c for c in text if c not in string.punctuation])
    tokens = re.split('\W+', text)
    # Those commonly used words, such as, am, is, the, and etc., are 
    # called the Stop Words
    text = [word for word in tokens if word not in stopwords]
    return text

# Remove the stop words (am, is, the, etc)
data['msg_nonstop'] = data['msg'].apply(lambda x: clean_text(x.lower()))
print('\nRemove stop words => data.head():')
print(data.head())

# Get the Stemmed the text
def stemming(tokenized_text):
    text = [ps.stem(word) for word in tokenized_text]
    return text

data['msg_stemmed'] = data['msg_nonstop'].apply (lambda x: stemming(x))
print ('\nGet stemmed (root) word => data.head():')
print(data.head())