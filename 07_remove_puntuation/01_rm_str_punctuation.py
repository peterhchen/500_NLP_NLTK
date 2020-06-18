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
    txt_nopunct = [c for c in txt if c not in string.punctuation]
    return txt_nopunct

data['msg_clean'] = data['msg'].apply(lambda x: remove_punctuation (x))
print('remove punctuation => data.head():')
print (data.head())