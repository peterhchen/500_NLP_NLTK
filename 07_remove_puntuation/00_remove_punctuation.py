import nltk
import pandas as pd

# Remove Punctuation
# Read dataset
pd.set_option('display.max_colwidth', 100)  # We can longer display
dataset = pd.read_csv('../data/smsspamcollection/SMSSpamCollection', \
    sep='\t', header=None)

dataset.columns=['label', 'sms']
print('set columns of dataset => dataset.head():')
print (dataset.head())

import string
print ('string.punctuation:', string.punctuation)