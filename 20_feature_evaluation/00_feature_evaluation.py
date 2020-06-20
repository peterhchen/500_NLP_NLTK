# Read in Raw Text
import pandas as pd

data = pd.read_csv('../data/smsspamcollection/SMSSpamCollection', \
    sep='\t', header=None)
data.columns = ['label', 'msg']
print(data.head())

# Create Feature: Message Length
data['msg_len'] = data['msg'].apply(lambda x: len(x))
print('\nCreate Feature => Message Length: ')
print(data.head())

# Create Feature: Punctuation Usage
import string
def punctuation_count(txt):
    # Loop all the character if character inside the punctuation list
    # then return 1. Take sum of all 1 to be the count.
    count = sum([1 for c in txt if c in string.punctuation])
    # return count / length of the text message * 100 for percentage (%)
    return 100*count/len(txt)

data['punctuation_%'] = data['msg'].apply(lambda x: punctuation_count (x))
print('\nCreate Feature => Punctuation Usage: ')
print(data.head())

# Evaluate Created Features
from matplotlib import pyplot
import numpy as np
#%matplotlib inline
