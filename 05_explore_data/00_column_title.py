import nltk
import pandas as pd

# Read dataset
dataset = pd.read_csv('../data/smsspamcollection/SMSSpamCollection', \
    sep='\t', header=None)
print('dataset.head():')
print(dataset.head()) 

dataset.columns=['label', 'sms']
print('set columns of dataset => dataset.head():')
print (dataset.head())