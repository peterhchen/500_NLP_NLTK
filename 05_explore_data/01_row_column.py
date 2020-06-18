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

# Explore Data
# 1) Shape of Data: How many rows and columns are in the dataset?
print(f'Input dataset {len(dataset)} row and {len(dataset.columns)}')
