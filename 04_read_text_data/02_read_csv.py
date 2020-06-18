import nltk
import pandas as pd

# Method 2: Read data with read_csv()
dataset = pd.read_csv('../data/smsspamcollection/SMSSpamCollection', \
    sep='\t', header=None)
print('dataset.head():')
print(dataset.head()) 