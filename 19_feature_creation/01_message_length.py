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
