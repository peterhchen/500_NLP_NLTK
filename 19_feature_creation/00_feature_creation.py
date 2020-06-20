# Read in Raw Text
import pandas as pd

data = pd.read_csv('../data/smsspamcollection/SMSSpamCollection', \
    sep='\t', header=None)
data.columns = ['label', 'msg']
print(data.head())

# Create Feature: Message Length

# Create Feature: Punctuation Usage
