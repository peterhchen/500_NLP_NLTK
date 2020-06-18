import nltk

# Read dataset
# 1) open()
# 2) pandas read_csv()

raw_data = open('../data/smsspamcollection/SMSSpamCollection').read()
print('\nraw_data[0:200]')
print(raw_data[0:200])

# Parse raw data into parsed array
parsed_data = raw_data.replace('\t', '\n').split('\n')
print('\nparsed_data[0:10]')
print(parsed_data[0:10])

# assign parsed array into label list and msg list.
label_list = parsed_data[0::2]  # start = 0, stop = end, step = 2
print('\nlabel_list[0:5]')
print(label_list[0:5])
msg_list = parsed_data[1::2]  # start = 1, stop = end, step = 2
print('\nmsg_list[0:5]')
print(msg_list[0:5])

# Now, combined the label list and message list into pandas DataFrame.
import pandas as pd
print ('len(label_list):', len(label_list))
print ('len(msg_list):', len(msg_list))
# https://stackoverflow.com/questions/509211/understanding-slice-notation
# -1 is the last element.
print('label_list[-3:]', label_list[-3:])   # print last [-3], [-2], [-1]

combined_df = pd.DataFrame ({
    'label': label_list[:-1],   # stop the before last element (-1).
    'sms': msg_list
})
print('\ncombined_df.head():')
print(combined_df.head())