import nltk
wn = nltk.WordNetLemmatizer ()
ps = nltk.PorterStemmer()
print ('\ndir(wn):')
print(dir(wn))
print("\nps.stem('goose'):", ps.stem('goose'))
print("ps.stem('geese'):", ps.stem('geese'))

print("\nwn.lemmatize('goose'):", wn.lemmatize('goose'))
print("wn.lemmatize('geese'):", wn.lemmatize('geese'))

print("\nps.stem('cactus'):", ps.stem('cactus'))
print("ps.stem('cacti'):", ps.stem('cacti'))

print("\nwn.lemmatize('cactus'):", wn.lemmatize('cactus'))
print("wn.lemmatize('cacti):", wn.lemmatize('cacti'))

print("\nps.stem('code'):", ps.stem('code'))
print("ps.stem('coder'):", ps.stem('coder'))
print("ps.stem('coding'):", ps.stem('coding'))

print("\nwn.lemmatize('code'):", wn.lemmatize('code'))
print("wn.lemmatize('coder):", wn.lemmatize('coder'))
print("wn.lemmatize('coding):", wn.lemmatize('coding'))