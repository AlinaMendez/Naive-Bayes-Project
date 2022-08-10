# Import libraries
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load data
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews_dataset.csv')

# Clean
df_raw.drop('package_name', axis=1, inplace= True)
df_raw['review'] = df_raw['review'].str.lower() 

# Separate target from feature
X = df_raw['review']
y = df_raw['polarity']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Vectorize
vec = CountVectorizer(stop_words='english')
X_train = vec.fit_transform(X_train).toarray()
X_test = vec.transform(X_test).toarray()

# Make the model
model = MultinomialNB()
model.fit(X_train, y_train)
model.score(X_test, y_test)

# Seave the model
filename = '../models/Naive_Bayes_Model.sav'
pickle.dump(model, open(filename, 'wb'))


