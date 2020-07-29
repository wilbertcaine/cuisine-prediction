import json
import pandas as pd
train = pd.read_json('../input/whats-cooking-kernels-only/train.json')
test = pd.read_json('../input/whats-cooking-kernels-only/test.json')

from sklearn.model_selection import train_test_split
X_train = list(train['ingredients'])
y_train = list(train['cuisine'])
X_test = list(test['ingredients'])

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

from nltk.stem.porter import *
stemmer = PorterStemmer()

from nltk.corpus import stopwords
def preprocess_ingredient(ingredients):
    for ingredient in ingredients:
        for i in range(len(ingredient)):
            x = ingredient[i] #'Bertolli® Classico Olive Oil', '(10 oz.) frozen chopped spinach, thawed and squeezed dry' ,'leg of lamb', 'lamb leg'
            x = x.lower() #'bertolli® classico olive oil', '(10 oz.) frozen chopped spinach, thawed and squeezed dry' ,'leg of lamb', 'lamb leg'
            x = re.sub("[^a-z ]", "", x) #'bertolli classico olive oil', ' oz frozen chopped spinach thawed and squeezed dry' ,'leg of lamb', 'lamb leg'
            word_tokens = word_tokenize(x)
            if 'oz' in word_tokens:
                word_tokens.remove('oz')
            filtered_words = [w for w in word_tokens if not w in stop_words] 
            filtered_words.sort() #['bertolli', 'classico', 'oil', 'olive'], ['chopped', 'dry', 'frozen', 'spinach', 'squeezed', 'thawed'], ['lamb', 'leg'], ['lamb', 'leg']
            stemmed_word = [stemmer.stem(word) for word in filtered_words]
            x = ' '.join(stemmed_word) #'bertolli classico oil oliv', 'chop dri frozen spinach squeez thaw' ,'lamb leg', 'lamb leg'
            ingredient[i] = x
            
preprocess_ingredient(X_train)
preprocess_ingredient(X_test)

vocabs = set()
for ingredient in X_train:
    vocabs.update(ingredient)

def create_bag_of_words(ingredients):
    data_features = list()
    for ingredient in ingredients:
        features = list()
        for item in vocabs:
            features.append(item in ingredient)
        data_features.append(features)
    return data_features

train_data_features = create_bag_of_words(X_train)

def train_logistic_regression(features, label):
    from sklearn.linear_model import LogisticRegression
    ml_model = LogisticRegression(C = 2,random_state = 0)
    ml_model.fit(features, label)
    return ml_model

ml_model = train_logistic_regression(train_data_features, y_train)

import pickle
with open('cuisine_prediction_model.pkl', 'wb') as file:
    pickle.dump(ml_model, file)
    
with open('cuisine_prediction_model.pkl', 'rb') as file:  
    pickled_ml_model = pickle.load(file)

test_data_features = create_bag_of_words(X_test)
predicted_y = pickled_ml_model.predict(test_data_features)

test['cuisine'] = predicted_y
test[['id', 'cuisine']].to_csv('submission.csv', index=False)
test[['id', 'cuisine']].head()