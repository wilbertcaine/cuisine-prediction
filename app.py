from __future__ import print_function # In python 2.7
import sys
from flask import Flask, render_template, request
import pandas as pd

# import pickle
# with open(f'model/cuisine_prediction_model.pkl', 'rb') as f:
#     model = pickle.load(f)
from joblib import dump, load
model = load(f'model/cuisine_prediction_model.joblib') 

app = Flask(__name__, template_folder='templates')

@app.route("/")
def home():
    return render_template("main.html")

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
def preprocess_ingredient(ingredients):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
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
            
with open(f'model/vocabs.pkl', 'rb') as f:
    vocabs = pickle.load(f)

def create_bag_of_words(ingredients):
    data_features = list()
    for ingredient in ingredients:
        features = list()
        for item in vocabs:
            features.append(item in ingredient)
        data_features.append(features)
    return data_features

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/png')

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    my_list = []
    my_list.append(list(userText.split(","))) 
    preprocess_ingredient(my_list)
    print(my_list, file=sys.stderr)
    test_data_features = create_bag_of_words(my_list)
    return str(model.predict(test_data_features))

if __name__ == '__main__':
    app.run()
