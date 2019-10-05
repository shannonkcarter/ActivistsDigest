from flask import Flask, request, flash, render_template
import pandas as pd
import numpy as np
import string
import re
import pickle
import scipy
import nltk
import nltk.data
from nltk.tokenize import word_tokenize
import gensim
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Load Data and Model
def load_data():
    data = pd.read_pickle('pickleddf2.pkl')
    filename = 'GoogleNews-vectors-negative300-SLIM.bin'
    emb_model = KeyedVectors.load_word2vec_format(filename, binary=True)
    return data, emb_model

# Function to clean and tokenize user input
def clean_text(my_str):
    str_low = my_str.lower()
    remove = string.punctuation
    finalword = nltk.word_tokenize(str_low)
    finalword = emb_model[finalword]
    return finalword

app = Flask(__name__)
data, emb_model = load_data()

# Home Page
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("home.html")

@app.route('/go', methods=['POST'])
def activistsdigest():
    global data
    global emb_model
    if request.method == 'POST':
        for aKey in request.form:
            if aKey == 'pitch':
                text = request.form[aKey]
        print(request.form['pitch'])
        print(text)

        # clean and tokenize user input
        clean_input = clean_text(text)

        # calculate cosine similarity
        cosine_list = []
        for vector in data['vector_average']:
            vector.reshape(1, -1)
            cosine_sim = cosine_similarity(vector.reshape(1, -1), clean_input[0].reshape(1, -1))[0, 0]
            #cosine_sim = get_cosine(meeting_data['vector_average'][0], vectorized_input[0])[0, 0]
            cosine_list.append(cosine_sim)

        # add cosine similarity to the data frame
        data['cosine_similarity'] = cosine_list

        # sort dataframe in ascending order by cosine similarity
        data1 = data.sort_values(by=['cosine_similarity'], ascending = False)
        data2 = data1.drop('vector_average', 1)
        data2 = data2.drop('cosine_similarity', 1)
        data2 = data2.head(15)
        data2 = data2.rename(columns={"text": "Public Speaker Comment", "date": "Meeting Date"})

    # return the created df and the user input
    return render_template("output.html", data2=data2, text=text)

if __name__ == '__main__':
    app.run(debug=True)
