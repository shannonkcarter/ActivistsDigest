from flask import Flask, request, flash, render_template
import pandas as pd
# importing the functions file
import sim_tool as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk.data
import pandas as pd
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

# Home Page
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("home.html")

@app.route('/pitchidea', methods=['POST'])
def activistsdigest():
    if request.method == 'POST':
        for aKey in request.form:
            if aKey == 'pitch':
                text = request.form[aKey]
        # text = request.form('pitch')
        print(request.form['pitch'])
        print(text)

        data, emb_model = st.load_data()
    # accept user input
        # text = input("Pitch your idea here: ")

    # clean and tokenize user input
        clean_input = st.clean_text(text)


        cosine_list = []
        for vector in data['vector_average']:
            vector.reshape(1, -1)
            cosine_sim = cosine_similarity(vector.reshape(1, -1), clean_input[0].reshape(1, -1))[0, 0]
            #cosine_sim = get_cosine(meeting_data['vector_average'][0], vectorized_input[0])[0, 0]

            cosine_list.append(cosine_sim)

        # add input_veclist to df
        data['cosine_similarity'] = cosine_list

        # sort dataframe in ascending order by distance
        data1 = data.sort_values(by=['cosine_similarity'], ascending = False)
        data2 = data1.drop('vector_average', 1)
        data2 = data2.drop('cosine_similarity', 1)
        data2 = data2.rename(columns={"text": "Public Speaker Comment", "date": "Meeting Date"})
        # return df for top 10 rows
        #results = data1.head(10)
        html_table = data2.head(10).to_html(index = False)
        #results = data1.head(10).to_html()

    # return print(results)
    return render_template("output.html", html_table=html_table, text=text)
        # if form.validate():
        #     flash(idea)
        # else:
        #     flash('Idea description is required. ')
        # return render_template("home.html", form = inputtext)


if __name__ == '__main__':
    app.run(debug=True)
