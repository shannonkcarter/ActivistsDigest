
import csv
import scipy
import six
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import string
import re
import smart_open as so
import gensim
from gensim.models import KeyedVectors
from flask import render_template
#from flaskexample import app
from flask import Flask, request, render_template
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk.data
import pandas as pd
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Load Data
def load_data():
    data = pd.read_pickle('pickleddf.pkl')
    filename = 'GoogleNews-vectors-negative300.bin'
    emb_model = KeyedVectors.load_word2vec_format(filename, binary=True)
    return data, emb_model

# Define a function that cleans strings: lowercase, remove select punctuation, tokenize
#def clean_text(my_str):
#    str_low = my_str.lower()
#    remove = string.punctuation
#    remove = remove.replace(".", " ").replace(",", " ").replace("'", " ")
#    pattern = r"[{}]".format(remove) # create the pattern
#    cleanstr = re.sub(pattern, " ", str_low)
#    finalword = word_tokenize(cleanstr)
#    return finalword
filename1 = 'GoogleNews-vectors-negative300.bin'
emb_model1 = KeyedVectors.load_word2vec_format(filename1, binary=True)

def clean_text(my_str):
    str_low = my_str.lower()
    remove = string.punctuation
    finalword = nltk.word_tokenize(str_low)
    finalword = emb_model1[finalword]
    return finalword
