import pandas as pd
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import string
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer



def clean_text(text):
    
    # cleaning "unnessesary" characters
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)      # Remove all special characters
    text = re.sub(r'\d+', '', text)                 # Remove numbers
    text = re.sub(r'\b[a-zA-Z]\b', '', text)        # Remove single characters
    text = re.sub(r'\s+', ' ', text)                # Substitute multiple spaces with single space
    text = text.lower()                             # Convert to lowercase
  
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))                # Get the list of English stopwords
    words = word_tokenize(text) 
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    
    return words


def lemmatize_text(text):
    
    # init lemmatizer
    lemmatizer = WordNetLemmatizer()

    # lemmatize and convert from list to string
    return ' '.join([lemmatizer.lemmatize(word) for word in text])



def tfidf_vectorization(data):
    
    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer(min_df = 0.1)
  
    tfidf_matrix = vectorizer.fit_transform(data)
    return tfidf_matrix



def count_vectorizer(data):
    
    bow_vect = CountVectorizer(max_features=1000)

    return bow_vect.fit_transform(data)

