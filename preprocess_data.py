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


#consider removing
def tfidf_vec_fit_transform(data):
    
    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
  
    tfidf_matrix = vectorizer.fit_transform(data)
    return tfidf_matrix

#consider removing
def tfidf_vec_transform(data):
    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    tfidf_matrix = vectorizer.transform(data)
    return tfidf_matrix


def count_vectorizer(data):
    
    bow_vect = CountVectorizer(max_features=1000)

    return bow_vect.fit_transform(data)

def show_top_words(data):

    # Count word frequency
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data)

    word_counts = X.sum(axis=0)

    # Convert to DataFrame for sorting
    word_freq = pd.DataFrame(
        [(word, word_counts[0, idx]) for word, idx in vectorizer.vocabulary_.items()],
        columns=["word", "count"]
    )

    # Sort and show Top 10
    top10 = word_freq.sort_values(by="count", ascending=False).head(10)
    print(top10)
