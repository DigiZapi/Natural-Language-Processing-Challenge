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
    
    """
    Clean and preprocess a text string for NLP tasks.

    This function performs several preprocessing steps on the input text:
    - Removes special characters, numbers, and single-character words
    - Converts text to lowercase
    - Removes extra whitespace
    - Tokenizes text into words
    - Removes English stopwords

    Parameters
    ----------
    text : str
        The raw text string to be cleaned.

    Returns
    -------
    words : list of str
        A list of cleaned, tokenized words with stopwords removed.
    """


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

    """
    Lemmatize a list of words and return a single cleaned string.

    This function applies WordNet lemmatization to each word in the input list,
    reducing words to their base or dictionary form, and then joins them into
    a single space-separated string.

    Parameters
    ----------
    text : list of str
        Tokenized words to be lemmatized.

    Returns
    -------
    lemmatized_text : str
        A single string containing all lemmatized words separated by spaces.
    """
    
    # init lemmatizer
    lemmatizer = WordNetLemmatizer()

    # lemmatize and convert from list to string
    return ' '.join([lemmatizer.lemmatize(word) for word in text])


def count_vectorizer(data):
    """
    Convert a collection of text documents into a Bag-of-Words (BoW) feature matrix.

    This function initializes a CountVectorizer with a maximum of 1000 features
    and transforms the input text data into a sparse BoW representation.

    Parameters
    ----------
    data : list of str or pandas.Series
        Collection of text documents to be vectorized.

    Returns
    -------
    X : scipy.sparse.csr_matrix
        Sparse matrix of shape (n_samples, n_features) containing token counts
    """
    bow_vect = CountVectorizer(max_features=1000)

    return bow_vect.fit_transform(data)

def show_top_words(data):

    """
    Display the top 10 most frequent words in a collection of text documents.

    This function vectorizes the input text using a CountVectorizer, counts the
    occurrences of each word, and prints a DataFrame containing the top 10 words
    sorted by frequency.

    Parameters
    ----------
    data : list of str or pandas.Series
        Collection of text documents to analyze.

    Returns
    -------
    None
        Prints a DataFrame of the top 10 words and their counts.
    """

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
