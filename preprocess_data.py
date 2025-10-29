import pandas as pd
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import string
from nltk.corpus import stopwords

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
    
    lemmatizer = WordNetLemmatizer()

    return [lemmatizer.lemmatize(word) for word in text]
    
    #train_data['lemmatized_tokens'] = train_data['cleaned_text'].apply(lambda x: [lemmatizer.lemmatize(token) for token in x])
    #test_data['lemmatized_tokens'] = test_data['cleaned_text'].apply(lambda x: [lemmatizer.lemmatize(token) for token in x])

    #print(train_data['lemmatized_tokens'])