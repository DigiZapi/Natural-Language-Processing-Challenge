import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler


# Random Forest
def model_rf_train(tfidf_matrix_train, tfidf_matrix_val, data_train_label, data_val_label):
    # Random forest model
    model_randomf = RandomForestClassifier(n_estimators=256, random_state=42)

    model_randomf.fit(tfidf_matrix_train, data_train_label)
    pred = model_randomf.predict(tfidf_matrix_val)

    # Evaluate random forest
    print("Random forest model")
    print("accuracy:", metrics.accuracy_score(data_val_label, pred))
    print("Classification report:\n", metrics.classification_report(data_val_label, pred))


# Multinomial Naive Bayes (MultinomialNB) classifier
def model_multinominalNB_train(data_train, data_val, data_train_label, data_val_label):
    
    # Multinomial Naive Bayes (MultinomialNB) classifier
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(data_train, data_train_label)
    pred = model.predict(data_val)

    # Evaluate model
    print("Test")
    print("accuracy:", metrics.accuracy_score(data_val_label, pred))
    print("Classification report:\n", metrics.classification_report(data_val_label, pred))


# Simple Feedforward NN
def model_sfnn_train(x_train, y_train, x_val, y_val):

    # Convert your TF-IDF matrices to dense arrays
    x_train_dense = x_train.toarray()
    x_val_dense = x_val.toarray()

    #scaler = StandardScaler()gt
    #x_train_dense = scaler.fit_transform(x_train_dense)
    #x_val_dense = scaler.transform(x_val_dense)

    #y_train_cat = to_categorical(y_train)
    #y_val_cat = to_categorical(y_val)

    # Build simple feedforward NN
    model = Sequential()
    model.add(Dense(1028, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(1028, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(1028, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

    # Train model
    history = model.fit(x_train_dense, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)
    # Evaluate
    loss, accuracy = model.evaluate(x_val_dense, y_val)
    print(f'Test Accuracy: {accuracy:.4f}')

    