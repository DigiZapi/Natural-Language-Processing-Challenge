import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.decomposition import TruncatedSVD

# Random Forest
def model_rf_train(tfidf_matrix_train, tfidf_matrix_val, data_train_label, data_val_label):
    # Random forest model
    model = RandomForestClassifier(n_estimators=80, random_state=42)

    model.fit(tfidf_matrix_train, data_train_label)
    pred_train = model.predict(tfidf_matrix_train)
    pred_val = model.predict(tfidf_matrix_val)

    # print train accuracy
    train_accuracy = accuracy_score(data_train_label, pred_train)
    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

    # Evaluate random forest
    print("Random forest model")
    print("accuracy:", metrics.accuracy_score(data_val_label, pred_val))
    print("Classification report:\n", metrics.classification_report(data_val_label, pred_val))

    return model


# Multinomial Naive Bayes (MultinomialNB) classifier
def model_multinominalNB_train(data_train, data_val, data_train_label, data_val_label):
    
    # Multinomial Naive Bayes (MultinomialNB) classifier
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(data_train, data_train_label)
    pred_val = model.predict(data_val)
    pred_train = model.predict(data_train)

    # print train accuracy
    train_accuracy = accuracy_score(data_train_label, pred_train)
    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

    # Evaluate model
    print("Test")
    print("accuracy:", metrics.accuracy_score(data_val_label, pred_val))
    print("Classification report:\n", metrics.classification_report(data_val_label, pred_val))

    return model


# Simple Feedforward NN
def model_sfnn_train(tfidf_train, y_train, tfidf_val, y_val, n_components=1000):
    
    # TruncatedSVD with randomized algorithm for large sparse matrices
    svd = TruncatedSVD(n_components=n_components, algorithm='randomized', random_state=42)
    X_train_svd = svd.fit_transform(tfidf_train)
    X_val_svd = svd.transform(tfidf_val)

    model = Sequential([
        Dense(256, activation='relu', input_shape=(n_components,)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # change to 'softmax' if multi-class
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',  # or 'sparse_categorical_crossentropy'
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('best_sfnn_model.keras', monitor='val_loss', save_best_only=True)
    ]

    history = model.fit(
        X_train_svd, y_train,
        validation_data=(X_val_svd, y_val),
        epochs=20,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )


    # Predictions
    train_pred = (model.predict(X_train_svd) > 0.5).astype(int)
    val_pred = (model.predict(X_val_svd) > 0.5).astype(int)

    # Train evaluation
    train_accuracy = accuracy_score(y_train, train_pred)
    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

    # Validation evaluation
    val_accuracy = accuracy_score(y_val, val_pred)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    print("Validation Classification Report:\n", classification_report(y_val, val_pred))


    return model

"""
LOGISTIC REGRESSION 
"""
def model_logistic_regression(x_train, y_train, x_val, y_val):

    # Initialize the classifier
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

    # Convert your TF-IDF matrices to dense arrays
    x_train_dense = x_train.toarray()
    x_val_dense = x_val.toarray()

    # Train the classifier
    model.fit(x_train, y_train)

    # Predict classes and get training accuracy
    y_train_pred = model.predict(x_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"\nTrain Accuracy: {train_accuracy * 100:.2f}%")

    # Predict classes for unseen validation set and get validation accuracy
    y_val_pred = model.predict(x_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"\nValidation Accuracy: {val_accuracy * 100:.2f}%")

    # Classification report
    print("\nClassification Report:\n", classification_report(y_val, y_val_pred))

    return model

"""
NAIVES BAYES
"""
def model_naives_bayes(x_train, y_train, x_val, y_val):

    nb = MultinomialNB(alpha=1.0)
    nb.fit(x_train, y_train)

    y_val_pred = nb.predict(x_val)
    acc = accuracy_score(y_val, y_val_pred)

    # Predict classes and get training accuracy
    y_train_pred = nb.predict(x_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print("Train accuracy:", train_accuracy)

    print(f"\n Validation Accuracy:{acc:4f}\n")
    print("classification_report:\n", classification_report(y_val, y_val_pred, digits=4))

    return nb



def predict_values(model, test_data, df_test, filepath):
    
    pred = model.predict(test_data)

    df_write = pd.DataFrame(columns=["label", "text"])
    df_write['text'] = df_test['text']
    df_write['label'] = pred

    df_write.to_csv(filepath, index=False, header=False) 

    print("✅ Predictions saved to:", filepath)