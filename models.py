import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
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
import xgboost as xgb

# Random Forest
def model_rf_train(tfidf_matrix_train, tfidf_matrix_val, data_train_label, data_val_label):
    
    """
    Train and evaluate a RandomForestClassifier on TF-IDF feature matrices.

    This function trains a Random Forest classifier on the provided training TF-IDF
    matrix and labels, evaluates it on both training and validation sets, and prints
    accuracy and a classification report for performance inspection.

    Parameters
    ----------
    tfidf_matrix_train : csr_matrix
        Sparse TF-IDF matrix for the training data.
    tfidf_matrix_val : csr_matrix
        Sparse TF-IDF matrix for the validation data.
    data_train_label : pandas.Series or array-like
        Labels corresponding to the training data.
    data_val_label : pandas.Series or array-like
        Labels corresponding to the validation data.

    Returns
    -------
    model : sklearn.ensemble.RandomForestClassifier
        The trained Random Forest classifier.

    Notes
    -----
    - Prints training and validation accuracy.
    - Prints a full classification report for the validation set.
    - Default parameters: n_estimators=100, random_state=42.
    """

    # Random forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

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


# Simple Feedforward NN
def model_sfnn_train(tfidf_train, y_train, tfidf_val, y_val, n_components=384):
    
    """
    Train a shallow feed-forward neural network on dimensionality-reduced TF-IDF features.

    This function applies TruncatedSVD to reduce the dimensionality of sparse TF-IDF
    matrices and then trains a neural network for binary text classification.
    The model includes dropout for regularization and early stopping/checkpointing
    to prevent overfitting.

    Parameters
    ----------
    tfidf_train : csr_matrix
        Sparse TF-IDF matrix for the training set.
    y_train : array-like
        Labels for the training data.
    tfidf_val : csr_matrix
        Sparse TF-IDF matrix for the validation set.
    y_val : array-like
        Labels for the validation data.
    n_components : int, optional (default=2000)
        Number of latent components to retain in TruncatedSVD.

    Returns
    -------
    model : keras.Model
        Trained Keras neural network model with saved best weights.
    """


    # TruncatedSVD with randomized algorithm for large sparse matrices
    svd = TruncatedSVD(n_components=n_components, algorithm='randomized', random_state=42)
    X_train_svd = svd.fit_transform(tfidf_train)
    X_val_svd = svd.transform(tfidf_val)

    model = Sequential([
        Dense(512, activation='relu', input_shape=(n_components,)),
        Dropout(0.5),
        Dense(256, activation='relu'),
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

    """
    Train and evaluate a Logistic Regression classifier on TF-IDF feature matrices.

    This function trains a balanced Logistic Regression model to handle datasets with 
    potential class imbalance. After training, it reports accuracy on both training 
    and validation sets, and provides a full classification report.

    Parameters
    ----------
    x_train : csr_matrix or array-like
        TF-IDF features for the training data.
    y_train : array-like
        Labels for the training set.
    x_val : csr_matrix or array-like
        TF-IDF features for the validation data.
    y_val : array-like
        Labels for the validation set.

    Returns
    -------
    model : sklearn.linear_model.LogisticRegression
        Trained logistic regression classifier.

    Notes
    -----
    - Uses `class_weight='balanced'` to adjust for potential class imbalance.
    - `max_iter=1000` increases the default iteration limit to ensure convergence.
    - No dimensionality reduction applied; works directly with sparse TF-IDF input.
    - Validation metrics are printed for quick evaluation.
    """

    # Initialize the classifier
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

    # Convert your TF-IDF matrices to dense arrays
    #x_train_dense = x_train.toarray()
    #x_val_dense = x_val.toarray()

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
    
    """
    Train and evaluate a Multinomial Naive Bayes classifier using hyperparameter tuning.

    This function performs a grid search over different smoothing values (`alpha`)
    to find the best Multinomial Naive Bayes model for sparse TF-IDF input features.
    The optimized model is evaluated on both training and validation sets, with 
    printed metrics for performance inspection.

    Parameters
    ----------
    x_train : csr_matrix or array-like
        TF-IDF matrix for the training data.
    y_train : array-like
        Labels for the training set.
    x_val : csr_matrix or array-like
        TF-IDF matrix for the validation data.
    y_val : array-like
        Labels for the validation set.

    Returns
    -------
    nb : sklearn.naive_bayes.MultinomialNB
        The best-performing Multinomial Naive Bayes classifier selected via GridSearchCV.

    Notes
    -----
    - Uses 5-fold cross-validation to optimize `alpha`, the Laplace smoothing parameter.
    - Printed metrics include: training accuracy, validation accuracy,
      and a full classification report.
    - Best estimator is stored in `gs.best_estimator_`.
    - Assumes input feature values >= 0 (required for MultinomialNB).
    """

    alphas = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5]  # list of numeric values
    params = {'alpha': alphas}
    
    gs = GridSearchCV(MultinomialNB(), params, cv=5)
    gs.fit(x_train, y_train)
    nb = gs.best_estimator_

    y_val_pred = nb.predict(x_val)
    
    # Predict classes and get training accuracy
    y_train_pred = nb.predict(x_train)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_accuracy = round(train_accuracy, 2)
    print(f"Train Accuracy:{train_accuracy:.2f}\n")

    acc = accuracy_score(y_val, y_val_pred)
    acc = round(acc, 2)
    print(f"Validation Accuracy: {acc:.2f}\n")
    print("classification_report:\n", classification_report(y_val, y_val_pred, digits=4))

    return nb


def model_xgboost(X_train, y_train, X_val, y_val):
    
    """
    Train and evaluate an XGBoost classifier with hyperparameter tuning.

    This function performs GridSearchCV to optimize the learning rate (eta)
    for an XGBClassifier configured for binary classification. After training,
    the function evaluates the model on both training and validation sets and
    prints accuracy and a classification report.

    Parameters
    ----------
    X_train : csr_matrix or array-like
        TF-IDF or other numerical features for the training set.
    y_train : array-like
        Labels for training samples.
    X_val : csr_matrix or array-like
        Validation feature matrix.
    y_val : array-like
        Labels for validation samples.

    Returns
    -------
    best_model : xgboost.XGBClassifier
        Best classifier chosen by GridSearchCV.

    Notes
    -----
    - Uses the binary:logistic objective → assumes binary classification.
    - Hyperparameter tuning performed on `eta` (learning rate) with 5-fold CV.
    - Accuracy and a detailed classification report are printed for validation.
    - Parameters in `params` are passed into the XGBClassifier initialization.
      GridSearchCV explores `eta` values defined in `params_grid`.
    """
    
    
    # Convert data to DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Set parameters for XGBoost
    params = {
        'objective': 'binary:logistic',
        'max_depth': 4,
        'eval_metric': 'logloss',
    }
    
    # Implement GridSearchCV for hyperparameter tuning
    alphas = [0.01, 0.1, 0.2, 0.3, 0.4]  # Example alpha values for tuning
    params_grid = {'eta': alphas}
    
    gs = GridSearchCV(xgb.XGBClassifier(**params), params_grid, cv=5)
    
    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_

    # Make predictions
    y_val_pred = best_model.predict(X_val)
    y_train_pred = best_model.predict(X_train)

    # Calculate and print accuracy and classification report
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_accuracy = round(train_accuracy, 2)
    print(f"Train Accuracy: {train_accuracy:.2f}\n")
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_accuracy = round(val_accuracy, 2)
    print(f"Validation Accuracy: {val_accuracy:.2f}\n")
    
    print("Classification Report:\n", classification_report(y_val, y_val_pred, digits=4))



def predict_values(model, test_data, df_test, filepath):

    """
    Generate predictions from a trained model and save them to a CSV file.

    This function uses the provided model to predict labels for the `test_data`,
    pairs the predictions with the corresponding text from `df_test`, and writes
    the results to a CSV file at the specified `filepath`.

    Parameters
    ----------
    model : object
        Trained machine learning model with a `predict` method.
    test_data : array-like or sparse matrix
        Feature representation of the test dataset.
    df_test : pandas.DataFrame
        DataFrame containing the test text in a column named 'text'.
    filepath : str
        File path where the CSV with predictions will be saved.

    Returns
    -------
    None
        Writes a CSV file containing two columns: 'label' and 'text'.
        Prints confirmation of file saving.

    Notes
    -----
    - CSV is saved without an index and without headers.
    - Make sure `df_test` has a 'text' column corresponding to `test_data`.
    - The order of predictions corresponds to the order of `df_test`.
    """
    
    pred = model.predict(test_data)

    df_write = pd.DataFrame(columns=["label", "text"])
    df_write['text'] = df_test['text']
    df_write['label'] = pred

    df_write.to_csv(filepath, index=False, header=False) 

    print("✅ Predictions saved to:", filepath)