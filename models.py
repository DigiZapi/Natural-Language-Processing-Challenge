import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline




# Random Forest
def model_rf_train(tfidf_matrix_train, tfidf_matrix_val, data_train_label, data_val_label):
    # Random forest model
    model_randomf = RandomForestClassifier(n_estimators=100, random_state=42)

    model_randomf.fit(tfidf_matrix_train, data_train_label)
    pred = model_randomf.predict(tfidf_matrix_val)

    # Evaluate random forest
    print("Random forest model")
    print("accuracy:", metrics.accuracy_score(data_val_label, pred))
    print("Classification report:\n", metrics.classification_report(data_val_label, pred))


def model_mnb_train():
    
    pass
