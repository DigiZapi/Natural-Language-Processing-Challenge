# Fake News Detection — AI Engineering Bootcamp Project

This project explores multiple machine learning and deep learning approaches to **classify news headlines as fake or real** using natural language processing (NLP) techniques.  

It was developed as part of an **AI Engineering Bootcamp**, focusing on experimentation with preprocessing pipelines, text vectorization methods, and model performance evaluation.

---

## 📁 Project Overview

The project includes:

- **Text preprocessing** (cleaning, lemmatization)
- **Feature extraction** using:
  - TF-IDF with and without stopword removal  
  - Sentence embeddings (`all-MiniLM-L6-v2` from SentenceTransformers)
- **Model training and evaluation** with:
  - Naive Bayes
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Simple Feedforward Neural Network (Keras)
- **Model comparison** across preprocessing variants

---

## ⚙️ Data

- **Training/validation data:** `data/training_data_lowercase.csv` (34,152 samples)  
- **Test data:** `data/testing_data_lowercase_nolabels.csv` (9,984 samples)  
- Data split: 92% training / 8% validation (stratified by label)

Each dataset includes:
- `label` — binary target (0 = real, 1 = fake)
- `text` — news headline or short text sample

---

## 🧹 Preprocessing Pipelines

| Setup | Description |
|--------|--------------|
| **1. Clean + Lemmatize + TF-IDF** | Custom cleaning and lemmatization, bigram TF-IDF |
| **2. TF-IDF (raw text)** | Unprocessed text with 1–3 grams |
| **3. TF-IDF (stopwords removed)** | Removes English stopwords |
| **4. Sentence Embeddings** | Encoded with `all-MiniLM-L6-v2` (384-dim vectors) |

---

## 🤖 Models & Results

| Model | Best Preprocessing | Train Accuracy | Validation Accuracy | Notes |
|--------|---------------------|----------------|---------------------|--------|
| **Naive Bayes** | TF-IDF (raw text) | 1.00 | **0.95** | Excellent baseline performance |
| **Logistic Regression** | TF-IDF (raw text) | 97.6% | **0.94** | Strong generalization; stable |
| **Random Forest** | TF-IDF | 100% | 0.91 | Overfitting tendency |
| **XGBoost** | Sentence Embeddings | 0.99 | **0.93** | Performs well on dense embeddings |
| **Simple Feedforward NN** | TF-IDF (raw text) | 95.8% | **0.94** | Matched Logistic Regression performance |

---

## 🏁 Key Findings

- **Naive Bayes** provided the **highest validation accuracy (95%)** using TF-IDF features on raw text — likely due to its suitability for sparse count-based features.  
- **Logistic Regression** and **Feedforward NN** closely followed with ~94% validation accuracy, showing robustness across setups.  
- **XGBoost with embeddings** performed best among gradient boosting models, reaching 93% accuracy with strong recall on both classes.  
- **Random Forests** showed overfitting, achieving perfect training accuracy but slightly lower validation results (≈90%).  
- **Text embeddings** (MiniLM) improved results for complex models but didn’t outperform TF-IDF for simpler classifiers.  

Overall, **TF-IDF vectorization with Logistic Regression or Naive Bayes** provided the most consistent and interpretable performance.

---

## 📦 Repository Structure

- data/
  - training_data_lowercase.csv
  - testing_data_lowercase_nolabels.csv
  - testing_data_lowercase_labels.csv
- preprocess_data.py # Cleaning, lemmatization, and vectorization functions
- models.py # Model training and evaluation functions
- main.ipynb / main.py # Main execution and experiments
- README.md # Project documentation

---

## 📊 Summary

| Category | Best Performing Model | Validation Accuracy |
|-----------|-----------------------|---------------------|
| Classical ML | **Naive Bayes (TF-IDF)** | **95%** |
| Neural Model | **Simple FFNN (TF-IDF)** | **94%** |
| Embedding-based | **XGBoost (MiniLM)** | **93%** |

➡️ **Conclusion:** Simpler TF-IDF-based models outperform more complex architectures for this dataset — emphasizing the effectiveness of well-tuned traditional NLP pipelines.
