# Fake-news-detector
fake news detector app (only for trained data)

# 📰 Fake News Detection Using NLP and Machine Learning

This project focuses on detecting **Fake News** using advanced Natural Language Processing (NLP) techniques and machine learning models. It takes in a news article and classifies it as either **Real** or **Fake**.

## 🚀 Project Overview

With the exponential growth of online content, the spread of misinformation (fake news) has become a major issue. This project aims to solve that problem using machine learning, text preprocessing, and NLP methods. The solution is trained on real-world data and can predict the authenticity of a news article.

---

## 🎯 Objective

- Classify news content as **Real** or **Fake**.
- Use **NLP preprocessing** to clean and transform text.
- Apply **machine learning models** like Logistic Regression, SVM, and Random Forest.
- Create a **Gradio interface** for easy interaction.
- Deploy the model on **Hugging Face Spaces**.

---

## 📚 Dataset

- **Name**: Fake and Real News Dataset
- **Source**: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Files**:
  - `Fake.csv`
  - `True.csv`

---

## 🛠️ Tech Stack

| Component        | Tool/Library        |
|------------------|---------------------|
| Programming      | Python              |
| NLP              | NLTK                |
| Vectorization    | TF-IDF              |
| ML Algorithms    | Logistic Regression, SVM, Random Forest |
| Model Training   | Scikit-learn        |
| Web Interface    | Gradio              |
| Deployment       | Hugging Face Spaces |
| Data Handling    | Pandas              |

---

## 🧪 Features

- Preprocessing using:
  - Lowercasing
  - Punctuation removal
  - Tokenization
  - Stopwords removal
  - Stemming
- ML Models:
  - Logistic Regression (default)
  - Optional: SVM, Random Forest
- TF-IDF based text vectorization
- Real-time prediction using Gradio
- Model saved using Pickle for reuse

---

## 🧼 NLP Preprocessing Pipeline

1. Convert text to lowercase
2. Remove punctuation
3. Tokenize sentences into words
4. Remove stopwords using NLTK
5. Apply stemming using PorterStemmer
6. Recombine tokens into cleaned text

---

## 🔍 Model Training Pipeline

1. Combine `Fake.csv` and `True.csv`
2. Preprocess the text using the NLP pipeline
3. Vectorize cleaned text using `TfidfVectorizer`
4. Train on split dataset using:
   - Logistic Regression (default)
   - (Optional) SVM or RandomForestClassifier
5. Save trained model and vectorizer to `model.pkl` and `vectorizer.pkl`

---
## Hugging face app
https://huggingface.co/spaces/jeevitha-app/Fake_news_detector_app

