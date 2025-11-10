🧠 Emotion Detection using NLP
📌 Overview

This project detects emotions from text using Natural Language Processing (NLP) techniques with Bag of Words (BoW) and TF-IDF features.
Models used: Logistic Regression and Multinomial Naive Bayes.

⚙️ Steps

Data Preprocessing

Cleaned text (lowercase, remove stopwords, punctuation).

Tokenized and lemmatized.

Feature Extraction

Used Bag of Words and TF-IDF to convert text into numerical form.

Model Training

Trained Logistic Regression and MultinomialNB models.

Evaluated using accuracy, precision, recall, and F1-score.

Prediction Example

text = "I am feeling very happy today!"
pred = model.predict(vectorizer.transform([text]))
print(pred[0])

🧩 Tools & Libraries

Python

Scikit-learn

NLTK

Pandas, NumPy

Matplotlib / Seaborn

📊 Results
Model	Accuracy
Logistic Regression	86%
MultinomialNB	76%
💡 Future Work

Use deep learning (LSTM/BERT).

Deploy using Flask or Streamlit.
