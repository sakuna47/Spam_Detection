Spam Detection Web App

Overview

This project is a Spam Detection Web App built using Python, Scikit-Learn, NLTK, and Streamlit. The model uses Natural Language Processing (NLP) techniques and Machine Learning to classify messages as either spam or ham (not spam).

Features

Text Preprocessing: Converts messages to lowercase, removes stopwords, and filters special characters.

TF-IDF Vectorization: Transforms text data into numerical vectors for machine learning.

Naïve Bayes Classification: A trained Multinomial Naïve Bayes model to detect spam.

Web Interface: A Streamlit-based UI where users can enter a message and get a classification result.

Model Persistence: The trained model and vectorizer are saved using Pickle for reuse.

Google Drive Integration: Loads dataset and saves model files directly on Google Drive.

Tech Stack

Python (pandas, numpy, nltk, scikit-learn, pickle)

Machine Learning (TF-IDF, Naïve Bayes classifier)

Natural Language Processing (NLP)

Streamlit (for web interface)

Flask (for backend deployment, if needed)

Installation & Setup

Clone the repository

git clone https://github.com/your-username/spam-detection-web-app.git
cd spam-detection-web-app

Install dependencies

pip install pandas numpy scikit-learn nltk flask streamlit

Download NLTK stopwords

import nltk
nltk.download("stopwords")

Run the Streamlit Web App

streamlit run app.py

Usage

Enter a text message in the web app.

Click Predict to classify the message as spam or ham.

The model will provide an instant prediction.

Dataset

The dataset is loaded from a CSV file stored in Google Drive. It contains labeled messages categorized as spam or ham.

Model Training

The dataset is preprocessed by removing stopwords and special characters.

Text data is transformed using TF-IDF Vectorization.

A Multinomial Naïve Bayes model is trained for classification.

The model and vectorizer are saved using Pickle.

Future Improvements

Expand dataset for better accuracy.

Improve text preprocessing with lemmatization.

Deploy using Flask & Docker for a production-ready API.

Implement Deep Learning models for better classification.

Contributors

Your Name (GitHub Profile)

License

This project is licensed under the MIT License.
