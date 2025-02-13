# Fake News Detection System

## Overview
This project is a **Fake News Detection System** that classifies news articles as either **Real** or **Fake** using **Natural Language Processing (NLP)** and **Machine Learning**. It includes:
- A **Jupyter Notebook** for training a **Logistic Regression** model using **TF-IDF** feature extraction.
- A **Streamlit Web App** for users to input news articles and get real-time predictions.

## Features
- **Text Preprocessing**: Converts text to lowercase, removes special characters, stems words, and eliminates stopwords.
- **TF-IDF Vectorization**: Extracts the top 5000 most important words.
- **Machine Learning Model**: Logistic Regression classifier for accurate predictions.
- **Model Deployment**: A **Streamlit** web app to check if a news article is real or fake.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download NLTK stopwords:
   ```python
   import nltk
   nltk.download('stopwords')
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Train the model using the Jupyter Notebook (`fake_news_detection.ipynb`).
2. Save the trained model and vectorizer.
3. Run `app.py` and enter a news article to check its authenticity.

## Dataset
- **Real News**: Sourced from legitimate news websites.
- **Fake News**: Collected from unreliable sources.
- Both datasets are labeled (`0 = Real`, `1 = Fake`).

## Technologies Used
- **Python**
- **Pandas, NumPy, Scikit-Learn** (for data processing & ML)
- **NLTK** (for text preprocessing)
- **Matplotlib, Seaborn** (for data visualization)
- **Streamlit** (for web app deployment)

## Results
- Model Accuracy: **~95%**
- **Confusion Matrix & Classification Report** provided in the notebook.

## Future Improvements
- Improve model with deep learning (**LSTM, BERT**).
- Add multilingual support for news detection.
- Deploy as a cloud-based API for broader use.

## Contributors
- **sakuna47** ([GitHub](https://github.com/sakuna47e))

## License
This project is open-source under the **MIT License**.

