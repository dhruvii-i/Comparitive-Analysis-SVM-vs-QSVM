pip install streamlit
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import gdown
from io import BytesIO
import nltk
import pickle 

# Download NLTK resources
nltk.download('punkt')

# Load the TF-IDF model
tfidi = joblib.load('tfidf_vectorizer1.pkl')

# Load the SVM model
svm_model = joblib.load('svm2_model.pkl')

# Load the TF-IDF vectorizer used for training
with open('tfidf_vectorizer1.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

def analysis(input_text, tfidf_vectorizer, svm_model):
    # Preprocess the input data using the same TF-IDF vectorizer
    input_data_features = tfidf_vectorizer.transform([input_text])
    data_features = pd.DataFrame(input_data_features.toarray())

    prediction = svm_model.predict(data_features)
    if prediction[0] == 0:
        return "Positive Sentiment 😍 🥂 🎉"
    elif prediction[0] == 1:
        return "Negative Sentiment 😤 😡 😠"
    else:
        return "Neutral Sentiment 😶 🙂"

def main():
    st.markdown("""
<style>
    /* Change the font size for all text within the Streamlit app */
    body {
        font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)

    st.title("Sentiment Analysis using SVM model")
    input_text = st.text_input("Enter the text for sentiment analysis 📝")
    
    dig = ""
    if st.button("Analyse my sentiment 	🤗"):
        dig = analysis(input_text, tfidf_vectorizer, svm_model)
    st.success(dig)

if __name__ == '__main__':
    main()
