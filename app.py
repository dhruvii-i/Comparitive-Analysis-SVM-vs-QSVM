import streamlit as st
import pickle
import sklearn

# Load the SVM model from the pickle file
with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

# Function to predict sentiment
def predict_sentiment(text):
    # Use the SVM model to predict sentiment
    predicted_label = svm_model.predict([text])[0]
    return predicted_label

# Streamlit app layout
st.title('Sentiment Analysis with SVM')
st.write('Enter a text to analyze its sentiment.')

# Text input for user
text_input = st.text_input('Input Text:', '')

# Button to predict sentiment
if st.button('Predict Sentiment'):
    if text_input.strip() == '':
        st.error('Please enter some text.')
    else:
        # Predict sentiment
        sentiment = predict_sentiment(text_input)
        if sentiment == 1:
            st.success('Sentiment: Positive')
        elif sentiment == 0:
            st.warning('Sentiment: Neutral')
        else:
            st.error('Sentiment: Negative')
