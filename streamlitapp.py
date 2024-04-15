import numpy as np
import pandas as pd
import streamlit as st
import joblib
import requests
from io import BytesIO

model = joblib.load('svm_model.pkl')

def download_file_from_google_drive(url):
    response = requests.get(url)
    return BytesIO(response.content)

tfidi_url = "https://drive.google.com/file/d/1ijX8Sn3OwFqx89fJsWcnNxWIN-lejz-Z/view?usp=sharing"
tfidi_file = download_file_from_google_drive(tfidi_url)
tfidi = joblib.load(tfidi_file)

def analysis(input_text):
    input_data_features = tfidi.transform(input_text)
    data_features = pd.DataFrame(input_data_features.toarray())
 
    prediction = model.predict(data_features)
    print(prediction)
    if (prediction[0] == 0):
        return "Positive Sentiment :heart_eyes: :champagne: :tada:"
    elif (prediction[0] == 1):
        return "Negetive Sentiment :sneezing_face: :angry: :rage:"
    else:
        return "Neutral Sentiment :no_mouth: :slightly_smiling_face:"

def main():
    st.markdown("""
<style>
    /* Change the font size for all text within the Streamlit app */
    body {
        font-size: 40px;
    }
</style>
""", unsafe_allow_html=True)
    def set_bg_hack_url():
        '''
        A function to unpack an image from url and set as bg.
        Returns
        -------
        The background.
        '''
            
        st.markdown(
             f"""
             <style>
             .stApp {{
                 background: url("https://c.ndtvimg.com/2020-07/1k0ddgo_flipkart650_625x300_28_July_20.jpg?ver-20230922.06");
                 background-size: cover
             }}
             </style>
             """,
             unsafe_allow_html=True
         )
    set_bg_hack_url()
    st.title("Sentiment Analysis on Flipkart Reviews :shopping_trolley: ")
    input_text = st.text_input("Enter a Review on You Experience :thinking_face: :thinking_face:")
    
    dig =""
    if st.button("Analyse my sentiment 	:hugging_face:"):
        dig = analysis([input_text])
    st.success(dig)
        


if __name__ == '__main__':
    main()
