import streamlit as st
import pickle

# Load the model
with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define your Streamlit app here
def main():
    st.title('Your Streamlit App')
    st.write('This is a simple Streamlit app.')

if __name__ == '__main__':
    main()
