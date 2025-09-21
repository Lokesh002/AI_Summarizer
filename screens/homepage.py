import streamlit as st

def homepage():
    st.set_page_config(page_title="AI Summarizer - Home", page_icon="🏠", layout="wide")    
    st.title("Welcome to the AI Summarizer App 🎉")
    st.subheader("Your Personal Assistant for PDF Summarization")

if __name__ == "__main__":
    homepage()
