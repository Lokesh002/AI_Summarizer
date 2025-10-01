import streamlit as st
import os
from utils.config import Config
from dotenv import load_dotenv
load_dotenv()

def main():
    os.makedirs(Config.Folders.PDF_DIR, exist_ok=True)
    os.makedirs(Config.Folders.VECTOR_STORE_DIR, exist_ok=True)
    #create a file .env if not present
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("GOOGLE_API_KEY=\n")    
    pg=st.navigation([st.Page("screens/homepage.py",title="Home"),
                      st.Page("screens/upload_data.py", title="Upload Data"),
                      st.Page("screens/summarizer.py", title="Summarize"),
                      st.Page("screens/chatbot.py", title="Chat with Data"),
                      st.Page("screens/file_manager.py", title="File Manager"),
                      st.Page("screens/data_lineage.py", title="Data Lineage Agent")])
    pg.run()

if __name__ == "__main__":
    main()