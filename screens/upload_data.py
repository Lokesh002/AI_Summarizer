import streamlit as st
import os
from function.vector_db_ops import create_and_save_vector_store
from function.pdf_reader import extract_text_from_pdf
def upload_data():
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
    if uploaded_file is not None:
        file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type, "filesize": f"{round(uploaded_file.size/1024/1024,2)} MB"}
        st.write(file_details)
        if st.button("Upload"):
            # save file in a folder "Audio" or Video based on file type
            if uploaded_file.type in ["application/pdf"]:
                try:
                    pdf_text=extract_text_from_pdf(uploaded_file.read())
                    create_and_save_vector_store(pdf_text)
                except RuntimeError as e:
                        st.error(str(e))
                
                with open(f"PDFs/{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.success("File Uploaded")
           
if __name__ == "__main__":

    upload_data()