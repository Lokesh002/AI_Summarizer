import streamlit as st
from utils.get_folder_contents import get_pdf_files
from utils.config import Config
import os
from function.pdf_reader import extract_text_from_pdf
from function.summarize import summarize_pdf
from function.vector_db_ops import create_and_save_vector_store

def process_and_summarize():
    st.title("PDF Summarization 🎙️")
    st.markdown("---")
    st.write("Select a PDF File to Process")

    pdf_files = get_pdf_files()
    if not pdf_files:
        st.info("No pdf files found. Please upload a media file first.")
    else:
        selected_pdf = st.selectbox("Choose a pdf file:", pdf_files, key="pdf_select")
        
        if st.button("Process and Summarize PDF"):
            pdf_path = os.path.join(Config.Folders.PDF_DIR, selected_pdf)
            # base_filename = get_base_filename(selected_pdf)
            
            st.info("Loading existing pdf.")
            with open(pdf_path, "rb") as f:
                pdf_text = extract_text_from_pdf(f.read())
                
            if not pdf_text:
                 st.error("Failed to load PDF.")
            else:
                with st.expander("View Full Contents"):
                    st.text_area("PDF", pdf_text, height=300)
                
                st.markdown("---")
                
                st.subheader("PDF Summary")
                with st.spinner("Generating summary..."):
                    try:
                        summary = summarize_pdf( pdf_text)
                        st.markdown(summary)
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")
if __name__ == "__main__":
    process_and_summarize()
