import streamlit as st
import os
import shutil
from utils.config import Config
from utils.get_folder_contents import get_pdf_files, get_base_filename, get_vector_store_dirs
def file_manager():
    st.header("File Management")

    def delete_associated_files(base_name):
        """Deletes all files associated with a base filename."""
        # Note: pdf file could be mp3, wav, or m4a, but extracted pdf is always mp3
        pdf_extensions = ['.pdf']
        for ext in pdf_extensions:
            pdf_path = os.path.join(Config.Folders.PDF_DIR, f"{base_name}{ext}")
            if os.path.exists(pdf_path): os.remove(pdf_path)
        
        
        vector_store_path = os.path.join(Config.Folders.VECTOR_STORE_DIR, f"{base_name}.faiss")
        if os.path.exists(vector_store_path): shutil.rmtree(vector_store_path)

    
    st.markdown("---")

    # --- pdf Files Section ---
    st.subheader("Uploaded PDF Files")
    pdf_files = get_pdf_files()
    if not pdf_files: st.info("No pdf files available.")
    else:
        for filename in pdf_files:
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1: st.text(filename)
            with col2:
                with open(os.path.join(Config.Folders.PDF_DIR, filename), "rb") as f:
                    st.download_button("⬇️", f, file_name=filename, mime="application/pdf", key=f"dl_aud_{filename}", help=f"Download {filename}")
            with col3:
                if st.button("🗑️", key=f"del_pdf_{filename}", help=f"Delete pdf and associated files (PDF, vector store)"):
                    base_name = get_base_filename(filename)
                    delete_associated_files(base_name)
                    st.success(f"Deleted '{filename}' and its associated files.")
                    st.rerun()

    st.markdown("---")
    
    # --- Vector Store Section ---
    st.subheader("Generated Vector Stores")
    vector_store_dirs = get_vector_store_dirs()
    if not vector_store_dirs:
        st.info("No vector stores have been created yet.")
    else:
        for dirname in vector_store_dirs:
            dir_path = os.path.join(Config.Folders.VECTOR_STORE_DIR, dirname)
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(dirname)
            with col2:
                if st.button("🗑️", key=f"delete_vs_{dirname}", help=f"Delete vector store '{dirname}'"):
                    shutil.rmtree(dir_path)
                    st.success(f"Deleted vector store '{dirname}'.")
                    st.rerun()

if __name__ == "__main__":
    file_manager()
