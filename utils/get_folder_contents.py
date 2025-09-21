from utils.config import Config
import os
def get_base_filename(file_path):
    """Returns the filename without the extension."""
    return os.path.splitext(os.path.basename(file_path))[0]

def get_pdf_files():
    """Returns a sorted list of transcript files."""
    return sorted([f for f in os.listdir(Config.Folders.PDF_DIR) if f.endswith('.pdf')])

def get_vector_store_dirs():
    """Returns a sorted list of vector store directories."""
    return sorted([d for d in os.listdir(Config.Folders.VECTOR_STORE_DIR) if os.path.isdir(os.path.join(Config.Folders.VECTOR_STORE_DIR, d))])

