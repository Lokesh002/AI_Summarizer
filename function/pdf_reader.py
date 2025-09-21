
import pymupdf # Import pymupdf (Fitz) for PDF processing

def extract_text_from_pdf(file_bytes_data):
    
    text = ""
    try:
        # Open the PDF document from the provided bytes data
        doc = pymupdf.open(stream=file_bytes_data, filetype="pdf")
        # Iterate through each page of the PDF
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num) # Load the current page
            text += page.get_text() # Extract text from the page and append to the total text
        doc.close() # Close the PDF document
        return text.strip() # Return the cleaned (whitespace stripped) extracted text
    except Exception:
        # Display an error message in the Streamlit app if text extraction fails
        
        return None

