import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from operator import itemgetter
import shutil
import pymupdf # Import pymupdf (Fitz) for PDF processing
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import asyncio

########### CONFIG PARAMETERS ############
class Config:
    class Folders:
        PDF_DIR = "PDFs"
        VECTOR_STORE_DIR = "Vector_Dbs"
    class Prompts:
        SUMMARIZER_PROMPT="""You are an expert at summarizing PDF files. Based on this file, provide a 'Title', and a bulleted list of 'Key Points'.\n\nTranscript:\n{transcript}"""
        CHATBOT_PROMPT="You are a helpful assistant. Answer the user question. You have some context also provided to you for answering. This is the context:\n\n{context}"
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

########### UTILITY FUNCTIONS ############

def get_embedding_001():
    
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

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
    
def summarize_pdf(transcript) -> str:
    """Summarizes a transcript using the Gemini model."""
    prompt = ChatPromptTemplate.from_template(Config.Prompts.SUMMARIZER_PROMPT)
    chain = prompt | llm | StrOutputParser()
    try:
        summary = chain.invoke({"transcript": transcript})
        return summary
    except Exception as e:
        raise RuntimeError(f"An error occurred during summarization: {e}")

def get_base_filename(file_path):
    """Returns the filename without the extension."""
    return os.path.splitext(os.path.basename(file_path))[0]

def get_pdf_files():
    """Returns a sorted list of transcript files."""
    return sorted([f for f in os.listdir(Config.Folders.PDF_DIR) if f.endswith('.pdf')])

def get_vector_store_dirs():
    """Returns a sorted list of vector store directories."""
    return sorted([d for d in os.listdir(Config.Folders.VECTOR_STORE_DIR) if os.path.isdir(os.path.join(Config.Folders.VECTOR_STORE_DIR, d))])

def create_and_save_vector_store(transcript):
    """Chunks transcript and saves it as a FAISS vector store."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    embedding=get_embedding_001()
    chunks = text_splitter.split_text(text=transcript)
    try:
        if not os.path.exists(f"{Config.Folders.VECTOR_STORE_DIR}/faiss_vdb.faiss"):
            vector_store = FAISS.from_texts(chunks, embedding=embedding)
        else:
            vector_store = FAISS.load_local(f"{Config.Folders.VECTOR_STORE_DIR}/faiss_vdb.faiss", embeddings=embedding, allow_dangerous_deserialization=True)
            vector_store.add_texts(chunks)
        
        vector_store_path = os.path.join(Config.Folders.VECTOR_STORE_DIR, f"faiss_vdb.faiss")       
        vector_store.save_local(vector_store_path)
    except Exception as e:
        raise RuntimeError(f"An error occured while creating vecotr db: {e}")
    
def get_vector_store():
    """Loads a FAISS vector store by name."""
    vector_store_path = os.path.join(Config.Folders.VECTOR_STORE_DIR, f"faiss_vdb.faiss")
    if not os.path.exists(vector_store_path):
        raise FileNotFoundError(f"Vector store 'faiss_vdb' not found.")
    try:
        embedding=get_embedding_001()
        vector_store = FAISS.load_local(vector_store_path, embeddings=embedding, allow_dangerous_deserialization=True)
        return vector_store
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading vector store: {e}")
    
def update_vector_store():
    #delete the existing vector store and create new with all the transcripts present
    if os.path.exists(f"{Config.Folders.VECTOR_STORE_DIR}/faiss_vdb.faiss"):
        shutil.rmtree(f"{Config.Folders.VECTOR_STORE_DIR}/faiss_vdb.faiss")
    transcripts = get_pdf_files()
    for transcript in transcripts:
        with open(os.path.join(Config.Folders.PDF_DIR, transcript), "r", encoding="utf-8") as f:
            transcript_text = f.read()
        create_and_save_vector_store(transcript_text)
    
########### SCREENS ############

def homepage():
    st.set_page_config(page_title="AI Summarizer - Home", page_icon="🏠", layout="wide")    
    st.title("Welcome to the AI Summarizer App 🎉")
    st.subheader("Your Personal Assistant for PDF Summarization")

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

def summarize():
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
def chatbot():
    st.header("Chat with Your PDFs💬")

    def get_rag_chain():
        try:
            # embeddings = get_embedding_001()
            vector_store = get_vector_store()
            retriever = vector_store.as_retriever()
            prompt = ChatPromptTemplate.from_messages([
                ("system", Config.Prompts.CHATBOT_PROMPT),
                MessagesPlaceholder(variable_name="history"),
                ("user", "{input}"),
            ])
            setup_and_retrieval = RunnableParallel(
                context=itemgetter("input") | retriever,
                history=itemgetter("history"),
                input=itemgetter("input"),
            )
            rag_chain = setup_and_retrieval | prompt | llm | StrOutputParser()
            return rag_chain
        except Exception as e:
            st.error(f"Failed to load vector store or build RAG chain. Make sure atleast one PDF is uploaded.")
            return None

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        rag_chain = get_rag_chain()
        if not rag_chain: st.stop()
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                
                response = rag_chain.invoke({"input":prompt, "history":st.session_state.messages})
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
from utils.get_folder_contents import get_pdf_files, get_base_filename, get_vector_store_dirs
def file_manager():
    st.header("File Management")

    def delete_associated_files(base_name):
        """Deletes all files associated with a base filename."""
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
def main():
    os.makedirs(Config.Folders.PDF_DIR, exist_ok=True)
    os.makedirs(Config.Folders.VECTOR_STORE_DIR, exist_ok=True)
    # create a file .env if not present
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("GOOGLE_API_KEY=\n")    
    pg=st.navigation([st.Page(homepage,title="Home"),
                      st.Page(upload_data, title="Upload Data"),
                      st.Page(summarize, title="Summarize"),
                      st.Page(chatbot, title="Chat with Data"),
                      st.Page(file_manager, title="File Manager")])
    pg.run()

if __name__ == "__main__":
    main()