class Config:
    class Folders:
        PDF_DIR = "PDFs"
        VECTOR_STORE_DIR = "Vector_Dbs"
    class Prompts:
        SUMMARIZER_PROMPT="""You are an expert at summarizing PDF files. Based on this file, provide a 'Title', and a bulleted list of 'Key Points'.\n\nTranscript:\n{transcript}"""
        CHATBOT_PROMPT="You are a helpful assistant. Answer the user question. You have some context also provided to you for answering. This is the context:\n\n{context}"