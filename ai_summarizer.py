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

from typing_extensions import TypedDict
from typing import Sequence, Literal
import dotenv
dotenv.load_dotenv()
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from pydantic import BaseModel
from langgraph.graph import StateGraph
from pydantic import BaseModel
import sqlite3
import os
import time
from typing import Optional
FOLDER_NAME="./generated_files"
RUN_COMPLETE=True
os.makedirs(FOLDER_NAME, exist_ok=True)
def get_chinook_schema(db_path="chinook.db"):
    """
    Connects to the Chinook database and retrieves its schema,
    storing it in a dictionary.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query to get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()



        schema_data = {}
        for table_name_tuple in tables:
            table_name = table_name_tuple[0]
            
            # Query to get column information for each table
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns_info = cursor.fetchall()

            # Store column details for the current table
            table_schema = []
            for col_id, col_name, col_type, notnull, default_value, is_primary_key in columns_info:
                table_schema.append({
                    "name": col_name,
                    "type": col_type,
                    "not_null": bool(notnull),
                    "default_value": default_value,
                    "is_primary_key": bool(is_primary_key)
                })

            # Get foreign key information
            cursor.execute(f"PRAGMA foreign_key_list({table_name});")
            foreign_keys = cursor.fetchall()
            fk_info = []
            for fk in foreign_keys:
                id, seq, table, from_col, to_col, on_update, on_delete, match = fk
                fk_info.append({
                    'id': id,
                    'seq': seq,
                    'parent_table': table,
                    'from_column': from_col,
                    'to_column': to_col,
                    'on_update': on_update,
                    'on_delete': on_delete,
                    'match': match
                })
            # Get sample data
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
            sample_rows = cursor.fetchall()
            sample_columns = [description[0] for description in cursor.description]
            sample_data = [dict(zip(sample_columns, row)) for row in sample_rows]   
            # Add sample data to the table schema
            sample_rows = sample_data               
            # Store the schema and foreign key info in the main dictionary
            schema_data[table_name] = {
                'columns': table_schema,
                'foreign_keys': fk_info,
                'sample_data': sample_rows
            }
        print("Created Schema")  
        
        return schema_data

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return None
    finally:
        if conn:
            conn.close()

class Metadata(BaseModel):
    table_description: Sequence[str]
    column_description: Sequence[str]
    ForeignKeys_description: Sequence[str]
    sampleData: Sequence[str]

class LineageMining(BaseModel):
    description: Sequence[str]
    trace: Sequence[str]
    source_column: Sequence[str]
    Derived_column: Sequence[str]
    code_snippet: Sequence[str]
    data_quality: Sequence[str]

class AgentState(TypedDict):
    schema: Optional[dict]
    metadata: Optional[Sequence[Metadata]]
    transform_code: Optional[str]
    lineage_json: Optional[LineageMining]    
    transform_code_language: Optional[Literal["python", "SQL", "HiveQL", "TechSpec_File", "Own_Script"]]
    own_script: Optional[str]
    techspec_file: Optional[str]
    transform_instructions: Optional[str]
    output_files:Optional[dict]

def metadata_agent(state: AgentState) -> AgentState:
    print("Entered metadata agent")
    if RUN_COMPLETE:
        schema_filename=state.get("output_files").get("schema")
        schema = get_chinook_schema()
        print(schema)
        with open(FOLDER_NAME+"/"+schema_filename, "w") as f:
            f.write(json.dumps(schema))
        llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        metadata = []
        for table in schema:
            # print(table) 
            table_name = table
            columns = schema[table_name]["columns"]
            foreign_keys = schema[table_name]["foreign_keys"]
            sample_data = schema[table_name]["sample_data"]
            metadata_prompt_template = """
            You are an expert data cataloger. Given the following table schema,
            generate a concise and informative description for the table and each of its columns.
            Focus on the purpose of the table and the meaning of each column. You may infer context based on sample data if provided.
            This metadata will be used to help users understand the data structure and its contents.Include any relevant relationships based on foreign keys.
            Include column constraints such as primary keys, foreign keys, and not-null constraints in the descriptions.
            Include data types for each column.
            Include any information that is included in create table statements.

            Table Name: {table_name}
            Schema:
            Column details:
            {columns} 
            Foreign Keys info: 
            {foreign_keys}
            Sample Data:
            {sample_data}

            """

            prompt = PromptTemplate(
                template=metadata_prompt_template,
                input_variables=["table_name", "columns", "foreign_keys", "sample_data"]
            )

            # Bind the Pydantic schema to the LLM using with_structured_output
            structured_llm = llm.with_structured_output(Metadata)
            # 3. Create an LLM chain
            metadata_chain = prompt | structured_llm

            # 6. Invoke the chain and receive the structured output
            generated_metadata = metadata_chain.invoke({"table_name":table_name,
                "columns":columns,
                "foreign_keys":foreign_keys,
                "sample_data":sample_data})
            
            metadata.append(generated_metadata.model_dump())
        
        metadata_file_name=state.get("output_files").get("metadata_agent")
        with open(FOLDER_NAME+"/"+metadata_file_name, "w") as f:
            f.write(json.dumps(metadata))
        print("created metadata")
        return {"metadata": metadata,
                "schema": schema}
    else:
        time.sleep(2)
        return {"metadata":"Empty metdata"}
def transformation_code_agent(state: AgentState) -> AgentState:
    
    print("entered transformation code agent")
    if RUN_COMPLETE:
        language=state.get("transform_code_language")
        # Get the schema and metadata from the current state
        schema = state.get("schema")
        metadata = state.get("metadata")
        if language=="TechSpec_File":
            techspec_text=state.get("techspec_file")
            filename=state.get("output_files").get("transform_code_agent")
            with open(FOLDER_NAME+"/"+filename, "w", encoding="utf-8") as f:
                f.write(techspec_text)
            return {**state, "transform_code": techspec_text}
        elif language=="Own_Script":
            own_script=state.get("own_script")
            filename=state.get("output_files").get("transform_code_agent")
            with open(FOLDER_NAME+"/"+filename, "w", encoding="utf-8") as f:
                f.write(own_script)
            return {**state, "transform_code": own_script}
        else:
        # Initialize the Language Model
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
            default_transform_instructions="""1.  **Extract:** Connect to the 'chinook.db' database and extract data from the 'customers', 'invoices', and 'invoice_items' tables into pandas DataFrames.
            2.  **Transform:**
                - Merge these Tables to link customers to their invoices and the items on each invoice.
                - Calculate a new column, 'TotalSpent', for each customer by summing the 'UnitPrice' * 'Quantity' for all their invoice items.
                - Create a final Table that shows 'CustomerId', 'FirstName', 'LastName', 'Country', and the calculated 'TotalSpent'.
                - Sort the results by 'TotalSpent' in descending order.
            3.  **Load:** Print the top 10 customers from the final transformed table to the console."""
            # Define a detailed prompt for the LLM to generate the ETL code
            prompt_template = """
            You are an expert {language} data engineer. Your task is to write a ETL script utilizing {language} to transform data based on the provided database schema and metadata.

            **Database Context:**
            - Database Name: chinook.db (a SQLite database)
            - Schema: {schema}
            - Metadata (descriptions of tables and columns): {metadata}

            **Task Requirements:**
            Write a complete {language} script that performs the following ETL process:
            {transform_instructions}
            **Output Format:**
            - Provide ONLY the raw {language} code.
            - Do not include any explanations, introductory text, or markdown code blocks (like ```{language}... ```).
            - The code should be fully functional and ready to execute.
            """
            # if state.get("transform_code_language","python").lower() == "python":
            #     filename="chinook_transform_code.py"
            # elif state.get("transform_code_language","python").lower() == "sql":
            #     filename="chinook_transform_code.sql"
            # elif state.get("transform_code_language","python").lower() == "hiveql":
            #     filename="chinook_transform_code.txt"
            filename=state.get("output_files").get("transform_code_agent")
            # Create a prompt instance from the template
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["schema", "metadata", "language","transform_instructions"]
            )
            
            # Create the chain to pipe the prompt to the LLM
            code_generation_chain = prompt | llm

            # Invoke the chain with the schema and metadata to generate the code
            # We use json.dumps to format the dictionary/list context into a clean string for the prompt
            response = code_generation_chain.invoke({
                "schema": json.dumps(schema, indent=2),
                "metadata": json.dumps(metadata, indent=2),
                "language": state.get("transform_code_language","python"),
                "transform_instructions": state.get("transform_instructions",default_transform_instructions)
            })
            
            # Extract the code string from the LLM's response content
            generated_code = response.content.strip()

        with open(FOLDER_NAME+"/"+filename, "w") as f:
            f.write(generated_code)

        print("completed python code")
        # Return the updated state with the new 'transform_code'
        return {**state, "transform_code": generated_code}
    else:
        time.sleep(2)
        return {**state, "transform_code": "Empty code"}
    

def lineage_builder_agent(state: AgentState)-> AgentState:
    print("entered lineage builder agent")
    if RUN_COMPLETE:
        sample_transform_script= state["transform_code"]
        llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        
        metadata_prompt_template = """
    You are an expert identifying data transformations for purposes of data lineage. Given the following code snippet{sample_transform_script},
    generate following as json format with keys as below:

    key: [description]
    "Describe the transformations applied to the '[Column Name]' column in the '[Table Name]' table during the ETL process. Include details on data type conversions, cleansing rules, aggregation methods, and any derived calculations."
    key : [trace]
    "Trace the lineage of the '[Column Name]' column from its source(s) to its target in the '[Table Name]' table. Identify all intermediate steps and transformations performed."
    key : [source_column]
    "Identify the original source column(s) from which the '[Column Name]' in the '[Table Name]' table is derived. Include source table names and any relevant joins or merges."
    key : [Derived_column]
    "Identify any new columns that are derived or calculated in the '[Table Name]' table, specifying the source columns and the logic used for derivation."
    key : [code_snippet]
    "Provide the code or logic snippets used to transform the '[Column Name]' column within the ETL process. Highlight any functions, expressions, or operators involved."
    key : [data_quality]
    "Explain how the '[Column Name]' column is handled during data quality checks and validation within the ETL, including any rules for identifying and handling invalid or missing values."
    """

        prompt = PromptTemplate(
            template=metadata_prompt_template,
            input_variables=["sample_transform_script"]
        )
        structured_llm = llm.with_structured_output(LineageMining)
        metadata_chain = prompt | structured_llm

        lineage_mining_response = metadata_chain.invoke({"sample_transform_script":sample_transform_script,})
        filename=state.get("output_files").get("lineage_builder_agent")
        with open(FOLDER_NAME+"/"+filename, "w") as f:
            f.write(lineage_mining_response.model_dump_json())
        print("created data lineage")
        return {**state, "lineage_json":lineage_mining_response}
    else:
        time.sleep(2)
        return{**state, "lineage_json":"Empty lineage"}


graph_builder= StateGraph(AgentState)
graph_builder.add_node("metadata_agent", metadata_agent)
graph_builder.add_node("transform_code_agent", transformation_code_agent)
graph_builder.add_node("lineage_builder_agent", lineage_builder_agent)
graph_builder.set_entry_point("metadata_agent")
graph_builder.add_edge("metadata_agent", "transform_code_agent")
graph_builder.add_edge("transform_code_agent", "lineage_builder_agent")
graph_builder.set_finish_point("lineage_builder_agent")
graph=graph_builder.compile()

########### AGENT SCREEN ############

mime_type={
            "json":"application/json",
            "py":"text/x-python",
            "sql":"text/x-sql",
            "txt":"text/plain",
            "pdf":"application/pdf"
        }


async def data_lineage_screen():
    st.subheader("Select type of transformation: ")
    selection=st.selectbox("Please select your method of transformation:", ["Python","SQL", "HiveQL","TechSpec File", "Own Script"])
    
    match selection:
        case "Python":
            transform_instructions=st.text_area("Please enter transformation steps for chinook db","""1.  **Extract:** Connect to the 'chinook.db' database and extract data from the 'customers', 'invoices', and 'invoice_items' tables into pandas DataFrames.
2.  **Transform:**
    - Merge these Tables to link customers to their invoices and the items on each invoice.
    - Calculate a new column, 'TotalSpent', for each customer by summing the 'UnitPrice' * 'Quantity' for all their invoice items.
    - Create a final Table that shows 'CustomerId', 'FirstName', 'LastName', 'Country', and the calculated 'TotalSpent'.
    - Sort the results by 'TotalSpent' in descending order.
3.  **Load:** Print the top 10 customers from the final transformed table to the console.""", key="python", height=210)    
        case "SQL":
            transform_instructions=st.text_area("Please enter transformation steps for chinook db","""1.  **Extract:** Connect to the 'chinook.db' database and extract data from the 'customers', 'invoices', and 'invoice_items' tables into pandas DataFrames.
2.  **Transform:**
    - Merge these Tables to link customers to their invoices and the items on each invoice.
    - Calculate a new column, 'TotalSpent', for each customer by summing the 'UnitPrice' * 'Quantity' for all their invoice items.
    - Create a final Table that shows 'CustomerId', 'FirstName', 'LastName', 'Country', and the calculated 'TotalSpent'.
    - Sort the results by 'TotalSpent' in descending order.
3.  **Load:** Print the top 10 customers from the final transformed table to the console.""", key="sql", height=210)
        case "HiveQL":
            transform_instructions=st.text_area("Please enter transformation steps for chinook db","""1.  **Extract:** Connect to the 'chinook.db' database and extract data from the 'customers', 'invoices', and 'invoice_items' tables into pandas DataFrames.
2.  **Transform:**
    - Merge these Tables to link customers to their invoices and the items on each invoice.
    - Calculate a new column, 'TotalSpent', for each customer by summing the 'UnitPrice' * 'Quantity' for all their invoice items.
    - Create a final Table that shows 'CustomerId', 'FirstName', 'LastName', 'Country', and the calculated 'TotalSpent'.
    - Sort the results by 'TotalSpent' in descending order.
3.  **Load:** Print the top 10 customers from the final transformed table to the console.""", key="hiveql", height=210)
        case "TechSpec File":
            techspec_file=st.file_uploader("Please upload your techspec file here.", key="techspec", type=["pdf"])
            if techspec_file is not None:
                techspec_text=extract_text_from_pdf(techspec_file.read())
                
                   
        case "Own Script":
            own_script=st.text_area("Please enter your transformation script here.", key="ownscript", height=210)
    if st.button("Generate Lineage"):
        if selection=="Python":
            streaming_response=graph.astream_events({"transform_code_language":"python",
            "transform_instructions":transform_instructions,
            "output_files":{"schema":"chinook_schema.json",
                "metadata_agent":"chinook_metadata.json",
                "transform_code_agent":"chinook_transform_code.py",
                "lineage_builder_agent":"chinook_lineage.json"
                }},
             )# stream_mode="events")
        elif selection=="SQL":
            streaming_response=graph.astream_events({"transform_code_language":"SQL",
            "transform_instructions":transform_instructions,
            "output_files":{"schema":"chinook_schema.json",
                "metadata_agent":"chinook_metadata.json",
                "transform_code_agent":"chinook_transform_code.sql",
                "lineage_builder_agent":"chinook_lineage.json"
                }}, 
            )
        elif selection=="HiveQL":
            streaming_response=graph.astream_events({"transform_code_language":"HiveQL",
            "transform_instructions":transform_instructions,
            "output_files":{"schema":"chinook_schema.json",
                "metadata_agent":"chinook_metadata.json",
                "transform_code_agent":"chinook_transform_code.txt",
                "lineage_builder_agent":"chinook_lineage.json"
                }}, 
            )
        elif selection=="TechSpec File":
            if techspec_file is None:
                st.error("Please upload a techspec file.")
                return
            else:
                if techspec_text is not None:
                    st.info("Techspec file uploaded successfully.")
                    streaming_response=graph.astream_events({"transform_code_language":"TechSpec_File",
                "techspec_file":techspec_text,
                "output_files":{"schema":"chinook_schema.json",
                                "metadata_agent":"chinook_metadata.json",
                                "transform_code_agent":"chinook_transform_code.txt",
                                "lineage_builder_agent":"chinook_lineage.json" }})
        elif selection=="Own Script":
            if own_script is None or own_script=="":
                st.error("Please enter your own script.")
                return
            else:
                streaming_response=graph.astream_events({"transform_code_language":"Own_Script",
                "own_script":own_script,
                "output_files":{"schema":"chinook_schema.json",
                                "metadata_agent":"chinook_metadata.json",
                                "transform_code_agent":"chinook_transform_code.txt",
                                "lineage_builder_agent":"chinook_lineage.json"
                                }})
        
        async for chunk in streaming_response:
            # show thinking spinner while chunk end is not retrieved
            
            if chunk.get("event")=="on_chain_start" and chunk.get("name") in ["metadata_agent","transform_code_agent","lineage_builder_agent"]:
                container=st.empty()
                agent=chunk.get("name")
                with container.expander(agent+" agent called: ", expanded=True):
                    st.write("Thinking...")
                                         
            elif chunk.get("event")=="on_chain_end" and chunk.get("name") in ["metadata_agent","transform_code_agent","lineage_builder_agent"]:
                agent=chunk.get("name")
                with container.expander(agent+" agent called: ", expanded=True):
                    filename=chunk["data"]["input"]["output_files"][agent]
                    if agent=="metadata_agent":
                        schema_filename=chunk["data"]["input"]["output_files"]["schema"]
                        with open(FOLDER_NAME+"/"+schema_filename, "rb") as f:
                            st.download_button(f"⬇️ Download {schema_filename}", f,on_click="ignore", file_name=schema_filename, mime=mime_type[schema_filename.split(".")[1]], key=f"{schema_filename}_dl", help=f"Download {schema_filename}")
                    with open(FOLDER_NAME+"/"+filename, "rb") as f:
                        st.download_button(f"⬇️ Download {filename}", f,on_click="ignore", file_name=filename, mime=mime_type[filename.split(".")[1]], key=f"{filename}_dl", help=f"Download {filename}")
                        # st.write(chunk["data"]["output"])
            
def data_lineage_sync():
    asyncio.run(data_lineage_screen())


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
                      st.Page(file_manager, title="File Manager"),
                      st.Page(data_lineage_sync, title="Data Lineage")])
    pg.run()

if __name__ == "__main__":
    main()