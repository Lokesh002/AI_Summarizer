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
    """
    Generates a Python/SQL/HiveQL ETL script using an LLM based on the database schema and metadata.

    This agent takes the schema and metadata from the state, formulates a detailed prompt
    instructing an LLM to act as a data engineer, and asks it to write a Python script.
    The script performs an ETL task, such as calculating total sales per customer.
    The resulting code string is then added to the state.

    Args:
        state (AgentState): The current state of the graph, containing schema and metadata.

    Returns:
        AgentState: The updated state containing the generated Python code as a string.
    """
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
graph_builder.add_edge( "transform_code_agent", "lineage_builder_agent")
graph_builder.set_finish_point("lineage_builder_agent")
graph=graph_builder.compile()
# transform_code_language can be (python, SQL, HiveQL)
# response= graph.invoke(input={"transform_code_language":"SQL"})