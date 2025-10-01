import streamlit as st
from function.data_lineage_agent import graph, FOLDER_NAME
from function.pdf_reader import extract_text_from_pdf
import asyncio
import time
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
                "transform_code":"chinook_transform_code.py",
                "lineage_builder":"chinook_lineage.json"
                }},
             )# stream_mode="events")
        elif selection=="SQL":
            streaming_response=graph.astream_events({"transform_code_language":"SQL",
            "transform_instructions":transform_instructions,
            "output_files":{"schema":"chinook_schema.json",
                "metadata_agent":"chinook_metadata.json",
                "transform_code":"chinook_transform_code.sql",
                "lineage_builder":"chinook_lineage.json"
                }}, 
            )
        elif selection=="HiveQL":
            streaming_response=graph.astream_events({"transform_code_language":"HiveQL",
            "transform_instructions":transform_instructions,
            "output_files":{"schema":"chinook_schema.json",
                "metadata_agent":"chinook_metadata.json",
                "transform_code":"chinook_transform_code.txt",
                "lineage_builder":"chinook_lineage.json"
                }}, 
            )
        elif selection=="TechSpec File":
            streaming_response=graph.astream_events({"transform_code_language":"TechSpec_File",
            "techspec_file":techspec_text,
            "output_files":{"schema":"chinook_schema.json",
                "metadata_agent":"chinook_metadata.json",
                "transform_code":"chinook_transform_code.txt",
                "lineage_builder":"chinook_lineage.json"
                }},
            )
        elif selection=="Own Script":
            streaming_response=graph.astream_events({"transform_code_language":"Own_Script",
            "own_script":own_script,
            "output_files":{"schema":"chinook_schema.json",
                "metadata_agent":"chinook_metadata.json",
                "transform_code":"chinook_transform_code.txt",
                "lineage_builder":"chinook_lineage.json"
                }}
            )
        
        async for chunk in streaming_response:
            # show thinking spinner while chunk end is not retrieved
            
            if chunk.get("event")=="on_chain_start" and chunk.get("name") in ["metadata_agent","transform_code","lineage_builder"]:
                container=st.empty()
                agent=chunk.get("name")
                with container.expander(agent+" agent called: ", expanded=True):
                    st.write("Thinking...")
                                         
            elif chunk.get("event")=="on_chain_end" and chunk.get("name") in ["metadata_agent","transform_code","lineage_builder"]:
                agent=chunk.get("name")
                with container.expander(agent+" agent called: ", expanded=True):
                    filename=chunk["data"]["input"]["output_files"][agent]
                    if agent=="metadata_agent":
                        schema_filename=chunk["data"]["input"]["output_files"]["schema"]
                        with open(FOLDER_NAME+"/"+schema_filename, "rb") as f:
                            st.download_button(f"⬇️ Download {schema_filename}", f,on_click="ignore", file_name=schema_filename, mime=mime_type[schema_filename.split(".")[1]], key=f"{schema_filename}_dl", help=f"Download {schema_filename}")
                    with open(FOLDER_NAME+"/"+filename, "rb") as f:
                        st.download_button(f"⬇️ Download {filename}", f,on_click="ignore", file_name=filename, mime=mime_type[filename.split(".")[1]], key=f"{filename}_dl", help=f"Download {filename}")
                        st.write(chunk["data"]["output"])
            
            

if __name__=="__main__":
    asyncio.run(data_lineage_screen())