import os
import streamlit as st
from utils.config import Config
from models.llms.google_llm import gemini_2_5_flash as llm
from function.vector_db_ops import get_vector_store
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from operator import itemgetter


def chatbot_page():
    st.header("Chat with Your PDFs💬")

    def get_rag_chain(messages):
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
            # return ({"context": retriever, "history": RunnablePassthrough(), "input":RunnablePassthrough()} | prompt | llm | StrOutputParser())
        except Exception as e:
            st.error(f"Failed to load vector store or build RAG chain: {e}")
            return None

    # mtime = get_vector_db_mtime(vector_store_path)
    

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        rag_chain = get_rag_chain(st.session_state.messages)
        if not rag_chain: st.stop()
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                
                response = rag_chain.invoke({"input":prompt, "history":st.session_state.messages})
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    chatbot_page()