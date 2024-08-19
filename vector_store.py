# Import necessary modules
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
import config

# Function to initialize vector store and LLM with enhanced session state management
def get_vector_store_and_llm(documents):
    if 'vectordb' not in st.session_state or 'llm' not in st.session_state:
        try:
            vectordb = FAISS.from_documents(documents, OllamaEmbeddings())
            llm = Ollama(model=config.LLM_MODEL)
            st.session_state['vectordb'] = vectordb
            st.session_state['llm'] = llm
        except Exception as e:
            raise RuntimeError("Failed to initialize vector store and LLM") from e
