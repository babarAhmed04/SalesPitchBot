# Import necessary modules
import streamlit as st
from document_loader import load_documents
from vector_store import get_vector_store_and_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Streamlit app definition with improved error handling
def main():
    st.title("Document Based Retrieval System for Sales Pitches")
    
    try:
        documents = load_documents()
        get_vector_store_and_llm(documents)
    except Exception as error:
        st.error(f"Error in initializing the application: {str(error)}")
        return

    input_query = st.text_input("Enter your query here:")
    if st.button("Clear Chat"):
        st.experimental_rerun()

    if st.button("Generate Pitch"):
        if input_query:
            try:
                prompt = ChatPromptTemplate.from_template("""
You role is of a successful and experienced Sales Manager...
...
""")
                document_chain = create_stuff_documents_chain(st.session_state.llm, prompt)
                retriever = st.session_state.vectordb.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                response = retrieval_chain.invoke({"input": input_query})
                st.write(response['answer'])
            except Exception as error:
                st.error(f"Failed to generate pitch: {str(error)}")
        else:
            st.error("Please enter a query to generate a pitch.")

if __name__ == "__main__":
    main()
