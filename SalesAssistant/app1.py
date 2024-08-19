import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

# Load and preprocess documents
@st.cache_data
def load_documents():
    json_loader = DirectoryLoader("./data/Deal Details", glob='**/*.json', show_progress=True, loader_cls=TextLoader)
    json_docs = json_loader.load()

    cs_loader = PyPDFDirectoryLoader("./data/Case Studies")
    cs_docs = cs_loader.load()

    pitch_loader = PyPDFDirectoryLoader("./data/Sales Pitch")
    pitch_docs = pitch_loader.load()

    all_docs = json_docs + cs_docs + pitch_docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(all_docs)
    return final_documents

# Initialize the vector store and LLM only once per session
def get_vector_store_and_llm(documents):
    if 'vectordb' not in st.session_state or 'llm' not in st.session_state:
        vectordb = FAISS.from_documents(documents, OllamaEmbeddings())
        llm = Ollama(model="llama3")
        st.session_state.vectordb = vectordb
        st.session_state.llm = llm

# Streamlit App
def main():
    st.title("Document Based Retrieval System for Sales Pitches")
    documents = load_documents()
    get_vector_store_and_llm(documents)
    
    input_query = st.text_input("Enter your query here:")
    if st.button("Clear Chat"):
        st.rerun()

    if st.button("Generate Pitch"):
        if input_query:  # Check if the query is not empty
            prompt = ChatPromptTemplate.from_template("""
You role is of a successful and experienced Sales Manager currently working for a leading computer hardware manufacturer. 
Your job is to help your team of Sales Executives an exceptional sales pitches based on the following points. 
0. ALWAYS FINISH THE OUTPUT. Never send partial responses
                                                      
1. Each generated response should be completely based on past data such as case studies, sales pitch and deal details. No made up data should feature in the generated responses.                                                      
                                                      
2. Ensure that the generated response should not include any real world entities such as HP, Dell                                                      

3. Extract the Customer name and product from the user query and build a context

4. Exhaustively explore past deal details, case studies and extract the customer name and our competitors who are in the same line of business.

5. Based on the extracted data in the previous step, provide a brief customer and competitor profile before the actual sales pitch

6. Use previous win reasons and deal details such as discounts to construct an effective sales pitch in a strictly professional tone. Highlight past deal details and ensure that the sales pitch includes some of the factors such as additional discounts, past relationships, extended warranties and support.

7. If you are unable to get the context or identify that it is not a Sales related query, do not make up an answer, reply with the following statement "Unable to generate pitch"
<context>
{context}
</context>
Question: {input}
""")
            document_chain = create_stuff_documents_chain(st.session_state.llm, prompt)
            retriever = st.session_state.vectordb.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            response = retrieval_chain.invoke({"input": input_query})
            st.write(response['answer'])
        else:
            st.error("Please enter a query to generate a pitch.")

if __name__ == "__main__":
    main()
