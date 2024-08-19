# Import necessary modules
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import config

# Function to load and preprocess documents with error handling
def load_documents():
    try:
        json_loader = DirectoryLoader(config.DEAL_DETAILS_DIR, glob=config.GLOB_PATTERN, show_progress=config.SHOW_PROGRESS, loader_cls=TextLoader)
        json_docs = json_loader.load()

        cs_loader = PyPDFDirectoryLoader(config.CASE_STUDIES_DIR)
        cs_docs = cs_loader.load()

        pitch_loader = PyPDFDirectoryLoader(config.SALES_PITCH_DIR)
        pitch_docs = pitch_loader.load()

        all_docs = json_docs + cs_docs + pitch_docs
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        final_documents = text_splitter.split_documents(all_docs)
        return final_documents
    except Exception as e:
        raise RuntimeError("Failed to load and process documents") from e
