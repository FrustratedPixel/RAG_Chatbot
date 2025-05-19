from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
import tempfile
import os

def process_document(file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as temp:
        temp.write(file.getvalue())
        temp_path = temp.name
    
    if file.name.endswith(".pdf"):
        loader = PyPDFLoader(temp_path)
    elif file.name.endswith(".docx"):
        loader = Docx2txtLoader(temp_path)
    elif file.name.endswith(".txt"):
        loader = TextLoader(temp_path)
    
    documents = loader.load()
    
    os.unlink(temp_path)
    
    return documents

# Chunking

from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

