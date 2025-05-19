import streamlit as st
from utils.doc_processor import process_document, chunk_documents
from utils.vector_store import create_embeddings, create_vector_db, search_documents
from utils.model import get_llm_model, initialize_llm, generate_response

# Page configuration
st.set_page_config(
    page_title="Document Q&A with RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    h1, h2, h3 {
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        margin-bottom: 1rem;
    }
    .upload-section {
        margin-bottom: 2rem;
    }
    .stChatFloatingInputContainer {
        padding-bottom: 1rem;
        padding-top: 1rem;
    }
    .chat-container {
        margin-top: 2rem;
        padding-bottom: 1rem;
    }
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 1.5rem;
    }
    .waiting-badge {
        background-color: #FFC107;
        color: #212529;
    }
    .ready-badge {
        background-color: #28A745;
        color: white;
    }
    .file-list {
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.title('Document Q&A with RAG')
st.write('Upload your documents and chat with them using AI-powered assistance')

# File upload section with better spacing
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded_docs = st.file_uploader(
    "Upload your documents (PDF, DOCX, TXT)", 
    type=["pdf", "docx", "txt"], 
    accept_multiple_files=True
)

if uploaded_docs:
    st.markdown('<div class="file-list">', unsafe_allow_html=True)
    st.write(f"Uploaded {len(uploaded_docs)} documents!")
    for file in uploaded_docs:
        st.write(f"- {file.name}")
    st.markdown('</div>', unsafe_allow_html=True)

all_docs = []

# Processing docs with better feedback
if uploaded_docs:
    with st.spinner("Processing documents..."):
        progress_bar = st.progress(0)
        for i, file in enumerate(uploaded_docs):
            documents = process_document(file)
            all_docs.extend(documents)
            progress_bar.progress((i + 1) / len(uploaded_docs))
        st.success(f"Processed {len(all_docs)} document chunks!")

# Chunking
if all_docs:
    chunks = chunk_documents(all_docs)
    st.write(f"Created {len(chunks)} chunks from your documents")

    # Creating Vector Embeddings and Database
    if 'chunks' in locals() and chunks:
        with st.spinner("Creating embeddings..."):
            embeddings = create_embeddings(chunks)
            st.success("Embeddings created successfully!")

        if 'embeddings' in locals() and len(embeddings) > 0:
            with st.spinner("Creating vector database..."):
                vector_db = create_vector_db(embeddings)
                st.success(f"Vector database created with {vector_db.ntotal} vectors!")
st.markdown('</div>', unsafe_allow_html=True)

# Status indicator
if 'vector_db' in locals():
    st.markdown('<span class="status-badge ready-badge">READY</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="status-badge waiting-badge">WAITING</span> Please upload and process documents first', unsafe_allow_html=True)

# Create tabs but with better spacing
tab1, tab2 = st.tabs(["💬 Chat", "ℹ️ About"])

with tab1:
    # LLM Model integration
    # Get model path
    if 'model_path' not in st.session_state:
        st.session_state.model_path = get_llm_model()

    # Initialize LLM (only once)
    if 'llm' not in st.session_state:
        with st.spinner("Loading the language model... This might take a minute."):
            st.session_state.llm = initialize_llm(st.session_state.model_path)
            st.success("Language model loaded successfully!")

    # Enhanced chat interface (removing the redundant text input)
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input - this is the ONLY chat input now
    user_query = st.chat_input("Ask a question about your documents...")

    if user_query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Generate response
        if 'vector_db' in locals() and 'chunks' in locals() and 'llm' in st.session_state:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Retrieve relevant documents
                    relevant_docs = search_documents(user_query, vector_db, chunks, k=3)
                    
                    # Combine the context from relevant documents
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                    
                    # Generate response using the LLM
                    response = generate_response(st.session_state.llm, user_query, context)
                    
                    # Display the response
                    st.write(response)
                    
                    # Add response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error("Please upload and process documents first.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.header("About this Application")
    st.write("""
    This application uses Retrieval Augmented Generation (RAG) to answer questions based on your documents.
    
    The system processes your documents, creates vector embeddings of the content, and uses those 
    embeddings to find the most relevant information when you ask questions. The AI model then 
    generates answers based on the retrieved content.
    """)

# Add helpful instructions with proper styling
with st.expander("How to use this application"):
    st.write("""
    1. Upload one or more documents using the file uploader at the top
    2. Wait for the system to process the documents and build the vector database
    3. Use the chat interface to ask questions about your documents
    4. The system will retrieve relevant information and provide AI-generated answers
    
    Supported document types:
    - PDF files (.pdf)
    - Word documents (.docx)
    - Text files (.txt)
    """)
