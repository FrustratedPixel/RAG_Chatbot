# Document Q&A with RAG

A Streamlit application that uses Retrieval Augmented Generation (RAG) to answer questions about uploaded documents. This tool enables users to upload multiple document types and have interactive conversations with their content.

![Document Q&A with RAG]: Support for PDF, DOCX, and TXT files
- **Interactive Chat Interface**: Ask questions about your documents in a conversational format
- **Document Processing Pipeline**: Automatic parsing, preprocessing, and chunking of document content
- **Vector Search**: Semantic search to retrieve the most relevant document fragments
- **Local LLM Integration**: Uses TinyLLama model for generating responses without requiring API keys (using better and powerful models gives better output)
- **Modern UI**: Clean, intuitive interface with visual feedback on processing status

## Technologies Used

- Streamlit
- LangChain
- FAISS (Facebook AI Similarity Search)
- Sentence Transformers
- llama-cpp-python
- HuggingFace Hub
- PyPDF2, python-docx (for document parsing)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/FrustratedPixel/RAG_Chatbot.git
   cd document-qa-rag
   ```

## Usage

1. Start the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Open the provided URL in your browser (typically http://localhost:8501)

3. Upload your documents using the file uploader

4. Wait for the system to process your documents

5. Ask questions about your documents using the chat interface

## Project Structure

```
RAG_Chatbot/
├── app.py                   # Main Streamlit application
├── utils/
│   ├── doc_processor.py     # Document processing utilities
│   ├── vector_store.py      # Vector database
|   |-- Retrival_sys.py      # Retrival functions
│   ├── model.py             # LLM model loading and inference
├── models/                  # Directory for storing downloaded models
└── README.md                # Project documentation
```

## How It Works

1. **Document Processing**: Uploaded documents are parsed based on their file type and split into meaningful chunks
2. **Vector Embedding**: Document chunks are converted into vector embeddings using Sentence Transformers
3. **Database Creation**: Vectors are stored in a FAISS index for efficient similarity search
4. **Query Processing**: User questions are converted to vectors and used to retrieve relevant document chunks
5. **Response Generation**: Retrieved chunks are provided as context to the LLM to generate accurate answers

## Dependencies

The main dependencies for this project are:
- streamlit
- langchain
- faiss-cpu
- sentence-transformers
- llama-cpp-python
- huggingface_hub
- pypdf2
- python-docx
- unstructured

## Future Improvements

- Add support for more document formats (e.g., CSV, PPTX)
- Implement document highlighting to show which portions were used in answers
- Add memory to maintain conversation context
- Improve document chunking strategies for better retrieval
- Add document metadata filtering options

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Streamlit team for creating an amazing tool for building data apps
- HuggingFace for providing pre-trained models
- The open-source community for developing the libraries that made this project possible

---

Built by Shiva S

---
