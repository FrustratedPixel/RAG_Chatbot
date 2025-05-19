import os
import streamlit as st
from huggingface_hub import hf_hub_download

def get_llm_model():
    model_name = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    model_file = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"  # Using a smaller quantized version
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Path to save the model
    model_path = os.path.join("models", model_file)
    
    # Download the model if it doesn't exist
    if not os.path.exists(model_path):
        with st.spinner(f"Downloading {model_file}... This might take a while."):
            model_path = hf_hub_download(
                repo_id=model_name,
                filename=model_file,
                local_dir="models",
                local_dir_use_symlinks=False
            )
        st.success("Model downloaded successfully!")
    
    return model_path


from llama_cpp import Llama

def initialize_llm(model_path):
    return Llama(
        model_path=model_path,
        n_ctx=2048,  # Context window size
        n_threads=4  # Number of CPU threads to use
    )

def generate_response(llm, query, context):
    prompt = f"""Please answer the question based on the context provided below.

Context:
{context}

Question: {query}

Answer:"""
    
    # Generate response
    response = llm(
        prompt,
        max_tokens=512,
        temperature=0.7,
        stop=["Question:", "Context:"],
        echo=False
    )
    
    return response['choices'][0]['text'].strip()