from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def create_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts)
    return embeddings

def create_vector_db(embeddings):
    embeddings = np.array([embedding for embedding in embeddings]).astype("float32")
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    index.add(embeddings)
    return index

def search_documents(query, vector_db, chunks, k=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])
    
    distances, indices = vector_db.search(query_embedding, k)
    
    results = [chunks[idx] for idx in indices[0]]
    
    return results
