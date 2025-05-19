import numpy as np

def process_query(query_text, model):
    # Convert user query to embedding vector
    query_embedding = model.encode([query_text])[0]
    return query_embedding

def retrieve_similar_chunks(query_embedding, index, chunks, top_k=3):
    # Search for similar vectors
    distances, indices = index.search(np.array([query_embedding]), top_k)
    
    # Get the corresponding document chunks
    retrieved_chunks = [chunks[i] for i in indices[0]]
    
    return retrieved_chunks, distances[0]

def assemble_context(retrieved_chunks, distances):
    context = ""
    for i, (chunk, distance) in enumerate(zip(retrieved_chunks, distances)):
        context += f"\nDocument chunk {i+1} (relevance: {1-distance:.2f}):\n{chunk.page_content}\n"
    
    return context
