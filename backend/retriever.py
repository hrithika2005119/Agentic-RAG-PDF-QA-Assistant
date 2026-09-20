import numpy as np
from embeddings import load_model
import ollama


def generate_query_embedding(query: str):
    model = load_model()

    response = ollama.embed(
        model=model, 
        input=query
        )
    
    embedding = response["embeddings"][0]

    embedding = np.array(
        embedding,
        dtype=np.float32
        )
    
    return embedding

if __name__ == "__main__":
    query = "What is the refund policy?"

    query_embedding = generate_query_embedding(query)

    print("Query embedding shape:" , query_embedding.shape)

def search(index, query_embedding, top_k=5):
    query_embedding = np.array([query_embedding],  dtype=np.float32)

    distances, indices = index.search(query_embedding, top_k)

    return indices[0], distances[0]

def get_relevant_chunks(indices, metadata_mapping):
    relevant_chunks = []

    for index in indices:
        if index == -1:
            continue
        chunk = metadata_mapping[int(index)]
        relevant_chunks.append(chunk)

    return relevant_chunks