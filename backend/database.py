import faiss
import numpy as np
import os

def create_index(embedding_dimension):
    index = faiss.IndexFlatL2(embedding_dimension)
    return index

def add_embeddings(index, embeddings):
    embeddings = np.array(embeddings, dtype="float32")
    index.add(embeddings)
    return index

def save_index(index, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    faiss.write_index(index, file_path)


def load_index(file_path):
    index = faiss.read_index(file_path)
    return index


index = create_index(768)
save_index(index, "vector_db/pdf_index.faiss")

loaded_index = load_index("vector_db/pdf_index.faiss")
print("Index dimensions:", loaded_index.d)
print("Number of vectors:", loaded_index.ntotal)