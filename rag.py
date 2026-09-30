from pathlib import Path

from pdf_processor import extract_text
from chunking import split_chunks
from embeddings import generate_embeddings
from database import create_index, add_embeddings, save_index
from retriever import (
    generate_query_embedding,
    search,
    get_relevant_chunks
)


VECTOR_DB_PATH = "vector_db/pdf_index.faiss"


def process_document(pdf_path: Path):
    """
    Process a PDF and store its embeddings in FAISS.

    Input:
        pdf_path: Path to the PDF file

    Output:
        index: FAISS index
        metadata_mapping: Mapping between vector index and PDF chunks
    """

    # 1. Extract text from PDF
    pages = extract_text(pdf_path)

    # 2. Split PDF text into chunks
    chunks = split_chunks(pages)

    # 3. Generate embeddings for each chunk
    embeddings = generate_embeddings(chunks)

    # 4. Create FAISS index
    embedding_dimension = len(embeddings[0])
    index = create_index(embedding_dimension)

    # 5. Add embeddings to FAISS
    add_embeddings(index, embeddings)

    # 6. Create mapping between FAISS index and original chunks
    metadata_mapping = {
        i: chunk for i, chunk in enumerate(chunks)
    }

    # 7. Save FAISS index
    save_index(index, VECTOR_DB_PATH)

    return index, metadata_mapping


def build_context(relevant_chunks):
    """
    Combine retrieved chunks into context for the LLM.

    Input:
        relevant_chunks: List of chunks returned by retriever.py

    Output:
        context: Combined text
    """

    context = ""

    for chunk in relevant_chunks:
        context += (
            f"Page {chunk['page']}:\n"
            f"{chunk['text']}\n\n"
        )

    return context


def retrieve_context(query, index, metadata_mapping, top_k=5):
    """
    Retrieve relevant PDF chunks for a user question.

    Input:
        query: User question
        index: FAISS index
        metadata_mapping: Chunk metadata
        top_k: Number of chunks to retrieve

    Output:
        relevant_chunks: List of relevant PDF chunks
    """

    # Generate embedding for user's question
    query_embedding = generate_query_embedding(query)

    # Search FAISS
    indices, distances = search(
        index,
        query_embedding,
        top_k
    )

    # Get original PDF chunks
    relevant_chunks = get_relevant_chunks(
        indices,
        metadata_mapping
    )

    return relevant_chunks


if __name__ == "__main__":

    pdf_path = Path("documents/sample.pdf")

    # Build the document vector database
    index, metadata_mapping = process_document(pdf_path)

    print("Number of vectors:", index.ntotal)

    # Test user question
    query = "What is this document about?"

    relevant_chunks = retrieve_context(
        query,
        index,
        metadata_mapping
    )

    # Build context for LLM
    context = build_context(relevant_chunks)

    print("\nRetrieved Context:\n")
    print(context)
