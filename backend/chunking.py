import re

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def create_metadata(chunk_id, page, text):
    return {
        "chunk_id": chunk_id,
        "page": page,
        "text": text
    }

def split_chunks(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    chunk_id = 1

    for page in pages:
        page_no = page["page"]
        text = clean_text(page["text"])
        words = text.split()
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])

            chunks.append(
                create_metadata(chunk_id, page_no, chunk_text)
            )

            chunk_id += 1
            start += chunk_size - overlap

    return chunks