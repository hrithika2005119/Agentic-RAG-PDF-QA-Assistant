import fitz
from pathlib import Path

def extract_text(pdf_path: Path):
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)) :
        page = doc.load_page(page_num)
        text = page.get_text()
        pages.append(
            {
                "page": page_num + 1,
                "text": text
            }
        )
        doc.close()
        return pages 
       
def extract_metadata(pdf_path: Path) :
    doc = fitz.open(pdf_path)
    metadata = doc.metadata
    doc.close()
    return metadata