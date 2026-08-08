from pathlib import Path
from fastapi import UploadFile, HTTPException
import shutil

MAX_FILE_SIZE = 20 * 1024 * 1024 #20 MB

def validate_pdf(file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400 ,
            detail="Only PDF files are allowed."
    )
    if file.size == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty"
        )
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400 ,
            detail="File size exceeds the 20 MB limit."
        )
    return True

async def save_pdf(file: UploadFile):
    documents_dir = Path("documents")
    documents_dir.mkdir(parents=True,exist_ok=True)
    file_path = documents_dir / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file_path
    
async def upload_pdf(file: UploadFile):
    validate_pdf(file)
    file_path = await save_pdf(file)
    return file_path 