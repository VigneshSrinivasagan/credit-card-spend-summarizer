import shutil
from pathlib import Path
from fastapi import UploadFile
from src.ingestion.ingestion import ingest_pdf
from src.api.v1.agents.agents import run_search_agent,run_search_agent_static


# receive the document as user input and save it inside the data directory
async def add_document(file: UploadFile):
    """
    Save uploaded document to the data directory
    """
    # Define the data directory path
    data_dir = Path(__file__).parent.parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save the file directly
    file_path = data_dir / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"file path : {file_path}")

    # ingestion method called
    ingest_pdf(file_path)

    return {
        "message": "Document added and ingested successfully",
        "filename": file.filename,
        "file_path": str(file_path),
    }

def query_documents_static(query: str):
    return run_search_agent_static(query)

async def query_documents(query: str):
    return run_search_agent(query)
