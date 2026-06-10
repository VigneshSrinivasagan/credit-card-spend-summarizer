from pathlib import Path
from fastapi import UploadFile, HTTPException
from starlette.concurrency import run_in_threadpool

from src.ingestion.ingestion import ingest_pdf
from src.core.guardrails import guard_input, guard_output
from src.api.v1.agents.agents import run_search_agent, run_search_agent_static

import os


DATA_DIR = Path("src/api/data").resolve()
ALLOWED_EXTENSIONS = {".pdf"}


async def add_document(file: UploadFile):
    """
    Save uploaded PDF document to the data directory and ingest it.
    """

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is missing.")

    ext = Path(file.filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"File type '{ext}' is not supported."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    safe_filename = Path(file.filename).name
    saved_path = DATA_DIR / safe_filename

    content = await file.read()

    with open(saved_path, "wb") as f:
        f.write(content)

    await run_in_threadpool(ingest_pdf, str(saved_path))

    return {
        "message": f"'{safe_filename}' uploaded and ingested successfully.",
        "saved_path": str(saved_path),
    }


def query_documents_static(query: str):
    return run_search_agent_static(query)



async def query_documents(query: str, chat_history: list = None):
    chat_history = chat_history or []
    guard_input(query)
    result = run_search_agent(query, chat_history)
    print(result)
    if isinstance(result, dict) and result.get("answer"):
        result["answer"] = guard_output(result["answer"])
    return result
