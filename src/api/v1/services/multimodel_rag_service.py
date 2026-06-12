from pathlib import Path
from fastapi import UploadFile, HTTPException
from starlette.concurrency import run_in_threadpool

from src.ingestion.ingestion import ingest_pdf
from src.core.guardrails import guard_input, guard_output
from src.api.v1.agents.agents import run_search_agent

import os, json

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
            status_code=415, detail=f"File type '{ext}' is not supported."
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


def apply_output_guard(value):
    """
    Recursively apply guard_output to every string inside the response.

    Handles:
    - str
    - dict
    - list
    - tuple

    Example:
        (True, "Robert Clarke")
        becomes
        (True, "<PERSON>")
    """

    if isinstance(value, str):
        return guard_output(value)

    if isinstance(value, dict):
        return {key: apply_output_guard(item) for key, item in value.items()}

    if isinstance(value, list):
        return [apply_output_guard(item) for item in value]

    if isinstance(value, tuple):
        return tuple(apply_output_guard(item) for item in value)

    return value


# async def query_documents(query: str, chat_history: list = None):
#     chat_history = chat_history or []
#     guard_input(query)
#     result = run_search_agent(query, chat_history)

#     print(f"RAW RESULT BEFORE OUTPUT GUARD: {result}")

#     if isinstance(result, dict):
#         print("Inside the isinstance(result, dict): ", result)
#         for key in ["answer", "output", "response", "final_answer"]:
#             if isinstance(result.get(key), str):
#                 print(f"BEFORE OUTPUT GUARD [{key}]:", result[key])
#                 result[key] = guard_output(result[key])
#                 print(f"AFTER OUTPUT GUARD [{key}]:", result[key])

#     elif isinstance(result, str):
#         result = guard_output(result)

#     return result


async def query_documents(query: str, chat_history: list = None):
    chat_history = chat_history or []
    try:
        guard_input(query)
    except Exception as e:
        return {"answer": "I understand this can be frustrating. Let me help resolve this for you."}    

    # Collect streamed chunks from the async generator
    collected_payload = None
    async for chunk in run_search_agent(query, chat_history):
        if chunk.startswith("data: ") and "[DONE]" not in chunk:
            raw_json = chunk.removeprefix("data: ").strip()
            try:
                collected_payload = json.loads(raw_json)
            except json.JSONDecodeError:
                pass

    if collected_payload is None:
        return {"answer": "No response received."}

    result = collected_payload
    print(f"RAW RESULT BEFORE OUTPUT GUARD: {result}")

    # Apply output guard
    if isinstance(result, dict):
        for key in ["answer", "output", "response", "final_answer"]:
            if isinstance(result.get(key), str):
                print(f"BEFORE OUTPUT GUARD [{key}]:", result[key])
                result[key] = guard_output(result[key])
                print(f"AFTER OUTPUT GUARD [{key}]:", result[key])
                break  # ← guard only the first matching key, not multiple
    elif isinstance(result, str):
        result = guard_output(result)

    return result
