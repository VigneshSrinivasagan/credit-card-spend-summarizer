import os
from typing import TypedDict, List
from langchain_core.documents import Document
from src.core.db import similarity_search


# ── State ──────────────────────────────────────────────────────────────────────
# The state is the shared data that flows through the entire graph.
# Each node reads from state and returns updated state.
class RAGState(TypedDict):
    query: str
    chat_history: list
    retrieved_docs: List[Document]  # Output of Node 1 — wide retrieval (k=20)
    reranked_docs: List[Document]  # Output of Node 2 — narrowed by reranker (top_n=10)
    response: dict  # Output of Node 3 — final structured answer
    route: str
    generated_sql: str
    sql_result: str
    score: float
    reasoning: str
    sufficient: bool


# ── Node 1: Vector Search ──────────────────────────────────────────────────────
# Uses a bi-encoder (Google Gemini embeddings) to find semantically similar chunks.
# We retrieve k=20 to cast a wide net — the reranker will narrow this down to top 10.
def vector_search_node(state: RAGState) -> RAGState:
    """Single-query vector search — used as a fallback if called directly."""
    docs = similarity_search(state["query"], k=20)
    print(f"[vector_search_node] Retrieved {len(docs)} chunks from multimodal_chunks")
    return {**state, "retrieved_docs": docs}
