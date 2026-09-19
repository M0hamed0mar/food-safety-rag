"""
Vector store package.

Exposes:
    - FAISSStore: vector similarity index (FAISS)
    - VectorStore: abstract base
    - MetadataStore: chunk/document metadata (SQLite)
    - ChatStore: chat sessions (SQLite)
"""

from app.vector_store.faiss_store import FAISSStore, VectorStore
from app.vector_store.metadata_store import MetadataStore
from app.vector_store.chat_store import ChatStore

__all__ = [
    "VectorStore",
    "FAISSStore",
    "MetadataStore",
    "ChatStore",
]
