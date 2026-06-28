"""
Vector store module for the Food Safety RAG System.

Provides abstraction for embedding storage and retrieval.
"""

from food_safety_rag.core.vector_store.faiss_store import FAISSStore
from food_safety_rag.core.vector_store.metadata_store import MetadataStore

__all__ = [
    "FAISSStore",
    "MetadataStore",
]
