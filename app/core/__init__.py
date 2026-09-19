"""
Core layer for the RAG system.

Exposes:
    - LLM client (Groq)
    - Embeddings generator (multilingual, local)
    - Custom exceptions
"""

from app.core.exceptions import (
    RAGError,
    LLMError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
    LLMValidationError,
    EmbeddingError,
)
from app.core.llm_client import LLMClient, llm_client
from app.core.embeddings import EmbeddingsGenerator, get_embeddings

__all__ = [
    "RAGError",
    "LLMError",
    "LLMConnectionError",
    "LLMRateLimitError",
    "LLMResponseError",
    "LLMValidationError",
    "EmbeddingError",
    "LLMClient",
    "llm_client",
    "EmbeddingsGenerator",
    "get_embeddings",
]
