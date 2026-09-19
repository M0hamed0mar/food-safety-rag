"""
Custom exceptions for the RAG system.

All exceptions inherit from RAGError and carry structured
`details` for logging and debugging.
"""

from __future__ import annotations

from typing import Any


class RAGError(Exception):
    """Base exception for all RAG errors."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        # Preserve any extra kwargs (e.g. document_path=..., model_name=...)
        if extra:
            self.details.update(extra)
            for k, v in extra.items():
                setattr(self, k, v)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details!r})"


# ============================================================
# LLM errors
# ============================================================

class LLMError(RAGError):
    """Base class for all LLM-related errors."""


class LLMConnectionError(LLMError):
    """Raised when the LLM provider cannot be reached."""


class LLMRateLimitError(LLMError):
    """Raised when the LLM provider returns HTTP 429."""


class LLMResponseError(LLMError):
    """Raised when the LLM returns an unusable response."""


class LLMValidationError(LLMError):
    """Raised when the LLM output fails schema validation or is truncated."""


# ============================================================
# Embedding errors
# ============================================================

class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""


class EmbeddingRateLimitError(EmbeddingError):
    """Raised when the embedding provider returns a rate-limit error."""


# ============================================================
# Document / ingestion errors
# ============================================================

class DocumentError(RAGError):
    """Base class for document-related errors."""


class DocumentNotFoundError(DocumentError):
    """Raised when a document file cannot be found."""


class DocumentValidationError(DocumentError):
    """Raised when a document fails validation."""


class DocumentParsingError(DocumentError):
    """Raised when a document cannot be parsed."""


class OCRError(DocumentError):
    """Raised when OCR fails."""


class ChunkingError(DocumentError):
    """Raised when document chunking fails."""


class DuplicateDocumentError(DocumentError):
    """Raised when attempting to ingest a duplicate document."""


class IngestionError(RAGError):
    """Raised during document ingestion orchestration."""


# ============================================================
# Retrieval errors
# ============================================================

class RetrievalError(RAGError):
    """Raised during retrieval."""


class VectorStoreError(RetrievalError):
    """Raised when the vector store fails."""


class BM25Error(RetrievalError):
    """Raised when BM25 sparse retrieval fails."""


class FusionError(RetrievalError):
    """Raised when result fusion (RRF) fails."""


class RerankerError(RetrievalError):
    """Raised when reranking fails."""


class QueryValidationError(RetrievalError):
    """Raised when a user query fails validation."""


# ============================================================
# Generation errors
# ============================================================

class GenerationError(RAGError):
    """Raised during answer generation."""


class ContextWindowExceededError(GenerationError):
    """Raised when the context exceeds the LLM's max context window."""


class StreamingError(GenerationError):
    """Raised when token streaming fails."""


# ============================================================
# Cache / config / service
# ============================================================

class CacheError(RAGError):
    """Raised when cache operations fail."""


class ConfigurationError(RAGError):
    """Raised when configuration is invalid or missing."""


class ServiceError(RAGError):
    """Raised when a service-layer operation fails."""


__all__ = [
    # Base
    "RAGError",
    # LLM
    "LLMError",
    "LLMConnectionError",
    "LLMRateLimitError",
    "LLMResponseError",
    "LLMValidationError",
    # Embedding
    "EmbeddingError",
    "EmbeddingRateLimitError",
    # Document / ingestion
    "DocumentError",
    "DocumentNotFoundError",
    "DocumentValidationError",
    "DocumentParsingError",
    "OCRError",
    "ChunkingError",
    "DuplicateDocumentError",
    "IngestionError",
    # Retrieval
    "RetrievalError",
    "VectorStoreError",
    "BM25Error",
    "FusionError",
    "RerankerError",
    "QueryValidationError",
    # Generation
    "GenerationError",
    "ContextWindowExceededError",
    "StreamingError",
    # Cache / config / service
    "CacheError",
    "ConfigurationError",
    "ServiceError",
]
