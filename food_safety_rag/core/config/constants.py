"""
Global constants for the Food Safety RAG System.

This module defines immutable configuration constants used throughout the system.
Values here represent system-wide defaults and boundaries.
"""

from enum import Enum


class RetrievalMode(str, Enum):
    """Enumeration of supported retrieval strategies."""
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


class ChunkSourceType(str, Enum):
    """Enumeration of document chunk source types."""
    PARAGRAPH = "paragraph"
    BULLET_LIST = "bullet_list"
    TABLE = "table"
    FIGURE = "figure"
    HEADING = "heading"
    SECTION = "section"


class DocumentLanguage(str, Enum):
    """Enumeration of supported document languages."""
    ENGLISH = "en"
    ARABIC = "ar"
    FRENCH = "fr"


# Retrieval Configuration
DEFAULT_DENSE_TOP_K: int = 50
"""Default number of candidates to retrieve from dense search."""

DEFAULT_BM25_TOP_K: int = 50
"""Default number of candidates to retrieve from BM25 search."""

DEFAULT_RRF_CONSTANT: int = 60
"""Reciprocal Rank Fusion constant for score computation."""

DEFAULT_RERANKER_TOP_K: int = 30
"""Number of candidates after initial fusion before reranking."""

DEFAULT_FINAL_CONTEXT_K: int = 6
"""Number of top-ranked chunks to include in final context."""

# Chunk Configuration
DEFAULT_CHUNK_SIZE: int = 500
"""Target chunk size in tokens (adaptive, not absolute)."""

DEFAULT_CHUNK_OVERLAP: int = 50
"""Number of tokens of overlap between adjacent chunks."""

MIN_CHUNK_SIZE: int = 100
"""Minimum chunk size in tokens."""

MAX_CHUNK_SIZE: int = 1000
"""Maximum chunk size in tokens."""

# Embedding Configuration
EMBEDDING_MODEL: str = "text-embedding-3-small"
"""Official OpenAI embedding model identifier."""

EMBEDDING_DIMENSION: int = 384
"""Dimension of embeddings from text-embedding-3-small."""

DEFAULT_BATCH_SIZE_EMBEDDINGS: int = 100
"""Batch size for embedding API requests."""

# Reranker Configuration
RERanker_MODEL: str = "BAAI/bge-reranker-v2-m3"
"""Official Hugging Face reranker model identifier."""

# LLM Configuration
LLM_MODEL: str = "gpt-4o-mini"
"""Official OpenAI language model identifier."""

LLM_TEMPERATURE: float = 0.3
"""Temperature setting for LLM generation (lower = more deterministic)."""

LLM_MAX_TOKENS: int = 2048
"""Maximum tokens for LLM response."""

# OCR Configuration
OCR_ENABLED: bool = True
"""Global flag to enable/disable OCR processing."""

OCR_BATCH_SIZE: int = 5
"""Number of pages to process with OCR simultaneously."""

# Cache Configuration
CACHE_EMBEDDING_TTL: int = 86400 * 7
"""Time-to-live for embedding cache in seconds (7 days)."""

CACHE_RETRIEVAL_TTL: int = 3600
"""Time-to-live for retrieval cache in seconds (1 hour)."""

CACHE_CONTEXT_TTL: int = 1800
"""Time-to-live for context cache in seconds (30 minutes)."""

CACHE_LLM_RESPONSE_TTL: int = 3600
"""Time-to-live for LLM response cache in seconds (1 hour)."""

# Validation Configuration
MAX_QUERY_LENGTH: int = 2000
"""Maximum allowed user query length in characters."""

MIN_QUERY_LENGTH: int = 3
"""Minimum allowed user query length in characters."""

MAX_DOCUMENT_SIZE_MB: int = 100
"""Maximum allowed document file size in megabytes."""

# Vector Store Configuration
FAISS_INDEX_FACTORY_STRING: str = "IDMap,Flat"
"""FAISS index factory specification for index creation."""

SIMILARITY_THRESHOLD: float = 0.5
"""Minimum similarity threshold for considering a chunk relevant."""

# Logging Configuration
LOG_LEVEL: str = "INFO"
"""Default logging level for the application."""

STRUCTURED_LOG_FORMAT: str = "json"
"""Format for structured logging (json or text)."""

# Pipeline Configuration
SKIP_OCR_CONFIDENCE_THRESHOLD: float = 0.8
"""Text confidence threshold above which OCR is skipped."""

DUPLICATE_DETECTION_THRESHOLD: float = 0.95
"""Content similarity threshold for duplicate detection."""

# Paths and Directories
UPLOADED_DOCS_DIR: str = "data/uploaded_docs"
"""Directory for storing uploaded documents."""

FAISS_INDEX_DIR: str = "data/faiss_index"
"""Directory for storing FAISS index files."""

CACHE_DIR: str = "data/cache"
"""Directory for storing cache files."""

LOGS_DIR: str = "logs"
"""Directory for storing application logs."""

# Document Structure Hierarchy Levels
DOCUMENT_HIERARCHY_LEVELS: list[str] = [
    "document",
    "chapter",
    "section",
    "subsection",
    "paragraph",
    "bullet_list",
    "table",
]
"""Hierarchy levels for semantic document structure."""

# Error Messages
ERR_INVALID_QUERY: str = "Query is invalid or too short."
ERR_DOCUMENT_NOT_FOUND: str = "Document not found."
ERR_CORRUPTED_DOCUMENT: str = "Document appears to be corrupted or encrypted."
ERR_UNSUPPORTED_FORMAT: str = "File format is not supported."
ERR_RETRIEVAL_FAILED: str = "Retrieval pipeline encountered an error."
ERR_GENERATION_FAILED: str = "Answer generation encountered an error."
ERR_NO_CONTEXT: str = "Insufficient context retrieved to answer the query."
