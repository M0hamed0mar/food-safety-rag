"""
Schemas package.

Pydantic models + dataclasses used across the RAG system:
    - Document: raw document + metadata
    - Chunk: chunked document content + metadata
    - Query: user query + expansions
    - Answer: generated answer + citations
    - RetrievalCandidate: retrieval result with all scores
    - Chat: session + message models
    - IngestionMetadata / RetrievalMetadata / CacheEntryMetadata
"""

from app.schemas.document import Document, DocumentMetadata, DocumentStructure
from app.schemas.chunk import Chunk, ChunkMetadata, HierarchicalChunk
from app.schemas.query import (
    ExpandedQuery,
    Query,
    QueryExpansionResult,
    RetrievalQuery,
)
from app.schemas.answer import (
    Answer,
    AnswerMetadata,
    Citation,
    StreamingChunk,
    UnsupportedAnswer,
)
from app.schemas.metadata import (
    CacheEntryMetadata,
    IngestionMetadata,
    RetrievalMetadata,
)
from app.schemas.retrieval import RetrievalCandidate, RetrievalResult
from app.schemas.chat import (
    ChatMessage,
    ChatSession,
    ChatCreateRequest,
    ChatListResponse,
)

__all__ = [
    # Document
    "Document",
    "DocumentMetadata",
    "DocumentStructure",
    # Chunk
    "Chunk",
    "ChunkMetadata",
    "HierarchicalChunk",
    # Query
    "Query",
    "ExpandedQuery",
    "QueryExpansionResult",
    "RetrievalQuery",
    # Answer
    "Answer",
    "AnswerMetadata",
    "Citation",
    "StreamingChunk",
    "UnsupportedAnswer",
    # Metadata
    "IngestionMetadata",
    "RetrievalMetadata",
    "CacheEntryMetadata",
    # Retrieval
    "RetrievalCandidate",
    "RetrievalResult",
    # Chat
    "ChatMessage",
    "ChatSession",
    "ChatCreateRequest",
    "ChatListResponse",
]
