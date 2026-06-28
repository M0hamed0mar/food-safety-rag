"""
Data schemas for the Food Safety RAG System.

Defines Pydantic models and dataclasses for all data structures.
"""

from food_safety_rag.core.schemas.metadata import (
    ChunkMetadata,
    DocumentMetadata,
    RetrievalMetadata,
)
from food_safety_rag.core.schemas.chunk import Chunk
from food_safety_rag.core.schemas.document import Document
from food_safety_rag.core.schemas.query import QueryRequest, QueryExpansion
from food_safety_rag.core.schemas.answer import AnswerResponse, Citation

__all__ = [
    "ChunkMetadata",
    "DocumentMetadata",
    "RetrievalMetadata",
    "Chunk",
    "Document",
    "QueryRequest",
    "QueryExpansion",
    "AnswerResponse",
    "Citation",
]
