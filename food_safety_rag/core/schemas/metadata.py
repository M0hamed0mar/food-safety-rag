"""
Metadata schema for document chunks and ingestion pipeline.

This module defines the structured metadata format for all chunks in the system.
Metadata quality is critical for retrieval, citation, evaluation, and debugging.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any

from food_safety_rag.core.config.constants import ChunkSourceType, DocumentLanguage


@dataclass
class ChunkMetadata:
    """
    Structured metadata for a single document chunk.
    
    This metadata is stored alongside each chunk and is essential for:
    - Citation generation
    - Retrieval filtering
    - Evaluation
    - Debugging
    - Analytics
    
    Never store meaningless values like "Unknown". Use None instead.
    All values should be actual, verifiable information.
    """

    # Document Information
    document_id: str
    """Unique identifier for the source document."""

    document_name: str
    """Human-readable document name (e.g., 'HACCP_Guidelines_2024.pdf')."""

    # Location Information
    page: Optional[int] = None
    """Page number where chunk originates (1-indexed)."""

    chapter: Optional[str] = None
    """Chapter title if document has hierarchical structure."""

    section: Optional[str] = None
    """Section title where chunk is located."""

    subsection: Optional[str] = None
    """Subsection title if applicable."""

    # Chunk Identification
    chunk_id: str = field(default_factory=str)
    """Unique identifier for this chunk (e.g., 'doc_001_chunk_042')."""

    chunk_index: int = 0
    """Sequential index of chunk within document (0-indexed)."""

    total_chunks: int = 0
    """Total number of chunks in document."""

    # Content Classification
    source_type: ChunkSourceType = ChunkSourceType.PARAGRAPH
    """Type of content: paragraph, bullet_list, table, figure, heading, etc."""

    title: Optional[str] = None
    """Title or heading of chunk content."""

    # Processing Information
    language: str = DocumentLanguage.ENGLISH
    """Language of chunk content (ISO 639-1 code: 'en', 'ar', 'fr', etc.)."""

    ocr_used: bool = False
    """Whether OCR was used to extract this chunk's text."""

    ocr_confidence: Optional[float] = None
    """Confidence score from OCR (0.0-1.0) if applicable."""

    # Relationship Information
    table_id: Optional[str] = None
    """Reference to parent table if chunk is extracted from table."""

    figure_id: Optional[str] = None
    """Reference to parent figure if chunk is extracted from figure."""

    parent_chunk_id: Optional[str] = None
    """Reference to parent chunk if hierarchically nested."""

    # Token/Size Information
    token_count: int = 0
    """Approximate token count for this chunk."""

    character_count: int = 0
    """Number of characters in chunk."""

    # Embedding Information
    embedding_model: str = "text-embedding-3-small"
    """Model used to generate embedding."""

    embedding_version: int = 1
    """Version of embedding model used."""

    # Timestamps
    ingestion_timestamp: datetime = field(default_factory=datetime.utcnow)
    """When chunk was ingested into system."""

    last_updated: datetime = field(default_factory=datetime.utcnow)
    """Last modification timestamp."""

    # Custom Metadata
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional domain-specific metadata."""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of metadata.
        """
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        data['ingestion_timestamp'] = self.ingestion_timestamp.isoformat()
        data['last_updated'] = self.last_updated.isoformat()
        # Convert enums to strings
        if isinstance(data.get('source_type'), ChunkSourceType):
            data['source_type'] = data['source_type'].value
        return data

    def to_citation_dict(self) -> Dict[str, Any]:
        """
        Get citation-relevant metadata only.
        
        Returns:
            Dict[str, Any]: Dictionary with only citation-relevant fields.
        """
        return {
            "document_name": self.document_name,
            "page": self.page,
            "section": self.section,
            "subsection": self.subsection,
            "chunk_id": self.chunk_id,
            "title": self.title,
        }


@dataclass
class DocumentMetadata:
    """
    Metadata for an entire document.
    
    Stored at document level and referenced by all chunks from that document.
    """

    document_id: str
    """Unique identifier for document."""

    document_name: str
    """File name of document."""

    file_path: str
    """Path to document file."""

    file_size_mb: float
    """File size in megabytes."""

    page_count: int
    """Total number of pages."""

    language: str = DocumentLanguage.ENGLISH
    """Document language (ISO 639-1 code)."""

    ingestion_timestamp: datetime = field(default_factory=datetime.utcnow)
    """When document was ingested."""

    chunk_count: int = 0
    """Total chunks generated from document."""

    total_tokens: int = 0
    """Total tokens in all chunks."""

    source_type: str = "document"
    """Type of document source."""

    encoding: str = "utf-8"
    """Text encoding of document."""

    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata specific to document."""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation.
        """
        data = asdict(self)
        data['ingestion_timestamp'] = self.ingestion_timestamp.isoformat()
        return data


@dataclass
class RetrievalMetadata:
    """
    Metadata tracking retrieval performance and caching.
    
    Used for monitoring and optimization.
    """

    query: str
    """Original user query."""

    normalized_query: str
    """Query after normalization."""

    expanded_queries: list[str] = field(default_factory=list)
    """Alternative queries generated for expansion."""

    dense_retrieval_latency_ms: float = 0.0
    """Time taken for dense retrieval in milliseconds."""

    bm25_retrieval_latency_ms: float = 0.0
    """Time taken for BM25 retrieval in milliseconds."""

    fusion_latency_ms: float = 0.0
    """Time taken for RRF fusion in milliseconds."""

    reranking_latency_ms: float = 0.0
    """Time taken for reranking in milliseconds."""

    cache_hit: bool = False
    """Whether result was cached."""

    dense_candidates_count: int = 0
    """Number of dense retrieval candidates."""

    bm25_candidates_count: int = 0
    """Number of BM25 candidates."""

    fusion_candidates_count: int = 0
    """Number of candidates after fusion."""

    final_context_count: int = 0
    """Number of chunks in final context."""

    total_retrieval_latency_ms: float = 0.0
    """Total retrieval pipeline latency."""

    timestamp: datetime = field(default_factory=datetime.utcnow)
    """When retrieval occurred."""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation.
        """
        return asdict(self)
