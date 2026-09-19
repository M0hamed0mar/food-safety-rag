"""
Chunk schema definitions.

This module defines Pydantic models for document chunks, which are the fundamental
units of information in the retrieval and generation pipeline.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """
    Rich metadata for a document chunk.
    
    Every chunk must include comprehensive metadata to support retrieval,
    citation, evaluation, and future filtering capabilities.
    
    Attributes:
        document_id: Unique identifier of the parent document.
        document_name: Original filename of the parent document.
        page: Page number where this chunk originates.
        chapter: Chapter title containing this chunk.
        section: Section title containing this chunk.
        subsection: Subsection title containing this chunk.
        title: Specific title or heading of this chunk.
        chunk_id: Globally unique identifier for this chunk.
        chunk_index: Sequential index of this chunk within the document.
        total_chunks: Total number of chunks in the parent document.
        language: Detected language code of this chunk.
        ocr: Whether this chunk was extracted via OCR.
        source_type: Type of content source (paragraph, table, list, etc.).
        table_id: Identifier if this chunk is a table, None otherwise.
        figure_id: Identifier if this chunk is a figure, None otherwise.
        token_count: Approximate token count of this chunk.
        embedding_model: Name of the embedding model used.
        parent_chunk_id: ID of the parent chunk in hierarchical chunking.
        child_chunk_ids: IDs of child chunks in hierarchical chunking.
        semantic_tags: Automatically extracted semantic tags.
        keywords: Key terms extracted from this chunk.
        confidence_score: Confidence in extraction quality (0.0 to 1.0).
        extraction_timestamp: When this chunk was created.
        table_data: JSON string containing table structure (headers, rows, etc.).
        
        # Phase 3: TableProcessor fields
        is_extracted_table: Whether this chunk was extracted from a table via TableProcessor.
        embedding_text: Pre-generated text for embedding (from TableProcessor).
        table_type: Detected table type (hazard_table, allergen_table, etc.).
    """
    
    document_id: str = Field(
        ...,
        description="Unique identifier of the parent document.",
        min_length=1,
    )
    document_name: str = Field(
        ...,
        description="Original filename of the parent document.",
        min_length=1,
    )
    page: Optional[int] = Field(
        default=None,
        description="Page number where this chunk originates.",
        ge=1,
    )
    chapter: Optional[str] = Field(
        default=None,
        description="Chapter title containing this chunk.",
    )
    section: Optional[str] = Field(
        default=None,
        description="Section title containing this chunk.",
    )
    subsection: Optional[str] = Field(
        default=None,
        description="Subsection title containing this chunk.",
    )
    title: Optional[str] = Field(
        default=None,
        description="Specific title or heading of this chunk.",
    )
    chunk_id: str = Field(
        ...,
        description="Globally unique identifier for this chunk.",
        min_length=1,
    )
    chunk_index: int = Field(
        ...,
        description="Sequential index of this chunk within the document.",
        ge=0,
    )
    total_chunks: int = Field(
        ...,
        description="Total number of chunks in the parent document.",
        ge=1,
    )
    language: Optional[str] = Field(
        default=None,
        description="Detected language code of this chunk (ISO 639-1).",
    )
    ocr: bool = Field(
        default=False,
        description="Whether this chunk was extracted via OCR.",
    )
    source_type: str = Field(
        default="paragraph",
        description="Type of content source (paragraph, table, list, figure, heading, etc.).",
    )
    table_id: Optional[str] = Field(
        default=None,
        description="Identifier if this chunk is a table, None otherwise.",
    )
    figure_id: Optional[str] = Field(
        default=None,
        description="Identifier if this chunk is a figure, None otherwise.",
    )
    token_count: int = Field(
        default=0,
        description="Approximate token count of this chunk.",
        ge=0,
    )
    embedding_model: Optional[str] = Field(
        default=None,
        description="Name of the embedding model used.",
    )
    parent_chunk_id: Optional[str] = Field(
        default=None,
        description="ID of the parent chunk in hierarchical chunking.",
    )
    child_chunk_ids: list[str] = Field(
        default_factory=list,
        description="IDs of child chunks in hierarchical chunking.",
    )
    semantic_tags: list[str] = Field(
        default_factory=list,
        description="Automatically extracted semantic tags.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Key terms extracted from this chunk.",
    )
    confidence_score: float = Field(
        default=1.0,
        description="Confidence in extraction quality (0.0 to 1.0).",
        ge=0.0,
        le=1.0,
    )
    extraction_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this chunk was created.",
    )
    table_data: Optional[str] = Field(
        default=None,
        description="JSON string containing table structure (headers, rows, etc.).",
    )
    
    # ==========================================================================
    # PHASE 3: TableProcessor Fields
    # ==========================================================================
    is_extracted_table: bool = Field(
        default=False,
        description="Whether this chunk was extracted from a table via TableProcessor.",
    )
    embedding_text: Optional[str] = Field(
        default=None,
        description="Pre-generated text for embedding (from TableProcessor).",
    )
    table_type: Optional[str] = Field(
        default=None,
        description="Detected table type (hazard_table, allergen_table, etc.).",
    )
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "document_id": "doc_abc123",
                "document_name": "HACCP_Manual_2024.pdf",
                "page": 17,
                "chapter": "Hazard Analysis",
                "section": "Biological Hazards",
                "subsection": "Pathogen Control",
                "title": "Salmonella Prevention Measures",
                "chunk_id": "chunk_abc123_042",
                "chunk_index": 42,
                "total_chunks": 381,
                "language": "en",
                "ocr": False,
                "source_type": "paragraph",
                "table_id": None,
                "figure_id": None,
                "token_count": 412,
                "embedding_model": "text-embedding-3-large",
                "parent_chunk_id": None,
                "child_chunk_ids": [],
                "semantic_tags": ["pathogen", "salmonella", "prevention"],
                "keywords": ["salmonella", "temperature", "cooking"],
                "confidence_score": 0.95,
                "table_data": None,
                "is_extracted_table": False,
                "embedding_text": None,
                "table_type": None,
            }
        }


class Chunk(BaseModel):
    """
    A document chunk containing text content and rich metadata.
    
    This is the fundamental unit of information in the RAG system.
    Chunks are created during ingestion and retrieved during query processing.
    
    Attributes:
        content: The actual text content of the chunk.
        metadata: Comprehensive metadata describing this chunk.
        embedding: Vector embedding of this chunk (optional, set after embedding).
        embedding_id: ID in the vector store (optional).
        bm25_score: BM25 relevance score (set during retrieval).
        dense_score: Dense retrieval similarity score (set during retrieval).
        reranker_score: Reranker relevance score (set during reranking).
        rrf_score: Reciprocal Rank Fusion score (set during fusion).
    """
    
    content: str = Field(
        ...,
        description="The actual text content of the chunk.",
        min_length=1,
    )
    metadata: ChunkMetadata = Field(
        ...,
        description="Comprehensive metadata describing this chunk.",
    )
    embedding: Optional[list[float]] = Field(
        default=None,
        description="Vector embedding of this chunk (set after embedding generation).",
    )
    embedding_id: Optional[int] = Field(
        default=None,
        description="ID in the vector store (set after indexing).",
    )
    bm25_score: Optional[float] = Field(
        default=None,
        description="BM25 relevance score (set during retrieval).",
    )
    dense_score: Optional[float] = Field(
        default=None,
        description="Dense retrieval similarity score (set during retrieval).",
    )
    reranker_score: Optional[float] = Field(
        default=None,
        description="Reranker relevance score (set during reranking).",
    )
    rrf_score: Optional[float] = Field(
        default=None,
        description="Reciprocal Rank Fusion score (set during fusion).",
    )
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "content": "Salmonella prevention requires maintaining cooking temperatures above 74°C...",
                "metadata": {
                    "document_id": "doc_abc123",
                    "document_name": "HACCP_Manual_2024.pdf",
                    "page": 17,
                    "chunk_id": "chunk_abc123_042",
                    "chunk_index": 42,
                    "total_chunks": 381,
                    "language": "en",
                    "ocr": False,
                    "source_type": "paragraph",
                    "token_count": 412,
                    "embedding_model": "text-embedding-3-large",
                },
            }
        }
    
    def to_retrieval_context(self) -> str:
        """
        Format this chunk as a retrieval context string for LLM consumption.
        
        Returns:
            str: Formatted context string with content and citation metadata.
        """
        citation_parts: list[str] = []
        
        if self.metadata.document_name:
            citation_parts.append(f"Document: {self.metadata.document_name}")
        if self.metadata.page:
            citation_parts.append(f"Page: {self.metadata.page}")
        if self.metadata.chapter:
            citation_parts.append(f"Chapter: {self.metadata.chapter}")
        if self.metadata.section:
            citation_parts.append(f"Section: {self.metadata.section}")
        if self.metadata.subsection:
            citation_parts.append(f"Subsection: {self.metadata.subsection}")
        if self.metadata.chunk_id:
            citation_parts.append(f"Chunk ID: {self.metadata.chunk_id}")
        
        citation = " | ".join(citation_parts)
        
        return f"[{citation}]\n{self.content}\n"
    
    def to_citation_string(self) -> str:
        """
        Generate a citation string for this chunk.
        
        Returns:
            str: Formatted citation string.
        """
        parts: list[str] = []
        
        if self.metadata.document_name:
            parts.append(self.metadata.document_name)
        if self.metadata.page:
            parts.append(f"p.{self.metadata.page}")
        if self.metadata.section:
            parts.append(self.metadata.section)
        
        if not parts:
            return f"[{self.metadata.chunk_id}]"
        
        return f"[{', '.join(parts)}]"


class HierarchicalChunk(Chunk):
    """
    A chunk that participates in a hierarchical document structure.
    
    Extends the base Chunk with hierarchical relationships to support
    parent-child navigation and context preservation.
    
    Attributes:
        level: Hierarchy level (0=document, 1=chapter, 2=section, etc.).
        parent: Direct parent chunk in the hierarchy.
        children: Direct child chunks in the hierarchy.
        siblings: Adjacent chunks at the same level.
        prev_sibling: Previous chunk at the same level.
        next_sibling: Next chunk at the same level.
        full_path: Hierarchical path string (e.g., "Doc > Ch1 > Sec2 > Para5").
    """
    
    level: int = Field(
        default=0,
        description="Hierarchy level (0=document, 1=chapter, 2=section, etc.).",
        ge=0,
    )
    parent: Optional["HierarchicalChunk"] = Field(
        default=None,
        description="Direct parent chunk in the hierarchy.",
    )
    children: list["HierarchicalChunk"] = Field(
        default_factory=list,
        description="Direct child chunks in the hierarchy.",
    )
    siblings: list["HierarchicalChunk"] = Field(
        default_factory=list,
        description="Adjacent chunks at the same level.",
    )
    prev_sibling: Optional["HierarchicalChunk"] = Field(
        default=None,
        description="Previous chunk at the same level.",
    )
    next_sibling: Optional["HierarchicalChunk"] = Field(
        default=None,
        description="Next chunk at the same level.",
    )
    full_path: str = Field(
        default="",
        description="Hierarchical path string (e.g., 'Doc > Ch1 > Sec2 > Para5').",
    )
    
    def get_ancestor_chain(self) -> list["HierarchicalChunk"]:
        """
        Get the full chain of ancestors from root to parent.
        
        Returns:
            list[HierarchicalChunk]: Ordered list of ancestor chunks.
        """
        ancestors: list[HierarchicalChunk] = []
        current: Optional[HierarchicalChunk] = self.parent
        
        while current is not None:
            ancestors.insert(0, current)
            current = current.parent
        
        return ancestors
    
    def get_descendants(self) -> list["HierarchicalChunk"]:
        """
        Get all descendant chunks recursively.
        
        Returns:
            list[HierarchicalChunk]: Flat list of all descendants.
        """
        descendants: list[HierarchicalChunk] = []
        
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        
        return descendants
    
    def get_sibling_context(self) -> str:
        """
        Get context from sibling chunks for enhanced retrieval.
        
        Returns:
            str: Combined content from previous and next siblings.
        """
        context_parts: list[str] = []
        
        if self.prev_sibling:
            context_parts.append(f"Previous: {self.prev_sibling.content[:200]}...")
        if self.next_sibling:
            context_parts.append(f"Next: {self.next_sibling.content[:200]}...")
        
        return "\n".join(context_parts)