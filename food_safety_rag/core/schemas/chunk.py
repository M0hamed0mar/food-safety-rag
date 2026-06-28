"""
Chunk schema for document chunks.

Represents a semantic unit of information extracted from a document.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from food_safety_rag.core.schemas.metadata import ChunkMetadata


@dataclass
class Chunk:
    """
    A semantic unit of information from a document.
    
    A chunk represents a meaningful piece of content that can be:
    - A paragraph
    - A bullet point
    - A table row
    - A complete table
    - A figure caption
    - A section heading
    
    Each chunk is independently retrievable and citable.
    """

    # Content
    text: str
    """The actual text content of the chunk."""

    # Metadata
    metadata: ChunkMetadata
    """Rich metadata for citation and filtering."""

    # Embedding
    embedding: Optional[List[float]] = None
    """Dense vector embedding (384-dimensional for text-embedding-3-small)."""

    # Relationships
    related_chunks: List[str] = field(default_factory=list)
    """List of chunk IDs that are semantically related."""

    def __post_init__(self) -> None:
        """
        Validate chunk after initialization.
        
        Raises:
            ValueError: If chunk text is empty.
        """
        if not self.text or not self.text.strip():
            raise ValueError("Chunk text cannot be empty")

    @property
    def chunk_id(self) -> str:
        """
        Get the unique chunk identifier.
        
        Returns:
            str: The chunk ID from metadata.
        """
        return self.metadata.chunk_id

    @property
    def document_id(self) -> str:
        """
        Get the source document ID.
        
        Returns:
            str: The document ID from metadata.
        """
        return self.metadata.document_id

    @property
    def document_name(self) -> str:
        """
        Get the source document name.
        
        Returns:
            str: The document name from metadata.
        """
        return self.metadata.document_name

    def has_embedding(self) -> bool:
        """
        Check if chunk has an embedding.
        
        Returns:
            bool: True if embedding exists and is non-empty.
        """
        return self.embedding is not None and len(self.embedding) > 0

    def get_citation_text(self) -> str:
        """
        Generate a citation string for this chunk.
        
        Returns:
            str: Human-readable citation.
        """
        parts = [f"Document: {self.metadata.document_name}"]
        if self.metadata.page:
            parts.append(f"Page: {self.metadata.page}")
        if self.metadata.section:
            parts.append(f"Section: {self.metadata.section}")
        parts.append(f"ID: {self.metadata.chunk_id}")
        return " | ".join(parts)
