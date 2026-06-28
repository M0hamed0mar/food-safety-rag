"""
Document schema for uploaded documents.

Represents a source document in the system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from food_safety_rag.core.schemas.metadata import DocumentMetadata


@dataclass
class Document:
    """
    A source document in the RAG system.
    
    Represents a file (typically PDF) that has been ingested
    and broken into semantic chunks.
    """

    # Document Identity
    document_id: str
    """Unique identifier for document."""

    file_name: str
    """Original file name."""

    file_path: Path
    """Path to the document file."""

    file_size_bytes: int
    """Size of document file in bytes."""

    # Document Content
    text_content: str
    """Full extracted text from document."""

    page_count: int
    """Total number of pages."""

    # Processing Information
    ingestion_timestamp: datetime = field(default_factory=datetime.utcnow)
    """When document was ingested."""

    language: str = "en"
    """Document language (ISO 639-1 code)."""

    # Chunks
    chunk_ids: List[str] = field(default_factory=list)
    """IDs of all chunks created from this document."""

    chunk_count: int = 0
    """Total number of chunks."""

    # Metadata
    metadata: Optional[DocumentMetadata] = None
    """Rich document metadata."""

    # Status
    is_indexed: bool = False
    """Whether document has been indexed in vector store."""

    def __post_init__(self) -> None:
        """
        Validate document after initialization.
        
        Raises:
            ValueError: If file path doesn't exist or text is empty.
        """
        if not self.file_path.exists():
            raise ValueError(f"Document file not found: {self.file_path}")
        if not self.text_content or not self.text_content.strip():
            raise ValueError("Document text content cannot be empty")

    @property
    def file_size_mb(self) -> float:
        """
        Get file size in megabytes.
        
        Returns:
            float: File size in MB.
        """
        return self.file_size_bytes / (1024 * 1024)

    @property
    def total_tokens_estimate(self) -> int:
        """
        Estimate total tokens in document.
        
        Uses approximate calculation of ~1 token per 4 characters.
        
        Returns:
            int: Estimated token count.
        """
        return len(self.text_content) // 4

    def mark_indexed(self) -> None:
        """
        Mark document as successfully indexed.
        """
        self.is_indexed = True
