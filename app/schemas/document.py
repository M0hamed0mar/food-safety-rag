"""
Document schema definitions.

This module defines Pydantic models for document representation throughout
the ingestion and retrieval pipeline.
"""

from datetime import datetime
from typing import Any, Optional
from dataclasses import dataclass, field

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """
    Metadata for a source document.
    
    This model captures high-level information about a document before
    it is processed into chunks.
    
    Attributes:
        document_id: Unique identifier for the document (UUID or hash).
        document_name: Original filename of the document.
        file_path: Absolute path to the document file.
        file_size_bytes: Size of the file in bytes.
        file_extension: File extension (e.g., pdf, docx).
        document_type: Classification of the document (Manual, Standard, etc.).
        title: Extracted or provided document title.
        author: Author or organization responsible for the document.
        language: Detected or specified language code (ISO 639-1).
        total_pages: Total number of pages in the document.
        creation_date: Document creation date if available.
        ingestion_date: Timestamp when the document was ingested.
        document_hash: SHA-256 hash of the file content for deduplication.
        ocr_required: Whether OCR was needed for any page.
        has_tables: Whether the document contains tables.
        has_figures: Whether the document contains figures.
        tags: Optional user-defined tags for categorization.
        custom_metadata: Additional flexible metadata fields.
    """
    
    document_id: str = Field(
        ...,
        description="Unique identifier for the document.",
        min_length=1,
    )
    document_name: str = Field(
        ...,
        description="Original filename of the document.",
        min_length=1,
    )
    file_path: str = Field(
        ...,
        description="Absolute path to the document file.",
    )
    file_size_bytes: int = Field(
        ...,
        description="Size of the file in bytes.",
        ge=0,
    )
    file_extension: str = Field(
        ...,
        description="File extension (e.g., pdf, docx).",
    )
    document_type: Optional[str] = Field(
        default=None,
        description="Classification of the document (Manual, Standard, Guideline, etc.).",
    )
    title: Optional[str] = Field(
        default=None,
        description="Extracted or provided document title.",
    )
    author: Optional[str] = Field(
        default=None,
        description="Author or organization responsible for the document.",
    )
    language: Optional[str] = Field(
        default=None,
        description="Detected or specified language code (ISO 639-1).",
    )
    total_pages: Optional[int] = Field(
        default=None,
        description="Total number of pages in the document.",
        ge=0,
    )
    creation_date: Optional[datetime] = Field(
        default=None,
        description="Document creation date if available.",
    )
    ingestion_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the document was ingested.",
    )
    document_hash: str = Field(
        ...,
        description="SHA-256 hash of the file content for deduplication.",
        min_length=64,
        max_length=64,
    )
    ocr_required: bool = Field(
        default=False,
        description="Whether OCR was needed for any page.",
    )
    has_tables: bool = Field(
        default=False,
        description="Whether the document contains tables.",
    )
    has_figures: bool = Field(
        default=False,
        description="Whether the document contains figures.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Optional user-defined tags for categorization.",
    )
    custom_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional flexible metadata fields.",
    )
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "document_id": "doc_abc123",
                "document_name": "HACCP_Manual_2024.pdf",
                "file_path": "/data/uploaded_docs/HACCP_Manual_2024.pdf",
                "file_size_bytes": 2457600,
                "file_extension": "pdf",
                "document_type": "Manual",
                "title": "HACCP Food Safety Manual 2024",
                "author": "Food Safety Department",
                "language": "en",
                "total_pages": 150,
                "document_hash": "a" * 64,
                "ocr_required": False,
                "has_tables": True,
                "has_figures": True,
                "tags": ["HACCP", "2024", "manual"],
            }
        }


class DocumentStructure(BaseModel):
    """
    Hierarchical structure extracted from a document.
    
    This model represents the document's organizational hierarchy
    which is used for semantic and hierarchical chunking.
    
    Attributes:
        document_id: Reference to the parent document.
        title: Document-level title.
        chapters: List of chapters in the document.
        sections: Flat list of all sections for quick access.
        tables: List of table identifiers and descriptions.
        figures: List of figure identifiers and descriptions.
        references: List of reference sections.
        appendices: List of appendix sections.
    """
    
    document_id: str = Field(
        ...,
        description="Reference to the parent document.",
    )
    title: Optional[str] = Field(
        default=None,
        description="Document-level title.",
    )
    chapters: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of chapters with title, page range, and child sections.",
    )
    sections: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Flat list of all sections for quick access.",
    )
    tables: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of table identifiers, page numbers, and descriptions.",
    )
    figures: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of figure identifiers, page numbers, and captions.",
    )
    references: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of reference sections.",
    )
    appendices: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of appendix sections.",
    )
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "document_id": "doc_abc123",
                "title": "HACCP Food Safety Manual 2024",
                "chapters": [
                    {
                        "title": "Introduction to HACCP",
                        "start_page": 1,
                        "end_page": 10,
                        "sections": [
                            {"title": "Background", "page": 1},
                            {"title": "Scope", "page": 3},
                        ],
                    }
                ],
                "tables": [
                    {"table_id": "T1", "page": 15, "description": "Hazard Analysis Table"},
                ],
            }
        }


# ============================================================================
# DOCUMENT CLASS WITH PROCESSED TABLES SUPPORT
# ============================================================================

# استيراد متأخر لتجنب Circular Import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.ingestion.table_processor import ProcessedTable


class Document(BaseModel):
    """
    Complete document representation including content and metadata.
    
    This is the primary model used after document loading and before chunking.
    
    Attributes:
        metadata: Document metadata.
        structure: Extracted hierarchical structure.
        raw_text: Complete extracted text from the document.
        pages: List of page contents with page numbers.
        tables: Extracted table contents.
        ocr_text: Text extracted via OCR (if applicable).
        language: Detected document language.
        processed_tables: Processed tables from TableProcessor (Phase 3).
    """
    
    metadata: DocumentMetadata = Field(
        ...,
        description="Document metadata.",
    )
    structure: DocumentStructure = Field(
        default_factory=lambda: DocumentStructure(document_id=""),
        description="Extracted hierarchical structure.",
    )
    raw_text: str = Field(
        default="",
        description="Complete extracted text from the document.",
    )
    pages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of page contents with page numbers and text.",
    )
    tables: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Extracted table contents with metadata.",
    )
    ocr_text: Optional[str] = Field(
        default=None,
        description="Text extracted via OCR (if applicable).",
    )
    language: Optional[str] = Field(
        default=None,
        description="Detected document language.",
    )
    processed_tables: list[Any] = Field(
        default_factory=list,
        description="Processed tables from TableProcessor (Phase 3).",
        exclude=True,  # استبعاد من serialization لتجنب مشاكل Pydantic
    )
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "metadata": {
                    "document_id": "doc_abc123",
                    "document_name": "HACCP_Manual_2024.pdf",
                    "file_path": "/data/uploaded_docs/HACCP_Manual_2024.pdf",
                    "file_size_bytes": 2457600,
                    "file_extension": "pdf",
                    "document_hash": "a" * 64,
                },
                "raw_text": "HACCP Food Safety Manual...",
                "language": "en",
            }
        }
    
    def add_processed_table(self, processed_table: Any) -> None:
        """
        Add a processed table to the document.
        
        Args:
            processed_table: ProcessedTable object from TableProcessor.
        """
        if self.processed_tables is None:
            self.processed_tables = []
        self.processed_tables.append(processed_table)
    
    def get_processed_tables(self) -> list[Any]:
        """
        Get all processed tables.
        
        Returns:
            list[Any]: List of ProcessedTable objects.
        """
        return self.processed_tables or []
    
    def has_processed_tables(self) -> bool:
        """
        Check if the document has processed tables.
        
        Returns:
            bool: True if there are processed tables.
        """
        return bool(self.processed_tables)