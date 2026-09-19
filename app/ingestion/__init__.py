"""
Ingestion pipeline.

Document → Load → (OCR) → Clean → Chunk → Metadata → Embed → Index

Exposes:
    - DocumentLoader: read raw documents
    - TextCleaner: normalize text
    - SmartOCR: OCR for scanned pages (opt-in)
    - SemanticChunker / HierarchicalChunker: split into chunks
    - MetadataGenerator: build chunk metadata
    - TableProcessor: extract + normalize tables
    - EmbeddingGenerator: legacy-compatible embedding helper
                       (delegates to app.core.embeddings)
"""

from app.ingestion.loader import DocumentLoader
from app.ingestion.cleaner import TextCleaner
from app.ingestion.ocr import SmartOCR
from app.ingestion.chunker import SemanticChunker, HierarchicalChunker
from app.ingestion.metadata import MetadataGenerator
from app.ingestion.embedding import EmbeddingGenerator
from app.ingestion.table_processor import TableProcessor, ProcessedTable

__all__ = [
    "DocumentLoader",
    "TextCleaner",
    "SmartOCR",
    "SemanticChunker",
    "HierarchicalChunker",
    "MetadataGenerator",
    "EmbeddingGenerator",
    "TableProcessor",
    "ProcessedTable",
]
