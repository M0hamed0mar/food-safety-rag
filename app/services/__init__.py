"""
Services package.

High-level orchestration:
    - IngestionService: document ingestion pipeline
    - RetrievalService: query → context pipeline
    - GenerationService: context → answer pipeline
"""

from app.services.ingestion_service import IngestionService
from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService

__all__ = [
    "IngestionService",
    "RetrievalService",
    "GenerationService",
]
