"""
Metadata schema definitions.

This module defines Pydantic models for metadata-related schemas
used across the ingestion, retrieval, and generation pipelines.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class IngestionMetadata(BaseModel):
    """
    Metadata captured during the document ingestion process.
    
    Tracks the complete lifecycle of document processing from loading
    through chunking and indexing.
    
    Attributes:
        ingestion_id: Unique identifier for this ingestion run.
        document_id: Reference to the ingested document.
        document_name: Name of the ingested document.
        start_time: When ingestion started.
        end_time: When ingestion completed.
        stages: List of completed processing stages.
        stage_timings: Timing information for each stage in milliseconds.
        chunk_count: Number of chunks generated.
        embedding_count: Number of embeddings generated.
        cache_hits: Number of embedding cache hits.
        cache_misses: Number of embedding cache misses.
        ocr_pages: Number of pages processed with OCR.
        skipped_ocr_pages: Number of pages where OCR was skipped.
        errors: List of errors encountered during ingestion.
        warnings: List of warnings encountered during ingestion.
        status: Current status of ingestion.
    """
    
    ingestion_id: str = Field(
        ...,
        description="Unique identifier for this ingestion run.",
        min_length=1,
    )
    document_id: str = Field(
        ...,
        description="Reference to the ingested document.",
        min_length=1,
    )
    document_name: str = Field(
        ...,
        description="Name of the ingested document.",
        min_length=1,
    )
    start_time: datetime = Field(
        default_factory=datetime.utcnow,
        description="When ingestion started.",
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="When ingestion completed.",
    )
    stages: list[str] = Field(
        default_factory=list,
        description="List of completed processing stages.",
    )
    stage_timings: dict[str, float] = Field(
        default_factory=dict,
        description="Timing information for each stage in milliseconds.",
    )
    chunk_count: int = Field(
        default=0,
        description="Number of chunks generated.",
        ge=0,
    )
    embedding_count: int = Field(
        default=0,
        description="Number of embeddings generated.",
        ge=0,
    )
    cache_hits: int = Field(
        default=0,
        description="Number of embedding cache hits.",
        ge=0,
    )
    cache_misses: int = Field(
        default=0,
        description="Number of embedding cache misses.",
        ge=0,
    )
    ocr_pages: int = Field(
        default=0,
        description="Number of pages processed with OCR.",
        ge=0,
    )
    skipped_ocr_pages: int = Field(
        default=0,
        description="Number of pages where OCR was skipped.",
        ge=0,
    )
    errors: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of errors encountered during ingestion.",
    )
    warnings: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of warnings encountered during ingestion.",
    )
    status: str = Field(
        default="in_progress",
        description="Current status of ingestion (in_progress, completed, failed, partial).",
    )
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "ingestion_id": "ing_001",
                "document_id": "doc_abc123",
                "document_name": "HACCP_Manual_2024.pdf",
                "start_time": "2024-01-15T10:00:00Z",
                "end_time": "2024-01-15T10:05:30Z",
                "stages": ["load", "validate", "parse", "ocr", "clean", "chunk", "embed", "index"],
                "stage_timings": {
                    "load": 150.0,
                    "parse": 800.0,
                    "ocr": 1200.0,
                    "clean": 200.0,
                    "chunk": 300.0,
                    "embed": 2500.0,
                    "index": 500.0,
                },
                "chunk_count": 381,
                "embedding_count": 381,
                "cache_hits": 0,
                "cache_misses": 381,
                "ocr_pages": 12,
                "skipped_ocr_pages": 138,
                "status": "completed",
            }
        }
    
    def add_stage(self, stage_name: str, duration_ms: float) -> None:
        """
        Record a completed processing stage.
        
        Args:
            stage_name: Name of the completed stage.
            duration_ms: Duration of the stage in milliseconds.
        """
        self.stages.append(stage_name)
        self.stage_timings[stage_name] = duration_ms
    
    def add_error(self, error_message: str, stage: Optional[str] = None, details: Optional[dict[str, Any]] = None) -> None:
        """
        Record an error encountered during ingestion.
        
        Args:
            error_message: Description of the error.
            stage: Stage where the error occurred.
            details: Additional error details.
        """
        error_entry: dict[str, Any] = {
            "message": error_message,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if stage:
            error_entry["stage"] = stage
        if details:
            error_entry["details"] = details
        
        self.errors.append(error_entry)
    
    def add_warning(self, warning_message: str, stage: Optional[str] = None, details: Optional[dict[str, Any]] = None) -> None:
        """
        Record a warning encountered during ingestion.
        
        Args:
            warning_message: Description of the warning.
            stage: Stage where the warning occurred.
            details: Additional warning details.
        """
        warning_entry: dict[str, Any] = {
            "message": warning_message,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if stage:
            warning_entry["stage"] = stage
        if details:
            warning_entry["details"] = details
        
        self.warnings.append(warning_entry)
    
    def complete(self, status: str = "completed") -> None:
        """
        Mark ingestion as complete.
        
        Args:
            status: Final status (completed, failed, partial).
        """
        self.end_time = datetime.utcnow()
        self.status = status
    
    def get_total_duration_ms(self) -> Optional[float]:
        """
        Calculate total ingestion duration.
        
        Returns:
            Optional[float]: Total duration in milliseconds, or None if not complete.
        """
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None


class RetrievalMetadata(BaseModel):
    """
    Metadata captured during the retrieval process.
    
    Tracks query processing, retrieval strategies, and result quality.
    
    Attributes:
        retrieval_id: Unique identifier for this retrieval run.
        query_id: Reference to the original query.
        query_text: The processed query text.
        start_time: When retrieval started.
        end_time: When retrieval completed.
        dense_results: Number of results from dense retrieval.
        sparse_results: Number of results from sparse retrieval.
        fused_results: Number of results after fusion.
        reranked_results: Number of results after reranking.
        final_results: Number of results in final context.
        dense_latency_ms: Dense retrieval latency.
        sparse_latency_ms: Sparse retrieval latency.
        fusion_latency_ms: Fusion latency.
        reranking_latency_ms: Reranking latency.
        total_latency_ms: Total retrieval latency.
        cache_hit: Whether retrieval results were cached.
        expanded_queries: Number of expanded queries used.
        filters_applied: Metadata filters applied during retrieval.
    """
    
    retrieval_id: str = Field(
        ...,
        description="Unique identifier for this retrieval run.",
        min_length=1,
    )
    query_id: str = Field(
        ...,
        description="Reference to the original query.",
        min_length=1,
    )
    query_text: str = Field(
        ...,
        description="The processed query text.",
        min_length=1,
    )
    start_time: datetime = Field(
        default_factory=datetime.utcnow,
        description="When retrieval started.",
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="When retrieval completed.",
    )
    dense_results: int = Field(
        default=0,
        description="Number of results from dense retrieval.",
        ge=0,
    )
    sparse_results: int = Field(
        default=0,
        description="Number of results from sparse retrieval.",
        ge=0,
    )
    fused_results: int = Field(
        default=0,
        description="Number of results after fusion.",
        ge=0,
    )
    reranked_results: int = Field(
        default=0,
        description="Number of results after reranking.",
        ge=0,
    )
    final_results: int = Field(
        default=0,
        description="Number of results in final context.",
        ge=0,
    )
    dense_latency_ms: Optional[float] = Field(
        default=None,
        description="Dense retrieval latency in milliseconds.",
        ge=0.0,
    )
    sparse_latency_ms: Optional[float] = Field(
        default=None,
        description="Sparse retrieval latency in milliseconds.",
        ge=0.0,
    )
    fusion_latency_ms: Optional[float] = Field(
        default=None,
        description="Fusion latency in milliseconds.",
        ge=0.0,
    )
    reranking_latency_ms: Optional[float] = Field(
        default=None,
        description="Reranking latency in milliseconds.",
        ge=0.0,
    )
    total_latency_ms: Optional[float] = Field(
        default=None,
        description="Total retrieval latency in milliseconds.",
        ge=0.0,
    )
    cache_hit: bool = Field(
        default=False,
        description="Whether retrieval results were cached.",
    )
    expanded_queries: int = Field(
        default=0,
        description="Number of expanded queries used.",
        ge=0,
    )
    filters_applied: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata filters applied during retrieval.",
    )
    
    def complete(self) -> None:
        """Mark retrieval as complete and calculate total latency."""
        self.end_time = datetime.utcnow()
        if self.dense_latency_ms and self.sparse_latency_ms and self.fusion_latency_ms and self.reranking_latency_ms:
            self.total_latency_ms = (
                self.dense_latency_ms
                + self.sparse_latency_ms
                + self.fusion_latency_ms
                + self.reranking_latency_ms
            )
    
    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of retrieval performance.
        
        Returns:
            dict[str, Any]: Retrieval performance summary.
        """
        return {
            "retrieval_id": self.retrieval_id,
            "query_id": self.query_id,
            "total_results_pipeline": {
                "dense": self.dense_results,
                "sparse": self.sparse_results,
                "fused": self.fused_results,
                "reranked": self.reranked_results,
                "final": self.final_results,
            },
            "latency_ms": {
                "dense": self.dense_latency_ms,
                "sparse": self.sparse_latency_ms,
                "fusion": self.fusion_latency_ms,
                "reranking": self.reranking_latency_ms,
                "total": self.total_latency_ms,
            },
            "cache_hit": self.cache_hit,
            "expanded_queries": self.expanded_queries,
        }


class CacheEntryMetadata(BaseModel):
    """
    Metadata for a cache entry.
    
    Tracks cache usage, expiration, and hit statistics.
    
    Attributes:
        key: Cache key.
        cache_type: Type of cache (embedding, retrieval, llm, context).
        created_at: When the entry was created.
        expires_at: When the entry expires.
        access_count: Number of times this entry was accessed.
        last_accessed: When the entry was last accessed.
        size_bytes: Approximate size of the cached data in bytes.
        hit_count: Number of cache hits for this entry.
    """
    
    key: str = Field(
        ...,
        description="Cache key.",
        min_length=1,
    )
    cache_type: str = Field(
        ...,
        description="Type of cache (embedding, retrieval, llm, context).",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the entry was created.",
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="When the entry expires.",
    )
    access_count: int = Field(
        default=0,
        description="Number of times this entry was accessed.",
        ge=0,
    )
    last_accessed: Optional[datetime] = Field(
        default=None,
        description="When the entry was last accessed.",
    )
    size_bytes: Optional[int] = Field(
        default=None,
        description="Approximate size of the cached data in bytes.",
        ge=0,
    )
    hit_count: int = Field(
        default=0,
        description="Number of cache hits for this entry.",
        ge=0,
    )
    
    def record_access(self) -> None:
        """Record an access to this cache entry."""
        self.access_count += 1
        self.hit_count += 1
        self.last_accessed = datetime.utcnow()
    
    def is_expired(self) -> bool:
        """
        Check if this cache entry has expired.
        
        Returns:
            bool: True if expired, False otherwise.
        """
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at