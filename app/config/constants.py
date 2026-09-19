"""
Global constants (domain-agnostic).

Nothing in this file assumes a specific use case.
Any domain-specific values must live in prompts/ or ingestion code.
"""

from __future__ import annotations

# ============================================================
# App metadata
# ============================================================

APP_NAME: str = "RAG System"
APP_VERSION: str = "0.2.0"


# ============================================================
# Supported document types
# ============================================================

SUPPORTED_EXTENSIONS: set[str] = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".html",
    ".xlsx",
    ".xls",
}

MAX_FILE_SIZE_BYTES: int = 100 * 1024 * 1024  # 100 MB


# ============================================================
# Query limits
# ============================================================

MIN_QUERY_LENGTH: int = 2
MAX_QUERY_LENGTH: int = 10_000


# ============================================================
# Unicode
# ============================================================

UNICODE_NORMALIZATION_FORM: str = "NFKC"


# ============================================================
# Retrieval
# ============================================================

DEFAULT_RRF_K: int = 60
DEFAULT_CHUNK_OVERLAP_PERCENT: float = 0.15

BM25_K1: float = 1.5
BM25_B: float = 0.75


# ============================================================
# Streaming
# ============================================================

STREAMING_CHUNK_SIZE: int = 1


# ============================================================
# Timeouts
# ============================================================

DEFAULT_API_TIMEOUT: float = 30.0


__all__ = [
    "APP_NAME",
    "APP_VERSION",
    "SUPPORTED_EXTENSIONS",
    "MAX_FILE_SIZE_BYTES",
    "MIN_QUERY_LENGTH",
    "MAX_QUERY_LENGTH",
    "UNICODE_NORMALIZATION_FORM",
    "DEFAULT_RRF_K",
    "DEFAULT_CHUNK_OVERLAP_PERCENT",
    "BM25_K1",
    "BM25_B",
    "STREAMING_CHUNK_SIZE",
    "DEFAULT_API_TIMEOUT",
]


# ============================================================
# Logging events
# ============================================================

from enum import Enum


class LogEvent(str, Enum):
    """Standardized log event identifiers."""

    # Ingestion
    DOCUMENT_LOAD_START = "document_load_start"
    DOCUMENT_LOAD_COMPLETE = "document_load_complete"
    DOCUMENT_VALIDATION_FAILED = "document_validation_failed"
    OCR_START = "ocr_start"
    OCR_COMPLETE = "ocr_complete"
    OCR_SKIPPED = "ocr_skipped"
    CLEANING_START = "cleaning_start"
    CLEANING_COMPLETE = "cleaning_complete"
    CHUNKING_START = "chunking_start"
    CHUNKING_COMPLETE = "chunking_complete"
    METADATA_GENERATION = "metadata_generation"
    EMBEDDING_START = "embedding_start"
    EMBEDDING_COMPLETE = "embedding_complete"

    # Index
    INDEX_START = "index_start"
    INDEX_COMPLETE = "index_complete"
    INDEX_UPDATE = "index_update"

    # Cache
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    CACHE_CLEAR = "cache_clear"
    CACHE_INVALIDATION = "cache_invalidation"

    # Model
    MODEL_LOAD_START = "model_load_start"
    MODEL_LOAD_COMPLETE = "model_load_complete"

    # Retrieval
    QUERY_RECEIVED = "query_received"
    QUERY_NORMALIZATION = "query_normalization"
    QUERY_EXPANSION = "query_expansion"
    QUERY_REFORMULATION = "query_reformulation"
    DENSE_RETRIEVAL_START = "dense_retrieval_start"
    DENSE_RETRIEVAL_COMPLETE = "dense_retrieval_complete"
    BM25_RETRIEVAL_START = "bm25_retrieval_start"
    BM25_RETRIEVAL_COMPLETE = "bm25_retrieval_complete"
    RRF_FUSION = "rrf_fusion"
    RERANKING_START = "reranking_start"
    RERANKING_COMPLETE = "reranking_complete"
    RERANKING_SKIPPED = "reranking_skipped"
    CONTEXT_BUILDING = "context_building"
    CONTEXT_EXPANSION = "context_expansion"
    METADATA_FILTER_APPLIED = "metadata_filter_applied"

    # Generation
    GENERATION_START = "generation_start"
    GENERATION_COMPLETE = "generation_complete"
    STREAMING_START = "streaming_start"
    STREAMING_COMPLETE = "streaming_complete"
    STREAMING_INTERRUPTED = "streaming_interrupted"

    # System
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    ERROR = "error"
    WARNING = "warning"

    # Latency
    LATENCY_RECORDED = "latency_recorded"
    TOTAL_REQUEST_LATENCY_MS = "total_request_latency_ms"


class MetricName(str, Enum):
    """Standardized metric names."""

    # Latency
    OCR_LATENCY_MS = "ocr_latency_ms"
    EMBEDDING_LATENCY_MS = "embedding_latency_ms"
    DENSE_RETRIEVAL_LATENCY_MS = "dense_retrieval_latency_ms"
    BM25_LATENCY_MS = "bm25_latency_ms"
    FUSION_LATENCY_MS = "fusion_latency_ms"
    RERANKING_LATENCY_MS = "reranking_latency_ms"
    GENERATION_LATENCY_MS = "generation_latency_ms"
    TOTAL_REQUEST_LATENCY_MS = "total_request_latency_ms"

    # Quality
    RECALL_AT_K = "recall_at_k"
    PRECISION_AT_K = "precision_at_k"
    MRR = "mean_reciprocal_rank"
    HIT_RATE = "hit_rate"

    # Throughput
    DOCUMENTS_PER_SECOND = "documents_per_second"
    CHUNKS_PER_SECOND = "chunks_per_second"
    QUERIES_PER_SECOND = "queries_per_second"


# ============================================================
# FAISS index types
# ============================================================

FAISS_INDEX_FLAT: str = "flat"
FAISS_INDEX_IVFFLAT: str = "ivf"
FAISS_INDEX_HNSW: str = "hnsw"
FAISS_INDEX_IVFPQ: str = "ivfpq"
FAISS_INDEX_DEFAULT: str = FAISS_INDEX_FLAT

# FAISS IVF parameters
FAISS_NLIST_DEFAULT: int = 100
FAISS_NPROBE_DEFAULT: int = 10

# FAISS HNSW parameters
FAISS_HNSW_M_DEFAULT: int = 32
FAISS_HNSW_EF_CONSTRUCTION_DEFAULT: int = 200
FAISS_HNSW_EF_SEARCH_DEFAULT: int = 128


# ============================================================
# Additional constants (auto-added for retrieval compatibility)
# ============================================================

RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
RERANKER_MAX_CANDIDATES: int = 25
RERANKER_BATCH_SIZE: int = 16
RERANKER_MAX_LENGTH: int = 512
RERANKER_LAZY_LOAD_DEFAULT: bool = True
RERANKER_OOM_RECOVERY_DEFAULT: bool = True
FUSION_CANDIDATES_BEFORE_MMR: int = 40
CANDIDATES_AFTER_MMR: int = 10
MMR_POSITION: str = "after_reranker"
VALIDATOR_OUTLIER_THRESHOLD: float = 2.5
VALIDATOR_ENABLE_SEMANTIC_DEDUP_DEFAULT: bool = True
VALIDATOR_SEMANTIC_DEDUP_THRESHOLD_DEFAULT: float = 0.95
VALIDATOR_MIN_QUALITY_SCORE_DEFAULT: float = 0.1
HYBRID_NORMALIZE_SCORES_DEFAULT: bool = True
HYBRID_NORMALIZATION_METHOD_DEFAULT: str = "minmax"
BM25_DOMAIN_AWARE_TOKENIZATION_DEFAULT: bool = True
BM25_ARABIC_NORMALIZATION_DEFAULT: bool = True
CHUNK_MAX_TOKENS_DEFAULT: int = 600
CHUNK_OVERLAP_RATIO_DEFAULT: float = 0.15
ADAPTIVE_DENSE_TOP_K_BASE: int = 30
ADAPTIVE_BM25_TOP_K_BASE: int = 30
ADAPTIVE_RRF_CANDIDATE_COUNT_BASE: int = 15
ADAPTIVE_RERANKER_TOP_K_BASE: int = 5
ADAPTIVE_DENSE_TOP_K_MAX: int = 100
ADAPTIVE_BM25_TOP_K_MAX: int = 100
ADAPTIVE_RRF_CANDIDATE_COUNT_MAX: int = 40
ADAPTIVE_RERANKER_TOP_K_MAX: int = 10
QUERY_COMPLEXITY_SIMPLE_THRESHOLD: int = 5
QUERY_COMPLEXITY_COMPLEX_THRESHOLD: int = 15
DOCUMENT_REFERENCE_KEYWORDS: set = set()
STANDARD_KEYWORDS: set = set()
SECTION_REFERENCE_KEYWORDS: set = set()
CITATION_FORMAT: str = "[{doc_name}, p.{page}, {section}]"
