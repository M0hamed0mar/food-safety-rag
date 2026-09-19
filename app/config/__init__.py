"""
Configuration layer.

Exposes:
    - settings: Pydantic settings loaded from environment
    - constants: Global constants + enums (LogEvent, MetricName)
    - prompts: Prompt templates (domain-agnostic defaults)
"""

from app.config.settings import settings, get_settings
from app.config.constants import (
    APP_NAME,
    APP_VERSION,
    SUPPORTED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    MIN_QUERY_LENGTH,
    MAX_QUERY_LENGTH,
    UNICODE_NORMALIZATION_FORM,
    DEFAULT_RRF_K,
    DEFAULT_CHUNK_OVERLAP_PERCENT,
    BM25_K1,
    BM25_B,
    STREAMING_CHUNK_SIZE,
    DEFAULT_API_TIMEOUT,
    FAISS_INDEX_FLAT,
    FAISS_INDEX_IVFFLAT,
    FAISS_INDEX_HNSW,
    FAISS_INDEX_IVFPQ,
    FAISS_INDEX_DEFAULT,
    FAISS_NLIST_DEFAULT,
    FAISS_NPROBE_DEFAULT,
    FAISS_HNSW_M_DEFAULT,
    FAISS_HNSW_EF_CONSTRUCTION_DEFAULT,
    FAISS_HNSW_EF_SEARCH_DEFAULT,
    LogEvent,
    MetricName,
)

__all__ = [
    "settings",
    "get_settings",
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
    "FAISS_INDEX_FLAT",
    "FAISS_INDEX_IVFFLAT",
    "FAISS_INDEX_HNSW",
    "FAISS_INDEX_IVFPQ",
    "FAISS_INDEX_DEFAULT",
    "FAISS_NLIST_DEFAULT",
    "FAISS_NPROBE_DEFAULT",
    "FAISS_HNSW_M_DEFAULT",
    "FAISS_HNSW_EF_CONSTRUCTION_DEFAULT",
    "FAISS_HNSW_EF_SEARCH_DEFAULT",
    "LogEvent",
    "MetricName",
]
