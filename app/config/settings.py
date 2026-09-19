"""
Application settings.

Loaded from environment variables (see .env.example).
Uses pydantic-settings v2 for validation + type coercion.

Phase A1: Deduplicated + cleaned.
All values preserved from the original. No behavior change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
LOGS_DIR: Path = PROJECT_ROOT / "logs"


# ============================================================
# Settings
# ============================================================

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --------------------------------------------------------
    # App
    # --------------------------------------------------------
    APP_NAME: str = "RAG System"
    APP_ENV: Literal["development", "staging", "production"] = "development"
    APP_DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # --------------------------------------------------------
    # LLM (Groq)
    # --------------------------------------------------------
    LLM_PROVIDER: str = "groq"
    GROQ_API_KEY: SecretStr = Field(default=SecretStr(""))
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_MODEL: str = "qwen/qwen3.8-27b"
    GROQ_TEMPERATURE: float = 0.7
    GROQ_MAX_TOKENS: int = 4096
    GROQ_TIMEOUT: int = 90
    GROQ_MAX_RETRIES: int = 5
    GROQ_MIN_REQUEST_INTERVAL: float = 2.5
    GROQ_RETRY_BACKOFF_MULTIPLIER: float = 2.0
    GROQ_RETRY_BACKOFF_MAX: float = 60.0

    # --------------------------------------------------------
    # Embeddings (local)
    # --------------------------------------------------------
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DIM: int = 1024
    EMBEDDING_BATCH_SIZE: int = 16
    EMBEDDING_DEVICE: str = "cpu"

    # --------------------------------------------------------
    # API Server
    # --------------------------------------------------------
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # --------------------------------------------------------
    # Data paths
    # --------------------------------------------------------
    DATA_DIR: Path = DATA_DIR
    LOGS_DIR: Path = LOGS_DIR

    # --------------------------------------------------------
    # Ingestion
    # --------------------------------------------------------
    MAX_FILE_SIZE_BYTES: int = 100 * 1024 * 1024
    CHUNK_MIN_TOKENS: int = 80
    CHUNK_MAX_TOKENS: int = 400
    CHUNK_OVERLAP_RATIO: float = 0.15

    # OCR (disabled by default)
    OCR_ENABLED: bool = False
    OCR_ZOOM_LEVEL: float = 1.0
    OCR_CONFIDENCE_THRESHOLD: float = 0.7
    OCR_RETRY_ZOOM: float = 2.0

    # --------------------------------------------------------
    # Retrieval (core)
    # --------------------------------------------------------
    DENSE_TOP_K: int = 50
    BM25_TOP_K: int = 50
    RRF_K: int = 60
    RRF_CONSTANT: int = 60
    RRF_CANDIDATE_COUNT: int = 80
    FUSION_CANDIDATES_BEFORE_MMR: int = 40
    CANDIDATES_AFTER_MMR: int = 10
    RERANKER_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.0

    # Hybrid retrieval
    HYBRID_NORMALIZE_SCORES: bool = True
    HYBRID_NORMALIZATION_METHOD: str = "minmax"
    HYBRID_DENSE_WEIGHT: float = 0.55
    HYBRID_BM25_WEIGHT: float = 0.45

    # BM25
    BM25_TOKENIZATION_DEBUG: bool = False
    BM25_DOMAIN_AWARE_TOKENIZATION_DEFAULT: bool = True
    BM25_ARABIC_NORMALIZATION_DEFAULT: bool = True

    # Adaptive retrieval
    ADAPTIVE_RETRIEVAL_ENABLED: bool = True
    ADAPTIVE_DENSE_TOP_K_MAX: int = 120
    ADAPTIVE_BM25_TOP_K_MAX: int = 60
    ADAPTIVE_RRF_CANDIDATE_COUNT_MAX: int = 80
    ADAPTIVE_RERANKER_TOP_K_MAX: int = 15

    # Query complexity
    QUERY_COMPLEXITY_SIMPLE_THRESHOLD: int = 5
    QUERY_COMPLEXITY_COMPLEX_THRESHOLD: int = 15

    # Query quality check
    ENABLE_QUERY_QUALITY_CHECK: bool = True
    MIN_CONTENT_WORDS: int = 1
    MIN_MEANINGFUL_CHARS: int = 5

    # Query expansion (disabled by default)
    QUERY_EXPANSION_NUM_VARIANTS: int = 3

    # Translation (disabled by default)
    TRANSLATION_ENABLED: bool = False
    TRANSLATION_MODE: str = "none"

    # --------------------------------------------------------
    # Reranker
    # --------------------------------------------------------
    RERANKER_ENABLED: bool = True
    RERANKER_USE_GPU: bool = False
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_BATCH_SIZE: int = 16
    RERANKER_MAX_LENGTH: int = 512
    RERANKER_MAX_CANDIDATES: int = 40
    RERANKER_LAZY_LOAD: bool = True
    RERANKER_OOM_RECOVERY: bool = True
    RERANKER_FALLBACK_ENABLED: bool = True
    RERANKER_FALLBACK_STRATEGY: str = "rrf"

    # --------------------------------------------------------
    # Validator (disabled by default)
    # --------------------------------------------------------
    ENABLE_VALIDATOR: bool = True
    ENABLE_OUTLIER_REMOVAL: bool = False
    VALIDATOR_ENABLE_SEMANTIC_DEDUP: bool = False
    VALIDATOR_SEMANTIC_DEDUP_THRESHOLD: float = 0.95
    VALIDATOR_MIN_QUALITY_SCORE: float = 0.3
    VALIDATOR_OUTLIER_THRESHOLD: float = 2.5

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------
    MAX_CONTEXT_CHUNKS: int = 8
    CONTEXT_EXPANSION_ENABLED: bool = False
    CONTEXT_EXPANSION_MAX_NEIGHBORS: int = 1
    CONTEXT_EXPANSION_INCLUDE_PARENT: bool = True
    CONTEXT_EXPANSION_INCLUDE_CHILDREN: bool = False

    # --------------------------------------------------------
    # Retrieval fallback
    # --------------------------------------------------------
    RETRIEVAL_FALLBACK_ENABLED: bool = False
    RETRIEVAL_FALLBACK_MIN_CHUNKS: int = 3
    RETRIEVAL_FALLBACK_MIN_SCORE: float = 0.15

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------
    GENERATION_MAX_TOKENS: int = 2048
    GENERATION_TEMPERATURE: float = 0.0
    CITATION_USE_LLM: bool = False

    # --------------------------------------------------------
    # FAISS / Vector Store
    # --------------------------------------------------------
    FAISS_INDEX_NAME: str = "rag_index"
    FAISS_INDEX_DIR: Path = PROJECT_ROOT / "data" / "faiss_index"
    FAISS_METRIC_TYPE: str = "cosine"
    FAISS_INDEX_TYPE: str = "flat"
    FAISS_NLIST: int = 100
    FAISS_NPROBE: int = 10
    FAISS_HNSW_M: int = 32
    FAISS_HNSW_EF_CONSTRUCTION: int = 200
    FAISS_HNSW_EF_SEARCH: int = 128
    FAISS_IVF_THRESHOLD: int = 10000

    # --------------------------------------------------------
    # Stores
    # --------------------------------------------------------
    METADATA_DB_PATH: Path = PROJECT_ROOT / "data" / "faiss_index" / "metadata.db"
    CHAT_DB_PATH: Path = PROJECT_ROOT / "data" / "chats.db"

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    def ensure_directories(self) -> None:
        """Create required runtime directories."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Singleton
# ============================================================

_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


settings: Settings = get_settings()


__all__ = ["Settings", "settings", "get_settings", "PROJECT_ROOT", "DATA_DIR", "LOGS_DIR"]
