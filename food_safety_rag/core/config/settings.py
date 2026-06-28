"""
Settings management for the Food Safety RAG System.

This module handles environment-based configuration using Pydantic.
All secrets and configurable values are read from environment variables.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Uses Pydantic's BaseSettings for validation and type coercion.
    All sensitive values (API keys) must be provided via environment variables.
    Never commit secrets to version control.
    """

    # API Keys and Credentials
    openai_api_key: str
    """OpenAI API key for embeddings and LLM generation. Required."""

    huggingface_api_key: Optional[str] = None
    """Hugging Face API key for reranker model (optional, for private models)."""

    # OpenAI Configuration
    openai_embedding_model: str = "text-embedding-3-small"
    """OpenAI embedding model to use."""

    openai_llm_model: str = "gpt-4o-mini"
    """OpenAI language model to use."""

    openai_embedding_batch_size: int = 100
    """Batch size for embedding API requests."""

    openai_request_timeout: int = 30
    """Timeout for OpenAI API requests in seconds."""

    openai_max_retries: int = 3
    """Maximum number of retries for OpenAI API calls."""

    # Retrieval Configuration
    dense_top_k: int = 50
    """Number of candidates from dense retrieval."""

    bm25_top_k: int = 50
    """Number of candidates from BM25 retrieval."""

    rrf_constant: int = 60
    """RRF fusion constant."""

    reranker_top_k: int = 30
    """Number of candidates after fusion before reranking."""

    final_context_k: int = 6
    """Number of final chunks in context."""

    similarity_threshold: float = 0.5
    """Minimum similarity threshold for relevance."""

    # Chunking Configuration
    chunk_size: int = 500
    """Target chunk size in tokens."""

    chunk_overlap: int = 50
    """Token overlap between chunks."""

    min_chunk_size: int = 100
    """Minimum chunk size."""

    max_chunk_size: int = 1000
    """Maximum chunk size."""

    # LLM Configuration
    llm_temperature: float = 0.3
    """LLM temperature for generation."""

    llm_max_tokens: int = 2048
    """Maximum tokens for LLM response."""

    # OCR Configuration
    enable_ocr: bool = True
    """Enable OCR for scanned pages."""

    ocr_batch_size: int = 5
    """Batch size for OCR processing."""

    skip_ocr_confidence_threshold: float = 0.8
    """Confidence threshold to skip OCR."""

    # Cache Configuration
    enable_embedding_cache: bool = True
    """Enable embedding cache."""

    enable_retrieval_cache: bool = True
    """Enable retrieval cache."""

    enable_context_cache: bool = True
    """Enable context cache."""

    enable_llm_response_cache: bool = True
    """Enable LLM response cache."""

    cache_embedding_ttl: int = 604800
    """Embedding cache TTL in seconds (7 days)."""

    cache_retrieval_ttl: int = 3600
    """Retrieval cache TTL in seconds (1 hour)."""

    cache_context_ttl: int = 1800
    """Context cache TTL in seconds (30 minutes)."""

    cache_llm_response_ttl: int = 3600
    """LLM response cache TTL in seconds (1 hour)."""

    # Logging Configuration
    log_level: str = "INFO"
    """Logging level."""

    structured_logs: bool = True
    """Use structured logging format."""

    # Path Configuration
    uploaded_docs_dir: str = "data/uploaded_docs"
    """Directory for uploaded documents."""

    faiss_index_dir: str = "data/faiss_index"
    """Directory for FAISS index."""

    cache_dir: str = "data/cache"
    """Directory for cache files."""

    logs_dir: str = "logs"
    """Directory for logs."""

    # FastAPI Configuration
    api_host: str = "0.0.0.0"
    """API host address."""

    api_port: int = 8000
    """API port."""

    api_debug: bool = False
    """Enable debug mode."""

    api_reload: bool = False
    """Enable auto-reload in development."""

    # System Configuration
    environment: str = "development"
    """Environment: development, staging, production."""

    enable_monitoring: bool = True
    """Enable performance monitoring."""

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def uploaded_docs_path(self) -> Path:
        """Get uploaded documents directory as Path object."""
        return Path(self.uploaded_docs_dir)

    @property
    def faiss_index_path(self) -> Path:
        """Get FAISS index directory as Path object."""
        return Path(self.faiss_index_dir)

    @property
    def cache_path(self) -> Path:
        """Get cache directory as Path object."""
        return Path(self.cache_dir)

    @property
    def logs_path(self) -> Path:
        """Get logs directory as Path object."""
        return Path(self.logs_dir)

    def ensure_directories(self) -> None:
        """
        Create all required directories if they don't exist.
        
        Called during application initialization to ensure the directory
        structure is ready for operations.
        """
        for directory in [
            self.uploaded_docs_path,
            self.faiss_index_path,
            self.cache_path,
            self.logs_path,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """
    Factory function to load and cache settings.
    
    Returns:
        Settings: Validated settings instance loaded from environment variables.
        
    Raises:
        ValidationError: If required environment variables are missing or invalid.
    """
    return Settings()
