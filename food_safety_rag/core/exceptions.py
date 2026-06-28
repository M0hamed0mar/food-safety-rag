"""
Custom exception definitions for the Food Safety RAG System.

All exceptions inherit from a base application exception for consistent error handling.
Exceptions include meaningful context to aid debugging.
"""


class FoodSafetyRAGException(Exception):
    """
    Base exception for all Food Safety RAG system errors.
    
    All specific exceptions inherit from this to allow catching all system errors.
    """

    def __init__(self, message: str, error_code: str | None = None) -> None:
        """
        Initialize exception with message and optional error code.
        
        Args:
            message: Human-readable error message.
            error_code: Optional error code for categorization.
        """
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        super().__init__(self.message)


# Document Loading and Validation Exceptions


class DocumentLoadingException(FoodSafetyRAGException):
    """Raised when document loading fails."""

    def __init__(self, message: str, file_path: str | None = None) -> None:
        """
        Initialize document loading exception.
        
        Args:
            message: Error description.
            file_path: Path to the problematic document.
        """
        self.file_path = file_path
        super().__init__(message, error_code="DOCUMENT_LOAD_ERROR")


class DocumentValidationException(FoodSafetyRAGException):
    """Raised when document validation fails."""

    def __init__(self, message: str, validation_reason: str | None = None) -> None:
        """
        Initialize document validation exception.
        
        Args:
            message: Error description.
            validation_reason: Specific validation failure reason.
        """
        self.validation_reason = validation_reason
        super().__init__(message, error_code="DOCUMENT_VALIDATION_ERROR")


class DocumentCorruptedException(FoodSafetyRAGException):
    """Raised when document is corrupted or encrypted."""

    def __init__(self, message: str) -> None:
        """
        Initialize document corruption exception.
        
        Args:
            message: Error description.
        """
        super().__init__(message, error_code="DOCUMENT_CORRUPTED")


class UnsupportedDocumentFormatException(FoodSafetyRAGException):
    """Raised when document format is not supported."""

    def __init__(self, message: str, file_extension: str | None = None) -> None:
        """
        Initialize unsupported format exception.
        
        Args:
            message: Error description.
            file_extension: The unsupported file extension.
        """
        self.file_extension = file_extension
        super().__init__(message, error_code="UNSUPPORTED_FORMAT")


# OCR Exceptions


class OCRException(FoodSafetyRAGException):
    """Raised when OCR processing fails."""

    def __init__(self, message: str, page_number: int | None = None) -> None:
        """
        Initialize OCR exception.
        
        Args:
            message: Error description.
            page_number: Page where OCR failed.
        """
        self.page_number = page_number
        super().__init__(message, error_code="OCR_ERROR")


# Chunking and Text Processing Exceptions


class ChunkingException(FoodSafetyRAGException):
    """Raised when chunking fails."""

    def __init__(self, message: str) -> None:
        """
        Initialize chunking exception.
        
        Args:
            message: Error description.
        """
        super().__init__(message, error_code="CHUNKING_ERROR")


class TextProcessingException(FoodSafetyRAGException):
    """Raised when text processing fails."""

    def __init__(self, message: str) -> None:
        """
        Initialize text processing exception.
        
        Args:
            message: Error description.
        """
        super().__init__(message, error_code="TEXT_PROCESSING_ERROR")


# Embedding Exceptions


class EmbeddingException(FoodSafetyRAGException):
    """Raised when embedding generation fails."""

    def __init__(self, message: str, model: str | None = None) -> None:
        """
        Initialize embedding exception.
        
        Args:
            message: Error description.
            model: The embedding model that failed.
        """
        self.model = model
        super().__init__(message, error_code="EMBEDDING_ERROR")


class EmbeddingAPIException(FoodSafetyRAGException):
    """Raised when embedding API call fails."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        """
        Initialize embedding API exception.
        
        Args:
            message: Error description.
            status_code: HTTP status code.
        """
        self.status_code = status_code
        super().__init__(message, error_code="EMBEDDING_API_ERROR")


# Vector Store Exceptions


class VectorStoreException(FoodSafetyRAGException):
    """Raised when vector store operation fails."""

    def __init__(self, message: str) -> None:
        """
        Initialize vector store exception.
        
        Args:
            message: Error description.
        """
        super().__init__(message, error_code="VECTOR_STORE_ERROR")


class IndexException(FoodSafetyRAGException):
    """Raised when FAISS index operation fails."""

    def __init__(self, message: str) -> None:
        """
        Initialize index exception.
        
        Args:
            message: Error description.
        """
        super().__init__(message, error_code="INDEX_ERROR")


# Retrieval Exceptions


class RetrievalException(FoodSafetyRAGException):
    """Raised when retrieval fails."""

    def __init__(self, message: str, query: str | None = None) -> None:
        """
        Initialize retrieval exception.
        
        Args:
            message: Error description.
            query: The query that failed retrieval.
        """
        self.query = query
        super().__init__(message, error_code="RETRIEVAL_ERROR")


class DenseRetrievalException(RetrievalException):
    """Raised when dense retrieval fails."""

    def __init__(self, message: str, query: str | None = None) -> None:
        """
        Initialize dense retrieval exception.
        
        Args:
            message: Error description.
            query: The query that failed.
        """
        super().__init__(message, query)
        self.error_code = "DENSE_RETRIEVAL_ERROR"


class BM25RetrievalException(RetrievalException):
    """Raised when BM25 retrieval fails."""

    def __init__(self, message: str, query: str | None = None) -> None:
        """
        Initialize BM25 retrieval exception.
        
        Args:
            message: Error description.
            query: The query that failed.
        """
        super().__init__(message, query)
        self.error_code = "BM25_RETRIEVAL_ERROR"


class RerankingException(FoodSafetyRAGException):
    """Raised when reranking fails."""

    def __init__(self, message: str, model: str | None = None) -> None:
        """
        Initialize reranking exception.
        
        Args:
            message: Error description.
            model: The reranker model used.
        """
        self.model = model
        super().__init__(message, error_code="RERANKING_ERROR")


# Query Exceptions


class QueryException(FoodSafetyRAGException):
    """Raised when query processing fails."""

    def __init__(self, message: str, query: str | None = None) -> None:
        """
        Initialize query exception.
        
        Args:
            message: Error description.
            query: The problematic query.
        """
        self.query = query
        super().__init__(message, error_code="QUERY_ERROR")


class QueryValidationException(QueryException):
    """Raised when query validation fails."""

    def __init__(self, message: str, query: str | None = None) -> None:
        """
        Initialize query validation exception.
        
        Args:
            message: Error description.
            query: The invalid query.
        """
        super().__init__(message, query)
        self.error_code = "QUERY_VALIDATION_ERROR"


class QueryNormalizationException(QueryException):
    """Raised when query normalization fails."""

    def __init__(self, message: str, query: str | None = None) -> None:
        """
        Initialize query normalization exception.
        
        Args:
            message: Error description.
            query: The query that failed normalization.
        """
        super().__init__(message, query)
        self.error_code = "QUERY_NORMALIZATION_ERROR"


# Generation Exceptions


class GenerationException(FoodSafetyRAGException):
    """Raised when answer generation fails."""

    def __init__(self, message: str, model: str | None = None) -> None:
        """
        Initialize generation exception.
        
        Args:
            message: Error description.
            model: The LLM model used.
        """
        self.model = model
        super().__init__(message, error_code="GENERATION_ERROR")


class GenerationAPIException(GenerationException):
    """Raised when LLM API call fails."""

    def __init__(
        self, message: str, status_code: int | None = None, model: str | None = None
    ) -> None:
        """
        Initialize generation API exception.
        
        Args:
            message: Error description.
            status_code: HTTP status code.
            model: The LLM model used.
        """
        self.status_code = status_code
        super().__init__(message, model)
        self.error_code = "GENERATION_API_ERROR"


# Cache Exceptions


class CacheException(FoodSafetyRAGException):
    """Raised when cache operation fails."""

    def __init__(self, message: str, cache_key: str | None = None) -> None:
        """
        Initialize cache exception.
        
        Args:
            message: Error description.
            cache_key: The cache key involved.
        """
        self.cache_key = cache_key
        super().__init__(message, error_code="CACHE_ERROR")


# Configuration Exceptions


class ConfigurationException(FoodSafetyRAGException):
    """Raised when configuration is invalid."""

    def __init__(self, message: str, setting: str | None = None) -> None:
        """
        Initialize configuration exception.
        
        Args:
            message: Error description.
            setting: The problematic setting.
        """
        self.setting = setting
        super().__init__(message, error_code="CONFIGURATION_ERROR")
