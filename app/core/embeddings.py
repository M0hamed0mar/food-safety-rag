"""
Local multilingual embeddings generator.

Uses `sentence-transformers` with the `intfloat/multilingual-e5-large`
model (1024-dim), which supports Arabic + English + 90+ other languages.

The model is cached on first use and reused for all subsequent calls
(process-wide singleton).
"""

from __future__ import annotations

import threading
from typing import Any

import structlog

from app.core.exceptions import EmbeddingError


logger = structlog.get_logger(__name__)


# ============================================================
# Configuration
# ============================================================

# Model configuration
MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024

# E5 models require a prefix for best results
#   - "query: " for search queries
#   - "passage: " for documents / chunks
QUERY_PREFIX = ""  # BGE does not require prefixes
PASSAGE_PREFIX = ""  # BGE does not require prefixes

# Batch size for encoding
DEFAULT_BATCH_SIZE = 16


# ============================================================
# Generator
# ============================================================

class EmbeddingsGenerator:
    """
    Local multilingual embeddings generator.

    Uses a process-wide singleton for the underlying model to avoid
    loading it multiple times (which would be slow + memory-heavy).
    """

    _model = None
    _model_lock = threading.Lock()

    def __init__(
        self,
        *,
        model_name: str = MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize: bool = True,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device
        self.dimension = EMBEDDING_DIM

    # --------------------------------------------------------
    # Model loader (lazy, thread-safe)
    # --------------------------------------------------------

    @classmethod
    def _get_model(cls, model_name: str, device: str | None = None):
        if cls._model is not None:
            return cls._model

        with cls._model_lock:
            if cls._model is not None:
                return cls._model

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers",
                    details={"missing": "sentence-transformers"},
                ) from exc

            logger.info("loading_embedding_model", model=model_name, device=device or "auto")

            try:
                cls._model = SentenceTransformer(model_name, device=device)
            except Exception as exc:
                raise EmbeddingError(
                    f"Failed to load embedding model '{model_name}'.",
                    details={"model": model_name, "error": str(exc)},
                ) from exc

            logger.info("embedding_model_loaded", model=model_name)
            return cls._model

    # --------------------------------------------------------
    # Encoding
    # --------------------------------------------------------

    def encode_queries(
        self,
        queries: list[str],
        *,
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """Encode search queries with the E5 'query: ' prefix."""
        prefixed = [QUERY_PREFIX + q.strip() for q in queries if q and q.strip()]
        return self._encode(prefixed, batch_size=batch_size)

    def encode_passages(
        self,
        passages: list[str],
        *,
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """Encode documents / chunks with the E5 'passage: ' prefix."""
        prefixed = [PASSAGE_PREFIX + p.strip() for p in passages if p and p.strip()]
        return self._encode(prefixed, batch_size=batch_size)

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """
        Encode a generic list of texts WITHOUT any prefix.

        Prefer `encode_queries` / `encode_passages` for best retrieval quality.
        """
        return self._encode(texts, batch_size=batch_size)

    # --------------------------------------------------------
    # Internal
    # --------------------------------------------------------

    def _encode(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []

        model = self._get_model(self.model_name, self.device)
        bs = batch_size or self.batch_size

        try:
            vectors = model.encode(
                texts,
                batch_size=bs,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise EmbeddingError(
                "Failed to encode texts.",
                details={"count": len(texts), "error": str(exc)},
            ) from exc

        return vectors.tolist()

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "normalize": self.normalize,
            "device": self.device or "auto",
            "batch_size": self.batch_size,
            "type": "local_multilingual",
        }


# ============================================================
# Singleton
# ============================================================

_embeddings_instance: EmbeddingsGenerator | None = None
_embeddings_lock = threading.Lock()


def get_embeddings() -> EmbeddingsGenerator:
    """Get the process-wide embeddings generator singleton."""
    global _embeddings_instance
    if _embeddings_instance is None:
        with _embeddings_lock:
            if _embeddings_instance is None:
                _embeddings_instance = EmbeddingsGenerator()
    return _embeddings_instance


__all__ = [
    "EmbeddingsGenerator",
    "get_embeddings",
    "MODEL_NAME",
    "EMBEDDING_DIM",
    "QUERY_PREFIX",
    "PASSAGE_PREFIX",
]
