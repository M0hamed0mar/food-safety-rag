"""
Embedding generator (ingestion-side).

This module is a thin wrapper around `app.core.embeddings.EmbeddingsGenerator`.
It preserves the API that the rest of the pipeline expects
(`embed_text`, `embed_texts`, `embed_chunks`, `_prepare_text_for_embedding`),
but the actual work is delegated to the local multilingual model.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Optional

from app.config import settings
from app.config.constants import LogEvent
from app.core.embeddings import (
    EmbeddingsGenerator as _CoreEmbeddingsGenerator,
    get_embeddings as _get_core_embeddings,
    EMBEDDING_DIM as _CORE_DIM,
)
from app.core.exceptions import EmbeddingError
from app.monitoring import get_logger, measure_latency
from app.schemas import Chunk


logger = get_logger("app.ingestion.embedding")


# ============================================================================
# Retry strategy (kept for API compatibility; not actively used since the
# local model has no rate limits)
# ============================================================================

class RetryStrategy(Enum):
    RATE_LIMIT = "rate_limit"
    TRANSIENT_ERROR = "transient_error"
    FATAL = "fatal"


# ============================================================================
# Constants
# ============================================================================

LANGUAGE_ARABIC = "ar"
SOURCE_TYPE_TABLE = "table"


# ============================================================================
# EmbeddingGenerator
# ============================================================================

class EmbeddingGenerator:
    """
    Embedding generator facade.

    Delegates to a process-wide `app.core.embeddings.EmbeddingsGenerator`
    singleton. Keeps the historical API used by the ingestion pipeline.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ) -> None:
        self.model_name = str(model_name or settings.EMBEDDING_MODEL)
        self.batch_size = int(batch_size or settings.EMBEDDING_BATCH_SIZE)
        self.max_retries = int(max_retries or 0)
        self.retry_delay = float(retry_delay or 0.0)
        self.dimension = int(settings.EMBEDDING_DIM)

        # Delegate to core embeddings singleton
        self._core: _CoreEmbeddingsGenerator = _get_core_embeddings()

    # ------------------------------------------------------------------
    # Basic embedding API
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text (generic; no E5 prefix)."""
        if not text or not text.strip():
            return [0.0] * self.dimension

        with measure_latency("embedding_generation") as latency:
            vecs = self._core.encode([text])
            latency.stop(text_length=len(text), model=self.model_name)

        return vecs[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts (generic; no E5 prefix)."""
        if not texts:
            return []

        with measure_latency("embedding_generation") as latency:
            vecs = self._core.encode(texts, batch_size=self.batch_size)
            latency.stop(total_texts=len(texts), model=self.model_name)

        return vecs

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query (uses E5 'query: ' prefix)."""
        if not query or not query.strip():
            return [0.0] * self.dimension
        return self._core.encode_queries([query])[0]

    # ------------------------------------------------------------------
    # Chunk embedding (used by the ingestion pipeline)
    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Generate embeddings for a list of chunks and attach them.

        Uses the E5 'passage: ' prefix for the embedding text — this is
        what the local multilingual model expects for best quality.
        """
        if not chunks:
            return []

        embedding_texts = [self._prepare_text_for_embedding(c) for c in chunks]

        with measure_latency("embedding_generation") as latency:
            embeddings = self._core.encode_passages(
                embedding_texts,
                batch_size=self.batch_size,
            )
            latency.stop(
                total_texts=len(chunks),
                model=self.model_name,
            )

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
            if chunk.metadata:
                chunk.metadata.embedding_model = self.model_name

        table_count = sum(1 for c in chunks if self._is_table_chunk(c))

        logger.log_ingestion(
            event=LogEvent.EMBEDDING_COMPLETE,
            document_id="chunks",
            document_name="chunk_embedding",
            message=f"Embedded {len(chunks)} chunks",
            details={
                "chunk_count": len(chunks),
                "model": self.model_name,
                "embedding_dimension": self.dimension,
                "table_count": table_count,
            },
        )

        return chunks

    # ------------------------------------------------------------------
    # Text preparation helpers
    # ------------------------------------------------------------------

    def _is_table_chunk(self, chunk: Chunk) -> bool:
        if chunk.metadata.source_type == SOURCE_TYPE_TABLE:
            return True
        if chunk.metadata.table_data:
            return True
        if getattr(chunk.metadata, "is_extracted_table", False):
            return True
        content = chunk.content or ""
        if "|" in content or "\t" in content:
            lines = content.split("\n")
            if len(lines) > 1:
                pipe_lines = sum(1 for l in lines if "|" in l or "\t" in l)
                if pipe_lines >= 2:
                    return True
        return False

    def _prepare_table_for_embedding(self, chunk: Chunk) -> str:
        # Prefer pre-generated embedding_text
        if getattr(chunk.metadata, "embedding_text", None):
            return chunk.metadata.embedding_text

        parts: list[str] = []

        if chunk.metadata.section:
            parts.append(f"[Section: {chunk.metadata.section}]")
        elif chunk.metadata.chapter:
            parts.append(f"[Chapter: {chunk.metadata.chapter}]")

        if chunk.metadata.title:
            parts.append(f"[Table: {chunk.metadata.title}]")

        if chunk.metadata.table_data:
            try:
                data = json.loads(chunk.metadata.table_data)
                headers = data.get("headers", [])
                rows = data.get("rows", [])
                parts.append(f"Structure: {data.get('row_count', len(rows))} rows, {data.get('column_count', len(headers))} columns")
                if headers:
                    parts.append(f"Headers: {', '.join(headers)}")
                for idx, row in enumerate(rows, 1):
                    parts.append(f"Row {idx}: " + " | ".join(str(c) for c in row))
            except (json.JSONDecodeError, TypeError):
                pass

        if not parts:
            parts.append(chunk.content)

        return "\n".join(parts)

    def _prepare_text_for_embedding(self, chunk: Chunk) -> str:
        """
        Prepare chunk text for embedding.

        - Table chunks: use structured representation
        - Text chunks: prepend light context (section / heading)
        """
        if self._is_table_chunk(chunk):
            return self._prepare_table_for_embedding(chunk)

        parts: list[str] = []

        if chunk.metadata.section:
            parts.append(f"[Section: {chunk.metadata.section}]")
        elif chunk.metadata.chapter:
            parts.append(f"[Chapter: {chunk.metadata.chapter}]")

        if chunk.metadata.title:
            parts.append(f"[Heading: {chunk.metadata.title}]")

        parts.append(chunk.content)
        return "\n".join(parts)


__all__ = [
    "EmbeddingGenerator",
    "RetryStrategy",
]
