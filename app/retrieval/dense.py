# ============================================================================
# FILE: core/retrieval/dense.py
# ============================================================================

"""
Dense retrieval module (Enhanced).

This module performs semantic dense retrieval using the vector store.
It converts queries to embeddings and searches for similar document chunks.
Supports metadata filtering for improved precision.
Supports multi-query retrieval for query expansion.

PHASE 4: Returns RetrievalCandidate instead of modifying Chunk.

Enhanced Features:
- Enhanced embedding validation with multiple criteria
- Deterministic cache key generation
- Better error handling and logging
- Embedding quality scoring
- Support for query reformulation fallback
- PRESERVES dense_score in RetrievalCandidate for tracing

PHASE 1: retrieve_multi_query accepts 4-tuple variants.
PHASE 4: Returns RetrievalCandidate with scores.
"""

from typing import Any, Optional, List, Tuple, Dict
import hashlib
import json
import numpy as np

from app.config import settings
from app.config.constants import LogEvent
from app.core.exceptions import RetrievalError
from app.ingestion.embedding import EmbeddingGenerator
from app.monitoring import get_logger, measure_latency
from app.schemas import Chunk, Query
from app.schemas.retrieval import RetrievalCandidate
from app.vector_store import FAISSStore


logger = get_logger("food_safety_rag.retrieval.dense")


class DenseRetriever:
    """
    Dense semantic retriever using FAISS vector similarity search.
    
    PHASE 4: Returns RetrievalCandidate instead of modifying Chunk.
    
    This retriever converts queries to embeddings using the same model
    used during ingestion, then searches the FAISS index for the most
    semantically similar chunks. The objective is maximizing recall while
    maintaining semantic relevance.
    
    Enhanced with:
    - Enhanced embedding validation (NaN, inf, L2 norm, zero-count checks)
    - Deterministic cache key generation with sorted filters
    - Embedding quality scoring
    - Query reformulation fallback
    - Better error handling and logging
    - PRESERVES dense_score in RetrievalCandidate for tracing
    
    PHASE 1: retrieve_multi_query accepts 4-tuple variants.
    PHASE 4: Returns RetrievalCandidate with scores.
    
    Attributes:
        vector_store: FAISS vector store instance.
        embedding_generator: Embedding generator for query embeddings.
        top_k: Number of results to retrieve.
        min_embedding_quality: Minimum acceptable embedding quality score.
        _embedding_cache: Cache for query embeddings.
        _last_results: Last retrieved results for tracing.
        _last_query_id: Last query ID for tracing.
    """

    def __init__(
        self,
        vector_store: Optional[FAISSStore] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        top_k: Optional[int] = None,
        min_embedding_quality: float = 0.1,
        enable_cache: bool = True,
    ) -> None:
        """
        Initialize the dense retriever.

        Args:
            vector_store: FAISS vector store. Creates new if None.
            embedding_generator: Embedding generator. Creates new if None.
            top_k: Number of results to retrieve. Defaults to settings.
            min_embedding_quality: Minimum acceptable embedding quality (0-1).
            enable_cache: Whether to cache query embeddings.
        """
        self.vector_store: FAISSStore = vector_store or FAISSStore()
        self.embedding_generator: EmbeddingGenerator = embedding_generator or EmbeddingGenerator()
        self.top_k: int = int(top_k or settings.DENSE_TOP_K)
        self.min_embedding_quality: float = min_embedding_quality
        self.enable_cache: bool = enable_cache
        
        # For tracing
        self._last_results: Optional[List[RetrievalCandidate]] = None
        self._last_query_id: Optional[str] = None
        self._embedding_cache: Dict[str, List[float]] = {}
        self._cache_stats: Dict[str, Any] = {
            "hits": 0,
            "misses": 0,
            "total": 0,
        }
        
        logger.log_event(
            event=LogEvent.SYSTEM_STARTUP,
            message="Dense retriever initialized (Enhanced + PHASE 1 + PHASE 4)",
            details={
                "top_k": self.top_k,
                "min_embedding_quality": self.min_embedding_quality,
                "enable_cache": self.enable_cache,
                "phase_1_supports_4tuple_variants": True,
                "phase_4_retrieval_candidate": True,
            },
        )

    # ==========================================================================
    # EMBEDDING VALIDATION
    # ==========================================================================

    def _validate_embedding(
        self,
        embedding: Optional[List[float]],
    ) -> Tuple[bool, str, float]:
        """
        Enhanced embedding validation with diagnostics.

        Checks:
        1. Not empty/None
        2. No NaN or inf values
        3. Not all near-zero
        4. Sufficient non-zero values (>10%)
        5. L2 norm is within reasonable range

        Args:
            embedding: Embedding vector to validate.

        Returns:
            Tuple[bool, str, float]: (is_valid, validation_message, quality_score)
        """
        if not embedding or len(embedding) == 0:
            return False, "Empty embedding", 0.0

        embedding_np = np.array(embedding)
        dim = len(embedding)

        # Check for NaN or inf
        if np.any(np.isnan(embedding_np)):
            return False, "Embedding contains NaN values", 0.0

        if np.any(np.isinf(embedding_np)):
            return False, "Embedding contains inf values", 0.0

        # Check if all values are near-zero
        abs_values = np.abs(embedding_np)
        max_abs = np.max(abs_values)

        if max_abs < 1e-7:
            return False, f"All values near-zero (max={max_abs:.2e})", 0.0

        # Check non-zero ratio
        non_zero_count = np.sum(abs_values > 1e-5)
        non_zero_ratio = non_zero_count / dim

        if non_zero_ratio < 0.05:  # Less than 5% non-zero
            return False, f"Very few non-zero values ({non_zero_ratio:.1%})", non_zero_ratio

        # Check L2 norm
        l2_norm = np.linalg.norm(embedding_np)

        if l2_norm < 0.1:
            return False, f"Very low L2 norm: {l2_norm:.4f}", non_zero_ratio

        if l2_norm > 100:
            return False, f"Very high L2 norm: {l2_norm:.1f}", non_zero_ratio

        # Calculate quality score (0-1)
        quality_score = min(1.0, non_zero_ratio * 2)  # Cap at 1.0

        return True, "Valid embedding", quality_score

    def _is_weak_embedding(self, embedding: List[float], threshold: float = 0.5) -> bool:
        """
        Check if embedding is weak (may need reformulation).

        Args:
            embedding: Embedding vector.
            threshold: Quality threshold.

        Returns:
            bool: True if embedding is weak.
        """
        is_valid, _, quality = self._validate_embedding(embedding)
        return not is_valid or quality < threshold

    # ==========================================================================
    # QUERY REFORMULATION
    # ==========================================================================

    def _reformulate_query(self, query_text: str) -> str:
        """
        Reformulate query if it's too specific or contains stop words.

        Args:
            query_text: Original query text.

        Returns:
            str: Reformulated query or original if no reform needed.
        """
        import re

        # Remove excessive punctuation
        reformulated = re.sub(r'[?!.,;:]{2,}', ' ', query_text)

        # Remove leading/trailing punctuation
        reformulated = re.sub(r'^[^a-zA-Zآ-ي]+|[^a-zA-Zآ-ي]+$', '', reformulated)

        # Normalize whitespace
        reformulated = re.sub(r'\s+', ' ', reformulated).strip()

        # If reformulation is too short, try keyword extraction
        if len(reformulated.split()) < 2:
            # Extract key terms (longer words)
            words = re.findall(r'[a-zA-Zآ-ي]{4,}', query_text)
            if words:
                reformulated = ' '.join(words)

        return reformulated if reformulated else query_text

    # ==========================================================================
    # CACHE MANAGEMENT
    # ==========================================================================

    def _make_cache_key(
        self,
        query_text: str,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate deterministic cache key.

        Args:
            query_text: Query text.
            filter_dict: Optional metadata filters.

        Returns:
            str: Deterministic cache key.
        """
        key_data = {
            'query': query_text.lower().strip(),
            'filters': filter_dict or {},
        }
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_json.encode()).hexdigest()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        stats = self._cache_stats.copy()
        stats['cache_size'] = len(self._embedding_cache)
        return stats

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        self._cache_stats = {"hits": 0, "misses": 0, "total": 0}
        logger.log_event(
            event=LogEvent.CACHE_CLEAR,
            message="Dense retriever embedding cache cleared",
        )

    # ==========================================================================
    # RETRIEVAL METHODS - PHASE 4: Returns RetrievalCandidate
    # ==========================================================================

    def retrieve(
        self,
        query: Query,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        return_trace: bool = False,
    ) -> List[RetrievalCandidate]:
        """
        Perform dense retrieval for a single query.

        PHASE 4: Returns RetrievalCandidate instead of modifying Chunk.

        Enhanced with:
        - Better embedding validation
        - Query reformulation fallback
        - Deterministic cache keys
        - PRESERVES dense_score in RetrievalCandidate

        Args:
            query: The user query.
            top_k: Override for number of results.
            filter_dict: Optional metadata filters.
            return_trace: Whether to return trace info with results.

        Returns:
            list[RetrievalCandidate]: List of retrieval candidates.

        Raises:
            RetrievalError: If retrieval fails completely.
        """
        with measure_latency("dense_retrieval") as latency:
            effective_top_k = int(top_k or self.top_k)
            query_text = query.get_effective_query()

            if not query_text:
                raise RetrievalError(
                    message="Query cannot be empty",
                    query=query_text,
                    error_code="RET_001",
                )

            logger.log_retrieval(
                event=LogEvent.DENSE_RETRIEVAL_START,
                query_id=query.query_id,
                message=f"PHASE 4: Starting dense retrieval: {query_text[:100]}...",
                details={
                    "query_text": query_text,
                    "top_k": effective_top_k,
                    "filters_applied": filter_dict is not None,
                    "phase_4_retrieval_candidate": True,
                },
            )

            try:
                # ============================================================
                # Step 1: Check cache
                # ============================================================
                query_embedding = None
                cache_key = self._make_cache_key(query_text, filter_dict)
                self._cache_stats['total'] += 1

                if self.enable_cache and cache_key in self._embedding_cache:
                    query_embedding = self._embedding_cache[cache_key]
                    self._cache_stats['hits'] += 1

                    logger.log_event(
                        event=LogEvent.CACHE_HIT,
                        message="Using cached embedding",
                        details={"cache_key": cache_key[:16]},
                    )

                # ============================================================
                # Step 2: Generate embedding (if not cached)
                # ============================================================
                if query_embedding is None:
                    self._cache_stats['misses'] += 1

                    query_embedding = self.embedding_generator.embed_text(query_text)

                    # Validate embedding
                    is_valid, validation_msg, quality = self._validate_embedding(
                        query_embedding
                    )

                    if not is_valid:
                        logger.log_event(
                            event=LogEvent.QUERY_REFORMULATION,
                            message=f"Invalid embedding: {validation_msg}. Trying reformulation.",
                            details={
                                "validation_message": validation_msg,
                                "quality_score": quality,
                            },
                        )

                        # Try reformulated query
                        reformulated = self._reformulate_query(query_text)

                        if reformulated != query_text:
                            query_embedding = self.embedding_generator.embed_text(
                                reformulated
                            )
                            is_valid, validation_msg, quality = self._validate_embedding(
                                query_embedding
                            )

                    # Final validation
                    if not is_valid:
                        raise RetrievalError(
                            message=f"Failed to generate valid query embedding: {validation_msg}",
                            query=query_text,
                            error_code="RET_001",
                        )

                    # Check for weak embedding
                    if self._is_weak_embedding(query_embedding):
                        logger.log_event(
                            event=LogEvent.QUERY_REFORMULATION,
                            message=f"Weak embedding (quality={quality:.2f}). Trying reformulation.",
                            details={"quality_score": quality},
                        )

                        reformulated = self._reformulate_query(query_text)
                        if reformulated != query_text:
                            alt_embedding = self.embedding_generator.embed_text(
                                reformulated
                            )
                            alt_valid, _, alt_quality = self._validate_embedding(
                                alt_embedding
                            )
                            if alt_valid and alt_quality > quality:
                                query_embedding = alt_embedding
                                quality = alt_quality
                                logger.log_event(
                                    event=LogEvent.QUERY_REFORMULATION,
                                    message=f"Using reformulated embedding (quality={quality:.2f})",
                                )

                    # Cache the embedding
                    if self.enable_cache:
                        self._embedding_cache[cache_key] = query_embedding

                    logger.log_event(
                        event=LogEvent.CACHE_MISS,
                        message="Embedding generated and cached",
                        details={
                            "cache_key": cache_key[:16],
                            "quality_score": round(quality, 3),
                            "embedding_dimension": len(query_embedding),
                        },
                    )

                # ============================================================
                # Step 3: Search vector store
                # ============================================================
                results = self.vector_store.search(
                    query_embedding=query_embedding,
                    top_k=effective_top_k,
                    filter_dict=filter_dict,
                )

                # ============================================================
                # Step 4: Build RetrievalCandidate results
                # ============================================================
                if not results:
                    logger.log_event(
                        event=LogEvent.WARNING,
                        message="Dense retrieval returned no results",
                        details={
                            "query": query_text[:100],
                            "top_k": effective_top_k,
                            "filters_applied": filter_dict is not None,
                        },
                    )
                    self._last_query_id = query.query_id
                    self._last_results = []
                    return []

                # PHASE 4: Build RetrievalCandidate
                candidates: List[RetrievalCandidate] = []
                for chunk, score in results:
                    candidate = RetrievalCandidate(
                        chunk=chunk,
                        dense_score=score,
                    )
                    candidates.append(candidate)

                # Sort by score descending
                candidates.sort(key=lambda x: x.dense_score or 0.0, reverse=True)

                # Add rank metadata
                for rank, candidate in enumerate(candidates, start=1):
                    candidate.rank = rank

                # Store last results for tracing
                self._last_results = candidates
                self._last_query_id = query.query_id

                latency.stop(
                    query_id=query.query_id,
                    results_count=len(candidates),
                    top_k=effective_top_k,
                    filters_applied=filter_dict is not None,
                    cache_hit=cache_key in self._embedding_cache if self.enable_cache else False,
                )

                logger.log_retrieval(
                    event=LogEvent.DENSE_RETRIEVAL_COMPLETE,
                    query_id=query.query_id,
                    message=f"PHASE 4: Dense retrieval returned {len(candidates)} candidates",
                    details={
                        "results_count": len(candidates),
                        "top_k": effective_top_k,
                        "duration_ms": latency.duration_ms,
                        "filters_applied": filter_dict is not None,
                        "cache_hit": cache_key in self._embedding_cache if self.enable_cache else False,
                        "phase_4_retrieval_candidate": True,
                        "top_results": [
                            {
                                "chunk_id": c.chunk.metadata.chunk_id if c.chunk.metadata else None,
                                "page": c.chunk.metadata.page if c.chunk.metadata else None,
                                "score": round(c.dense_score or 0.0, 4),
                            }
                            for c in candidates[:10]
                            if c.chunk.metadata and c.chunk.metadata.page is not None
                        ],
                    },
                )

                return candidates

            except RetrievalError:
                raise
            except Exception as exc:
                raise RetrievalError(
                    message=f"Dense retrieval failed: {str(exc)}",
                    query=query_text,
                    error_code="RET_001",
                    original_exception=exc,
                )

    # ==========================================================================
    # MULTI-QUERY RETRIEVAL - PHASE 4: Returns RetrievalCandidate
    # ==========================================================================

    def retrieve_multi_query(
        self,
        query: Query,
        query_variants: List[Tuple[str, float, int, int]],
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, List[RetrievalCandidate]]]:
        """
        Perform dense retrieval for multiple query variants.

        PHASE 4: Returns RetrievalCandidate.

        Each variant is a 4-tuple:
        - variant_text: The variant text
        - weight: The weight for this variant (0.0 to 1.0)
        - dense_top_k: Number of results for dense retrieval
        - bm25_top_k: Number of results for BM25 retrieval (ignored in dense)

        PRESERVES dense_score in RetrievalCandidate.

        Args:
            query: The original user query (for logging).
            query_variants: List of (query_text, weight, dense_top_k, bm25_top_k) tuples.
            filter_dict: Optional metadata filters to apply.

        Returns:
            list[tuple[str, float, list[RetrievalCandidate]]]: 
                List of (query_text, weight, results) for each variant.

        Raises:
            RetrievalError: If all variants fail.
        """
        with measure_latency("dense_multi_query_retrieval") as latency:
            results: List[Tuple[str, float, List[RetrievalCandidate]]] = []
            successful_variants = 0

            logger.log_retrieval(
                event=LogEvent.DENSE_RETRIEVAL_START,
                query_id=query.query_id,
                message=f"PHASE 4: Starting multi-query dense retrieval with {len(query_variants)} variants",
                details={
                    "num_variants": len(query_variants),
                    "variants": [q for q, _, _, _ in query_variants],
                    "weights": [w for _, w, _, _ in query_variants],
                    "dense_top_k": [dtk for _, _, dtk, _ in query_variants],
                    "phase_4_retrieval_candidate": True,
                },
            )

            for variant_text, weight, dense_top_k, bm25_top_k in query_variants:
                try:
                    dense_top_k = int(dense_top_k)

                    temp_query = Query(
                        query_id=f"{query.query_id}_variant",
                        original_text=variant_text,
                    )

                    variant_results = self.retrieve(
                        query=temp_query,
                        top_k=dense_top_k,
                        filter_dict=filter_dict,
                    )

                    # Apply weight to scores (create new candidates with weighted scores)
                    weighted_results = []
                    for candidate in variant_results:
                        weighted_candidate = RetrievalCandidate(
                            chunk=candidate.chunk,
                            dense_score=(candidate.dense_score or 0.0) * weight,
                        )
                        weighted_results.append(weighted_candidate)

                    if weighted_results:
                        successful_variants += 1

                    results.append((variant_text, weight, weighted_results))

                except RetrievalError as exc:
                    logger.log_error(
                        event=LogEvent.ERROR,
                        message=f"Dense retrieval failed for variant: {variant_text[:50]}...",
                        exception=exc,
                    )
                    results.append((variant_text, weight, []))

            if successful_variants == 0:
                raise RetrievalError(
                    message="All query variants failed in dense retrieval",
                    query=query.get_effective_query(),
                    error_code="RET_001",
                )

            latency.stop(
                query_id=query.query_id,
                num_variants=len(query_variants),
                successful_variants=successful_variants,
                phase_1_4tuple_variants=True,
                phase_4_retrieval_candidate=True,
            )

            return results

    # ==========================================================================
    # BATCH RETRIEVAL - PHASE 4: Returns RetrievalCandidate
    # ==========================================================================

    def retrieve_batch(
        self,
        queries: List[Query],
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[List[RetrievalCandidate]]:
        """
        Perform dense retrieval for multiple queries.

        PHASE 4: Returns RetrievalCandidate.

        Args:
            queries: List of queries.
            top_k: Override for number of results.
            filter_dict: Optional metadata filters to apply.

        Returns:
            list[list[RetrievalCandidate]]: List of results per query.
        """
        results: List[List[RetrievalCandidate]] = []

        for query in queries:
            try:
                result = self.retrieve(query, top_k=top_k, filter_dict=filter_dict)
                results.append(result)
            except RetrievalError as exc:
                logger.log_error(
                    event=LogEvent.ERROR,
                    message=f"Dense retrieval failed for query {query.query_id}: {str(exc)}",
                    exception=exc,
                )
                results.append([])

        return results

    # ==========================================================================
    # STATISTICS METHODS
    # ==========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get dense retriever statistics."""
        return {
            "top_k": self.top_k,
            "min_embedding_quality": self.min_embedding_quality,
            "enable_cache": self.enable_cache,
            "cache_stats": self.get_cache_stats(),
            "vector_store": self.vector_store.get_stats() if hasattr(self.vector_store, 'get_stats') else {},
            "last_results_count": len(self._last_results) if self._last_results else 0,
            "phase_1_4tuple_variants": True,
            "phase_4_retrieval_candidate": True,
        }

    def get_last_results(self) -> Optional[List[RetrievalCandidate]]:
        """Get the last retrieved results for tracing."""
        return self._last_results

    def get_last_query_id(self) -> Optional[str]:
        """Get the last query ID for tracing."""
        return self._last_query_id