# ============================================================================
# FILE: core/retrieval/reranker.py
# ============================================================================

"""
Cross-encoder reranker module (Enhanced with Fallback).

This module reranks retrieved chunks using a cross-encoder model.
The reranker receives (Query, Chunk) pairs and produces semantic relevance scores
to improve precision before context construction.

Uses BAAI/bge-reranker-v2-m3 for better quality.
"""

from typing import Any, Optional, List, Tuple, Dict
import torch

from app.config import settings
from app.config.constants import (
    RERANKER_BATCH_SIZE,
    RERANKER_MAX_LENGTH,
    RERANKER_MODEL,
    RERANKER_MAX_CANDIDATES,
    LogEvent,
)
from app.core.exceptions import RerankerError
from app.monitoring import get_logger, measure_latency
from app.schemas import Chunk, Query
from app.schemas.retrieval import RetrievalCandidate


logger = get_logger("food_safety_rag.retrieval.reranker")


class CrossEncoderReranker:
    """
    Cross-encoder reranker using BGE reranker v2 m3.

    All instances share the same model via _shared_instance.

    Attributes:
        model_name: Name or path of the reranker model.
        batch_size: Base batch size for processing.
        max_length: Maximum input length per chunk.
        max_candidates: Maximum candidates to rerank.
        use_gpu: Whether to attempt GPU usage.
        device: Detected device ('cuda' or 'cpu').
        _model: Lazy-loaded model instance.
        _available: Whether model is loaded and available.
        _fallback_used: Whether fallback was used in last rerank.
        _last_fallback_reason: Reason for fallback.
        _cuda_memory_warning_logged: Whether CUDA memory warning was logged.
    """

    # ==========================================================================
    # Singleton Pattern
    # ==========================================================================
    
    _shared_instance: Optional["CrossEncoderReranker"] = None
    _shared_model: Optional[Any] = None
    _shared_available: bool = False
    _shared_initialized: bool = False

    @classmethod
    def get_shared_instance(cls) -> "CrossEncoderReranker":
        """
        Get the shared singleton instance of the reranker.
        """
        if cls._shared_instance is None:
            cls._shared_instance = CrossEncoderReranker(
                lazy_load=True,
                oom_recovery=True,
            )
            logger.log_event(
                event=LogEvent.SYSTEM_STARTUP,
                message="Created shared singleton reranker instance with fallback",
                details={"instance_id": id(cls._shared_instance)},
            )
        return cls._shared_instance

    @classmethod
    def is_model_loaded(cls) -> bool:
        """Check if the shared model is already loaded."""
        return cls._shared_available

    # ==========================================================================
    # Instance Methods
    # ==========================================================================

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        max_candidates: Optional[int] = None,
        use_gpu: Optional[bool] = None,
        lazy_load: bool = True,
        oom_recovery: bool = True,
        fallback_enabled: bool = True,
        fallback_strategy: str = "rrf",
    ) -> None:
        """
        Initialize the cross-encoder reranker.
        """
        self.model_name: str = str(model_name or RERANKER_MODEL)
        self.base_batch_size: int = int(batch_size or RERANKER_BATCH_SIZE)
        self.max_length: int = int(max_length or RERANKER_MAX_LENGTH)
        
        # ✅ FIX: Ensure max_candidates is read from settings correctly
        if max_candidates is not None:
            self.max_candidates = int(max_candidates)
        else:
            self.max_candidates = int(RERANKER_MAX_CANDIDATES)
        
        self.lazy_load: bool = lazy_load
        self.oom_recovery: bool = oom_recovery
        
        self.fallback_enabled: bool = fallback_enabled
        self.fallback_strategy: str = fallback_strategy
        self._fallback_used: bool = False
        self._last_fallback_reason: Optional[str] = None

        # PHASE 2: Use shared model if available
        if CrossEncoderReranker._shared_model is not None:
            self._model = CrossEncoderReranker._shared_model
            self._available = CrossEncoderReranker._shared_available
            self._load_attempted = CrossEncoderReranker._shared_initialized
        else:
            self._model = None
            self._available = False
            self._load_attempted = False

        # GPU configuration
        if use_gpu is None:
            self._use_gpu: bool = bool(settings.RERANKER_USE_GPU)
        else:
            self._use_gpu = bool(use_gpu)

        self._device: str = self._detect_device()
        self._use_memory_efficient_mode: bool = self._device == "cuda"
        self._cuda_memory_warning_logged: bool = False

        # Statistics
        self._stats: Dict[str, Any] = {
            "total_reranks": 0,
            "total_candidates": 0,
            "total_batches": 0,
            "oom_events": 0,
            "fallback_count": 0,
            "avg_batch_size": 0.0,
            "avg_latency_ms": 0.0,
        }

        # Load model immediately if not lazy
        if not self.lazy_load:
            self._load_model()

        logger.log_event(
            event=LogEvent.SYSTEM_STARTUP,
            message="Reranker initialized",
            details={
                "device": self._device,
                "use_gpu": self._use_gpu,
                "base_batch_size": self.base_batch_size,
                "max_length": self.max_length,
                "max_candidates": self.max_candidates,
                "lazy_load": self.lazy_load,
                "oom_recovery": self.oom_recovery,
                "fallback_enabled": self.fallback_enabled,
                "fallback_strategy": self.fallback_strategy,
                "memory_efficient_mode": self._use_memory_efficient_mode,
                "available": self._available,
                "using_shared_model": CrossEncoderReranker._shared_model is not None,
                "instance_id": id(self),
            },
        )

    # ==========================================================================
    # DEVICE DETECTION
    # ==========================================================================

    def _detect_device(self) -> str:
        """Detect the best available device for inference."""
        if self._use_gpu:
            if torch.cuda.is_available():
                try:
                    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    vram_available = torch.cuda.memory_allocated(0) / (1024**3)

                    logger.log_event(
                        event=LogEvent.SYSTEM_STARTUP,
                        message=f"GPU detected: {torch.cuda.get_device_name(0)} ({vram_total:.1f}GB)",
                        details={
                            "device": torch.cuda.get_device_name(0),
                            "vram_total_gb": round(vram_total, 2),
                            "vram_available_gb": round(vram_available, 2),
                        },
                    )

                    if vram_total < 4.0 and not self._cuda_memory_warning_logged:
                        self._cuda_memory_warning_logged = True
                        logger.log_error(
                            event=LogEvent.WARNING,
                            message=f"Low VRAM detected ({vram_total:.1f}GB). "
                                    f"Using memory-efficient settings (batch_size={self.base_batch_size}, "
                                    f"max_candidates={self.max_candidates})",
                        )
                except Exception as exc:
                    logger.log_error(
                        event=LogEvent.WARNING,
                        message=f"Failed to query GPU memory: {str(exc)}",
                        exception=exc,
                    )

                return "cuda"
            else:
                logger.log_error(
                    event=LogEvent.WARNING,
                    message="GPU requested but not available, falling back to CPU",
                )

        return "cpu"

    # ==========================================================================
    # MODEL LOADING
    # ==========================================================================

    def _load_model(self) -> bool:
        """Load the reranker model lazily using sentence-transformers."""
        # Check if shared model is already loaded
        if CrossEncoderReranker._shared_model is not None:
            self._model = CrossEncoderReranker._shared_model
            self._available = CrossEncoderReranker._shared_available
            self._load_attempted = CrossEncoderReranker._shared_initialized
            
            logger.log_event(
                event=LogEvent.SYSTEM_STARTUP,
                message="Reusing shared reranker model (already loaded)",
                details={
                    "instance_id": id(self),
                    "shared_instance_id": id(CrossEncoderReranker._shared_instance),
                },
            )
            return self._available

        if self._available:
            return True

        if self._load_attempted:
            return False

        self._load_attempted = True

        try:
            from sentence_transformers import CrossEncoder

            logger.log_event(
                event=LogEvent.SYSTEM_STARTUP,
                message=f"Loading reranker model: {self.model_name}",
                details={
                    "device": self._device,
                    "batch_size": self.base_batch_size,
                    "max_candidates": self.max_candidates,
                    "lazy_load": self.lazy_load,
                },
            )

            self._model = CrossEncoder(
                self.model_name,
                device=self._device,
                max_length=self.max_length
            )
            self._available = True

            # Store in shared cache
            CrossEncoderReranker._shared_model = self._model
            CrossEncoderReranker._shared_available = True
            CrossEncoderReranker._shared_initialized = True

            if self._device == "cuda":
                torch.cuda.empty_cache()

            logger.log_event(
                event=LogEvent.SYSTEM_STARTUP,
                message="Reranker model loaded successfully",
                details={
                    "instance_id": id(self),
                    "shared_instance_id": id(CrossEncoderReranker._shared_instance) if CrossEncoderReranker._shared_instance else None,
                },
            )

            return True

        except ImportError as exc:
            logger.log_error(
                event=LogEvent.ERROR,
                message="sentence-transformers not installed. Install with: pip install sentence-transformers",
                exception=exc,
            )
            self._available = False
            return False
        except torch.cuda.OutOfMemoryError as exc:
            logger.log_error(
                event=LogEvent.ERROR,
                message="CUDA Out of Memory. Try reducing batch_size or max_candidates.",
                exception=exc,
                details={
                    "batch_size": self.base_batch_size,
                    "max_candidates": self.max_candidates,
                    "max_length": self.max_length,
                },
            )
            self._available = False
            return False
        except Exception as exc:
            logger.log_error(
                event=LogEvent.ERROR,
                message=f"Failed to load reranker model: {str(exc)}",
                exception=exc,
            )
            self._available = False
            return False

    def _ensure_model_loaded(self) -> bool:
        """Ensure model is loaded (lazy loading)."""
        if self._available:
            return True

        if self.lazy_load:
            return self._load_model()

        return False

    # ==========================================================================
    # BATCH SIZE OPTIMIZATION
    # ==========================================================================

    def _estimate_optimal_batch_size(self, num_pairs: int) -> int:
        """Estimate optimal batch size based on GPU memory."""
        if self._device == "cpu":
            return min(self.base_batch_size, num_pairs)

        try:
            import torch

            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                reserved_memory = torch.cuda.memory_reserved(0)
                available_memory = (total_memory - reserved_memory) / (1024**3)

                if available_memory < 2:
                    safe_batch = 2
                elif available_memory < 4:
                    safe_batch = 4
                elif available_memory < 6:
                    safe_batch = 8
                elif available_memory < 10:
                    safe_batch = 12
                else:
                    safe_batch = self.base_batch_size

                safe_batch = min(safe_batch, self.base_batch_size)
                safe_batch = min(safe_batch, num_pairs)

                return max(1, safe_batch)

        except Exception:
            pass

        return min(self.base_batch_size, num_pairs)

    def _free_cuda_memory(self) -> None:
        """Free CUDA memory cache."""
        if self._device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # ==========================================================================
    # FALLBACK STRATEGY
    # ==========================================================================

    def _apply_fallback(
        self,
        query: Query,
        candidates: List[RetrievalCandidate],
        top_k: int,
        reason: str,
    ) -> List[RetrievalCandidate]:
        """
        Apply fallback strategy when reranker fails.
        """
        self._fallback_used = True
        self._last_fallback_reason = reason
        self._stats["fallback_count"] += 1

        logger.log_event(
            event=LogEvent.RERANKING_SKIPPED,
            message=f"Reranker fallback triggered - {reason}",
            details={
                "query_id": query.query_id,
                "candidates_count": len(candidates),
                "top_k": top_k,
                "fallback_strategy": self.fallback_strategy,
                "reason": reason,
            },
        )

        # Strategy 1: RRF Score
        if self.fallback_strategy == "rrf" and candidates and candidates[0].rrf_score is not None:
            logger.log_event(
                event=LogEvent.RERANKING_SKIPPED,
                message="Fallback: Using RRF scores",
            )
            sorted_candidates = sorted(
                candidates,
                key=lambda x: x.rrf_score or 0.0,
                reverse=True
            )
            return sorted_candidates[:top_k]

        # Strategy 2: Dense Score
        if self.fallback_strategy in ["dense", "rrf"] and candidates and candidates[0].dense_score is not None:
            logger.log_event(
                event=LogEvent.RERANKING_SKIPPED,
                message="Fallback: Using Dense scores",
            )
            sorted_candidates = sorted(
                candidates,
                key=lambda x: x.dense_score or 0.0,
                reverse=True
            )
            return sorted_candidates[:top_k]

        # Strategy 3: BM25 Score
        if self.fallback_strategy in ["bm25", "dense", "rrf"] and candidates and candidates[0].bm25_score is not None:
            logger.log_event(
                event=LogEvent.RERANKING_SKIPPED,
                message="Fallback: Using BM25 scores",
            )
            sorted_candidates = sorted(
                candidates,
                key=lambda x: x.bm25_score or 0.0,
                reverse=True
            )
            return sorted_candidates[:top_k]

        # Strategy 4: AS-IS
        logger.log_event(
            event=LogEvent.RERANKING_SKIPPED,
            message="Fallback: No scores available, returning as-is",
        )
        return candidates[:top_k]

    # ==========================================================================
    # RERANKING
    # ==========================================================================

    def rerank(
        self,
        query: Query,
        candidates: List[RetrievalCandidate],
        top_k: Optional[int] = None,
        fallback_enabled: Optional[bool] = None,
    ) -> List[RetrievalCandidate]:
        """
        Rerank candidates using the cross-encoder model with fallback.
        """
        effective_fallback = fallback_enabled if fallback_enabled is not None else self.fallback_enabled
        
        with measure_latency("reranking") as latency:
            effective_top_k = int(top_k or settings.RERANKER_TOP_K)
            query_text = query.get_effective_query()

            logger.log_retrieval(
                event=LogEvent.RERANKING_START,
                query_id=query.query_id,
                message=f"Starting reranking for {len(candidates)} candidates (max_candidates={self.max_candidates})",
                details={
                    "query_text": query_text[:100],
                    "candidates_count": len(candidates),
                    "max_candidates": self.max_candidates,
                    "top_k": effective_top_k,
                    "base_batch_size": self.base_batch_size,
                    "device": self._device,
                    "available": self._available,
                    "fallback_enabled": effective_fallback,
                    "fallback_strategy": self.fallback_strategy,
                },
            )

            try:
                if not candidates:
                    logger.log_event(
                        event=LogEvent.WARNING,
                        message="No candidates to rerank",
                    )
                    return []

                if not self._ensure_model_loaded():
                    logger.log_event(
                        event=LogEvent.WARNING,
                        message="Reranker model unavailable, using fallback",
                    )
                    if effective_fallback:
                        return self._apply_fallback(
                            query, candidates, effective_top_k,
                            "Model not available"
                        )
                    candidates.sort(key=lambda x: x.get_best_score(), reverse=True)
                    return candidates[:effective_top_k]

                # ✅ FIX: Limit candidates to max_candidates
                candidates_to_rerank = candidates[:self.max_candidates]

                logger.log_event(
                    event=LogEvent.RERANKING_START,
                    message=f"Reranking {len(candidates_to_rerank)} candidates (limited from {len(candidates)})",
                    details={
                        "original_count": len(candidates),
                        "limited_count": len(candidates_to_rerank),
                        "max_candidates": self.max_candidates,
                    },
                )

                pairs = []
                candidate_objects = []
                for candidate in candidates_to_rerank:
                    chunk_text = candidate.chunk.content[:self.max_length] if candidate.chunk.content else ""
                    pairs.append([query_text, chunk_text])
                    candidate_objects.append(candidate)

                if not pairs:
                    return []

                safe_batch = self._estimate_optimal_batch_size(len(pairs))

                scores = []
                total_batches = (len(pairs) + safe_batch - 1) // safe_batch

                for batch_idx, batch_start in enumerate(range(0, len(pairs), safe_batch)):
                    batch_end = min(batch_start + safe_batch, len(pairs))
                    batch = pairs[batch_start:batch_end]

                    try:
                        with torch.no_grad():
                            if self._device == "cuda":
                                torch.cuda.empty_cache()

                            batch_scores = self._model.predict(batch)

                            if isinstance(batch_scores, (list, tuple)):
                                scores.extend(batch_scores)
                            elif hasattr(batch_scores, 'tolist'):
                                scores.extend(batch_scores.tolist())
                            else:
                                scores.extend([float(s) for s in batch_scores])

                    except torch.cuda.OutOfMemoryError as exc:
                        logger.log_error(
                            event=LogEvent.ERROR,
                            message=f"CUDA OOM at batch {batch_idx+1}/{total_batches}. "
                                    f"Falling back to CPU for this batch.",
                            exception=exc,
                            details={"batch_size": safe_batch},
                        )

                        self._stats["oom_events"] += 1

                        if self.oom_recovery:
                            smaller_batch = max(1, safe_batch // 2)
                            if smaller_batch < safe_batch:
                                logger.log_event(
                                    event=LogEvent.WARNING,
                                    message=f"Reducing batch size to {smaller_batch}",
                                )
                                for i in range(batch_start, batch_end, smaller_batch):
                                    sub_end = min(i + smaller_batch, batch_end)
                                    sub_batch = pairs[i:sub_end]
                                    try:
                                        sub_scores = self._model.predict(sub_batch)
                                        scores.extend(sub_scores)
                                    except Exception:
                                        for j in range(i, sub_end):
                                            scores.append(candidate_objects[j].get_best_score())
                                continue
                            else:
                                for idx in range(batch_start, batch_end):
                                    if idx < len(candidate_objects):
                                        scores.append(candidate_objects[idx].get_best_score())
                                self._stats["fallback_count"] += 1
                        else:
                            for idx in range(batch_start, batch_end):
                                if idx < len(candidate_objects):
                                    scores.append(candidate_objects[idx].get_best_score())
                            self._stats["fallback_count"] += 1

                        self._free_cuda_memory()

                    except Exception as exc:
                        logger.log_error(
                            event=LogEvent.ERROR,
                            message=f"Batch reranking failed at batch {batch_idx+1}: {str(exc)}",
                            exception=exc,
                        )
                        for idx in range(batch_start, batch_end):
                            if idx < len(candidate_objects):
                                scores.append(candidate_objects[idx].get_best_score())
                        self._stats["fallback_count"] += 1

                if len(scores) < len(candidate_objects):
                    for i in range(len(scores), len(candidate_objects)):
                        scores.append(candidate_objects[i].get_best_score())

                # Create new RetrievalCandidate with reranker_score
                reranked_results = []
                for i, candidate in enumerate(candidate_objects):
                    reranker_score = float(scores[i]) if i < len(scores) else candidate.get_best_score()
                    
                    new_candidate = RetrievalCandidate(
                        chunk=candidate.chunk,
                        dense_score=candidate.dense_score,
                        bm25_score=candidate.bm25_score,
                        rrf_score=candidate.rrf_score,
                        reranker_score=reranker_score,
                        fusion_score=candidate.fusion_score,
                        metadata=candidate.metadata.copy() if candidate.metadata else {},
                        rank=candidate.rank,
                    )
                    reranked_results.append(new_candidate)

                reranked_results.sort(key=lambda x: x.reranker_score or 0.0, reverse=True)

                for rank, candidate in enumerate(reranked_results, start=1):
                    candidate.rank = rank

                if effective_top_k > 0:
                    reranked_results = reranked_results[:effective_top_k]

                if self._device == "cuda":
                    self._free_cuda_memory()

                # Update stats
                self._stats["total_reranks"] += 1
                self._stats["total_candidates"] += len(candidates)
                self._stats["total_batches"] += total_batches
                
                prev_avg = self._stats["avg_batch_size"]
                prev_count = self._stats["total_reranks"] - 1
                if prev_count > 0:
                    self._stats["avg_batch_size"] = (prev_avg * prev_count + safe_batch) / self._stats["total_reranks"]
                else:
                    self._stats["avg_batch_size"] = float(safe_batch)
                
                duration_ms = latency.duration_ms if latency.duration_ms is not None else 0.0
                if prev_count > 0:
                    self._stats["avg_latency_ms"] = (self._stats["avg_latency_ms"] * prev_count + duration_ms) / self._stats["total_reranks"]
                else:
                    self._stats["avg_latency_ms"] = duration_ms

                latency.stop(
                    query_id=query.query_id,
                    input_candidates=len(candidates),
                    reranked_candidates=len(candidates_to_rerank),
                    output_candidates=len(reranked_results),
                    device=self._device,
                    batch_size=safe_batch,
                    using_shared_model=CrossEncoderReranker._shared_model is not None,
                    fallback_used=self._fallback_used,
                )

                logger.log_retrieval(
                    event=LogEvent.RERANKING_COMPLETE,
                    query_id=query.query_id,
                    message=f"Reranking complete: {len(reranked_results)} candidates",
                    details={
                        "input_candidates": len(candidates),
                        "reranked_candidates": len(candidates_to_rerank),
                        "output_candidates": len(reranked_results),
                        "max_candidates": self.max_candidates,
                        "duration_ms": duration_ms,
                        "device": self._device,
                        "batch_size": safe_batch,
                        "batches_processed": total_batches,
                        "oom_events": self._stats["oom_events"],
                        "fallback_used": self._fallback_used,
                        "fallback_reason": self._last_fallback_reason,
                        "using_shared_model": CrossEncoderReranker._shared_model is not None,
                    },
                )

                return reranked_results

            except Exception as exc:
                if effective_fallback:
                    logger.log_error(
                        event=LogEvent.WARNING,
                        message=f"Reranking failed: {str(exc)}. Using fallback.",
                        exception=exc,
                    )
                    return self._apply_fallback(
                        query, candidates, effective_top_k,
                        f"Exception: {str(exc)[:100]}"
                    )
                
                raise RerankerError(
                    message=f"Reranking failed: {str(exc)}",
                    num_candidates=len(candidates),
                    original_exception=exc,
                )

    # ==========================================================================
    # BATCH RERANKING
    # ==========================================================================

    def rerank_batch(
        self,
        queries: List[Query],
        candidates_per_query: List[List[RetrievalCandidate]],
        top_k: Optional[int] = None,
    ) -> List[List[RetrievalCandidate]]:
        """
        Rerank multiple queries' candidates.
        """
        results: List[List[RetrievalCandidate]] = []

        for query, candidates in zip(queries, candidates_per_query):
            try:
                result = self.rerank(query, candidates, top_k=top_k)
                results.append(result)
            except RerankerError as exc:
                logger.log_error(
                    event=LogEvent.ERROR,
                    message=f"Reranking failed for query {query.query_id}: {str(exc)}",
                    exception=exc,
                )
                candidates.sort(key=lambda x: x.get_best_score(), reverse=True)
                effective_top_k = int(top_k or settings.RERANKER_TOP_K)
                results.append(candidates[:effective_top_k])

        return results

    # ==========================================================================
    # STATISTICS AND UTILITY METHODS
    # ==========================================================================

    def get_config(self) -> Dict[str, Any]:
        """Get reranker configuration."""
        return {
            "model": self.model_name,
            "device": self._device,
            "base_batch_size": self.base_batch_size,
            "max_length": self.max_length,
            "max_candidates": self.max_candidates,
            "lazy_load": self.lazy_load,
            "oom_recovery": self.oom_recovery,
            "fallback_enabled": self.fallback_enabled,
            "fallback_strategy": self.fallback_strategy,
            "available": self._available,
            "memory_efficient_mode": self._use_memory_efficient_mode,
            "using_shared_model": CrossEncoderReranker._shared_model is not None,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get reranker statistics."""
        return {
            **self._stats,
            "available": self._available,
            "device": self._device,
            "max_candidates": self.max_candidates,
            "fallback_used_total": self._stats["fallback_count"],
            "last_fallback_used": self._fallback_used,
            "last_fallback_reason": self._last_fallback_reason,
            "memory_efficient_mode": self._use_memory_efficient_mode,
            "using_shared_model": CrossEncoderReranker._shared_model is not None,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self._available and self._model is not None:
            return {
                "model_name": self.model_name,
                "available": True,
                "device": self._device,
                "max_length": self.max_length,
                "max_candidates": self.max_candidates,
                "model_type": type(self._model).__name__,
                "using_shared_model": CrossEncoderReranker._shared_model is not None,
                "fallback_enabled": self.fallback_enabled,
                "fallback_strategy": self.fallback_strategy,
            }
        return {
            "model_name": self.model_name,
            "available": False,
            "device": self._device,
            "max_candidates": self.max_candidates,
        }

    def reload_model(self) -> bool:
        """Force reload the model."""
        self._model = None
        self._available = False
        self._load_attempted = False
        
        CrossEncoderReranker._shared_model = None
        CrossEncoderReranker._shared_available = False
        CrossEncoderReranker._shared_initialized = False
        
        return self._load_model()


def get_reranker() -> CrossEncoderReranker:
    """Get the shared singleton reranker instance."""
    return CrossEncoderReranker.get_shared_instance()