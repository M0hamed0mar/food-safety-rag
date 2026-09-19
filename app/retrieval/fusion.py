# ============================================================================
# FILE: core/retrieval/fusion.py
# ============================================================================

"""
Reciprocal Rank Fusion (RRF) module.

This module combines results from multiple retrieval strategies using
Reciprocal Rank Fusion. RRF is robust across different retrieval algorithms
and does not require score normalization.

PHASE 4: Uses RetrievalCandidate instead of modifying Chunk.

Enhanced with:
- Score normalization before fusion (Min-Max, Rank-based, Z-score)
- Adaptive K selection based on result distribution
- Strategy-aware RRF with quality-based weighting
- Support for multiple normalization methods
- PRESERVES rrf_score in RetrievalCandidate for tracing
"""

from typing import Any, Optional, List, Tuple, Dict
import statistics

from app.config import settings
from app.config.constants import DEFAULT_RRF_K, LogEvent
from app.core.exceptions import FusionError
from app.monitoring import get_logger, measure_latency
from app.schemas import Chunk
from app.schemas.retrieval import RetrievalCandidate


logger = get_logger("food_safety_rag.retrieval.fusion")


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion implementation for combining retrieval results.
    
    RRF combines results from multiple retrieval strategies by using the
    reciprocal of the rank position, weighted by a constant k.
    
    PHASE 4: Uses RetrievalCandidate instead of modifying Chunk.
    
    Enhanced with:
    - Score normalization before fusion
    - Adaptive K based on result distribution
    - Strategy-aware weighting
    - Multiple normalization methods
    
    PRESERVES rrf_score in RetrievalCandidate for tracing.
    
    Attributes:
        k: RRF constant. Higher values reduce the impact of rank differences.
        normalize_scores: Whether to normalize scores before fusion.
        normalization_method: Method for normalization ('minmax', 'rank', 'zscore').
        adaptive_k: Whether to use adaptive K adjustment.
        strategy_weighting: Whether to weight strategies by quality.
    """

    def __init__(
        self,
        k: Optional[int] = None,
        normalize_scores: bool = True,
        normalization_method: str = "minmax",
        adaptive_k: bool = True,
        strategy_weighting: bool = True,
    ) -> None:
        """
        Initialize the RRF fusion engine.

        Args:
            k: RRF constant. Defaults to settings or DEFAULT_RRF_K (60).
            normalize_scores: Whether to normalize scores before fusion.
            normalization_method: Method for normalization ('minmax', 'rank', 'zscore').
            adaptive_k: Whether to use adaptive K adjustment.
            strategy_weighting: Whether to weight strategies by quality.
        """
        self.k: int = int(k or settings.RRF_CONSTANT or DEFAULT_RRF_K)
        self.normalize_scores: bool = normalize_scores
        self.normalization_method: str = normalization_method
        self.adaptive_k: bool = adaptive_k
        self.strategy_weighting: bool = strategy_weighting

        logger.log_event(
            event=LogEvent.SYSTEM_STARTUP,
            message="RRF fusion engine initialized (Enhanced + PHASE 4)",
            details={
                "k": self.k,
                "normalize_scores": self.normalize_scores,
                "normalization_method": self.normalization_method,
                "adaptive_k": self.adaptive_k,
                "strategy_weighting": self.strategy_weighting,
                "phase_4_retrieval_candidate": True,
            },
        )

    # ==========================================================================
    # SCORE NORMALIZATION
    # ==========================================================================

    def _normalize_scores(
        self,
        candidates: List[RetrievalCandidate],
        score_type: str = "best",
        method: str = "minmax",
    ) -> List[RetrievalCandidate]:
        """
        Normalize scores to (0, 1) range using specified method.

        PHASE 4: Works with RetrievalCandidate.

        Args:
            candidates: List of RetrievalCandidate objects.
            score_type: Type of score to normalize ('dense', 'bm25', 'best').
            method: Normalization method ('minmax', 'rank', 'zscore').

        Returns:
            List[RetrievalCandidate]: Candidates with normalized scores.
        """
        if not candidates:
            return candidates

        # Get scores from candidates
        if score_type == "best":
            scores = [c.get_best_score() for c in candidates]
        else:
            scores = [c.get_score(score_type) or 0.0 for c in candidates]

        if method == "minmax":
            min_score = min(scores) if scores else 0.0
            max_score = max(scores) if scores else 1.0
            range_val = max_score - min_score

            if range_val < 1e-6:  # All scores are equal
                normalized = []
                for candidate in candidates:
                    # Create new candidate with normalized score
                    new_candidate = RetrievalCandidate(
                        chunk=candidate.chunk,
                        dense_score=candidate.dense_score,
                        bm25_score=candidate.bm25_score,
                        rrf_score=candidate.rrf_score,
                        reranker_score=candidate.reranker_score,
                        fusion_score=candidate.fusion_score,
                        metadata=candidate.metadata.copy() if candidate.metadata else {},
                        rank=candidate.rank,
                    )
                    normalized.append(new_candidate)
                return normalized
            else:
                normalized = []
                for candidate, score in zip(candidates, scores):
                    norm_score = (score - min_score) / range_val
                    # Create new candidate with normalized score
                    new_candidate = RetrievalCandidate(
                        chunk=candidate.chunk,
                        dense_score=candidate.dense_score,
                        bm25_score=candidate.bm25_score,
                        rrf_score=candidate.rrf_score,
                        reranker_score=candidate.reranker_score,
                        fusion_score=norm_score if score_type == "fusion" else candidate.fusion_score,
                        metadata=candidate.metadata.copy() if candidate.metadata else {},
                        rank=candidate.rank,
                    )
                    normalized.append(new_candidate)
                return normalized

        elif method == "rank":
            # Rank-based normalization (0 to 1)
            total = len(candidates)
            if total <= 1:
                normalized = []
                for candidate in candidates:
                    new_candidate = RetrievalCandidate(
                        chunk=candidate.chunk,
                        dense_score=candidate.dense_score,
                        bm25_score=candidate.bm25_score,
                        rrf_score=candidate.rrf_score,
                        reranker_score=candidate.reranker_score,
                        fusion_score=0.5 if score_type == "fusion" else candidate.fusion_score,
                        metadata=candidate.metadata.copy() if candidate.metadata else {},
                        rank=candidate.rank,
                    )
                    normalized.append(new_candidate)
                return normalized
            else:
                normalized = []
                for idx, candidate in enumerate(candidates):
                    norm_score = 1.0 - (idx / (total - 1))
                    new_candidate = RetrievalCandidate(
                        chunk=candidate.chunk,
                        dense_score=candidate.dense_score,
                        bm25_score=candidate.bm25_score,
                        rrf_score=candidate.rrf_score,
                        reranker_score=candidate.reranker_score,
                        fusion_score=norm_score if score_type == "fusion" else candidate.fusion_score,
                        metadata=candidate.metadata.copy() if candidate.metadata else {},
                        rank=candidate.rank,
                    )
                    normalized.append(new_candidate)
                return normalized

        elif method == "zscore":
            mean = statistics.mean(scores) if scores else 0.0
            std = statistics.stdev(scores) if len(scores) > 1 else 1.0

            if std < 1e-6:
                normalized = []
                for candidate in candidates:
                    new_candidate = RetrievalCandidate(
                        chunk=candidate.chunk,
                        dense_score=candidate.dense_score,
                        bm25_score=candidate.bm25_score,
                        rrf_score=candidate.rrf_score,
                        reranker_score=candidate.reranker_score,
                        fusion_score=0.5 if score_type == "fusion" else candidate.fusion_score,
                        metadata=candidate.metadata.copy() if candidate.metadata else {},
                        rank=candidate.rank,
                    )
                    normalized.append(new_candidate)
                return normalized
            else:
                normalized = []
                for candidate, score in zip(candidates, scores):
                    # Clip to [0, 1] range using 3-sigma rule
                    norm_score = max(0.0, min(1.0, (score - mean) / (std * 2) + 0.5))
                    new_candidate = RetrievalCandidate(
                        chunk=candidate.chunk,
                        dense_score=candidate.dense_score,
                        bm25_score=candidate.bm25_score,
                        rrf_score=candidate.rrf_score,
                        reranker_score=candidate.reranker_score,
                        fusion_score=norm_score if score_type == "fusion" else candidate.fusion_score,
                        metadata=candidate.metadata.copy() if candidate.metadata else {},
                        rank=candidate.rank,
                    )
                    normalized.append(new_candidate)
                return normalized

        else:
            # Fallback: no normalization
            return candidates

    # ==========================================================================
    # ADAPTIVE K CALCULATION
    # ==========================================================================

    def _calculate_adaptive_k(
        self,
        candidate_lists: List[List[RetrievalCandidate]],
    ) -> int:
        """
        Calculate adaptive K based on result quality and distribution.

        Args:
            candidate_lists: Lists of candidates from each strategy.

        Returns:
            int: Adaptive K value.
        """
        if not candidate_lists or all(len(r) == 0 for r in candidate_lists):
            return self.k

        non_empty = [r for r in candidate_lists if r]

        # 1. K based on number of strategies
        num_strategies = len(non_empty)
        if num_strategies == 1:
            return self.k

        # 2. K based on average results per strategy
        total_results = sum(len(r) for r in non_empty)
        avg_results_per_strategy = total_results / num_strategies

        if avg_results_per_strategy < 10:
            # Very few results: increase K to reduce rank impact
            return min(int(self.k * 2.5), 150)
        elif avg_results_per_strategy < 20:
            return min(int(self.k * 2.0), 120)
        elif avg_results_per_strategy < 50:
            return int(self.k * 1.5)

        # 3. Check score distribution variance
        try:
            all_scores = []
            for candidate_list in non_empty:
                if len(candidate_list) > 1:
                    scores = [c.get_best_score() for c in candidate_list]
                    all_scores.extend(scores)

            if all_scores and len(all_scores) > 1:
                variance = statistics.variance(all_scores)
                mean_score = statistics.mean(all_scores)

                # High variance: scores are spread out
                if variance > 0.1 and mean_score > 0.3:
                    return min(int(self.k * 1.2), 100)

                # Low variance: scores are similar
                if variance < 0.01:
                    return min(int(self.k * 1.8), 120)

        except (ValueError, statistics.StatisticsError):
            pass

        return self.k

    # ==========================================================================
    # STRATEGY QUALITY WEIGHTING
    # ==========================================================================

    def _calculate_strategy_weights(
        self,
        candidate_lists: List[List[RetrievalCandidate]],
    ) -> List[float]:
        """
        Calculate quality-based weights for each strategy.

        Args:
            candidate_lists: Lists of candidates from each strategy.

        Returns:
            List[float]: Weights for each strategy.
        """
        if not self.strategy_weighting:
            return [1.0] * len(candidate_lists)

        weights = []
        for candidate_list in candidate_lists:
            if not candidate_list:
                weights.append(0.0)
                continue

            # Quality metrics:
            # 1. Average score
            scores = [c.get_best_score() for c in candidate_list]
            avg_score = statistics.mean(scores) if scores else 0.0

            # 2. Score spread (higher spread = more discriminative)
            spread = statistics.stdev(scores) if len(scores) > 1 else 0.0

            # 3. Number of results (more results = more coverage)
            count_weight = min(1.0, len(candidate_list) / 100)

            # Combined quality score
            quality = (avg_score * 0.5) + (spread * 0.3) + (count_weight * 0.2)
            weights.append(max(0.1, min(1.0, quality)))

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)

        return weights

    # ==========================================================================
    # FUSION METHODS - PHASE 4: Uses RetrievalCandidate
    # ==========================================================================

    def fuse(
        self,
        *candidate_lists: List[RetrievalCandidate],
        candidate_count: Optional[int] = None,
    ) -> List[RetrievalCandidate]:
        """
        Fuse multiple candidate lists using RRF with normalization.

        PHASE 4: Uses RetrievalCandidate instead of modifying Chunk.

        Each input list should be pre-sorted by relevance (most relevant first).
        The fusion produces a unified ranking that considers all strategies.

        PRESERVES rrf_score in RetrievalCandidate.

        Args:
            *candidate_lists: Variable number of candidate lists from different strategies.
            candidate_count: Number of top candidates to return.

        Returns:
            list[RetrievalCandidate]: Fused results sorted by RRF score descending.

        Raises:
            FusionError: If fusion fails.
        """
        with measure_latency("rrf_fusion") as latency:
            lists = list(candidate_lists)

            logger.log_event(
                event=LogEvent.RRF_FUSION,
                message=f"PHASE 4: Starting RRF fusion with {len(lists)} strategies",
                details={
                    "num_strategies": len(lists),
                    "base_k": self.k,
                    "normalize_scores": self.normalize_scores,
                    "normalization_method": self.normalization_method,
                    "adaptive_k": self.adaptive_k,
                    "strategy_sizes": [len(r) for r in lists],
                    "phase_4_retrieval_candidate": True,
                },
            )

            try:
                # 1. Handle empty results
                non_empty_lists = [r for r in lists if len(r) > 0]

                if not non_empty_lists:
                    logger.log_event(
                        event=LogEvent.RRF_FUSION,
                        message="All candidate lists are empty",
                        level=30,
                    )
                    return []

                # 2. Single strategy: return as-is with RRF score
                if len(non_empty_lists) == 1:
                    logger.log_event(
                        event=LogEvent.RRF_FUSION,
                        message="Only one strategy returned results, using directly",
                        details={"results_count": len(non_empty_lists[0])},
                    )

                    effective_candidate_count = int(
                        candidate_count or settings.RRF_CANDIDATE_COUNT
                    )

                    # Normalize if requested
                    if self.normalize_scores:
                        normalized = self._normalize_scores(
                            non_empty_lists[0],
                            score_type="best",
                            method=self.normalization_method,
                        )
                    else:
                        normalized = non_empty_lists[0]

                    # PHASE 4: Create new candidates with RRF score
                    fused_results = []
                    for rank, candidate in enumerate(normalized, start=1):
                        rrf_score = 1.0 / (self.k + rank)
                        new_candidate = RetrievalCandidate(
                            chunk=candidate.chunk,
                            dense_score=candidate.dense_score,
                            bm25_score=candidate.bm25_score,
                            rrf_score=rrf_score,
                            reranker_score=candidate.reranker_score,
                            fusion_score=rrf_score,
                            metadata=candidate.metadata.copy() if candidate.metadata else {},
                            rank=rank,
                        )
                        fused_results.append(new_candidate)

                    if effective_candidate_count > 0:
                        fused_results = fused_results[:effective_candidate_count]

                    return fused_results

                # 3. Multiple strategies: normalize scores
                normalized_lists = []
                for candidate_list in non_empty_lists:
                    if self.normalize_scores:
                        normalized = self._normalize_scores(
                            candidate_list,
                            score_type="best",
                            method=self.normalization_method,
                        )
                    else:
                        normalized = candidate_list
                    normalized_lists.append(normalized)

                # 4. Calculate adaptive K
                adaptive_k = self._calculate_adaptive_k(normalized_lists) if self.adaptive_k else self.k

                # 5. Calculate strategy weights
                strategy_weights = self._calculate_strategy_weights(normalized_lists)

                # 6. Apply RRF with weights
                candidate_scores: Dict[str, Tuple[RetrievalCandidate, float]] = {}

                for strategy_idx, strategy_candidates in enumerate(normalized_lists):
                    if not strategy_candidates:
                        continue

                    weight = strategy_weights[strategy_idx] if strategy_idx < len(strategy_weights) else 1.0

                    for rank, candidate in enumerate(strategy_candidates, start=1):
                        chunk_id = candidate.chunk.metadata.chunk_id

                        rrf_score = (1.0 / (adaptive_k + rank)) * weight

                        if chunk_id not in candidate_scores:
                            candidate_scores[chunk_id] = (candidate, 0.0)

                        candidate_scores[chunk_id] = (
                            candidate_scores[chunk_id][0],
                            candidate_scores[chunk_id][1] + rrf_score,
                        )

                # 7. Build fused results
                fused_results = []
                for chunk_id, (candidate, score) in candidate_scores.items():
                    new_candidate = RetrievalCandidate(
                        chunk=candidate.chunk,
                        dense_score=candidate.dense_score,
                        bm25_score=candidate.bm25_score,
                        rrf_score=score,
                        reranker_score=candidate.reranker_score,
                        fusion_score=score,
                        metadata=candidate.metadata.copy() if candidate.metadata else {},
                        rank=None,
                    )
                    fused_results.append(new_candidate)

                fused_results.sort(key=lambda x: x.rrf_score or 0.0, reverse=True)

                # Add rank metadata
                for rank, candidate in enumerate(fused_results, start=1):
                    candidate.rank = rank

                # 8. Apply candidate count limit
                effective_candidate_count = int(
                    candidate_count or settings.RRF_CANDIDATE_COUNT
                )
                if effective_candidate_count > 0:
                    fused_results = fused_results[:effective_candidate_count]

                latency.stop(
                    input_strategies=len(lists),
                    output_candidates=len(fused_results),
                    k=adaptive_k,
                )

                logger.log_event(
                    event=LogEvent.RRF_FUSION,
                    message=f"PHASE 4: RRF fusion complete: {len(fused_results)} candidates",
                    details={
                        "input_strategies": len(lists),
                        "output_candidates": len(fused_results),
                        "k": adaptive_k,
                        "normalization": self.normalization_method,
                        "strategy_weights": [
                            round(w, 3) for w in strategy_weights
                        ],
                        "duration_ms": latency.duration_ms,
                        "phase_4_retrieval_candidate": True,
                    },
                )

                return fused_results

            except Exception as exc:
                raise FusionError(
                    message=f"RRF fusion failed: {str(exc)}",
                    strategies=[f"strategy_{i}" for i in range(len(lists))],
                    original_exception=exc,
                )

    def fuse_weighted(
        self,
        candidates_with_weights: List[Tuple[List[RetrievalCandidate], float]],
        candidate_count: Optional[int] = None,
    ) -> List[RetrievalCandidate]:
        """
        Fuse results with custom weights for each strategy.

        PHASE 4: Uses RetrievalCandidate.

        Args:
            candidates_with_weights: List of (candidates_list, weight) tuples.
            candidate_count: Number of candidates to return.

        Returns:
            list[RetrievalCandidate]: Fused results.
        """
        with measure_latency("rrf_weighted_fusion") as latency:
            logger.log_event(
                event=LogEvent.RRF_FUSION,
                message=f"PHASE 4: Starting weighted RRF fusion with {len(candidates_with_weights)} strategies",
                details={
                    "num_strategies": len(candidates_with_weights),
                    "weights": [w for _, w in candidates_with_weights],
                    "phase_4_retrieval_candidate": True,
                },
            )

            try:
                # Normalize scores for each strategy
                normalized_lists = []
                for candidate_list, weight in candidates_with_weights:
                    if self.normalize_scores and candidate_list:
                        normalized = self._normalize_scores(
                            candidate_list,
                            score_type="best",
                            method=self.normalization_method,
                        )
                    else:
                        normalized = candidate_list
                    normalized_lists.append((normalized, weight))

                candidate_scores: Dict[str, Tuple[RetrievalCandidate, float]] = {}

                for strategy_candidates, weight in normalized_lists:
                    if not strategy_candidates:
                        continue

                    for rank, candidate in enumerate(strategy_candidates, start=1):
                        chunk_id = candidate.chunk.metadata.chunk_id

                        rrf_score = (1.0 / (self.k + rank)) * weight

                        if chunk_id not in candidate_scores:
                            candidate_scores[chunk_id] = (candidate, 0.0)

                        candidate_scores[chunk_id] = (
                            candidate_scores[chunk_id][0],
                            candidate_scores[chunk_id][1] + rrf_score,
                        )

                fused_results = []
                for chunk_id, (candidate, score) in candidate_scores.items():
                    new_candidate = RetrievalCandidate(
                        chunk=candidate.chunk,
                        dense_score=candidate.dense_score,
                        bm25_score=candidate.bm25_score,
                        rrf_score=score,
                        reranker_score=candidate.reranker_score,
                        fusion_score=score,
                        metadata=candidate.metadata.copy() if candidate.metadata else {},
                        rank=None,
                    )
                    fused_results.append(new_candidate)

                fused_results.sort(key=lambda x: x.rrf_score or 0.0, reverse=True)

                # Add rank metadata
                for rank, candidate in enumerate(fused_results, start=1):
                    candidate.rank = rank

                effective_candidate_count = int(
                    candidate_count or settings.RRF_CANDIDATE_COUNT
                )
                if effective_candidate_count > 0:
                    fused_results = fused_results[:effective_candidate_count]

                latency.stop(
                    input_strategies=len(candidates_with_weights),
                    output_candidates=len(fused_results),
                )

                return fused_results

            except Exception as exc:
                raise FusionError(
                    message=f"Weighted RRF fusion failed: {str(exc)}",
                    strategies=[f"strategy_{i}" for i in range(len(candidates_with_weights))],
                    original_exception=exc,
                )

    def fuse_multi_query(
        self,
        query_results: List[Tuple[str, float, List[RetrievalCandidate]]],
        candidate_count: Optional[int] = None,
    ) -> List[RetrievalCandidate]:
        """
        Fuse results from multiple query variants.

        PHASE 4: Uses RetrievalCandidate.

        Each variant has:
        - query_text: The variant text
        - weight: The weight for this variant (0.0 to 1.0)
        - results: List of RetrievalCandidate for this variant

        Args:
            query_results: List of (query_text, weight, results) tuples.
            candidate_count: Number of candidates to return.

        Returns:
            list[RetrievalCandidate]: Fused results.
        """
        with measure_latency("rrf_multi_query_fusion") as latency:
            logger.log_event(
                event=LogEvent.RRF_FUSION,
                message=f"PHASE 4: Starting multi-query RRF fusion with {len(query_results)} variants",
                details={
                    "num_variants": len(query_results),
                    "weights": [w for _, w, _ in query_results],
                    "phase_4_retrieval_candidate": True,
                },
            )

            try:
                # Normalize scores for each variant
                normalized_variants = []
                for query_text, weight, results in query_results:
                    if self.normalize_scores and results:
                        normalized = self._normalize_scores(
                            results,
                            score_type="best",
                            method=self.normalization_method,
                        )
                    else:
                        normalized = results
                    normalized_variants.append((query_text, weight, normalized))

                candidate_scores: Dict[str, Tuple[RetrievalCandidate, float]] = {}

                for query_text, weight, results in normalized_variants:
                    if not results:
                        continue

                    for rank, candidate in enumerate(results, start=1):
                        chunk_id = candidate.chunk.metadata.chunk_id

                        rrf_score = (1.0 / (self.k + rank)) * weight

                        if chunk_id not in candidate_scores:
                            candidate_scores[chunk_id] = (candidate, 0.0)

                        candidate_scores[chunk_id] = (
                            candidate_scores[chunk_id][0],
                            candidate_scores[chunk_id][1] + rrf_score,
                        )

                fused_results = []
                for chunk_id, (candidate, score) in candidate_scores.items():
                    new_candidate = RetrievalCandidate(
                        chunk=candidate.chunk,
                        dense_score=candidate.dense_score,
                        bm25_score=candidate.bm25_score,
                        rrf_score=score,
                        reranker_score=candidate.reranker_score,
                        fusion_score=score,
                        metadata=candidate.metadata.copy() if candidate.metadata else {},
                        rank=None,
                    )
                    fused_results.append(new_candidate)

                fused_results.sort(key=lambda x: x.rrf_score or 0.0, reverse=True)

                # Add rank metadata
                for rank, candidate in enumerate(fused_results, start=1):
                    candidate.rank = rank

                effective_candidate_count = int(
                    candidate_count or settings.RRF_CANDIDATE_COUNT
                )
                if effective_candidate_count > 0:
                    fused_results = fused_results[:effective_candidate_count]

                latency.stop(
                    input_variants=len(query_results),
                    output_candidates=len(fused_results),
                )

                logger.log_event(
                    event=LogEvent.RRF_FUSION,
                    message=f"PHASE 4: Multi-query RRF fusion complete: {len(fused_results)} candidates",
                    details={
                        "input_variants": len(query_results),
                        "output_candidates": len(fused_results),
                        "duration_ms": latency.duration_ms,
                        "phase_4_retrieval_candidate": True,
                    },
                )

                return fused_results

            except Exception as exc:
                raise FusionError(
                    message=f"Multi-query RRF fusion failed: {str(exc)}",
                    strategies=[f"variant_{i}" for i in range(len(query_results))],
                    original_exception=exc,
                )

    def get_config(self) -> Dict[str, Any]:
        """Get RRF configuration."""
        return {
            "k": self.k,
            "adaptive_k": self.adaptive_k,
            "normalize_scores": self.normalize_scores,
            "normalization_method": self.normalization_method,
            "strategy_weighting": self.strategy_weighting,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get RRF statistics."""
        return {
            "config": self.get_config(),
            "name": "ReciprocalRankFusion_Enhanced",
            "version": "2.0",
            "phase_4_retrieval_candidate": True,
        }