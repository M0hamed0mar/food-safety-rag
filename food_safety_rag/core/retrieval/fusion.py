"""
Reciprocal Rank Fusion for combining retrieval results.

Fuses dense and sparse retrieval results using RRF algorithm.
"""

from typing import List, Tuple, Dict

from food_safety_rag.core.monitoring.logger import get_logger

logger = get_logger(__name__)


class RRFFusion:
    """
    Reciprocal Rank Fusion algorithm.
    
    Combines results from multiple retrievers (dense + BM25) using
    reciprocal rank formula: RRF(d) = Σ (1 / (k + rank(d)))
    
    This approach is robust across different retrieval algorithms
    and doesn't require score normalization.
    """

    def __init__(self, rrf_constant: int = 60) -> None:
        """
        Initialize RRF fusion.
        
        Args:
            rrf_constant: Constant k in RRF formula (higher = less weight to rank).
        """
        self.rrf_constant = rrf_constant
        logger.info(
            "RRF Fusion initialized",
            rrf_constant=rrf_constant,
        )

    def fuse(
        self,
        dense_results: Tuple[List[str], List[float]],
        bm25_results: Tuple[List[str], List[float]],
    ) -> Tuple[List[str], List[float]]:
        """
        Fuse dense and BM25 retrieval results using RRF.
        
        Args:
            dense_results: (chunk_ids, scores) from dense retrieval.
            bm25_results: (chunk_ids, scores) from BM25 retrieval.
            
        Returns:
            Tuple[List[str], List[float]]: Fused results sorted by RRF score.
        """
        dense_ids, dense_scores = dense_results
        bm25_ids, bm25_scores = bm25_results

        # Compute RRF scores
        rrf_scores: Dict[str, float] = {}

        # Add scores from dense retrieval
        for rank, chunk_id in enumerate(dense_ids, 1):
            rrf_score = 1.0 / (self.rrf_constant + rank)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + rrf_score

        # Add scores from BM25 retrieval
        for rank, chunk_id in enumerate(bm25_ids, 1):
            rrf_score = 1.0 / (self.rrf_constant + rank)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + rrf_score

        # Sort by RRF score (descending)
        sorted_results = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )

        fused_ids = [chunk_id for chunk_id, _ in sorted_results]
        fused_scores = [score for _, score in sorted_results]

        logger.debug(
            "RRF fusion completed",
            dense_results_count=len(dense_ids),
            bm25_results_count=len(bm25_ids),
            fused_results_count=len(fused_ids),
            unique_chunks=len(rrf_scores),
        )

        return fused_ids, fused_scores

    @staticmethod
    def get_rrf_score(rank: int, constant: int = 60) -> float:
        """
        Calculate RRF score for a given rank.
        
        Args:
            rank: 1-based rank position.
            constant: RRF constant k.
            
        Returns:
            float: RRF score.
        """
        return 1.0 / (constant + rank)
