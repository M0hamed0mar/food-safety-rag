"""
Reranking layer for improving retrieval result quality.

Uses cross-encoder models to rerank retrieval results based on query-document relevance.
"""

from typing import List, Tuple, Optional
import json

from food_safety_rag.core.exceptions import RerankingException
from food_safety_rag.core.monitoring.logger import get_logger

logger = get_logger(__name__)

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


class Reranker:
    """
    Cross-encoder-based reranker for result refinement.
    
    Uses a cross-encoder model to compute relevance scores between
    queries and documents, enabling more accurate ranking than
    retrieval-time scores alone.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        batch_size: int = 32,
    ) -> None:
        """
        Initialize reranker.
        
        Args:
            model_name: HuggingFace model name for cross-encoder.
            batch_size: Batch size for scoring.
            
        Raises:
            RerankingException: If model loading fails.
        """
        if CrossEncoder is None:
            raise RerankingException(
                "sentence-transformers library not installed. "
                "Install with: pip install sentence-transformers"
            )

        try:
            self.model = CrossEncoder(model_name)
            self.model_name = model_name
            self.batch_size = batch_size
            logger.info(
                "Reranker initialized",
                model_name=model_name,
                batch_size=batch_size,
            )
        except Exception as e:
            raise RerankingException(
                f"Failed to load reranker model: {e}", model=model_name
            ) from e

    def rerank(
        self,
        query: str,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Rerank documents for a query.
        
        Args:
            query: The query text.
            documents: List of document texts to rerank.
            doc_ids: Optional document IDs (uses indices if not provided).
            top_k: Optional number of top results to return.
            
        Returns:
            Tuple[List[str], List[float]]: Reranked doc IDs and scores.
            
        Raises:
            RerankingException: If reranking fails.
        """
        if not documents:
            return [], []

        if doc_ids is None:
            doc_ids = [str(i) for i in range(len(documents))]

        try:
            # Create query-document pairs
            pairs = [[query, doc] for doc in documents]

            # Score pairs
            scores = self.model.predict(pairs, batch_size=self.batch_size)

            # Rank by score
            ranked = sorted(
                zip(doc_ids, scores, documents),
                key=lambda x: x[1],
                reverse=True,
            )

            # Optionally limit to top_k
            if top_k:
                ranked = ranked[:top_k]

            reranked_ids = [item[0] for item in ranked]
            reranked_scores = [float(item[1]) for item in ranked]

            logger.debug(
                "Reranking completed",
                documents_scored=len(documents),
                top_k=top_k or len(ranked),
            )

            return reranked_ids, reranked_scores
        except Exception as e:
            raise RerankingException(
                f"Reranking failed: {e}",
                model=self.model_name,
            ) from e

    def batch_rerank(
        self,
        queries: List[str],
        documents_per_query: List[List[str]],
        doc_ids_per_query: Optional[List[List[str]]] = None,
        top_k: Optional[int] = None,
    ) -> List[Tuple[List[str], List[float]]]:
        """
        Rerank results for multiple queries.
        
        Args:
            queries: List of queries.
            documents_per_query: Nested list of documents per query.
            doc_ids_per_query: Optional nested list of doc IDs.
            top_k: Optional number of top results per query.
            
        Returns:
            List[Tuple[List[str], List[float]]]: Reranked results per query.
        """
        results = []
        for i, query in enumerate(queries):
            docs = documents_per_query[i]
            doc_ids = doc_ids_per_query[i] if doc_ids_per_query else None
            reranked = self.rerank(query, docs, doc_ids, top_k)
            results.append(reranked)
        return results
