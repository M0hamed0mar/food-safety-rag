"""
Hybrid retrieval combining dense and sparse search.

Coordinates dense and BM25 retrievers with RRF fusion.
"""

from typing import List, Tuple, Optional

from food_safety_rag.core.exceptions import RetrievalException
from food_safety_rag.core.monitoring.logger import get_logger
from food_safety_rag.core.retrieval.dense import DenseRetriever
from food_safety_rag.core.retrieval.bm25 import BM25Retriever
from food_safety_rag.core.retrieval.fusion import RRFFusion
from food_safety_rag.core.vector_store.faiss_store import FAISSStore

logger = get_logger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining dense and sparse retrieval.
    
    Uses both semantic embeddings and keyword matching to maximize
    recall and precision.
    """

    def __init__(
        self,
        faiss_store: FAISSStore,
        dense_top_k: int = 50,
        bm25_top_k: int = 50,
        rrf_constant: int = 60,
    ) -> None:
        """
        Initialize hybrid retriever.
        
        Args:
            faiss_store: FAISS vector store instance.
            dense_top_k: Number of results from dense retrieval.
            bm25_top_k: Number of results from BM25 retrieval.
            rrf_constant: RRF fusion constant.
        """
        self.dense_retriever = DenseRetriever(faiss_store)
        self.bm25_retriever = BM25Retriever()
        self.rrf_fusion = RRFFusion(rrf_constant)
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        logger.info(
            "Hybrid retriever initialized",
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            rrf_constant=rrf_constant,
        )

    def retrieve(
        self,
        query_text: str,
        query_embedding: List[float],
        k: int = 30,
        threshold: Optional[float] = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Retrieve chunks using hybrid approach.
        
        Args:
            query_text: Query text for BM25.
            query_embedding: Query embedding for dense search.
            k: Number of final results to return.
            threshold: Optional similarity threshold for dense search.
            
        Returns:
            Tuple[List[str], List[float]]: Fused and ranked results.
            
        Raises:
            RetrievalException: If retrieval fails.
        """
        try:
            # Dense retrieval
            dense_results = self.dense_retriever.retrieve(
                query_embedding,
                k=self.dense_top_k,
                threshold=threshold,
            )

            # BM25 retrieval
            bm25_results = self.bm25_retriever.retrieve(
                query_text,
                k=self.bm25_top_k,
            )

            # Fuse results
            fused_ids, fused_scores = self.rrf_fusion.fuse(
                dense_results, bm25_results
            )

            # Return top k
            final_ids = fused_ids[:k]
            final_scores = fused_scores[:k]

            logger.debug(
                "Hybrid retrieval completed",
                dense_results=len(dense_results[0]),
                bm25_results=len(bm25_results[0]),
                fused_results=len(fused_ids),
                final_results=len(final_ids),
            )

            return final_ids, final_scores
        except Exception as e:
            raise RetrievalException(
                f"Hybrid retrieval failed: {e}", query=query_text
            ) from e
