"""
Dense retrieval using FAISS semantic search.

Performs similarity-based retrieval using pre-computed embeddings.
"""

from typing import List, Tuple, Optional

from food_safety_rag.core.exceptions import DenseRetrievalException
from food_safety_rag.core.monitoring.logger import get_logger
from food_safety_rag.core.vector_store.faiss_store import FAISSStore

logger = get_logger(__name__)


class DenseRetriever:
    """
    Dense semantic retrieval using FAISS.
    
    Performs similarity search on embedding vectors to find semantically
    similar document chunks.
    """

    def __init__(self, faiss_store: FAISSStore) -> None:
        """
        Initialize dense retriever.
        
        Args:
            faiss_store: The FAISS vector store instance.
        """
        self.faiss_store = faiss_store
        logger.info("Dense retriever initialized")

    def retrieve(
        self,
        query_embedding: List[float],
        k: int = 50,
        threshold: Optional[float] = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Retrieve similar chunks using dense search.
        
        Args:
            query_embedding: Query embedding vector.
            k: Number of results to retrieve.
            threshold: Optional similarity threshold.
            
        Returns:
            Tuple[List[str], List[float]]: Chunk IDs and similarity scores.
            
        Raises:
            DenseRetrievalException: If retrieval fails.
        """
        try:
            if self.faiss_store.get_index_size() == 0:
                logger.warning("Dense retrieval attempted on empty index")
                return [], []

            chunk_ids, scores = self.faiss_store.search(
                query_embedding, k=k, threshold=threshold
            )

            logger.debug(
                "Dense retrieval completed",
                results_count=len(chunk_ids),
                k_requested=k,
            )
            return chunk_ids, scores
        except Exception as e:
            raise DenseRetrievalException(
                f"Dense retrieval failed: {e}"
            ) from e

    def batch_retrieve(
        self,
        query_embeddings: List[List[float]],
        k: int = 50,
        threshold: Optional[float] = None,
    ) -> List[Tuple[List[str], List[float]]]:
        """
        Retrieve similar chunks for multiple queries.
        
        Args:
            query_embeddings: List of query embedding vectors.
            k: Number of results per query.
            threshold: Optional similarity threshold.
            
        Returns:
            List[Tuple[List[str], List[float]]]: Results for each query.
            
        Raises:
            DenseRetrievalException: If batch retrieval fails.
        """
        results = []
        for embedding in query_embeddings:
            chunk_ids, scores = self.retrieve(embedding, k, threshold)
            results.append((chunk_ids, scores))
        return results
