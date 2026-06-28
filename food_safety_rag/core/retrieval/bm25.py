"""
BM25 sparse retrieval for keyword-based search.

Uses BM25 algorithm for lexical matching and exact terminology retrieval.
"""

from typing import List, Tuple, Optional, Dict
import re

from food_safety_rag.core.exceptions import BM25RetrievalException
from food_safety_rag.core.monitoring.logger import get_logger

logger = get_logger(__name__)

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


class BM25Retriever:
    """
    BM25-based sparse retrieval.
    
    Uses keyword frequency and document length normalization to retrieve
    chunks matching specific terminology and concepts.
    """

    def __init__(self) -> None:
        """
        Initialize BM25 retriever.
        
        Raises:
            BM25RetrievalException: If BM25 library is not installed.
        """
        if BM25Okapi is None:
            raise BM25RetrievalException(
                "rank_bm25 library not installed. Install with: pip install rank_bm25"
            )
        self.bm25 = None
        self.chunk_ids = []
        logger.info("BM25 retriever initialized")

    def index_chunks(
        self, chunk_texts: List[str], chunk_ids: List[str]
    ) -> None:
        """
        Build BM25 index from chunks.
        
        Args:
            chunk_texts: List of chunk text content.
            chunk_ids: Corresponding chunk IDs.
            
        Raises:
            BM25RetrievalException: If indexing fails.
        """
        if len(chunk_texts) != len(chunk_ids):
            raise BM25RetrievalException(
                "chunk_texts and chunk_ids must have same length"
            )

        try:
            # Tokenize chunks
            tokenized_chunks = [
                self._tokenize(text) for text in chunk_texts
            ]

            # Build BM25 index
            self.bm25 = BM25Okapi(tokenized_chunks)
            self.chunk_ids = chunk_ids

            logger.info(
                "BM25 index built",
                chunk_count=len(chunk_texts),
            )
        except Exception as e:
            raise BM25RetrievalException(
                f"Failed to build BM25 index: {e}"
            ) from e

    def retrieve(
        self,
        query: str,
        k: int = 50,
    ) -> Tuple[List[str], List[float]]:
        """
        Retrieve chunks using BM25.
        
        Args:
            query: Search query text.
            k: Number of results to retrieve.
            
        Returns:
            Tuple[List[str], List[float]]: Chunk IDs and BM25 scores.
            
        Raises:
            BM25RetrievalException: If retrieval fails.
        """
        if self.bm25 is None:
            logger.warning("BM25 retrieval attempted on uninitialized index")
            return [], []

        try:
            # Tokenize query
            query_tokens = self._tokenize(query)

            if not query_tokens:
                return [], []

            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)

            # Sort by score (descending) and get top k
            ranked = sorted(
                enumerate(scores), key=lambda x: x[1], reverse=True
            )[:k]

            chunk_ids = [self.chunk_ids[idx] for idx, _ in ranked]
            bm25_scores = [float(score) for _, score in ranked]

            logger.debug(
                "BM25 retrieval completed",
                results_count=len(chunk_ids),
                k_requested=k,
                query_tokens_count=len(query_tokens),
            )
            return chunk_ids, bm25_scores
        except Exception as e:
            raise BM25RetrievalException(
                f"BM25 retrieval failed: {e}", query=query
            ) from e

    def batch_retrieve(
        self, queries: List[str], k: int = 50
    ) -> List[Tuple[List[str], List[float]]]:
        """
        Retrieve for multiple queries.
        
        Args:
            queries: List of query texts.
            k: Number of results per query.
            
        Returns:
            List[Tuple[List[str], List[float]]]: Results for each query.
        """
        results = []
        for query in queries:
            chunk_ids, scores = self.retrieve(query, k)
            results.append((chunk_ids, scores))
        return results

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Simple tokenization for BM25.
        
        Args:
            text: Text to tokenize.
            
        Returns:
            List[str]: List of tokens.
        """
        # Convert to lowercase and remove punctuation
        text = text.lower()
        # Split on whitespace and remove empty tokens
        tokens = text.split()
        # Remove punctuation from tokens
        tokens = [
            re.sub(r"[^a-z0-9]", "", token)
            for token in tokens
        ]
        # Filter out empty tokens
        return [t for t in tokens if t]
