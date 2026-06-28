"""
Retrieval router for selecting appropriate retrieval strategy.

Routes queries to optimal retrieval methods based on characteristics.
"""

from enum import Enum
from typing import Optional

from food_safety_rag.core.monitoring.logger import get_logger

logger = get_logger(__name__)


class RetrievalStrategy(Enum):
    """
    Enumeration of available retrieval strategies.
    """

    DENSE = "dense"
    """Use dense semantic retrieval only."""

    BM25 = "bm25"
    """Use BM25 keyword-based retrieval only."""

    HYBRID = "hybrid"
    """Use hybrid dense + BM25 with fusion."""


class RetrievalRouter:
    """
    Routes queries to optimal retrieval strategy.
    
    Analyzes query characteristics and selects the best retrieval method.
    """

    def __init__(self) -> None:
        """
        Initialize retrieval router.
        """
        logger.info("Retrieval router initialized")

    def route(
        self,
        query: str,
        default_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
    ) -> RetrievalStrategy:
        """
        Route query to retrieval strategy.
        
        Args:
            query: The query text.
            default_strategy: Strategy to use if analysis is inconclusive.
            
        Returns:
            RetrievalStrategy: Recommended retrieval strategy.
        """
        characteristics = self._analyze_query(query)

        # Decision logic
        if characteristics["has_specific_terms"]:
            strategy = RetrievalStrategy.HYBRID
            reason = "Query contains specific food safety terms"
        elif characteristics["is_conceptual"]:
            strategy = RetrievalStrategy.DENSE
            reason = "Query is conceptual, using dense retrieval"
        elif characteristics["is_keyword_heavy"]:
            strategy = RetrievalStrategy.BM25
            reason = "Query is keyword-heavy, using BM25"
        else:
            strategy = default_strategy
            reason = "Using default strategy"

        logger.debug(
            "Query routed",
            strategy=strategy.value,
            reason=reason,
            query_length=len(query),
        )

        return strategy

    @staticmethod
    def _analyze_query(query: str) -> dict:
        """
        Analyze query characteristics.
        
        Args:
            query: Query to analyze.
            
        Returns:
            dict: Analysis results.
        """
        query_lower = query.lower()

        # Check for specific food safety terms
        food_safety_terms = {
            "haccp", "pathogen", "bacteria", "contamination",
            "temperature", "storage", "hygiene", "allergen",
            "cross-contamination", "sanitization", "shelf-life",
        }
        has_specific_terms = any(
            term in query_lower for term in food_safety_terms
        )

        # Check if conceptual (asking "what", "how", "why")
        is_conceptual = any(
            word in query_lower
            for word in ["what is", "how do", "why", "explain", "understand"]
        )

        # Check if keyword-heavy (many quoted terms or "and"/"or")
        keyword_operators = query_lower.count(" and ") + query_lower.count(
            " or "
        )
        is_keyword_heavy = keyword_operators > 2 or "\"" in query

        return {
            "has_specific_terms": has_specific_terms,
            "is_conceptual": is_conceptual,
            "is_keyword_heavy": is_keyword_heavy,
        }
