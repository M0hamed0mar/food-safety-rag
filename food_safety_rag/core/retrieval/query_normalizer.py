"""
Query normalization and preprocessing.

Cleans and standardizes queries before retrieval.
"""

import re
from typing import Optional

from food_safety_rag.core.exceptions import QueryValidationException
from food_safety_rag.core.monitoring.logger import get_logger

logger = get_logger(__name__)


class QueryNormalizer:
    """
    Normalizes and validates user queries.
    
    Performs cleaning, validation, and standardization of input queries
    before retrieval and generation.
    """

    # Minimum and maximum query lengths
    MIN_QUERY_LENGTH = 3
    MAX_QUERY_LENGTH = 2000

    @staticmethod
    def normalize(query: str) -> str:
        """
        Normalize query text.
        
        Args:
            query: Input query.
            
        Returns:
            str: Normalized query.
            
        Raises:
            QueryValidationException: If query is invalid.
        """
        if not query:
            raise QueryValidationException("Query cannot be empty")

        # Remove leading/trailing whitespace
        query = query.strip()

        # Check minimum length
        if len(query) < QueryNormalizer.MIN_QUERY_LENGTH:
            raise QueryValidationException(
                f"Query too short (minimum {QueryNormalizer.MIN_QUERY_LENGTH} characters)"
            )

        # Check maximum length
        if len(query) > QueryNormalizer.MAX_QUERY_LENGTH:
            raise QueryValidationException(
                f"Query too long (maximum {QueryNormalizer.MAX_QUERY_LENGTH} characters)"
            )

        # Remove multiple spaces
        query = re.sub(r"\s+", " ", query)

        # Remove control characters
        query = "".join(char for char in query if ord(char) >= 32)

        logger.debug("Query normalized", normalized_length=len(query))
        return query

    @staticmethod
    def remove_stopwords(query: str) -> str:
        """
        Remove common stopwords from query.
        
        Args:
            query: Input query.
            
        Returns:
            str: Query with stopwords removed.
        """
        # Common English stopwords
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for",
            "from", "has", "he", "in", "is", "it", "its", "of", "on",
            "or", "that", "the", "to", "was", "will", "with",
        }

        words = query.lower().split()
        filtered = [w for w in words if w.lower() not in stopwords]
        return " ".join(filtered)

    @staticmethod
    def extract_keywords(query: str) -> list[str]:
        """
        Extract important keywords from query.
        
        Args:
            query: Input query.
            
        Returns:
            list[str]: Extracted keywords.
        """
        # Remove punctuation and split
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", query)
        words = cleaned.lower().split()
        # Filter short words
        keywords = [w for w in words if len(w) > 3]
        return list(set(keywords))

    @staticmethod
    def validate(query: str) -> bool:
        """
        Validate if query meets requirements.
        
        Args:
            query: Query to validate.
            
        Returns:
            bool: True if valid.
        """
        if not query:
            return False
        if len(query.strip()) < QueryNormalizer.MIN_QUERY_LENGTH:
            return False
        if len(query) > QueryNormalizer.MAX_QUERY_LENGTH:
            return False
        return True
