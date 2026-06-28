"""
Query schemas for user questions.

Represents user queries and query expansion results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class QueryRequest:
    """
    User query request.
    
    Represents a question from a user to be processed by the RAG system.
    """

    query: str
    """The user's question or query string."""

    query_id: str = field(default_factory=str)
    """Unique identifier for this query."""

    timestamp: datetime = field(default_factory=datetime.utcnow)
    """When query was received."""

    user_id: Optional[str] = None
    """Optional user identifier for tracking."""

    session_id: Optional[str] = None
    """Optional session identifier for multi-turn conversations."""

    metadata: dict = field(default_factory=dict)
    """Additional metadata about the query."""

    def __post_init__(self) -> None:
        """
        Validate query after initialization.
        
        Raises:
            ValueError: If query is empty or too short.
        """
        if not self.query or not self.query.strip():
            raise ValueError("Query cannot be empty")
        if len(self.query.strip()) < 3:
            raise ValueError("Query must be at least 3 characters")

    def get_normalized_query(self) -> str:
        """
        Get normalized version of query.
        
        Removes extra whitespace and normalizes case.
        
        Returns:
            str: Normalized query.
        """
        return " ".join(self.query.strip().split())


@dataclass
class QueryExpansion:
    """
    Result of query expansion.
    
    Contains the original query and generated alternative queries
    for improved retrieval.
    """

    original_query: str
    """The original user query."""

    normalized_query: str
    """Normalized version of original query."""

    expanded_queries: List[str]
    """Alternative queries generated for retrieval."""

    expansion_method: str = "llm"
    """Method used for expansion (llm, keyword, synonym, etc.)."""

    expansion_timestamp: datetime = field(default_factory=datetime.utcnow)
    """When expansion was performed."""

    def all_queries(self) -> List[str]:
        """
        Get all queries (original + expanded).
        
        Returns:
            List[str]: List of all queries including original.
        """
        return [self.normalized_query] + self.expanded_queries

    def unique_queries(self) -> List[str]:
        """
        Get unique queries (removing duplicates).
        
        Returns:
            List[str]: List of unique queries.
        """
        seen = set()
        unique = []
        for query in self.all_queries():
            normalized = query.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(query)
        return unique
