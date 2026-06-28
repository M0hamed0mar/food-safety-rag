"""
Query expansion for improved retrieval coverage.

Generates alternative query formulations to improve recall.
"""

from typing import List

from food_safety_rag.core.exceptions import QueryNormalizationException
from food_safety_rag.core.monitoring.logger import get_logger
from food_safety_rag.core.llm.openai_client import OpenAIClient
from food_safety_rag.core.schemas.query import QueryExpansion

logger = get_logger(__name__)


class QueryExpander:
    """
    Expands queries to improve retrieval coverage.
    
    Generates alternative query formulations using LLM and keyword analysis.
    """

    def __init__(self, llm_client: OpenAIClient) -> None:
        """
        Initialize query expander.
        
        Args:
            llm_client: OpenAI client for generating expansions.
        """
        self.llm_client = llm_client
        logger.info("Query expander initialized")

    def expand(
        self,
        query: str,
        num_expansions: int = 3,
    ) -> QueryExpansion:
        """
        Expand a query to generate alternatives.
        
        Args:
            query: Original query.
            num_expansions: Number of alternative queries to generate.
            
        Returns:
            QueryExpansion: Expansion result with alternatives.
            
        Raises:
            QueryNormalizationException: If expansion fails.
        """
        try:
            # Normalize the query
            normalized = self._normalize_query(query)

            # Generate expansions using LLM
            prompt = f"""Generate {num_expansions} alternative ways to express this question for a food safety knowledge base:

Original question: {query}

Provide alternative phrasings that might help retrieve more relevant documents. Format as a JSON list of strings.
Example format: ["alternative 1", "alternative 2", "alternative 3"]

Alternative questions:"""

            response = self.llm_client.generate_completion(
                system_prompt="You are a query expansion assistant. Generate alternative phrasings of questions to improve retrieval. Always respond with valid JSON.",
                user_message=prompt,
                temperature=0.7,
                max_tokens=500,
            )

            # Parse expansions
            try:
                expansions = self._parse_expansions(response)
            except Exception as e:
                logger.warning(
                    f"Failed to parse LLM expansion response: {e}. Using fallback."
                )
                expansions = self._fallback_expand(query, num_expansions)

            result = QueryExpansion(
                original_query=query,
                normalized_query=normalized,
                expanded_queries=expansions[:num_expansions],
                expansion_method="llm",
            )

            logger.debug(
                "Query expansion completed",
                original_query=query,
                expansions_count=len(result.expanded_queries),
            )

            return result
        except Exception as e:
            logger.error(f"Query expansion failed: {e}", query=query)
            # Return query with just normalization
            return QueryExpansion(
                original_query=query,
                normalized_query=self._normalize_query(query),
                expanded_queries=[],
                expansion_method="fallback",
            )

    def _normalize_query(self, query: str) -> str:
        """
        Normalize query by removing extra whitespace and standardizing format.
        
        Args:
            query: Input query.
            
        Returns:
            str: Normalized query.
        """
        # Remove extra whitespace
        normalized = " ".join(query.split())
        # Remove trailing punctuation
        normalized = normalized.rstrip(".?!")
        return normalized.strip()

    def _parse_expansions(self, response: str) -> List[str]:
        """
        Parse LLM response to extract expansions.
        
        Args:
            response: LLM response text.
            
        Returns:
            List[str]: Parsed expansion queries.
            
        Raises:
            ValueError: If parsing fails.
        """
        import json

        # Try to find JSON array in response
        import re

        # Look for JSON array pattern
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            json_str = match.group()
            expansions = json.loads(json_str)
            if isinstance(expansions, list):
                return [str(e).strip() for e in expansions if e]

        raise ValueError("Could not parse expansions from response")

    @staticmethod
    def _fallback_expand(
        query: str, num_expansions: int
    ) -> List[str]:
        """
        Fallback query expansion using simple techniques.
        
        Args:
            query: Original query.
            num_expansions: Number of variations to generate.
            
        Returns:
            List[str]: Generated query variations.
        """
        expansions = []

        # Add question form
        if not query.endswith("?"):
            expansions.append(query + "?")

        # Add without question mark
        if query.endswith("?"):
            expansions.append(query[:-1])

        # Add common variations
        if "how" in query.lower():
            expansions.append(query.replace("how", "what is the method for"))

        if "what" in query.lower():
            expansions.append(query.replace("what", "explain"))

        return expansions[:num_expansions]
