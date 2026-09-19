"""
Query schema definitions.

This module defines Pydantic models for user queries and their processed
variants throughout the retrieval pipeline.

"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class Query(BaseModel):
    """
    A user query submitted to the RAG system.
    
    This is the primary input model for the retrieval pipeline.
    
    PHASE 14: Added translated_text field for Arabic → English translation.
    The retrieval pipeline uses translated_text while generation uses original_text.
    
    PHASE 15.7 FIX: is_arabic is set explicitly during query creation.
    The field_validator ensures consistent behavior.
    
    Attributes:
        query_id: Unique identifier for this query.
        original_text: The exact text submitted by the user.
        normalized_text: Query after normalization processing.
        translated_text: PHASE 14 - English translation for retrieval.
        language: Detected language code of the query.
        is_arabic: PHASE 14 - Whether the query is in Arabic.
        timestamp: When the query was received.
        user_id: Optional identifier of the submitting user.
        session_id: Optional session identifier for conversation tracking.
        history: Previous messages for context (max 3-4 recent turns).
        metadata: Additional query metadata.
    """
    
    query_id: str = Field(
        ...,
        description="Unique identifier for this query.",
        min_length=1,
    )
    original_text: str = Field(
        ...,
        description="The exact text submitted by the user.",
        min_length=1,
    )
    normalized_text: Optional[str] = Field(
        default=None,
        description="Query after normalization processing.",
    )
    translated_text: Optional[str] = Field(
        default=None,
        description="PHASE 14: English translation of the query for retrieval.",
    )
    language: Optional[str] = Field(
        default=None,
        description="Detected language code of the query (ISO 639-1).",
    )
    is_arabic: bool = Field(
        default=False,
        description="PHASE 14: Whether the query is in Arabic. "
                    "PHASE 15.7 FIX: Set explicitly during query creation.",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the query was received.",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Optional identifier of the submitting user.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier for conversation tracking.",
    )
    history: Optional[list[dict[str, str]]] = Field(
        default=None,
        description="Previous messages for context (max 3-4 recent turns).",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional query metadata.",
    )
    
    @field_validator("original_text")
    @classmethod
    def validate_query_not_empty(cls, value: str) -> str:
        """
        Ensure the query text is not empty or whitespace-only.
        
        Args:
            value: The query text to validate.
        
        Returns:
            str: The validated query text.
        
        Raises:
            ValueError: If the query is empty or whitespace-only.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("Query text cannot be empty or whitespace-only.")
        return value
    
    @field_validator("original_text")
    @classmethod
    def validate_query_length(cls, value: str) -> str:
        """
        Ensure the query does not exceed maximum length.
        
        Args:
            value: The query text to validate.
        
        Returns:
            str: The validated query text.
        
        Raises:
            ValueError: If the query exceeds maximum length.
        """
        if len(value) > 10000:
            raise ValueError(f"Query exceeds maximum length of 10000 characters: {len(value)}")
        return value
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "query_id": "qry_xyz789",
                "original_text": "ما هي نقاط التحكم الحرجة للوقاية من السالمونيلا؟",
                "translated_text": "What are the critical control points for Salmonella prevention?",
                "language": "ar",
                "is_arabic": True,
                "timestamp": "2024-01-15T10:30:00Z",
                "session_id": "chat_20240115_103000",
            }
        }
    
    def _detect_language(self) -> str:
        """
        Detect the language of the query text.
        
        Returns:
            str: Language code ('ar' for Arabic, 'en' for English, 'unknown' otherwise).
        """
        text = self.original_text.strip()
        if not text:
            return "unknown"
        
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        ratio = arabic_chars / len(text) if len(text) > 0 else 0
        
        if ratio >= 0.3:
            return "ar"
        elif ratio == 0:
            # Check if it's English (ASCII characters)
            english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
            if english_chars / len(text) > 0.5:
                return "en"
        return "unknown"
    
    def _is_query_arabic(self) -> bool:
        """
        Determine if the query is in Arabic.
        
        Returns:
            bool: True if the query is Arabic, False otherwise.
        """
        # PHASE 15.7 FIX: Use the explicitly set is_arabic if available
        if self.is_arabic:
            return True
        # Fallback to detection
        return self._detect_language() == "ar"
    
    def get_effective_query(self) -> str:
        """
        Get the effective query for RETRIEVAL.
        
        PHASE 14: Returns translated text if available, otherwise original.
        This ensures the retrieval pipeline works with English text when possible.
        
        PHASE 15.7: Preserves original language for generation.
        - Use get_effective_query() for RETRIEVAL (embedding, BM25)
        - Use get_original_query() for GENERATION (LLM prompt)
        
        Returns:
            str: The effective query string for retrieval.
        """
        if self.translated_text:
            return self.translated_text.strip()
        return self.original_text.strip()
    
    def get_original_query(self) -> str:
        """
        Get the original query text for GENERATION.
        
        PHASE 14: Returns the original text to preserve the user's language.
        The LLM should respond in the same language as the user's question.
        
        PHASE 15.7 FIX: This is the CRITICAL method for language preservation.
        The generator.py uses this to build the prompt in the user's language.
        
        Returns:
            str: The original query string for generation.
        """
        return self.original_text.strip()
    
    def get_language(self) -> str:
        """
        Get the detected language of the query.
        
        Returns:
            str: Language code ('ar', 'en', or 'unknown').
        """
        if self.language:
            return self.language
        self.language = self._detect_language()
        return self.language
    
    def get_history_text(self) -> str:
        """
        Get formatted conversation history for prompt context.
        
        Returns:
            str: Formatted history text, or empty string if no history.
        """
        if not self.history:
            return ""
        
        lines = ["Previous conversation:"]
        for msg in self.history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    def get_last_turn(self) -> Optional[dict[str, str]]:
        """
        Get the last message from history if available.
        
        Returns:
            Optional[dict[str, str]]: The last message, or None.
        """
        if not self.history:
            return None
        return self.history[-1]
    
    def has_translation(self) -> bool:
        """
        Check if the query has been translated.
        
        Returns:
            bool: True if translated_text is set and different from original.
        """
        return (
            self.translated_text is not None
            and self.translated_text != self.original_text
            and self.translated_text.strip() != ""
        )
    
    def to_retrieval_dict(self) -> dict[str, Any]:
        """
        Convert to a dictionary suitable for retrieval logging.
        
        Returns:
            dict[str, Any]: Dictionary with retrieval-relevant fields.
        """
        return {
            "query_id": self.query_id,
            "query_text": self.get_effective_query(),
            "original_text": self.original_text,
            "translated_text": self.translated_text,
            "language": self.get_language(),
            "is_arabic": self.is_arabic,
            "session_id": self.session_id,
        }


class ExpandedQuery(BaseModel):
    """
    An expanded variant of a user query for improved retrieval recall.
    
    PHASE 14: Works with translated English queries only.
    
    Attributes:
        original_query: Reference to the parent query.
        variant_text: The expanded query text (English).
        expansion_type: Type of expansion applied (synonym, rewrite, etc.).
        confidence: Confidence score for this expansion (0.0 to 1.0).
        weight: Weight for this variant in multi-query retrieval (0.0 to 1.0).
    """
    
    original_query: Query = Field(
        ...,
        description="Reference to the parent query.",
    )
    variant_text: str = Field(
        ...,
        description="The expanded query text.",
        min_length=1,
    )
    expansion_type: str = Field(
        default="general",
        description="Type of expansion applied (synonym, rewrite, abbreviation, etc.).",
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score for this expansion (0.0 to 1.0).",
        ge=0.0,
        le=1.0,
    )
    weight: float = Field(
        default=0.25,
        description="Weight for this variant in multi-query retrieval (0.0 to 1.0).",
        ge=0.0,
        le=1.0,
    )
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "variant_text": "What are the CCPs for Salmonella control in food processing?",
                "expansion_type": "abbreviation",
                "confidence": 0.9,
                "weight": 0.25,
            }
        }


class QueryExpansionResult(BaseModel):
    """
    Complete result of query expansion containing all variants.
    
    PHASE 14: All variants are in English (translated).
    
    Attributes:
        original_query: The original user query.
        expanded_queries: List of expanded query variants.
        total_variants: Total number of variants generated.
        expansion_method: Method used for expansion.
    """
    
    original_query: Query = Field(
        ...,
        description="The original user query.",
    )
    expanded_queries: list[ExpandedQuery] = Field(
        default_factory=list,
        description="List of expanded query variants.",
    )
    total_variants: int = Field(
        default=0,
        description="Total number of variants generated.",
        ge=0,
    )
    expansion_method: str = Field(
        default="rule_based",
        description="Method used for expansion (rule_based, llm_based, hybrid).",
    )
    
    def get_all_queries(self) -> list[str]:
        """
        Get all query texts including the original.
        
        Returns:
            list[str]: List of all query strings for retrieval.
        """
        # Use the effective query (translated if available)
        queries: list[str] = [self.original_query.get_effective_query()]
        queries.extend([eq.variant_text for eq in self.expanded_queries])
        return queries
    
    def get_unique_queries(self) -> list[str]:
        """
        Get unique query texts to avoid duplicate retrieval.
        
        Returns:
            list[str]: Deduplicated list of query strings.
        """
        all_queries = self.get_all_queries()
        seen: set[str] = set()
        unique: list[str] = []
        
        for query in all_queries:
            normalized = query.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(query)
        
        return unique
    
    def get_weighted_queries(self) -> list[tuple[str, float]]:
        """
        Get all queries with their weights for multi-query retrieval.
        
        Original query gets weight 1.0 (max).
        Variants get their configured weights.
        
        Returns:
            list[tuple[str, float]]: List of (query_text, weight) tuples.
        """
        result: list[tuple[str, float]] = [
            (self.original_query.get_effective_query(), 1.0)
        ]
        
        for variant in self.expanded_queries:
            result.append((variant.variant_text, variant.weight))
        
        return result
    
    def get_weighted_queries_with_topk(
        self,
        base_dense_top_k: int = 80,
        base_bm25_top_k: int = 50,
    ) -> list[tuple[str, float, int, int]]:
        """
        Get all queries with weights and adaptive Top-K values.
        
        Original query gets full Top-K.
        Variants get reduced Top-K based on their weight.
        
        Args:
            base_dense_top_k: Base Dense Top-K for original query.
            base_bm25_top_k: Base BM25 Top-K for original query.
        
        Returns:
            list[tuple[str, float, int, int]]: List of (query, weight, dense_top_k, bm25_top_k)
        """
        base_dense_top_k = int(base_dense_top_k)
        base_bm25_top_k = int(base_bm25_top_k)
        
        result: list[tuple[str, float, int, int]] = [
            (self.original_query.get_effective_query(), 1.0, base_dense_top_k, base_bm25_top_k)
        ]
        
        for variant in self.expanded_queries:
            # Scale Top-K by weight (minimum 10 for dense, 10 for BM25)
            dense_top_k = max(10, int(base_dense_top_k * variant.weight))
            bm25_top_k = max(10, int(base_bm25_top_k * variant.weight))
            result.append((variant.variant_text, variant.weight, dense_top_k, bm25_top_k))
        
        return result


class RetrievalQuery(BaseModel):
    """
    A query ready for the retrieval pipeline.
    
    PHASE 14: Uses translated_text for embedding generation.
    
    Attributes:
        query: The processed query with translation.
        embedding: Query embedding vector.
        expanded_queries: Expanded query variants.
        filters: Optional metadata filters for pre-filtering.
        top_k_dense: Number of results for dense retrieval.
        top_k_sparse: Number of results for sparse retrieval.
        similarity_threshold: Minimum similarity score threshold.
    """
    
    query: Query = Field(
        ...,
        description="The processed query.",
    )
    embedding: Optional[list[float]] = Field(
        default=None,
        description="Query embedding vector.",
    )
    expanded_queries: list[ExpandedQuery] = Field(
        default_factory=list,
        description="Expanded query variants.",
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata filters for pre-filtering.",
    )
    top_k_dense: int = Field(
        default=80,
        description="Number of results for dense retrieval.",
        ge=1,
    )
    top_k_sparse: int = Field(
        default=50,
        description="Number of results for sparse retrieval.",
        ge=1,
    )
    similarity_threshold: float = Field(
        default=0.0,
        description="Minimum similarity score threshold.",
        ge=0.0,
        le=1.0,
    )
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "query": {
                    "query_id": "qry_xyz789",
                    "original_text": "ما هي نقاط التحكم الحرجة للوقاية من السالمونيلا؟",
                    "translated_text": "What are the critical control points for Salmonella prevention?",
                    "is_arabic": True,
                },
                "top_k_dense": 80,
                "top_k_sparse": 50,
            }
        }
    
    def get_all_query_texts(self) -> list[str]:
        """
        Get all query texts for multi-strategy retrieval.
        
        Returns:
            list[str]: Combined list of original and expanded query texts.
        """
        texts: list[str] = [self.query.get_effective_query()]
        texts.extend([eq.variant_text for eq in self.expanded_queries])
        return list(dict.fromkeys(texts))  # Preserve order, remove duplicates
    
    def get_all_embeddings(self) -> list[list[float]]:
        """
        Get all query embeddings for multi-vector retrieval.
        
        Returns:
            list[list[float]]: List of embedding vectors.
        """
        embeddings: list[list[float]] = []
        if self.embedding:
            embeddings.append(self.embedding)
        return embeddings
    
    def get_weighted_query_list(self) -> list[tuple[str, float]]:
        """
        Get weighted queries for multi-query retrieval.
        
        Returns:
            list[tuple[str, float]]: List of (query_text, weight) tuples.
        """
        result: list[tuple[str, float]] = [
            (self.query.get_effective_query(), 1.0)
        ]
        
        for variant in self.expanded_queries:
            result.append((variant.variant_text, variant.weight))
        
        return result