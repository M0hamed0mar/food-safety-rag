# ============================================================================
# FILE: core/retrieval/expander.py
# ============================================================================
"""
Query expansion module (Simplified).

This module expands user queries to improve retrieval recall by generating
alternative phrasings, synonym variants, and domain-specific terminology.

PHASE 14: Simplified to work with English-only queries.
- Removed all Arabic-specific logic
- Removed Arabic keyword expansion
- Removed language detection
- Removed MAX_ARABIC_VARIANTS
- Pure English expansion only

The original query is always preserved and used for answer generation.
"""

import re
from typing import Any, Optional, List, Set, Dict, Tuple

from app.config import settings
from app.config.constants import LogEvent
from app.core.exceptions import RetrievalError
from app.monitoring import get_logger, measure_latency
from app.schemas import ExpandedQuery, Query, QueryExpansionResult


logger = get_logger("food_safety_rag.retrieval.expander")


class QueryExpander:
    """
    Query expander for improving retrieval recall.

    PHASE 14: Pure English expansion only.
    - Generates multiple variants of the query
    - Uses synonyms, abbreviations, and rewrites
    - All variants are in English

    Attributes:
        num_variants: Number of expanded variants to generate.
        use_llm: Whether to use LLM for expansion.
        min_query_length: Minimum query length for expansion.
        variant_weights: Dictionary of weights by expansion type.
    """

    # ==========================================================================
    # CONFIGURATION CONSTANTS
    # ==========================================================================

    MIN_QUERY_LENGTH: int = 2
    MAX_VARIANTS: int = 3

    # Variant weights by type (higher = more important)
    VARIANT_WEIGHTS: Dict[str, float] = {
        'original': 1.0,
        'abbreviation': 0.9,
        'synonym': 0.85,
        'keyword_focused': 0.8,
        'rewrite': 0.75,
        'domain_specific': 0.85,
    }

    # ==========================================================================
    # DOMAIN DATA
    # ==========================================================================

    # Food Safety domain abbreviations and their expansions
    # -----------------------------------------------------------------------------
    # Domain abbreviations (EMPTY BY DEFAULT).
    #
    # This is intentionally empty so the expander is DOMAIN-AGNOSTIC.
    # To add domain-specific abbreviations, populate this dict, e.g.:
    #
    #   DOMAIN_ABBREVIATIONS = {
    #       "api": ["application programming interface"],
    #       "rag": ["retrieval augmented generation"],
    #   }
    # -----------------------------------------------------------------------------
    DOMAIN_ABBREVIATIONS: dict[str, list[str]] = {}


    # English domain synonyms
    # -----------------------------------------------------------------------------
    # Domain synonyms (EMPTY BY DEFAULT).
    #
    # This is intentionally empty so the expander is DOMAIN-AGNOSTIC.
    # To add domain-specific synonyms, populate this dict, e.g.:
    #
    #   DOMAIN_SYNONYMS = {
    #       "bug": ["defect", "error", "issue"],
    #       "fast": ["quick", "rapid", "speedy"],
    #   }
    # -----------------------------------------------------------------------------
    DOMAIN_SYNONYMS: dict[str, list[str]] = {}


    def __init__(
        self,
        num_variants: Optional[int] = None,
        use_llm: Optional[bool] = None,
        min_query_length: Optional[int] = None,
    ) -> None:
        """
        Initialize the query expander.

        PHASE 14: Simplified - no Arabic-specific parameters.

        Args:
            num_variants: Number of variants to generate. Defaults to 3.
            use_llm: Whether to use LLM for expansion. Defaults to False.
            min_query_length: Minimum query length for expansion. Defaults to 2.
        """
        self.num_variants: int = num_variants or settings.QUERY_EXPANSION_NUM_VARIANTS
        self.use_llm: bool = use_llm or False
        self.min_query_length: int = min_query_length or self.MIN_QUERY_LENGTH

        logger.log_event(
            event=LogEvent.SYSTEM_STARTUP,
            message="Query expander initialized (PHASE 14 - English only)",
            details={
                "num_variants": self.num_variants,
                "use_llm": self.use_llm,
                "min_query_length": self.min_query_length,
                "phase": "14_simplified",
            },
        )

    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================

    def _get_variant_weight(self, expansion_type: str) -> float:
        """
        Get weight for a variant type.

        Args:
            expansion_type: Type of expansion.

        Returns:
            float: Weight for this type.
        """
        return self.VARIANT_WEIGHTS.get(expansion_type, 0.5)

    # ==========================================================================
    # EXPANSION STRATEGIES
    # ==========================================================================

    def _expand_abbreviations(self, query_text: str) -> List[str]:
        """
        Expand domain abbreviations in the query.

        Args:
            query_text: Query text.

        Returns:
            list[str]: Expanded variants.
        """
        variants = []
        expanded_text = query_text

        for abbrev, expansions in self.DOMAIN_ABBREVIATIONS.items():
            pattern = rf'\b{abbrev}\b'
            if re.search(pattern, query_text, re.IGNORECASE):
                for expansion in expansions:
                    new_text = re.sub(pattern, expansion, expanded_text, flags=re.IGNORECASE)
                    if new_text not in variants and new_text != query_text:
                        variants.append(new_text)

        return variants

    def _expand_synonyms(self, query_text: str) -> List[str]:
        """
        Expand domain synonyms in the query.

        Args:
            query_text: Query text.

        Returns:
            list[str]: Expanded variants.
        """
        variants = []

        for keyword, synonyms in self.DOMAIN_SYNONYMS.items():
            pattern = rf'\b{keyword}\b'
            if re.search(pattern, query_text, re.IGNORECASE):
                for synonym in synonyms:
                    new_text = re.sub(pattern, synonym, query_text, flags=re.IGNORECASE)
                    if new_text not in variants and new_text != query_text:
                        variants.append(new_text)

        return variants

    def _keyword_focused_query(self, query_text: str) -> str:
        """
        Create a keyword-focused variant by extracting main keywords.

        Args:
            query_text: Query text.

        Returns:
            str: Keyword-focused variant.
        """
        words = query_text.split()
        keywords = []

        for w in words:
            # Keep words with length > 3 (English)
            if len(w) > 3:
                keywords.append(w)

        if len(keywords) > 0:
            return " ".join(keywords)
        return query_text

    def _domain_specific_expansion(self, query_text: str) -> List[str]:
        """
        Generate domain-specific expansions for food safety terms.

        Args:
            query_text: Query text.

        Returns:
            list[str]: Domain-specific expansions.
        """
        variants = []
        query_lower = query_text.lower()

        domain_patterns = {
            'temperature_related': ['temperature', 'heat', 'cold', 'thermal'],
            'hazard_related': ['hazard', 'risk', 'danger'],
            'control_related': ['control', 'measure', 'prevent'],
            'quality_related': ['quality', 'standard', 'specification'],
        }

        for domain, keywords in domain_patterns.items():
            if any(kw in query_lower for kw in keywords):
                domain_terms = {
                    'temperature_related': ['thermal treatment'],
                    'hazard_related': ['hazard analysis'],
                    'control_related': ['control measures'],
                    'quality_related': ['quality control'],
                }
                for term in domain_terms.get(domain, [])[:1]:
                    new_text = f"{query_text} {term}"
                    if new_text not in variants:
                        variants.append(new_text)

        return variants

    def _rewrite_query(self, query_text: str) -> str:
        """
        Rewrite query for better retrieval.

        Args:
            query_text: Query text.

        Returns:
            str: Rewritten query.
        """
        rewritten = re.sub(r'[?!]+', '', query_text)
        rewritten = re.sub(r'\bwhat\b|\bwhere\b|\bwhen\b|\bhow\b', '', rewritten, flags=re.IGNORECASE)
        rewritten = rewritten.strip()

        return rewritten if rewritten else query_text

    # ==========================================================================
    # MAIN EXPANSION METHOD
    # ==========================================================================

    def expand(self, query: Query) -> QueryExpansionResult:
        """
        Expand a query into multiple variants for improved retrieval.

        PHASE 14: Works with English-only queries.
        Uses the effective query (translated to English if available).

        Args:
            query: The user query.

        Returns:
            QueryExpansionResult: Expansion result with variants.

        Raises:
            RetrievalError: If expansion fails.
        """
        with measure_latency("query_expansion") as latency:
            # PHASE 14: Use effective query (translated if available)
            query_text = query.get_effective_query()

            logger.log_retrieval(
                event=LogEvent.QUERY_EXPANSION,
                query_id=query.query_id,
                message=f"Starting query expansion: {query_text[:100]}...",
                details={
                    "query_text": query_text,
                    "has_translation": query.has_translation(),
                    "phase": "14_simplified",
                },
            )

            try:
                # Check if query is long enough
                if len(query_text.split()) < self.min_query_length:
                    logger.log_event(
                        event=LogEvent.QUERY_EXPANSION,
                        message=f"Query too short (< {self.min_query_length} words), skipping expansion",
                    )
                    return QueryExpansionResult(
                        original_query=query,
                        expanded_queries=[
                            ExpandedQuery(
                                original_query=query,
                                variant_text=query_text,
                                expansion_type="original",
                                confidence=1.0,
                                weight=1.0,
                            )
                        ],
                        total_variants=1,
                        expansion_method="none",
                    )

                variants: List[ExpandedQuery] = []
                seen: Set[str] = set()

                # ============================================================
                # ORIGINAL QUERY (Always include first)
                # ============================================================
                variants.append(ExpandedQuery(
                    original_query=query,
                    variant_text=query_text,
                    expansion_type="original",
                    confidence=1.0,
                    weight=self._get_variant_weight("original"),
                ))
                seen.add(query_text.lower().strip())

                # ============================================================
                # ENGLISH QUERY EXPANSION - Full expansion
                # ============================================================

                # 1. Abbreviation expansion
                abbrev_variants = self._expand_abbreviations(query_text)
                for variant_text in abbrev_variants[:2]:
                    normalized = variant_text.lower().strip()
                    if normalized not in seen and len(variants) < self.num_variants:
                        seen.add(normalized)
                        variants.append(ExpandedQuery(
                            original_query=query,
                            variant_text=variant_text,
                            expansion_type="abbreviation",
                            confidence=0.9,
                            weight=self._get_variant_weight("abbreviation"),
                        ))

                # 2. Synonym expansion
                if len(variants) < self.num_variants:
                    synonym_variants = self._expand_synonyms(query_text)
                    for variant_text in synonym_variants[:2]:
                        normalized = variant_text.lower().strip()
                        if normalized not in seen:
                            seen.add(normalized)
                            variants.append(ExpandedQuery(
                                original_query=query,
                                variant_text=variant_text,
                                expansion_type="synonym",
                                confidence=0.85,
                                weight=self._get_variant_weight("synonym"),
                            ))

                # 3. Keyword-focused
                if len(variants) < self.num_variants:
                    keyword_variant = self._keyword_focused_query(query_text)
                    normalized = keyword_variant.lower().strip()
                    if normalized not in seen and normalized != query_text.lower().strip():
                        seen.add(normalized)
                        variants.append(ExpandedQuery(
                            original_query=query,
                            variant_text=keyword_variant,
                            expansion_type="keyword_focused",
                            confidence=0.8,
                            weight=self._get_variant_weight("keyword_focused"),
                        ))

                # 4. Rewritten
                if len(variants) < self.num_variants:
                    rewritten_variant = self._rewrite_query(query_text)
                    normalized = rewritten_variant.lower().strip()
                    if normalized not in seen and normalized != query_text.lower().strip():
                        seen.add(normalized)
                        variants.append(ExpandedQuery(
                            original_query=query,
                            variant_text=rewritten_variant,
                            expansion_type="rewrite",
                            confidence=0.75,
                            weight=self._get_variant_weight("rewrite"),
                        ))

                # 5. Domain-specific expansion
                if len(variants) < self.num_variants:
                    domain_variants = self._domain_specific_expansion(query_text)
                    for variant_text in domain_variants[:1]:
                        normalized = variant_text.lower().strip()
                        if normalized not in seen:
                            seen.add(normalized)
                            variants.append(ExpandedQuery(
                                original_query=query,
                                variant_text=variant_text,
                                expansion_type="domain_specific",
                                confidence=0.85,
                                weight=self._get_variant_weight("domain_specific"),
                            ))

                # Ensure original is always first
                variants.sort(key=lambda v: 0 if v.expansion_type == "original" else 1)

                latency.stop(
                    query_id=query.query_id,
                    variants_generated=len(variants),
                    phase="14_simplified",
                )

                logger.log_retrieval(
                    event=LogEvent.QUERY_EXPANSION,
                    query_id=query.query_id,
                    message=f"Query expansion complete: {len(variants)} variants (PHASE 14)",
                    details={
                        "variants_generated": len(variants),
                        "variant_types": [v.expansion_type for v in variants],
                        "weights": [v.weight for v in variants],
                        "duration_ms": latency.duration_ms,
                        "phase": "14_simplified",
                        "has_translation": query.has_translation(),
                    },
                )

                return QueryExpansionResult(
                    original_query=query,
                    expanded_queries=variants,
                    total_variants=len(variants),
                    expansion_method="phase14_simplified",
                )

            except Exception as exc:
                raise RetrievalError(
                    message=f"Query expansion failed: {str(exc)}",
                    query=query_text,
                    error_code="RET_001",
                    original_exception=exc,
                )

    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================

    def get_config(self) -> Dict[str, Any]:
        """Get expander configuration."""
        return {
            "num_variants": self.num_variants,
            "use_llm": self.use_llm,
            "min_query_length": self.min_query_length,
            "variant_weights": self.VARIANT_WEIGHTS,
            "phase": "14_simplified",
            "arabic_support": False,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get expander statistics."""
        return {
            "config": self.get_config(),
            "domain_abbreviations": len(self.DOMAIN_ABBREVIATIONS),
            "domain_synonyms": len(self.DOMAIN_SYNONYMS),
            "phase": "14_simplified",
        }