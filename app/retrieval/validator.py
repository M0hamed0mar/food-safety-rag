# ============================================================================
# FILE: core/retrieval/validator.py
# ============================================================================

"""
Result Validation module (Phase 15 - Enhanced).

Validates retrieval results for quality, consistency, and correctness.
Phase 15: REMOVED Outlier Removal (was causing false negatives).
Now uses strict min_score filtering only.

Phase 15 Changes:
- COMPLETELY REMOVED Outlier Removal (_detect_outliers is gone)
- Removed enable_outlier_removal parameter
- Simplified validation: duplicates + min_score + content quality
- Better logging and statistics
- Improved semantic duplicate detection

Phase 14: Fixed max_score to allow reranker scores > 1.0.
Phase 4: Supports RetrievalCandidate.
"""

from typing import Any, Optional, List, Tuple, Dict
import statistics
import re

from app.config import settings
from app.config.constants import LogEvent
from app.monitoring import get_logger
from app.schemas import Chunk, Query
from app.schemas.retrieval import RetrievalCandidate


logger = get_logger("food_safety_rag.retrieval.validator")


class ResultValidator:
    """
    Validates retrieval results for quality and consistency.
    
    Phase 15: No Outlier Removal - only min_score and content quality.
    
    Validation steps:
    1. Remove duplicates (ID-based + content-based + semantic)
    2. Apply min_score threshold
    3. Check content quality (length, diversity, structure)
    4. Check metadata completeness
    
    Attributes:
        min_score: Minimum score threshold for inclusion.
        max_score: Maximum score threshold (None = no upper bound).
        min_content_length: Minimum content length.
        enable_duplicate_removal: Whether to remove duplicates.
        enable_semantic_dedup: Whether to use semantic deduplication.
        semantic_dedup_threshold: Threshold for semantic similarity.
        min_quality_score: Minimum content quality score.
    """

    def __init__(
        self,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        min_content_length: Optional[int] = None,
        enable_duplicate_removal: Optional[bool] = None,
        enable_semantic_dedup: bool = True,
        semantic_dedup_threshold: float = 0.95,
        min_quality_score: float = 0.1,
    ) -> None:
        """
        Initialize the result validator.
        
        Phase 15: Removed outlier_removal parameters.
        """
        self.min_score = float(min_score or 0.0)
        self.max_score = float(max_score) if max_score is not None else None
        self.min_content_length = int(min_content_length or 10)
        self.enable_duplicate_removal = enable_duplicate_removal if enable_duplicate_removal is not None else True
        
        self.enable_semantic_dedup = enable_semantic_dedup
        self.semantic_dedup_threshold = semantic_dedup_threshold
        self.min_quality_score = min_quality_score
        
        # Statistics
        self._stats: Dict[str, Any] = {
            "total_validated": 0,
            "duplicates_removed": 0,
            "semantic_duplicates_removed": 0,
            "invalid_removed": 0,
            "total_removed": 0,
        }

        logger.log_event(
            event=LogEvent.SYSTEM_STARTUP,
            message="Result validator initialized (Phase 15 - No Outlier Removal)",
            details={
                "min_score": self.min_score,
                "max_score": self.max_score,
                "min_content_length": self.min_content_length,
                "enable_duplicate_removal": self.enable_duplicate_removal,
                "enable_semantic_dedup": self.enable_semantic_dedup,
                "semantic_dedup_threshold": self.semantic_dedup_threshold,
                "min_quality_score": self.min_quality_score,
                "phase_15_outlier_removal_removed": True,
            },
        )

    # ==========================================================================
    # CONTENT QUALITY
    # ==========================================================================

    def _compute_content_quality(self, content: str) -> float:
        """
        Compute content quality score (0.0 to 1.0).
        
        Metrics:
        - Length score: longer content = higher score (capped)
        - Diversity: unique words / total words
        - Structure: punctuation, newlines, capitalization
        
        Args:
            content: The text content.
        
        Returns:
            float: Quality score between 0.0 and 1.0.
        """
        if not content:
            return 0.0

        stripped = content.strip()
        if not stripped:
            return 0.0

        # Length score (capped at 200 chars)
        length_score = min(1.0, len(stripped) / 200)

        # Word diversity
        words = re.findall(r'\b[a-zA-Zآ-ي]{2,}\b', stripped)
        if not words:
            return 0.0

        unique_words = len(set(words))
        total_words = len(words)
        diversity_score = unique_words / total_words if total_words > 0 else 0.0

        # Structure score
        structure_score = 0.0
        if re.search(r'[.!?]', stripped):
            structure_score += 0.3
        if re.search(r'\n', stripped):
            structure_score += 0.2
        if re.search(r'[0-9]', stripped):
            structure_score += 0.2
        if re.search(r'[A-Z]', stripped):
            structure_score += 0.3

        structure_score = min(1.0, structure_score)

        # Combined score
        quality = (length_score * 0.3) + (diversity_score * 0.3) + (structure_score * 0.4)
        return max(0.0, min(1.0, quality))

    # ==========================================================================
    # DUPLICATE DETECTION
    # ==========================================================================

    def _compute_semantic_similarity(
        self,
        candidate1: RetrievalCandidate,
        candidate2: RetrievalCandidate,
    ) -> float:
        """
        Compute semantic similarity between two candidates.
        
        Uses embeddings if available, otherwise content-based.
        
        Args:
            candidate1: First candidate.
            candidate2: Second candidate.
        
        Returns:
            float: Similarity score between 0.0 and 1.0.
        """
        chunk1 = candidate1.chunk
        chunk2 = candidate2.chunk
        
        if not chunk1.embedding or not chunk2.embedding:
            return self._compute_content_similarity(chunk1.content, chunk2.content)

        import numpy as np
        v1 = np.array(chunk1.embedding)
        v2 = np.array(chunk2.embedding)

        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        return max(0.0, min(1.0, dot_product / (norm1 * norm2)))

    def _compute_content_similarity(self, content1: str, content2: str) -> float:
        """
        Compute content-based similarity using Jaccard similarity.
        
        Args:
            content1: First content.
            content2: Second content.
        
        Returns:
            float: Similarity score between 0.0 and 1.0.
        """
        c1 = content1.lower().strip()
        c2 = content2.lower().strip()

        if not c1 or not c2:
            return 0.0

        if c1 == c2:
            return 1.0

        words1 = set(re.findall(r'\b[a-zA-Zآ-ي]{2,}\b', c1))
        words2 = set(re.findall(r'\b[a-zA-Zآ-ي]{2,}\b', c2))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _remove_duplicates(self, candidates: List[RetrievalCandidate]) -> List[RetrievalCandidate]:
        """
        Remove duplicate candidates.
        
        Uses three strategies:
        1. Exact chunk_id match
        2. Exact content match
        3. Semantic similarity (if embeddings available)
        
        Args:
            candidates: List of candidates.
        
        Returns:
            List[RetrievalCandidate]: Deduplicated candidates.
        """
        seen_ids: set[str] = set()
        seen_contents: set[str] = set()
        seen_candidates: List[RetrievalCandidate] = []
        deduplicated: List[RetrievalCandidate] = []
        removed_by_id = 0
        removed_by_content = 0
        removed_by_semantic = 0

        for candidate in candidates:
            chunk_id = candidate.chunk.metadata.chunk_id if candidate.chunk.metadata else None

            # ID-based deduplication
            if chunk_id in seen_ids:
                removed_by_id += 1
                continue

            # Content-based deduplication
            content_key = candidate.chunk.content[:200].lower().strip() if candidate.chunk.content else ""
            if content_key in seen_contents:
                removed_by_content += 1
                continue

            # Semantic deduplication
            if self.enable_semantic_dedup and candidate.chunk.embedding is not None:
                is_semantic_duplicate = False
                for existing_candidate in seen_candidates:
                    similarity = self._compute_semantic_similarity(candidate, existing_candidate)
                    if similarity >= self.semantic_dedup_threshold:
                        is_semantic_duplicate = True
                        removed_by_semantic += 1
                        break

                if is_semantic_duplicate:
                    continue

            seen_ids.add(chunk_id)
            seen_contents.add(content_key)
            if candidate.chunk.embedding is not None:
                seen_candidates.append(candidate)
            deduplicated.append(candidate)

        total_removed = removed_by_id + removed_by_content + removed_by_semantic
        if total_removed > 0:
            self._stats["duplicates_removed"] += removed_by_id + removed_by_content
            self._stats["semantic_duplicates_removed"] += removed_by_semantic
            self._stats["total_removed"] += total_removed

            logger.log_event(
                event=LogEvent.WARNING,
                message=f"Phase 15: Removed {total_removed} duplicate candidates",
                details={
                    "removed_by_id": removed_by_id,
                    "removed_by_content": removed_by_content,
                    "removed_by_semantic": removed_by_semantic,
                    "total_removed": total_removed,
                    "remaining": len(deduplicated),
                    "phase_15_no_outlier": True,
                },
            )

        return deduplicated

    # ==========================================================================
    # INDIVIDUAL VALIDATION - Phase 15: No Outlier Removal
    # ==========================================================================

    def _validate_single_candidate(self, candidate: RetrievalCandidate) -> Tuple[bool, Optional[str]]:
        """
        Validate a single candidate.

        Phase 15: No outlier removal - only min_score and quality checks.
        
        Checks:
        1. Score >= min_score
        2. Score <= max_score (if set)
        3. Content exists and has minimum length
        4. Content quality >= min_quality_score
        5. Metadata is complete
        
        Args:
            candidate: The candidate to validate.
        
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, reason)
        """
        chunk = candidate.chunk
        score = candidate.get_best_score()

        # Score validation
        if score < self.min_score:
            return False, f"Score {score:.4f} below minimum {self.min_score}"
        if self.max_score is not None and score > self.max_score:
            return False, f"Score {score:.4f} above maximum {self.max_score}"

        # Content validation
        if not chunk.content or len(chunk.content.strip()) < self.min_content_length:
            return False, f"Content too short or missing (length: {len(chunk.content or '')})"

        # Content quality
        quality = self._compute_content_quality(chunk.content)
        if quality < self.min_quality_score:
            return False, f"Content quality too low: {quality:.2f} < {self.min_quality_score}"

        # Metadata validation
        if not chunk.metadata:
            return False, "Missing metadata"

        if not chunk.metadata.chunk_id:
            return False, "Missing chunk_id in metadata"

        if not chunk.metadata.document_id:
            return False, "Missing document_id in metadata"

        # Score type validation
        try:
            if score != score or score == float('inf') or score == float('-inf'):
                return False, f"Invalid score value: {score}"
        except (ValueError, TypeError):
            return False, f"Score type error: {type(score)}"

        return True, None

    # ==========================================================================
    # MAIN VALIDATION - Phase 15: No Outlier Removal
    # ==========================================================================

    def validate(self, query: Query, candidates: List[RetrievalCandidate]) -> List[RetrievalCandidate]:
        """
        Validate a list of candidates.

        Phase 15: No Outlier Removal.
        
        Validation pipeline:
        1. Remove duplicates (if enabled)
        2. Apply min_score threshold
        3. Check content quality
        4. Check metadata completeness
        
        Args:
            query: The query (for logging).
            candidates: List of candidates to validate.
        
        Returns:
            List[RetrievalCandidate]: Validated candidates.
        """
        if not candidates:
            logger.log_event(
                event=LogEvent.WARNING,
                message="Phase 15: Validation: No candidates to validate",
            )
            return []

        original_count = len(candidates)
        self._stats["total_validated"] += 1

        logger.log_retrieval(
            event=LogEvent.DENSE_RETRIEVAL_COMPLETE,
            query_id=query.query_id,
            message=f"Phase 15: Validating {len(candidates)} candidates (no outlier removal)",
            details={
                "input_count": original_count,
                "phase_15_no_outlier": True,
                "min_score": self.min_score,
                "min_quality_score": self.min_quality_score,
            },
        )

        # Step 1: Remove duplicates
        working_candidates = candidates
        if self.enable_duplicate_removal:
            working_candidates = self._remove_duplicates(working_candidates)

        # Step 2: Apply score and quality filters (NO Outlier Removal)
        validated_candidates: List[RetrievalCandidate] = []
        invalid_count = 0
        invalid_reasons: List[str] = []

        for candidate in working_candidates:
            is_valid, reason = self._validate_single_candidate(candidate)

            if is_valid:
                validated_candidates.append(candidate)
            else:
                invalid_count += 1
                self._stats["invalid_removed"] += 1
                self._stats["total_removed"] += 1
                if invalid_count <= 5:
                    invalid_reasons.append(reason or "Unknown reason")

        # Step 3: Sort by score
        validated_candidates.sort(key=lambda x: x.get_best_score(), reverse=True)

        final_count = len(validated_candidates)

        logger.log_retrieval(
            event=LogEvent.DENSE_RETRIEVAL_COMPLETE,
            query_id=query.query_id,
            message=f"Phase 15: Validation complete: {final_count} valid candidates",
            details={
                "input_count": original_count,
                "after_dedup": len(working_candidates),
                "invalid_removed": invalid_count,
                "final_count": final_count,
                "semantic_dedup_enabled": self.enable_semantic_dedup,
                "total_removed": original_count - final_count,
                "phase_15_no_outlier": True,
                "min_score": self.min_score,
                "min_quality_score": self.min_quality_score,
                "sample_reasons": invalid_reasons[:3],
            },
        )

        return validated_candidates

    # ==========================================================================
    # BATCH VALIDATION
    # ==========================================================================

    def validate_batch(
        self,
        queries: List[Query],
        candidates_per_query: List[List[RetrievalCandidate]],
    ) -> List[List[RetrievalCandidate]]:
        """
        Validate multiple queries' candidates.
        
        Args:
            queries: List of queries.
            candidates_per_query: List of candidate lists per query.
        
        Returns:
            List[List[RetrievalCandidate]]: Validated results per query.
        """
        validated: List[List[RetrievalCandidate]] = []

        for query, candidates in zip(queries, candidates_per_query):
            try:
                valid_candidates = self.validate(query, candidates)
                validated.append(valid_candidates)
            except Exception as exc:
                logger.log_error(
                    event=LogEvent.ERROR,
                    message=f"Batch validation failed for query {query.query_id}: {str(exc)}",
                    exception=exc,
                )
                validated.append([])

        return validated

    # ==========================================================================
    # STATISTICS AND UTILITY METHODS
    # ==========================================================================

    def get_config(self) -> Dict[str, Any]:
        """Get validator configuration."""
        return {
            "min_score": self.min_score,
            "max_score": self.max_score,
            "min_content_length": self.min_content_length,
            "enable_duplicate_removal": self.enable_duplicate_removal,
            "enable_semantic_dedup": self.enable_semantic_dedup,
            "semantic_dedup_threshold": self.semantic_dedup_threshold,
            "min_quality_score": self.min_quality_score,
            "phase": "15_no_outlier",
            "outlier_removal_enabled": False,
            "phase_15": True,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get validator statistics."""
        stats = self._stats.copy()
        stats["config"] = self.get_config()
        stats["phase_15_no_outlier"] = True
        return stats

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self._stats = {
            "total_validated": 0,
            "duplicates_removed": 0,
            "semantic_duplicates_removed": 0,
            "invalid_removed": 0,
            "total_removed": 0,
        }