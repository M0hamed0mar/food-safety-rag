"""
Retrieval schema definitions.

This module defines Pydantic models for retrieval results and candidates.
PHASE 4: Immutable retrieval results to prevent side effects.

Key Features:
- RetrievalCandidate: Immutable container for chunk + all scores
- Prevents direct modification of Chunk objects
- All scores are stored separately for traceability
- Easy conversion to (Chunk, float) tuples for compatibility
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any
from app.schemas.chunk import Chunk


@dataclass
class RetrievalCandidate:
    """
    Immutable retrieval candidate with all scores.
    
    PHASE 4: This replaces direct modification of Chunk objects.
    All scores are stored here, not on the Chunk itself.
    
    Attributes:
        chunk: The original chunk (read-only, never modified).
        dense_score: Score from dense retrieval (FAISS similarity).
        bm25_score: Score from BM25 lexical retrieval.
        rrf_score: Score from Reciprocal Rank Fusion.
        reranker_score: Score from cross-encoder reranker.
        fusion_score: Combined score from fusion stage.
        metadata: Additional metadata for tracing.
        rank: Rank position after each stage (optional).
    """
    
    chunk: Chunk
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    reranker_score: Optional[float] = None
    fusion_score: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    rank: Optional[int] = None
    
    def get_score(self, score_type: str) -> Optional[float]:
        """
        Get score by type name.
        
        Args:
            score_type: Type of score ('dense', 'bm25', 'rrf', 'reranker', 'fusion').
        
        Returns:
            Optional[float]: The score value, or None if not set.
        """
        score_map = {
            'dense': self.dense_score,
            'bm25': self.bm25_score,
            'rrf': self.rrf_score,
            'reranker': self.reranker_score,
            'fusion': self.fusion_score,
        }
        return score_map.get(score_type)
    
    def get_best_score(self) -> float:
        """
        Get the highest available score.
        
        Priority order:
        1. reranker_score (most accurate)
        2. rrf_score
        3. fusion_score
        4. dense_score
        5. bm25_score
        
        Returns:
            float: The highest available score, or 0.0 if none.
        """
        scores = [
            self.reranker_score,
            self.rrf_score,
            self.fusion_score,
            self.dense_score,
            self.bm25_score,
        ]
        valid_scores = [s for s in scores if s is not None]
        return max(valid_scores) if valid_scores else 0.0
    
    def to_tuple(self) -> Tuple[Chunk, float]:
        """
        Convert to (chunk, score) tuple for compatibility.
        
        Returns:
            Tuple[Chunk, float]: (chunk, best_score) tuple.
        """
        return (self.chunk, self.get_best_score())
    
    def get_all_scores(self) -> dict[str, Optional[float]]:
        """
        Get all scores as a dictionary.
        
        Returns:
            dict[str, Optional[float]]: All scores by type.
        """
        return {
            'dense': self.dense_score,
            'bm25': self.bm25_score,
            'rrf': self.rrf_score,
            'reranker': self.reranker_score,
            'fusion': self.fusion_score,
        }
    
    def get_score_summary(self) -> str:
        """
        Get a human-readable summary of scores.
        
        Returns:
            str: Summary string.
        """
        parts = []
        for name, score in self.get_all_scores().items():
            if score is not None:
                parts.append(f"{name}={score:.4f}")
        return " | ".join(parts) if parts else "no scores"
    
    def with_score(self, score_type: str, value: float) -> "RetrievalCandidate":
        """
        Create a new candidate with an additional score.
        
        PHASE 4: Returns a new instance (immutable).
        
        Args:
            score_type: Type of score ('dense', 'bm25', 'rrf', 'reranker', 'fusion').
            value: The score value.
        
        Returns:
            RetrievalCandidate: New candidate with the added score.
        """
        kwargs = {
            'chunk': self.chunk,
            'dense_score': self.dense_score,
            'bm25_score': self.bm25_score,
            'rrf_score': self.rrf_score,
            'reranker_score': self.reranker_score,
            'fusion_score': self.fusion_score,
            'metadata': self.metadata.copy(),
            'rank': self.rank,
        }
        
        # Set the new score
        if score_type == 'dense':
            kwargs['dense_score'] = value
        elif score_type == 'bm25':
            kwargs['bm25_score'] = value
        elif score_type == 'rrf':
            kwargs['rrf_score'] = value
        elif score_type == 'reranker':
            kwargs['reranker_score'] = value
        elif score_type == 'fusion':
            kwargs['fusion_score'] = value
        else:
            raise ValueError(f"Unknown score type: {score_type}")
        
        return RetrievalCandidate(**kwargs)
    
    def with_rank(self, rank: int) -> "RetrievalCandidate":
        """
        Create a new candidate with rank set.
        
        Args:
            rank: The rank position.
        
        Returns:
            RetrievalCandidate: New candidate with rank set.
        """
        return RetrievalCandidate(
            chunk=self.chunk,
            dense_score=self.dense_score,
            bm25_score=self.bm25_score,
            rrf_score=self.rrf_score,
            reranker_score=self.reranker_score,
            fusion_score=self.fusion_score,
            metadata=self.metadata.copy(),
            rank=rank,
        )
    
    def with_metadata(self, key: str, value: Any) -> "RetrievalCandidate":
        """
        Create a new candidate with additional metadata.
        
        Args:
            key: Metadata key.
            value: Metadata value.
        
        Returns:
            RetrievalCandidate: New candidate with metadata added.
        """
        new_metadata = self.metadata.copy()
        new_metadata[key] = value
        return RetrievalCandidate(
            chunk=self.chunk,
            dense_score=self.dense_score,
            bm25_score=self.bm25_score,
            rrf_score=self.rrf_score,
            reranker_score=self.reranker_score,
            fusion_score=self.fusion_score,
            metadata=new_metadata,
            rank=self.rank,
        )


@dataclass
class RetrievalResult:
    """
    Complete retrieval result with all candidates and metadata.
    
    Attributes:
        query_id: The query ID.
        candidates: List of retrieval candidates.
        dense_candidates: Raw dense retrieval results (optional).
        bm25_candidates: Raw BM25 results (optional).
        fusion_candidates: Raw fusion results (optional).
        reranker_candidates: Raw reranker results (optional).
        metadata: Additional metadata for tracing.
    """
    
    query_id: str
    candidates: List[RetrievalCandidate] = field(default_factory=list)
    dense_candidates: Optional[List[RetrievalCandidate]] = None
    bm25_candidates: Optional[List[RetrievalCandidate]] = None
    fusion_candidates: Optional[List[RetrievalCandidate]] = None
    reranker_candidates: Optional[List[RetrievalCandidate]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def get_top_k(self, k: int) -> List[RetrievalCandidate]:
        """
        Get top K candidates.
        
        Args:
            k: Number of candidates to return.
        
        Returns:
            List[RetrievalCandidate]: Top K candidates.
        """
        return self.candidates[:k]
    
    def to_tuples(self) -> List[Tuple[Chunk, float]]:
        """
        Convert all candidates to (chunk, score) tuples.
        
        Returns:
            List[Tuple[Chunk, float]]: List of (chunk, score) tuples.
        """
        return [c.to_tuple() for c in self.candidates]
    
    def get_chunks(self) -> List[Chunk]:
        """
        Get all chunks from candidates.
        
        Returns:
            List[Chunk]: List of chunks.
        """
        return [c.chunk for c in self.candidates]
    
    def get_scores(self, score_type: str = 'best') -> List[float]:
        """
        Get scores from all candidates.
        
        Args:
            score_type: Type of score ('best', 'dense', 'bm25', 'rrf', 'reranker', 'fusion').
        
        Returns:
            List[float]: List of scores.
        """
        if score_type == 'best':
            return [c.get_best_score() for c in self.candidates]
        else:
            return [c.get_score(score_type) or 0.0 for c in self.candidates]
    
    def get_stage_results(self) -> dict[str, Any]:
        """
        Get results from all stages for tracing.
        
        Returns:
            dict[str, Any]: Stage results.
        """
        return {
            'dense': [c.to_tuple() for c in self.dense_candidates] if self.dense_candidates else [],
            'bm25': [c.to_tuple() for c in self.bm25_candidates] if self.bm25_candidates else [],
            'fusion': [c.to_tuple() for c in self.fusion_candidates] if self.fusion_candidates else [],
            'reranker': [c.to_tuple() for c in self.reranker_candidates] if self.reranker_candidates else [],
            'final': [c.to_tuple() for c in self.candidates],
        }