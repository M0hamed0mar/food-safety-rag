"""Unit tests for the retrieval layer.

Run with:
    pytest tests/test_retrieval.py -v
"""

from __future__ import annotations

import pytest


class TestBM25Retriever:
    """Tests for BM25Retriever tokenization + index behavior."""

    def test_tokenization_keeps_numeric_tokens(self) -> None:
        """Numeric tokens must survive tokenization (critical for food safety)."""
        from app.retrieval.bm25 import BM25Retriever

        retriever = BM25Retriever()
        tokens = retriever._tokenize("temperature 165 degrees for 4 hours")

        assert "165" in tokens, "Numeric limits must be preserved"
        assert "4" in tokens, "Numeric durations must be preserved"

    def test_tokenization_keeps_domain_acronyms(self) -> None:
        """Short domain acronyms (CCP, GMP, pH) must be kept."""
        from app.retrieval.bm25 import BM25Retriever

        retriever = BM25Retriever()
        tokens = retriever._tokenize("CCP HACCP GMP pH SOP")

        for acronym in ["ccp", "haccp", "gmp", "ph", "sop"]:
            assert acronym in tokens, f"Acronym '{acronym}' must be preserved"

    def test_stop_words_removed(self) -> None:
        """Common English stop words must be filtered out."""
        from app.retrieval.bm25 import BM25Retriever

        retriever = BM25Retriever()
        tokens = retriever._tokenize("the quick brown fox")

        assert "the" not in tokens
        assert "quick" in tokens


class TestRRFFusion:
    """Tests for ReciprocalRankFusion."""

    def test_single_list_passthrough(self) -> None:
        """Fusing a single list must preserve order."""
        from app.retrieval.fusion import ReciprocalRankFusion
        from app.schemas import Chunk, ChunkMetadata
        from app.schemas.retrieval import RetrievalCandidate

        fusion = ReciprocalRankFusion()

        def make_candidate(idx: int) -> RetrievalCandidate:
            meta = ChunkMetadata(
                document_id="d1",
                document_name="doc.pdf",
                chunk_id=f"c{idx}",
                chunk_index=idx,
                total_chunks=3,
            )
            return RetrievalCandidate(
                chunk=Chunk(content=f"content {idx}", metadata=meta),
                dense_score=1.0 - idx * 0.1,
            )

        candidates = [make_candidate(i) for i in range(3)]
        fused = fusion.fuse(candidates)

        assert len(fused) == 3
        # First candidate should remain first
        assert fused[0].chunk.metadata.chunk_id == "c0"

    def test_empty_input(self) -> None:
        """Fusing empty list must return empty list."""
        from app.retrieval.fusion import ReciprocalRankFusion

        fusion = ReciprocalRankFusion()
        result = fusion.fuse([])
        assert result == []


class TestRetrievalCandidate:
    """Tests for the RetrievalCandidate dataclass."""

    def test_get_best_score_priority(self) -> None:
        """get_best_score must prefer reranker > rrf > dense > bm25."""
        from app.schemas import Chunk, ChunkMetadata
        from app.schemas.retrieval import RetrievalCandidate

        meta = ChunkMetadata(
            document_id="d1",
            document_name="doc.pdf",
            chunk_id="c1",
            chunk_index=0,
            total_chunks=1,
        )
        chunk = Chunk(content="test", metadata=meta)

        c = RetrievalCandidate(
            chunk=chunk,
            dense_score=0.5,
            bm25_score=0.3,
            rrf_score=0.7,
            reranker_score=0.9,
        )
        assert c.get_best_score() == 0.9

    def test_get_best_score_with_none(self) -> None:
        """get_best_score must ignore None values."""
        from app.schemas import Chunk, ChunkMetadata
        from app.schemas.retrieval import RetrievalCandidate

        meta = ChunkMetadata(
            document_id="d1",
            document_name="doc.pdf",
            chunk_id="c1",
            chunk_index=0,
            total_chunks=1,
        )
        chunk = Chunk(content="test", metadata=meta)

        c = RetrievalCandidate(chunk=chunk, dense_score=0.5)
        assert c.get_best_score() == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
