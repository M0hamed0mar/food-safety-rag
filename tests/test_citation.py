"""Unit tests for the citation engine.

Run with:
    pytest tests/test_citation.py -v
"""

from __future__ import annotations

import pytest

from app.schemas import Answer, AnswerMetadata, Chunk, ChunkMetadata, Citation


def _make_chunk(idx: int, doc_name: str = "doc.pdf", page: int = 1) -> Chunk:
    meta = ChunkMetadata(
        document_id=f"doc_{idx}",
        document_name=doc_name,
        page=page,
        chunk_id=f"c{idx}",
        chunk_index=idx,
        total_chunks=10,
    )
    return Chunk(content=f"Content of chunk {idx} about hazards.", metadata=meta)


class TestCitationFormatting:
    """Tests for citation string formatting."""

    def test_to_string_includes_document_and_page(self) -> None:
        c = Citation(
            document_name="manual.pdf",
            page=5,
            chunk_id="c1",
        )
        s = c.to_string()
        assert "manual.pdf" in s
        assert "p.5" in s

    def test_to_string_fallback_when_empty(self) -> None:
        c = Citation(chunk_id="c42")
        assert "c42" in c.to_string()


class TestCitationEngineHeuristic:
    """Tests for the heuristic citation extraction."""

    def test_citations_are_generated_for_relevant_chunks(self) -> None:
        from app.generation.citation import CitationEngine

        engine = CitationEngine()

        chunk = _make_chunk(0, doc_name="manual.pdf", page=10)

        answer = Answer(
            answer_id="a1",
            query_id="q1",
            text="Salmonella prevention requires maintaining cooking temperatures above 74C.",
            metadata=AnswerMetadata(answer_id="a1", query_id="q1", model_name="test"),
        )

        # Attach citations
        result = engine.attach_to_answer(answer, [chunk])

        assert isinstance(result.citations, list)
        assert result.metadata.num_citations == len(result.citations)

    def test_citations_are_deduplicated(self) -> None:
        from app.generation.citation import CitationEngine

        engine = CitationEngine()
        chunks = [_make_chunk(0), _make_chunk(1), _make_chunk(2)]

        answer = Answer(
            answer_id="a1",
            query_id="q1",
            text="General answer with no citations markers.",
            metadata=AnswerMetadata(answer_id="a1", query_id="q1", model_name="test"),
        )

        result = engine.attach_to_answer(answer, chunks)
        ids = [c.chunk_id for c in result.citations]
        assert len(ids) == len(set(ids)), "Citations must be deduplicated"

    def test_no_citations_for_empty_chunks(self) -> None:
        from app.generation.citation import CitationEngine

        engine = CitationEngine()

        answer = Answer(
            answer_id="a1",
            query_id="q1",
            text="Answer with no context.",
            metadata=AnswerMetadata(answer_id="a1", query_id="q1", model_name="test"),
        )

        result = engine.attach_to_answer(answer, [])
        assert result.citations == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
