"""Unit tests for the context builder.

Run with:
    pytest tests/test_context_builder.py -v
"""

from __future__ import annotations

import pytest

from app.schemas import Chunk, ChunkMetadata


def _make_chunk(idx: int, content: str = None, page: int = 1) -> Chunk:
    meta = ChunkMetadata(
        document_id="d1",
        document_name="doc.pdf",
        page=page,
        chunk_id=f"c{idx}",
        chunk_index=idx,
        total_chunks=5,
    )
    return Chunk(
        content=content or f"This is the content of chunk {idx} with enough length.",
        metadata=meta,
    )


class TestContextBuilder:
    """Tests for the ContextBuilder."""

    def test_build_empty_returns_empty_string(self) -> None:
        from app.generation.context_builder import ContextBuilder

        builder = ContextBuilder()
        result = builder.build([])
        assert result == ""

    def test_build_includes_chunk_content(self) -> None:
        from app.generation.context_builder import ContextBuilder

        builder = ContextBuilder()
        chunk = _make_chunk(0, content="Unique test content marker XYZ123")

        result = builder.build([chunk])
        assert "XYZ123" in result

    def test_build_deduplicates_identical_chunks(self) -> None:
        from app.generation.context_builder import ContextBuilder

        builder = ContextBuilder()
        chunks = [
            _make_chunk(0, content="Same content repeated."),
            _make_chunk(1, content="Same content repeated."),
        ]

        result = builder.build(chunks)
        # The deduplication should reduce this to one occurrence of the content
        assert result.count("Same content repeated.") == 1

    def test_build_respects_max_chunks(self) -> None:
        from app.generation.context_builder import ContextBuilder

        builder = ContextBuilder(max_chunks=2)
        chunks = [_make_chunk(i, content=f"Unique content {i} long enough") for i in range(5)]

        result = builder.build(chunks)
        # At most 2 unique markers should appear
        markers_found = sum(1 for i in range(5) if f"Unique content {i}" in result)
        assert markers_found <= 2


class TestContextCompressor:
    """Tests for the ContextCompressor."""

    def test_compress_empty(self) -> None:
        from app.generation.compressor import ContextCompressor

        compressor = ContextCompressor()
        result = compressor.compress([])
        assert result == []

    def test_compress_preserves_unique_content(self) -> None:
        from app.generation.compressor import ContextCompressor

        compressor = ContextCompressor()
        chunk = _make_chunk(0, content="Unique marker ABC789 with enough content to survive.")
        result = compressor.compress([chunk])

        assert len(result) >= 1
        assert "ABC789" in result[0].content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
