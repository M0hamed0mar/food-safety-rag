"""
Citation engine module.

This module generates deterministic citations for answers based on
retrieved chunk metadata. Every citation is traceable to its source
document and never fabricates information.

Phase 15.8: Enhanced citation display with page numbers.
- Preserves page numbers from chunk metadata
- Supports multiple citations with deduplication
- Clean formatting for UI display
"""

import re
from typing import Any, Optional

from app.config import settings
from app.config.constants import LogEvent
from app.monitoring import get_logger
from app.schemas import Answer, Chunk, Citation


logger = get_logger("food_safety_rag.generation.citation")


class CitationEngine:
    """
    Citation engine that creates traceable references to source documents.

    Every answer citation includes:
    - Document name
    - Page number (if available)
    - Section (if available)
    - Chunk identifier (internal)

    Supports two strategies:
    1. Heuristic-based (default): Uses word overlap to identify cited chunks.
    2. LLM-based (optional): Uses the LLM to generate citations via structured output.

    Attributes:
        citation_format: Format string for inline citations.
        include_quotes: Whether to include quoted text in citations.
        use_llm: Whether to use LLM-based citation generation.
    """

    def __init__(
        self,
        citation_format: Optional[str] = None,
        include_quotes: bool = True,
        use_llm: Optional[bool] = None,
    ) -> None:
        """
        Initialize the citation engine.

        Args:
            citation_format: Custom citation format. Uses default if None.
            include_quotes: Whether to include source quotes. Defaults to True.
            use_llm: Whether to use LLM-based citation generation.
                Defaults to settings.CITATION_USE_LLM or False.
        """
        self.citation_format: str = citation_format or "[{doc_name}, p.{page}]"
        self.include_quotes: bool = include_quotes
        self.use_llm: bool = use_llm if use_llm is not None else getattr(settings, "CITATION_USE_LLM", False)

    def _create_citation(self, chunk: Chunk, quoted_text: Optional[str] = None) -> Citation:
        """
        Create a citation from a chunk's metadata.

        Phase 15.8: Preserves page numbers from metadata.

        Args:
            chunk: Source chunk.
            quoted_text: Optional quoted text supporting the claim.

        Returns:
            Citation: Structured citation object.
        """
        return Citation(
            document_name=chunk.metadata.document_name,
            document_id=chunk.metadata.document_id,
            page=chunk.metadata.page,
            chapter=chunk.metadata.chapter,
            section=chunk.metadata.section,
            subsection=chunk.metadata.subsection,
            chunk_id=chunk.metadata.chunk_id,
            chunk_index=chunk.metadata.chunk_index,
            relevance_score=chunk.reranker_score,
            quoted_text=quoted_text,
            confidence=1.0,
        )

    def _format_inline_citation(self, citation: Citation) -> str:
        """
        Format a citation for inline display in the answer.

        Phase 15.8: Includes document name and page number.

        Args:
            citation: Citation to format.

        Returns:
            str: Formatted citation string.
        """
        parts: list[str] = []

        if citation.document_name:
            parts.append(citation.document_name)
        if citation.page:
            parts.append(f"p.{citation.page}")
        if citation.section:
            parts.append(citation.section)

        if not parts:
            return f"[{citation.chunk_id}]"

        return f"[{', '.join(parts)}]"

    def _extract_cited_chunks_heuristic(self, answer_text: str, chunks: list[Chunk]) -> list[Chunk]:
        """
        Identify which chunks are actually cited in the answer text using heuristics.

        Uses multiple signals:
        1. Direct citation markers in the text ([doc, p. page, section])
        2. Keyword overlap between answer and chunk content
        3. Reranker scores as a confidence signal

        Args:
            answer_text: Generated answer text.
            chunks: List of source chunks.

        Returns:
            list[Chunk]: Chunks that appear to be cited in the answer.
        """
        cited: list[Chunk] = []
        answer_lower = answer_text.lower()

        citation_pattern = r"\[([^\]]+)\]"
        cited_refs = re.findall(citation_pattern, answer_text)

        cited_doc_names = set()
        for ref in cited_refs:
            parts = [p.strip() for p in ref.split(",")]
            if parts:
                cited_doc_names.add(parts[0].lower())

        stop_words = {
            "the", "this", "that", "these", "those", "there", "their",
            "they", "them", "then", "than", "such", "some", "would",
            "could", "should", "will", "shall", "might", "must",
        }
        answer_words = set(
            w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", answer_text)
            if w.lower() not in stop_words
        )

        chunk_scores: dict[str, tuple[Chunk, float, bool]] = {}

        for chunk in chunks:
            chunk_text = chunk.content.lower()
            chunk_words = set(
                w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", chunk_text)
                if w.lower() not in stop_words
            )

            overlap_ratio = 0.0
            if answer_words and chunk_words:
                overlap = len(answer_words & chunk_words)
                total = len(chunk_words)
                overlap_ratio = overlap / total if total > 0 else 0

            doc_name = chunk.metadata.document_name.lower() if chunk.metadata.document_name else ""
            citation_bonus = 1.0 if doc_name in cited_doc_names else 0.0

            reranker_signal = chunk.reranker_score if chunk.reranker_score is not None else 0.5

            combined_score = (overlap_ratio * 0.5) + (citation_bonus * 0.3) + (reranker_signal * 0.2)

            sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
            phrase_matches = 0
            for sent in sentences[:3]:
                sent_clean = sent.strip()
                if len(sent_clean) > 20 and sent_clean in answer_lower:
                    phrase_matches += 1

            if phrase_matches > 0:
                combined_score = min(1.0, combined_score + 0.2 * phrase_matches)

            chunk_scores[chunk.metadata.chunk_id] = (chunk, combined_score, citation_bonus > 0)

        sorted_chunks = sorted(
            chunk_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )

        threshold = 0.15
        for chunk, score, has_citation in sorted_chunks:
            if score >= threshold or has_citation:
                cited.append(chunk)

        if not cited and chunks:
            sorted_by_reranker = sorted(chunks, key=lambda c: c.reranker_score if c.reranker_score is not None else 0, reverse=True)
            cited = sorted_by_reranker[:3]

        return cited

    def _extract_cited_chunks_llm(self, answer_text: str, chunks: list[Chunk]) -> list[Chunk]:
        """
        Identify cited chunks using LLM-based structured output.

        This method prompts the LLM to generate citations as part of the answer.
        It expects citations in the format [chunk_id] within the answer text.

        Args:
            answer_text: Generated answer text.
            chunks: List of source chunks.

        Returns:
            list[Chunk]: Chunks that appear to be cited in the answer.
        """
        cited_chunks: list[Chunk] = []
        chunk_id_map = {c.metadata.chunk_id: c for c in chunks}

        pattern = r"\[(chunk_[a-zA-Z0-9_]+)\]"
        matches = re.findall(pattern, answer_text)

        for chunk_id in matches:
            if chunk_id in chunk_id_map and chunk_id_map[chunk_id] not in cited_chunks:
                cited_chunks.append(chunk_id_map[chunk_id])

        if not cited_chunks:
            citation_pattern = r"\[([^\]]+)\]"
            refs = re.findall(citation_pattern, answer_text)

            for ref in refs:
                parts = [p.strip() for p in ref.split(",")]
                if parts:
                    doc_name = parts[0].lower()
                    for chunk in chunks:
                        if chunk.metadata.document_name and chunk.metadata.document_name.lower() == doc_name:
                            if chunk not in cited_chunks:
                                cited_chunks.append(chunk)
                                break

        if not cited_chunks and chunks:
            logger.log_event(
                event=LogEvent.WARNING,
                message="No LLM-generated citations found, falling back to heuristic",
                level=30,
            )
            return self._extract_cited_chunks_heuristic(answer_text, chunks)

        return cited_chunks

    def generate_citations(
        self,
        answer_text: str,
        chunks: list[Chunk],
        use_llm_override: Optional[bool] = None,
    ) -> list[Citation]:
        """
        Generate citations for an answer based on source chunks.

        Phase 15.8: Preserves page numbers and section info.

        Args:
            answer_text: Generated answer text.
            chunks: Source chunks used in context.
            use_llm_override: Override the use_llm setting for this call.

        Returns:
            list[Citation]: List of structured citations.
        """
        logger.log_event(
            event=LogEvent.CONTEXT_BUILDING,
            message=f"Generating citations for answer from {len(chunks)} chunks",
            details={"source_chunks": len(chunks), "use_llm": self.use_llm},
        )

        effective_use_llm = use_llm_override if use_llm_override is not None else self.use_llm

        if effective_use_llm:
            cited_chunks = self._extract_cited_chunks_llm(answer_text, chunks)
        else:
            cited_chunks = self._extract_cited_chunks_heuristic(answer_text, chunks)

        citations: list[Citation] = []
        seen_chunk_ids: set[str] = set()

        for chunk in cited_chunks:
            chunk_id = chunk.metadata.chunk_id

            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            quoted_text: Optional[str] = None
            if self.include_quotes:
                sentences = re.split(r'(?<=[.!?])\s+', chunk.content)
                if sentences:
                    for sent in sentences:
                        stripped = sent.strip()
                        if len(stripped) > 20:
                            quoted_text = stripped[:200]
                            break

            citation = self._create_citation(chunk, quoted_text)
            citations.append(citation)

        logger.log_event(
            event=LogEvent.CONTEXT_BUILDING,
            message=f"Generated {len(citations)} citations",
            details={
                "citation_count": len(citations),
                "strategy": "llm" if effective_use_llm else "heuristic",
                "has_page_numbers": any(c.page is not None for c in citations),
            },
        )

        return citations

    def format_citations_section(self, citations: list[Citation]) -> str:
        """
        Format citations as a references section for appending to answers.

        Phase 15.8: Shows document name and page number.

        Args:
            citations: List of citations.

        Returns:
            str: Formatted references section.
        """
        if not citations:
            return ""

        # Deduplicate by document name and page
        seen_sources = set()
        unique_citations = []
        for c in citations:
            source_key = f"{c.document_name}_{c.page}" if c.document_name else c.chunk_id
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                unique_citations.append(c)

        if not unique_citations:
            return ""

        lines: list[str] = ["\n\n---\n**Sources:**"]

        for i, citation in enumerate(unique_citations, start=1):
            parts = []
            if citation.document_name:
                parts.append(citation.document_name)
            if citation.page:
                parts.append(f"p.{citation.page}")
            if citation.section:
                parts.append(citation.section)

            if parts:
                lines.append(f"{i}. {' | '.join(parts)}")
            else:
                lines.append(f"{i}. {citation.chunk_id}")

        return "\n".join(lines)

    def inject_citations(self, answer_text: str, citations: list[Citation]) -> str:
        """
        Inject inline citations into answer text at appropriate positions.

        Phase 15.8: Uses enhanced citation format with page numbers.

        Args:
            answer_text: Generated answer text.
            citations: List of citations to inject.

        Returns:
            str: Answer with inline citations.
        """
        if not citations or not answer_text:
            return answer_text

        citations_section = self.format_citations_section(citations)

        return answer_text + citations_section

    def attach_to_answer(self, answer: Answer, chunks: list[Chunk]) -> Answer:
        """
        Generate and attach citations to an answer object.

        Phase 15.8: Preserves page numbers and section info.

        Args:
            answer: The generated answer.
            chunks: Source chunks.

        Returns:
            Answer: Answer with citations attached.
        """
        citations = self.generate_citations(answer.text, chunks)
        answer.citations = citations
        answer.metadata.num_citations = len(citations)

        # Ensure citations have page numbers where available
        for citation in answer.citations:
            if citation.page is None:
                # Try to find page from associated chunk
                for chunk in chunks:
                    if chunk.metadata.chunk_id == citation.chunk_id and chunk.metadata.page is not None:
                        citation.page = chunk.metadata.page
                        break

        # Inject the citations section once (avoid duplicating if already present).
        if citations and "**Sources:**" not in answer.text:
            answer.text = self.inject_citations(answer.text, citations)

        return answer