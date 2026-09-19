"""
Context builder module.

This module constructs the final context sent to the LLM by combining
retrieved chunks, removing duplicates, preserving citations, and
optimizing token usage.

Enhanced with table formatting that preserves ALL rows.
Table chunks get highest priority and are displayed in Markdown format
with structure indicators and row counts.
"""

import json
import re
from typing import Any, Optional

from app.config import settings
from app.config.constants import LogEvent
from app.monitoring import get_logger, measure_latency
from app.schemas import Chunk


logger = get_logger("food_safety_rag.generation.context_builder")


class ContextBuilder:
    """
    Context builder that constructs optimized context for LLM consumption.

    Responsibilities:
    - Remove duplicated information across chunks (content-based only)
    - Remove irrelevant or low-confidence chunks
    - Preserve ranking order from reranker
    - Preserve citation metadata
    - Reduce unnecessary token usage
    - Maintain readable structure
    - Format tables for readability with ALL rows preserved
    - Optimize ordering to reduce Lost-in-the-Middle effects
    - Prioritize table chunks

    Attributes:
        max_context_tokens: Maximum tokens for the context window.
        max_chunks: Maximum number of chunks to include.
        similarity_threshold: Minimum reranker score for inclusion.
    """

    def __init__(
        self,
        max_context_tokens: Optional[int] = None,
        max_chunks: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> None:
        """
        Initialize the context builder.

        Args:
            max_context_tokens: Maximum context tokens. Defaults to settings.
            max_chunks: Maximum chunks to include. Defaults to settings.
            similarity_threshold: Minimum score threshold. Defaults to settings.
        """
        self.max_context_tokens: int = max_context_tokens or 8192
        self.max_chunks: int = max_chunks or settings.RERANKER_TOP_K
        self.similarity_threshold: float = similarity_threshold or settings.SIMILARITY_THRESHOLD

    def _remove_duplicates(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Remove duplicate or near-duplicate chunks.
        
        Uses exact match + Jaccard similarity > 0.85 for near-duplicates.

        Args:
            chunks: List of chunks.

        Returns:
            list[Chunk]: Deduplicated chunks.
        """
        seen_contents: list[str] = []
        deduplicated: list[Chunk] = []

        for chunk in chunks:
            normalized = " ".join(chunk.content.lower().strip().split())

            if not normalized:
                continue

            is_duplicate = False
            
            # Check exact duplicate
            for seen in seen_contents:
                if normalized == seen:
                    is_duplicate = True
                    break

            # Check near-duplicate (Jaccard similarity > 0.85)
            if not is_duplicate:
                for seen in seen_contents:
                    if len(normalized) > 20 and len(seen) > 20:
                        words_norm = set(normalized.split())
                        words_seen = set(seen.split())
                        if words_norm and words_seen:
                            overlap = len(words_norm & words_seen) / max(len(words_norm), len(words_seen))
                            if overlap > 0.85:
                                is_duplicate = True
                                break

            # Check substring containment
            if not is_duplicate:
                for seen in seen_contents:
                    if len(normalized) > 30 and len(seen) > 30:
                        if normalized in seen or seen in normalized:
                            is_duplicate = True
                            break

            if not is_duplicate:
                seen_contents.append(normalized)
                deduplicated.append(chunk)

        if len(deduplicated) < len(chunks):
            logger.log_event(
                event=LogEvent.CONTEXT_BUILDING,
                message=f"Removed {len(chunks) - len(deduplicated)} duplicate chunks",
                details={
                    "original_count": len(chunks),
                    "deduplicated_count": len(deduplicated),
                    "method": "content_based",
                },
            )

        return deduplicated

    def _remove_low_confidence(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Remove chunks with low reranker confidence scores.

        Args:
            chunks: List of ranked chunks.

        Returns:
            list[Chunk]: Filtered chunks.
        """
        if self.similarity_threshold <= 0:
            return chunks

        filtered = [
            chunk for chunk in chunks
            if chunk.reranker_score is None or chunk.reranker_score >= self.similarity_threshold
        ]

        if len(filtered) < len(chunks):
            logger.log_event(
                event=LogEvent.CONTEXT_BUILDING,
                message=f"Filtered {len(chunks) - len(filtered)} low-confidence chunks",
                details={
                    "original_count": len(chunks),
                    "filtered_count": len(filtered),
                    "threshold": self.similarity_threshold,
                },
            )

        return filtered

    def _is_table_chunk(self, chunk: Chunk) -> bool:
        """
        Check if a chunk is a table chunk.

        Args:
            chunk: The chunk to check.

        Returns:
            bool: True if the chunk is a table chunk.
        """
        # Check source_type
        if chunk.metadata.source_type == "table":
            return True

        # Check is_extracted_table flag
        if hasattr(chunk.metadata, 'is_extracted_table') and chunk.metadata.is_extracted_table:
            return True

        # Check table_data
        if chunk.metadata.table_data:
            return True

        # Check content for table indicators
        content = chunk.content
        if "|" in content or "\t" in content:
            lines = content.split("\n")
            if len(lines) > 1:
                # Check if multiple lines have similar structure
                column_counts = []
                for line in lines[:5]:
                    if "|" in line:
                        cols = [c.strip() for c in line.split("|") if c.strip()]
                        column_counts.append(len(cols))
                    elif "\t" in line:
                        cols = [c.strip() for c in line.split("\t") if c.strip()]
                        column_counts.append(len(cols))
                if column_counts and len(set(column_counts)) <= 2:
                    return True

        return False

    def _get_table_type_label(self, chunk: Chunk) -> str:
        """
        Get a human-readable table type label.

        Args:
            chunk: The table chunk.

        Returns:
            str: Table type label.
        """
        if hasattr(chunk.metadata, 'table_type') and chunk.metadata.table_type:
            type_map = {
                "hazard_table": "📊 Hazard Analysis Table",
                "allergen_table": "📊 Allergen List Table",
                "process_table": "📊 Process Step Table",
                "temperature_table": "📊 Temperature Requirements Table",
                "pathogen_table": "📊 Pathogen Information Table",
                "nutrition_table": "📊 Nutritional Information Table",
                "additive_table": "📊 Additives/Preservatives Table",
                "control_table": "📊 Control Measures Table",
                "ingredient_table": "📊 Ingredients/Composition Table",
                "specification_table": "📊 Specifications/Standards Table",
                "general_table": "📊 Data Table",
            }
            return type_map.get(chunk.metadata.table_type, "📊 Data Table")
        
        return "📊 Data Table"

    def _format_table(self, chunk: Chunk) -> str:
        """
        Format a table chunk for readable display with Markdown.

        Preserves ALL rows and includes structure information.

        Args:
            chunk: The table chunk.

        Returns:
            str: Formatted table string.
        """
        # Get table type label
        table_label = self._get_table_type_label(chunk)
        
        # Check if we have structured table data
        if chunk.metadata.table_data:
            try:
                table_data = json.loads(chunk.metadata.table_data)
                headers = table_data.get("headers", [])
                rows = table_data.get("rows", [])
                semantic_description = table_data.get("semantic_description", "")
                table_type = table_data.get("table_type", "general")
                keywords = table_data.get("keywords", [])
                row_count = table_data.get("row_count", 0)
                column_count = table_data.get("column_count", 0)
                is_extracted = table_data.get("is_extracted_table", False)

                if headers and rows:
                    result_parts = []
                    
                    # Add table header with type
                    result_parts.append(f"**{table_label}**")
                    
                    # Add structure info with row count
                    structure_info = f"*{row_count} rows, {column_count} columns*"
                    result_parts.append(structure_info)
                    
                    # Add semantic description if available
                    if semantic_description:
                        result_parts.append(f"*{semantic_description}*")
                    
                    # Add keywords if available
                    if keywords:
                        result_parts.append(f"*Keywords: {', '.join(keywords[:10])}*")
                    
                    # Add table in Markdown format
                    result_parts.append("")
                    
                    # Build Markdown table with ALL rows
                    table_lines = []
                    
                    # Header row
                    header_line = "| " + " | ".join(str(h) for h in headers) + " |"
                    table_lines.append(header_line)
                    
                    # Separator
                    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
                    table_lines.append(separator)
                    
                    # All data rows (no limit - preserve all data)
                    for row in rows:
                        # Pad row to match header length
                        padded_row = row + [""] * (len(headers) - len(row))
                        row_line = "| " + " | ".join(str(cell) for cell in padded_row) + " |"
                        table_lines.append(row_line)
                    
                    result_parts.append("\n".join(table_lines))
                    
                    # Add note if this is an extracted table
                    if is_extracted:
                        result_parts.append("*Table extracted with high confidence*")
                    
                    # Add row count indicator for large tables
                    if row_count > 20:
                        result_parts.append(f"*Table contains {row_count} total rows*")
                    
                    return "\n".join(result_parts)

            except (json.JSONDecodeError, TypeError, KeyError):
                pass

        # Fallback: try to detect table from content
        content = chunk.content
        lines = content.split("\n")

        # Check if this is a pipe-separated table
        if any("|" in line for line in lines):
            result_parts = []
            result_parts.append(f"**{table_label}**")
            
            table_lines = []
            for line in lines:
                if "|" in line:
                    cells = [c.strip() for c in line.split("|") if c.strip()]
                    if cells:
                        table_lines.append("| " + " | ".join(cells) + " |")
            
            if table_lines:
                # Try to add a separator if it doesn't exist
                if len(table_lines) > 1 and not any("---" in line for line in table_lines):
                    num_cols = table_lines[0].count("|") - 1
                    separator = "| " + " | ".join(["---"] * num_cols) + " |"
                    table_lines.insert(1, separator)
                
                result_parts.append("\n".join(table_lines))
                return "\n".join(result_parts)

        # Check for multi-column patterns
        if len(lines) > 2:
            column_counts = []
            for line in lines[:5]:
                if line.strip():
                    cols = re.split(r'\s{2,}', line.strip())
                    if len(cols) > 1:
                        column_counts.append(len(cols))
            if column_counts and len(set(column_counts)) <= 2:
                result_parts = []
                result_parts.append(f"**{table_label}**")
                result_parts.append(content)
                return "\n".join(result_parts)

        # Default: return content with label
        return f"**{table_label}**\n{content}"

    def _order_chunks_for_context(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Order chunks intelligently to reduce Lost-in-the-Middle effects.

        1. Group by section (keep related chunks together)
        2. Sort by relevance within each section
        3. Table chunks get highest priority (placed at top)
        4. Heading chunks get second priority

        Args:
            chunks: List of chunks.

        Returns:
            list[Chunk]: Ordered chunks.
        """
        if len(chunks) <= 1:
            return chunks

        # Step 1: Group chunks by section
        groups: dict[str, list[Chunk]] = {}
        group_keys: list[str] = []

        for chunk in chunks:
            doc_id = chunk.metadata.document_id or "unknown"
            section = chunk.metadata.section or "none"
            group_key = f"{doc_id}_{section}"
            
            if group_key not in groups:
                groups[group_key] = []
                group_keys.append(group_key)
            groups[group_key].append(chunk)

        # Step 2: Sort within each group by chunk_index
        for group_key in groups:
            groups[group_key].sort(key=lambda c: c.metadata.chunk_index or 0)

        # Step 3: Score each group by its best chunk's reranker score
        def get_group_score(group: list[Chunk]) -> float:
            max_score = max((c.reranker_score or 0.0) for c in group)
            # Boost groups that contain tables
            has_table = any(self._is_table_chunk(c) for c in group)
            return max_score + (1.0 if has_table else 0.0)

        sorted_groups = sorted(groups.items(), key=lambda x: get_group_score(x[1]), reverse=True)

        # Step 4: Build ordered list with priority
        ordered: list[Chunk] = []

        for group_key, group_chunks in sorted_groups:
            # Sort chunks within group by priority
            def chunk_priority(chunk: Chunk) -> tuple[int, float]:
                # Tables get highest priority (0)
                if self._is_table_chunk(chunk):
                    priority = 0
                elif chunk.metadata.source_type == "heading":
                    priority = 1
                elif chunk.metadata.source_type == "list":
                    priority = 2
                elif chunk.metadata.source_type == "paragraph" and chunk.metadata.semantic_tags:
                    priority = 3
                else:
                    priority = 4

                score = chunk.reranker_score or 0.0
                return (priority, -score)

            group_chunks.sort(key=chunk_priority)
            ordered.extend(group_chunks)

        # Step 5: Limit to max_chunks
        if len(ordered) > self.max_chunks:
            ordered = ordered[:self.max_chunks]

        return ordered

    def _format_chunk(self, chunk: Chunk, index: int) -> str:
        """
        Format a single chunk as context text with citation.

        Args:
            chunk: The chunk to format.
            index: The chunk index.

        Returns:
            str: Formatted chunk text.
        """
        parts: list[str] = []

        citation_parts: list[str] = []
        if chunk.metadata.document_name:
            citation_parts.append(f"Document: {chunk.metadata.document_name}")
        if chunk.metadata.page:
            citation_parts.append(f"Page: {chunk.metadata.page}")
        if chunk.metadata.section:
            citation_parts.append(f"Section: {chunk.metadata.section}")
        if chunk.metadata.subsection:
            citation_parts.append(f"Subsection: {chunk.metadata.subsection}")

        citation = " | ".join(citation_parts)
        parts.append(f"[{index + 1}] {citation}")
        parts.append("-" * 40)

        # Check if this is a table chunk and format accordingly
        if self._is_table_chunk(chunk):
            parts.append(self._format_table(chunk))
        else:
            parts.append(chunk.content.strip())

        parts.append("")

        return "\n".join(parts)

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using ~4 chars per token.
        """
        return max(1, len(text) // 4)

    def build(self, chunks: list[Chunk]) -> str:
        """
        Build the final context from retrieved chunks.

        Tables are formatted with ALL rows preserved in Markdown format.

        Args:
            chunks: List of chunks to build context from.

        Returns:
            str: Formatted context string.
        """
        with measure_latency("context_building") as latency:
            logger.log_event(
                event=LogEvent.CONTEXT_BUILDING,
                message=f"Building context from {len(chunks)} chunks",
                details={"input_chunks": len(chunks)},
            )

            if not chunks:
                logger.log_event(
                    event=LogEvent.CONTEXT_BUILDING,
                    message="No chunks provided, returning empty context",
                )
                return ""

            # Step 1: Remove duplicates (content-based only)
            chunks = self._remove_duplicates(chunks)

            # Step 2: Remove low-confidence chunks
            chunks = self._remove_low_confidence(chunks)

            # Step 3: Order chunks for context (group by section, prioritize tables)
            chunks = self._order_chunks_for_context(chunks)

            # Step 4: Limit to max_chunks
            chunks = chunks[:self.max_chunks]

            # Step 5: Build context
            context_parts: list[str] = []
            context_parts.append("=== RETRIEVED DOCUMENTS ===\n")

            total_tokens = 0
            included_chunks: list[Chunk] = []

            for i, chunk in enumerate(chunks):
                formatted = self._format_chunk(chunk, i)
                estimated_tokens = self._estimate_tokens(formatted)

                reserved_tokens = 2000
                if total_tokens + estimated_tokens + reserved_tokens > self.max_context_tokens:
                    logger.log_event(
                        event=LogEvent.CONTEXT_BUILDING,
                        message=f"Context token limit reached, stopping at {i} chunks",
                        details={"included_chunks": i, "estimated_tokens": total_tokens},
                    )
                    break

                context_parts.append(formatted)
                total_tokens += estimated_tokens
                included_chunks.append(chunk)

            context = "\n".join(context_parts)

            # Log table count in context
            table_count = sum(1 for c in included_chunks if self._is_table_chunk(c))
            
            # Log total rows in tables
            total_table_rows = 0
            for c in included_chunks:
                if self._is_table_chunk(c) and c.metadata.table_data:
                    try:
                        table_data = json.loads(c.metadata.table_data)
                        total_table_rows += table_data.get("row_count", 0)
                    except (json.JSONDecodeError, TypeError):
                        pass

            latency.stop(
                input_chunks=len(chunks),
                output_chunks=len(included_chunks),
                estimated_tokens=total_tokens,
                table_count=table_count,
                total_table_rows=total_table_rows,
            )

            logger.log_event(
                event=LogEvent.CONTEXT_BUILDING,
                message=f"Context built with {len(included_chunks)} chunks, ~{total_tokens} tokens, {table_count} tables ({total_table_rows} total rows)",
                details={
                    "input_chunks": len(chunks),
                    "output_chunks": len(included_chunks),
                    "estimated_tokens": total_tokens,
                    "table_count": table_count,
                    "total_table_rows": total_table_rows,
                    "duration_ms": latency.duration_ms,
                },
            )

            return context

    def build_with_metadata(self, chunks: list[Chunk]) -> dict[str, Any]:
        """
        Build context and return with metadata for debugging.

        Args:
            chunks: List of chunks.

        Returns:
            dict[str, Any]: Context with metadata.
        """
        context = self.build(chunks)
        table_count = sum(1 for c in chunks if self._is_table_chunk(c))
        
        # Count total table rows
        total_table_rows = 0
        for c in chunks:
            if self._is_table_chunk(c) and c.metadata.table_data:
                try:
                    table_data = json.loads(c.metadata.table_data)
                    total_table_rows += table_data.get("row_count", 0)
                except (json.JSONDecodeError, TypeError):
                    pass
        
        return {
            "context": context,
            "chunk_count": len(chunks),
            "table_count": table_count,
            "total_table_rows": total_table_rows,
            "estimated_tokens": self._estimate_tokens(context),
        }