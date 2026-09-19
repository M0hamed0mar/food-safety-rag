"""
Table Processor module for advanced table extraction and processing.

This module handles all table-related operations including:
- Extraction from PDFs (using pdfplumber and camelot) and DOCX
- Cleaning and restructuring
- Semantic description generation
- Table classification
- Formatting for embedding and context display

Enhanced with support for merged cells from pdfplumber.
"""

import json
import re
from typing import Any, Optional
from dataclasses import dataclass, field

from app.config.constants import LogEvent
from app.monitoring import get_logger, measure_latency


logger = get_logger("food_safety_rag.ingestion.table_processor")


@dataclass
class ProcessedTable:
    """
    A processed table with all metadata preserved.
    
    Attributes:
        table_id: Unique identifier for the table.
        page: Page number where the table appears.
        headers: List of column headers.
        rows: List of rows (each row is a list of cells).
        row_count: Number of rows.
        column_count: Number of columns.
        raw_text: Original table text.
        table_type: Detected table type (hazard, allergen, process, etc.).
        semantic_description: Generated semantic description.
        keywords: Extracted keywords.
        normalized_markdown: Table in Markdown format.
        embedding_text: Text representation for embedding.
        confidence: Extraction confidence score.
    """
    
    table_id: str
    page: int
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    raw_text: str = ""
    table_type: str = "general_table"
    semantic_description: str = ""
    keywords: list[str] = field(default_factory=list)
    normalized_markdown: str = ""
    embedding_text: str = ""
    confidence: float = 1.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "table_id": self.table_id,
            "page": self.page,
            "headers": self.headers,
            "rows": self.rows[:20],  # Limit for storage
            "row_count": self.row_count,
            "column_count": self.column_count,
            "raw_text": self.raw_text[:1000],  # Limit for storage
            "table_type": self.table_type,
            "semantic_description": self.semantic_description,
            "keywords": self.keywords,
            "normalized_markdown": self.normalized_markdown[:1000],
            "embedding_text": self.embedding_text[:2000],
            "confidence": self.confidence,
        }
    
    def get_markdown(self, max_rows: int = 50) -> str:
        """
        Get table in Markdown format with optional row limit.
        
        Args:
            max_rows: Maximum rows to include.
        
        Returns:
            str: Markdown table.
        """
        if not self.headers or not self.rows:
            return self.raw_text
        
        lines = []
        col_count = max(len(self.headers), max(len(row) for row in self.rows) if self.rows else 0)
        
        # Header row with proper padding
        padded_headers = self.headers + [""] * (col_count - len(self.headers))
        header_line = "| " + " | ".join(str(h).strip() for h in padded_headers) + " |"
        lines.append(header_line)
        
        # Separator with proper column count
        sep_line = "| " + " | ".join(["---"] * col_count) + " |"
        lines.append(sep_line)
        
        # Data rows
        display_rows = self.rows[:max_rows]
        for row in display_rows:
            # Pad row to match column count
            padded_row = row + [""] * (col_count - len(row))
            row_line = "| " + " | ".join(str(cell).strip() for cell in padded_row) + " |"
            lines.append(row_line)
        
        if len(self.rows) > max_rows:
            lines.append(f"| ... ({len(self.rows) - max_rows} more rows) |")
        
        return "\n".join(lines)
    
    def get_embedding_text(self, max_rows: int = 50) -> str:
        """
        Get text representation for embedding.
        
        Args:
            max_rows: Maximum rows to include.
        
        Returns:
            str: Text for embedding.
        """
        parts = []
        
        # Table metadata
        parts.append(f"[TABLE] Table ID: {self.table_id}")
        parts.append(f"Table Type: {self.table_type}")
        parts.append(f"Structure: {self.row_count} rows, {self.column_count} columns")
        
        if self.headers:
            parts.append(f"Columns: {', '.join(h.strip() for h in self.headers)}")
        
        # Table description
        if self.semantic_description:
            parts.append(f"Description: {self.semantic_description}")
        
        # Keywords
        if self.keywords:
            parts.append(f"Keywords: {', '.join(self.keywords)}")
        
        # Data (all rows for embedding)
        data_lines = []
        for i, row in enumerate(self.rows[:max_rows], start=1):
            padded_row = row + [""] * (self.column_count - len(row))
            row_text = " | ".join(str(cell).strip() for cell in padded_row)
            data_lines.append(f"Row {i}: {row_text}")
        
        if data_lines:
            parts.append("Data:")
            parts.extend(data_lines)
        
        if len(self.rows) > max_rows:
            parts.append(f"... ({len(self.rows) - max_rows} more rows)")
        
        return "\n".join(parts)


class TableProcessor:
    """
    Advanced table processor for extraction, cleaning, and formatting.
    
    Enhanced with support for merged cells from pdfplumber.
    """
    
    # Table type keywords for classification
    # -----------------------------------------------------------------------------
    # Table type keywords (EMPTY BY DEFAULT).
    #
    # This is intentionally empty so the classifier is DOMAIN-AGNOSTIC.
    # To add domain-specific table classifiers, populate this dict:
    #
    #   TABLE_TYPE_KEYWORDS = {
    #       "invoice_table": ["invoice", "amount", "total"],
    #       "specification_table": ["spec", "parameter", "value"],
    #       ...
    #   }
    #
    # The classifier will return "general_table" when no keywords match.
    # -----------------------------------------------------------------------------
    TABLE_TYPE_KEYWORDS: dict[str, list[str]] = {}

    
    def __init__(self) -> None:
        """Initialize the table processor."""
        self._table_counter: int = 0
    
    def _generate_table_id(self) -> str:
        """Generate a unique table ID."""
        self._table_counter += 1
        return f"T{self._table_counter:04d}"
    
    def _safe_cell_to_string(self, cell: Any) -> str:
        """
        Safely convert a cell value to string.
        
        Args:
            cell: Cell value of any type.
        
        Returns:
            str: Clean string representation.
        """
        if cell is None:
            return ""
        
        cell_str = str(cell).strip()
        
        # Handle common problematic values
        if cell_str in ("None", "null", "NULL", "undefined"):
            return ""
        
        # Remove extra whitespace
        cell_str = re.sub(r'\s+', ' ', cell_str)
        
        return cell_str
    
    def _detect_table_type(self, headers: list[str], rows: list[list[str]]) -> str:
        """
        Detect the type of table based on content with improved sampling.
        
        Args:
            headers: Column headers.
            rows: Table rows.
        
        Returns:
            str: Detected table type.
        """
        # Combine header text
        all_text = " ".join(h.lower() for h in headers)
        
        # Sample rows: take first 3, middle 3, and last 3
        sample_indices = set()
        sample_indices.update(range(min(3, len(rows))))  # First 3
        if len(rows) > 6:
            mid_idx = len(rows) // 2
            sample_indices.update(range(mid_idx - 1, min(mid_idx + 2, len(rows))))  # Middle 3
        sample_indices.update(range(max(0, len(rows) - 3), len(rows)))  # Last 3
        
        for idx in sorted(sample_indices):
            row = rows[idx]
            all_text += " " + " ".join(self._safe_cell_to_string(cell) for cell in row).lower()
        
        # Check each type with priority (more specific types first)
        type_scores = {}
        for table_type, keywords in self.TABLE_TYPE_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in all_text)
            if matches > 0:
                type_scores[table_type] = matches
        
        # Return type with highest score, or general if no matches
        if type_scores:
            return max(type_scores, key=type_scores.get)
        
        return "general_table"
    
    def _extract_keywords(self, headers: list[str], rows: list[list[str]]) -> list[str]:
        """
        Extract meaningful keywords from table headers and sample data.
        
        Args:
            headers: Column headers.
            rows: Table rows.
        
        Returns:
            list[str]: List of keywords.
        """
        keywords = set()
        
        # Add headers as keywords (cleaned)
        for header in headers:
            clean_header = self._safe_cell_to_string(header)
            if clean_header and len(clean_header) > 2:
                keywords.add(clean_header)
        
        # Extract from first 5 rows
        for row in rows[:5]:
            for cell in row:
                clean_cell = self._safe_cell_to_string(cell)
                
                # Only add substantial content
                if clean_cell and len(clean_cell) > 3:
                    # Split on common delimiters and take meaningful parts
                    words = re.split(r'[,;/\-]', clean_cell)
                    for word in words:
                        word = word.strip()
                        if len(word) > 3 and len(word) < 50:  # Reasonable length
                            keywords.add(word)
        
        # Limit and return sorted
        return sorted(list(keywords))[:15]
    
    def _generate_semantic_description(
        self,
        headers: list[str],
        rows: list[list[str]],
        table_type: str,
    ) -> str:
        """
        Generate a semantic description of the table.
        
        Args:
            headers: Column headers.
            rows: Table rows.
            table_type: Detected table type.
        
        Returns:
            str: Semantic description.
        """
        parts = []
        
        # Type description
        type_description = table_type.replace("_", " ").title()
        parts.append(f"This is a {type_description} with domain relevance to food safety.")
        
        # Structure
        parts.append(f"It contains {len(rows)} rows and {len(headers) if headers else 0} columns.")
        
        # Headers
        if headers:
            clean_headers = [self._safe_cell_to_string(h) for h in headers]
            parts.append(f"The columns are: {', '.join(clean_headers)}.")
        
        # Sample data (first 3 rows)
        if rows:
            sample_rows = rows[:3]
            parts.append("Sample data:")
            for i, row in enumerate(sample_rows, start=1):
                clean_row = [self._safe_cell_to_string(cell) for cell in row]
                row_text = " | ".join(clean_row)
                parts.append(f"Row {i}: {row_text}")
            
            if len(rows) > 3:
                parts.append(f"({len(rows) - 3} more rows)")
        
        return " ".join(parts)
    
    def _normalize_markdown(self, headers: list[str], rows: list[list[str]]) -> str:
        """
        Convert table to normalized Markdown format.
        
        Args:
            headers: Column headers.
            rows: Table rows.
        
        Returns:
            str: Markdown table.
        """
        if not headers or not rows:
            return ""
        
        lines = []
        col_count = len(headers)
        
        # Header
        clean_headers = [self._safe_cell_to_string(h) for h in headers]
        header_line = "| " + " | ".join(clean_headers) + " |"
        lines.append(header_line)
        
        # Separator
        sep_line = "| " + " | ".join(["---"] * col_count) + " |"
        lines.append(sep_line)
        
        # Rows
        for row in rows:
            clean_row = [self._safe_cell_to_string(cell) for cell in row]
            padded_row = clean_row + [""] * (col_count - len(clean_row))
            row_line = "| " + " | ".join(padded_row) + " |"
            lines.append(row_line)
        
        return "\n".join(lines)
    
    def _calculate_confidence_score(
        self,
        headers: list[str],
        rows: list[list[str]],
    ) -> float:
        """
        Calculate confidence score based on table quality metrics.
        
        Args:
            headers: Column headers.
            rows: Table rows.
        
        Returns:
            float: Confidence score between 0.0 and 1.0.
        """
        score = 0.95  # Base score
        
        # Penalize if insufficient data
        if len(rows) < 2:
            score -= 0.15
        
        # Penalize if headers are sparse
        if len(headers) < 2:
            score -= 0.10
        
        # Penalize if many empty cells
        empty_cell_count = 0
        total_cells = 0
        
        for row in rows:
            for cell in row:
                total_cells += 1
                if not self._safe_cell_to_string(cell):
                    empty_cell_count += 1
        
        if total_cells > 0:
            empty_ratio = empty_cell_count / total_cells
            if empty_ratio > 0.3:
                score -= (empty_ratio - 0.3) * 0.5
        
        # Penalize if inconsistent column counts
        col_lengths = [len(row) for row in rows]
        if col_lengths and max(col_lengths) != min(col_lengths):
            score -= 0.05
        
        return max(0.5, min(1.0, score))  # Keep between 0.5 and 1.0
    
    def _clean_table_data(
        self,
        headers: list[str],
        rows: list[list[str]],
    ) -> tuple[list[str], list[list[str]]]:
        """
        Clean table data by removing empty rows and normalizing.
        
        Args:
            headers: Column headers.
            rows: Table rows.
        
        Returns:
            tuple[list[str], list[list[str]]]: Cleaned headers and rows.
        """
        # Clean headers
        cleaned_headers = []
        for i, h in enumerate(headers):
            cleaned = self._safe_cell_to_string(h)
            if cleaned:
                cleaned_headers.append(cleaned)
            else:
                cleaned_headers.append(f"Column_{i + 1}")
        
        # If no headers, generate them
        if not cleaned_headers and rows:
            col_count = max(len(row) for row in rows)
            cleaned_headers = [f"Column_{i + 1}" for i in range(col_count)]
        
        # Clean rows
        cleaned_rows = []
        for row in rows:
            cleaned_row = [self._safe_cell_to_string(cell) for cell in row]
            # Skip completely empty rows
            if any(cleaned_row):
                cleaned_rows.append(cleaned_row)
        
        return cleaned_headers, cleaned_rows
    
    def process_table_data(
        self,
        raw_headers: list[str],
        raw_rows: list[list[str]],
        page: int,
        raw_text: str = "",
    ) -> ProcessedTable:
        """
        Process raw table data into a structured ProcessedTable.
        
        Args:
            raw_headers: Raw column headers.
            raw_rows: Raw table rows.
            page: Page number.
            raw_text: Original table text.
        
        Returns:
            ProcessedTable: Processed table with all metadata.
        """
        with measure_latency("table_processing") as latency:
            # Clean data
            headers, rows = self._clean_table_data(raw_headers, raw_rows)
            
            if not rows:
                logger.log_event(
                    event=LogEvent.WARNING,
                    message="Table has no rows after cleaning",
                    level=30,
                )
                return ProcessedTable(
                    table_id=self._generate_table_id(),
                    page=page,
                    raw_text=raw_text,
                )
            
            # Detect type
            table_type = self._detect_table_type(headers, rows)
            
            # Extract keywords
            keywords = self._extract_keywords(headers, rows)
            
            # Generate description
            semantic_description = self._generate_semantic_description(headers, rows, table_type)
            
            # Normalize Markdown
            normalized_markdown = self._normalize_markdown(headers, rows)
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(headers, rows)
            
            # Create ProcessedTable
            processed_table = ProcessedTable(
                table_id=self._generate_table_id(),
                page=page,
                headers=headers,
                rows=rows,
                row_count=len(rows),
                column_count=len(headers),
                raw_text=raw_text,
                table_type=table_type,
                semantic_description=semantic_description,
                keywords=keywords,
                normalized_markdown=normalized_markdown,
                confidence=confidence,
            )
            
            # Generate embedding text
            processed_table.embedding_text = processed_table.get_embedding_text(max_rows=50)
            
            latency.stop(
                table_id=processed_table.table_id,
                rows=len(rows),
                columns=len(headers),
                table_type=table_type,
            )
            
            logger.log_ingestion(
                event=LogEvent.OCR_COMPLETE,
                document_id="table_processor",
                document_name="table_processing",
                message=f"Processed table {processed_table.table_id} with {len(rows)} rows",
                details={
                    "table_id": processed_table.table_id,
                    "rows": len(rows),
                    "columns": len(headers),
                    "table_type": table_type,
                    "keywords": keywords[:5],
                    "confidence": round(confidence, 3),
                },
            )
            
            return processed_table
    
    def process_table_from_dict(
        self,
        table_dict: dict[str, Any],
        page: int,
    ) -> ProcessedTable:
        """
        Process a table from a dictionary (from PDF/DOCX extraction).
        
        Args:
            table_dict: Table dictionary with headers and rows.
            page: Page number.
        
        Returns:
            ProcessedTable: Processed table.
        """
        headers = table_dict.get("headers", [])
        rows = table_dict.get("rows", [])
        raw_text = table_dict.get("text", "")
        
        return self.process_table_data(headers, rows, page, raw_text)
    
    def tables_to_chunks(
        self,
        tables: list[ProcessedTable],
    ) -> list[dict[str, Any]]:
        """
        Convert processed tables to chunk dictionaries for ingestion.
        
        Args:
            tables: List of processed tables.
        
        Returns:
            list[dict[str, Any]]: Chunk dictionaries for ingestion.
        """
        chunks = []
        
        for table in tables:
            chunk = {
                "content": table.normalized_markdown or table.raw_text,
                "page_number": table.page,
                "has_table": True,
                "source_type": "table",
                "table_id": table.table_id,
                "headers": table.headers,
                "rows": table.rows,
                "row_count": table.row_count,
                "column_count": table.column_count,
                "semantic_description": table.semantic_description,
                "table_type": table.table_type,
                "keywords": table.keywords,
                "embedding_text": table.embedding_text,
                "is_extracted_table": True,
                "confidence": table.confidence,
            }
            chunks.append(chunk)
        
        return chunks