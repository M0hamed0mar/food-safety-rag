"""
Semantic and Hierarchical Chunking module.

This module provides intelligent document chunking that preserves semantic meaning
and document hierarchy. It NEVER uses naive fixed-size chunking.

Tables are always extracted as complete entities with all rows preserved.
Uses TableProcessor for advanced table handling and embedding text generation.
"""

import re
import json
from dataclasses import dataclass
from typing import Any, Optional

from app.config import settings
from app.config.constants import DEFAULT_CHUNK_OVERLAP_PERCENT, LogEvent
from app.monitoring import get_logger, measure_latency
from app.schemas import Chunk, ChunkMetadata, Document, HierarchicalChunk
from app.ingestion.table_processor import TableProcessor, ProcessedTable


logger = get_logger("food_safety_rag.ingestion.chunker")


@dataclass
class ChunkBoundary:
    """Represents a semantic boundary in text."""
    position: int
    boundary_type: str
    level: int
    text: str


@dataclass
class TableData:
    """Represents extracted table data."""
    start: int
    end: int
    text: str
    headers: list[str]
    rows: list[list[str]]
    row_count: int
    column_count: int
    table_id: str
    page: Optional[int] = None
    semantic_description: str = ""
    table_type: str = "general"


class BoundaryDetector:
    """
    Handles detection of semantic boundaries in text.
    
    Uses pre-compiled patterns for efficiency and supports multiple languages.
    """
    
    # Compiled patterns for efficiency
    HEADING_PATTERNS = [
        (re.compile(r"\n#{1,6}\s+(.+?)(?=\n)"), "heading", 1),
        (re.compile(r"\n[A-Z][A-Z\s\d\.\-]{3,}(?=\n)"), "heading", 1),
        (re.compile(r"\n\d+\.\s+[A-Z][^\n]{2,}(?=\n)"), "heading", 2),
        (re.compile(r"\n\d+\.\d+\.\s+[A-Z][^\n]{2,}(?=\n)"), "heading", 3),
        (re.compile(r"\n\d+\.\d+\.\d+\.\s+[A-Z][^\n]{2,}(?=\n)"), "heading", 4),
        (re.compile(r"\n(?:Chapter|CHAPTER|Section|SECTION)\s+\d+[:\s]+([^\n]+)"), "heading", 1),
        (re.compile(r"\n(?:Table|TABLE|Figure|FIGURE)\s+\d+[:\s]+([^\n]+)"), "heading", 2),
        # Arabic heading patterns
        (re.compile(r"\n[أ-ي][أ-ي\s\d\.\-]{3,}(?=\n)"), "heading", 1),
        (re.compile(r"\n(?:الفصل|الباب|القسم)\s+\d+[:\s]+([^\n]+)"), "heading", 1),
    ]
    
    PARAGRAPH_PATTERN = re.compile(r"\n\s*\n")
    LIST_ITEM_PATTERN = re.compile(r"\n\s*[\*\-\+\d\.\)]\s+")
    
    TABLE_PATTERNS = [
        re.compile(r"\n\s*\|"),
        re.compile(r"\n\s*[A-Za-z]+\s+[A-Za-z]+\s+[A-Za-z]+\s*\n"),
        re.compile(r"\n\s*[A-Za-z]+\s+[A-Za-z]+\s+[A-Za-z]+\s+[A-Za-z]+\s*\n"),
        re.compile(r"\n\s*[أ-ي]+\s+[أ-ي]+\s+[أ-ي]+\s*\n"),
    ]
    
    @classmethod
    def find_boundaries(cls, text: str) -> list[ChunkBoundary]:
        """
        Find all semantic boundaries in text.
        
        Args:
            text: Input text to analyze.
        
        Returns:
            list[ChunkBoundary]: Sorted list of boundaries.
        """
        boundaries_dict: dict[int, ChunkBoundary] = {}
        
        # Document start boundary
        boundaries_dict[0] = ChunkBoundary(0, "document_start", 0, "")
        
        # Heading boundaries
        for pattern, btype, level in cls.HEADING_PATTERNS:
            for match in pattern.finditer(text):
                pos = match.start()
                if pos not in boundaries_dict:
                    boundaries_dict[pos] = ChunkBoundary(
                        position=pos,
                        boundary_type=btype,
                        level=level,
                        text=match.group().strip(),
                    )
        
        # Paragraph boundaries
        for match in cls.PARAGRAPH_PATTERN.finditer(text):
            pos = match.start()
            if pos not in boundaries_dict:
                boundaries_dict[pos] = ChunkBoundary(pos, "paragraph", 4, "")
        
        # List item boundaries
        for match in cls.LIST_ITEM_PATTERN.finditer(text):
            pos = match.start()
            if pos not in boundaries_dict:
                boundaries_dict[pos] = ChunkBoundary(pos, "list_item", 4, "")
        
        # Table boundaries
        for pattern in cls.TABLE_PATTERNS:
            for match in pattern.finditer(text):
                pos = match.start()
                if pos not in boundaries_dict:
                    boundaries_dict[pos] = ChunkBoundary(pos, "table_row", 4, "")
        
        # Return sorted boundaries
        return sorted(boundaries_dict.values(), key=lambda b: b.position)


class TableExtractor:
    """
    Handles table extraction and parsing from text.
    
    Extracts tables as complete entities and preserves structure.
    """
    
    def __init__(self) -> None:
        """Initialize table extractor with counter."""
        self._counter: int = 0
    
    def extract_tables(self, text: str) -> list[TableData]:
        """
        Extract tables from text as complete entities.
        
        Args:
            text: Input text containing tables.
        
        Returns:
            list[TableData]: Extracted table data.
        """
        tables: list[TableData] = []
        lines = text.split("\n")
        
        in_table = False
        table_start_char = 0
        table_start_line = 0
        table_lines: list[str] = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            is_table_line = self._is_table_line(stripped, line)
            
            if is_table_line and not in_table:
                # Start of new table
                in_table = True
                table_start_line = i
                table_start_char = sum(len(lines[j]) + 1 for j in range(i))
                table_lines = [line]
            
            elif is_table_line and in_table:
                # Continue table
                table_lines.append(line)
            
            elif not is_table_line and in_table:
                # End of table
                in_table = False
                table_text = "\n".join(table_lines)
                
                table_data = self._parse_table(table_text, table_start_char)
                if table_data:
                    self._counter += 1
                    table_data.table_id = f"T{self._counter:04d}"
                    table_data.page = self._estimate_page_from_position(table_start_char, text)
                    table_data.semantic_description = self._generate_description(table_data)
                    table_data.table_type = self._detect_type(table_data)
                    tables.append(table_data)
                    
                    logger.log_ingestion(
                        event=LogEvent.OCR_COMPLETE,
                        document_id="table_extractor",
                        document_name="table_extraction",
                        message=f"Extracted table {table_data.table_id} with {table_data.row_count} rows",
                        details={
                            "table_id": table_data.table_id,
                            "rows": table_data.row_count,
                            "columns": table_data.column_count,
                            "table_type": table_data.table_type,
                        },
                    )
                
                table_lines = []
        
        # Handle final table if document ends in table
        if in_table and table_lines:
            table_text = "\n".join(table_lines)
            table_data = self._parse_table(table_text, table_start_char)
            if table_data:
                self._counter += 1
                table_data.table_id = f"T{self._counter:04d}"
                table_data.semantic_description = self._generate_description(table_data)
                table_data.table_type = self._detect_type(table_data)
                tables.append(table_data)
        
        return tables
    
    @staticmethod
    def _is_table_line(stripped: str, original: str) -> bool:
        """
        Check if a line is part of a table.
        
        Args:
            stripped: Stripped line content.
            original: Original line content.
        
        Returns:
            bool: True if line appears to be a table line.
        """
        # Pipe-separated or tab-separated
        if stripped.startswith("|") or "\t" in original:
            return True
        
        # Multi-column patterns (English or Arabic)
        columns = re.findall(r'\b[a-zA-Z]+\b', stripped)
        arabic_columns = re.findall(r'[\u0600-\u06FF]+', stripped)
        all_columns = columns + arabic_columns
        
        return len(all_columns) >= 3 and len(stripped) > 30
    
    @staticmethod
    def _parse_table(table_text: str, start_pos: int) -> Optional[TableData]:
        """
        Parse table text into structured format.
        
        Args:
            table_text: Raw table text.
            start_pos: Starting character position.
        
        Returns:
            Optional[TableData]: Parsed table data or None.
        """
        lines = [l.strip() for l in table_text.strip().split("\n") if l.strip()]
        
        if not lines or len(lines) < 2:
            return None
        
        # Detect separator pattern (pipe or tab)
        separator = "|" if "|" in lines[0] else "\t"
        
        # Extract headers (first line)
        headers = [h.strip() for h in lines[0].split(separator) if h.strip()]
        
        if not headers or len(headers) < 2:
            # Try to detect multi-column text table
            headers = re.split(r'\s{2,}', lines[0].strip())
            if len(headers) < 2:
                return None
        
        # Extract rows
        rows: list[list[str]] = []
        for line in lines[1:]:
            if separator in line:
                row = [cell.strip() for cell in line.split(separator) if cell.strip()]
            else:
                row = re.split(r'\s{2,}', line.strip())
            
            if row:
                # Pad or truncate to match header length
                if len(row) < len(headers):
                    row += [""] * (len(headers) - len(row))
                else:
                    row = row[:len(headers)]
                rows.append(row)
        
        if not rows:
            return None
        
        return TableData(
            start=start_pos,
            end=start_pos + len(table_text),
            text=table_text,
            headers=headers,
            rows=rows,
            row_count=len(rows),
            column_count=len(headers),
            table_id="",
        )
    
    @staticmethod
    def _estimate_page_from_position(position: int, text: str) -> int:
        """
        Estimate page number from character position.
        
        Args:
            position: Character position in text.
            text: Full document text.
        
        Returns:
            int: Estimated page number.
        """
        # Rough estimate: ~2500 characters per page
        chars_per_page = 2500
        return max(1, (position // chars_per_page) + 1)
    
    @staticmethod
    def _generate_description(table: TableData) -> str:
        """
        Generate semantic description of table.
        
        Args:
            table: Table data.
        
        Returns:
            str: Semantic description.
        """
        header_str = ", ".join(table.headers[:3])
        if table.row_count > 0:
            return f"Table with columns: {header_str}. Contains {table.row_count} rows of data."
        return f"Table with columns: {header_str}."
    
    @staticmethod
    def _detect_type(table: TableData) -> str:
        """
        Detect table type based on structure and content.
        
        Args:
            table: Table data.
        
        Returns:
            str: Detected table type.
        """
        # Check headers for indicators
        all_headers = " ".join(h.lower() for h in table.headers)
        
        type_keywords = {
            "hazard_table": ["hazard", "risk", "danger", "خطر", "مخاطر"],
            "allergen_table": ["allergen", "allergy", "حساسية", "مسبب حساسية"],
            "process_table": ["process", "step", "procedure", "خطوة", "معالجة"],
            "temperature_table": ["temperature", "thermal", "heat", "حرارة", "درجة حرارة"],
            "pathogen_table": ["pathogen", "bacteria", "virus", "بكتيريا", "فيروس"],
            "nutrition_table": ["nutrition", "vitamin", "mineral", "فيتامين", "معادن"],
            "additive_table": ["additive", "preservative", "مضاف", "مواد حافظة"],
            "control_table": ["control", "measure", "prevent", "تحكم", "إجراء"],
            "ingredient_table": ["ingredient", "composition", "component", "مكون", "تركيب"],
            "specification_table": ["specification", "standard", "requirement", "مواصفة", "معيار"],
        }
        
        for table_type, keywords in type_keywords.items():
            if any(kw in all_headers for kw in keywords):
                return table_type
        
        # If numeric headers, consider numerical table
        numeric_headers = sum(1 for h in table.headers if re.match(r'^[\d\.]+$', h.strip()))
        if numeric_headers >= len(table.headers) // 2:
            return "numerical_table"
        
        return "general_table"


class HierarchyExtractor:
    """
    Handles document hierarchy extraction.
    
    Extracts chapters, sections, and subsections from document text.
    Supports both numbered sections and Markdown headers.
    """
    
    # ==========================================================================
    # Numbered Section Patterns
    # ==========================================================================
    CHAPTER_PATTERN = re.compile(
        r"(?:^|\n)\s*(?:Chapter|CHAPTER|الفصل)\s+(\d+|[IVX]+)[\s:.-]+([^\n]+)",
        re.IGNORECASE
    )
    SECTION_PATTERN = re.compile(r"(?:^|\n)\s*(\d+\.\d+)[\s.]+([A-Z][^\n]{2,})")
    SUBSECTION_PATTERN = re.compile(r"(?:^|\n)\s*(\d+\.\d+\.\d+)[\s.]+([A-Z][^\n]{2,})")
    
    # ==========================================================================
    # Markdown Header Patterns
    # ==========================================================================
    MARKDOWN_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    
    @classmethod
    def extract_hierarchy(cls, text: str) -> list[dict[str, Any]]:
        """
        Extract document hierarchy from text.
        Supports both numbered sections and Markdown headers.
        
        Args:
            text: Document text.
        
        Returns:
            list[dict[str, Any]]: Hierarchy elements with position and title.
        """
        hierarchy: list[dict[str, Any]] = []
        
        # 1. Extract numbered sections (Chapter, Section, Subsection)
        for match in cls.CHAPTER_PATTERN.finditer(text):
            hierarchy.append({
                "type": "chapter",
                "level": 1,
                "position": match.start(),
                "number": match.group(1),
                "title": match.group(2).strip(),
            })
        
        for match in cls.SECTION_PATTERN.finditer(text):
            hierarchy.append({
                "type": "section",
                "level": 2,
                "position": match.start(),
                "number": match.group(1),
                "title": match.group(2).strip(),
            })
        
        for match in cls.SUBSECTION_PATTERN.finditer(text):
            hierarchy.append({
                "type": "subsection",
                "level": 3,
                "position": match.start(),
                "number": match.group(1),
                "title": match.group(2).strip(),
            })
        
        # 2. Extract Markdown headers (#, ##, ###)
        for match in cls.MARKDOWN_HEADING_PATTERN.finditer(text):
            level = len(match.group(1))
            title = match.group(2).strip()
            position = match.start()
            
            # Only add if not already captured by numbered section at same position
            existing = any(
                h.get("position") == position and h.get("title") == title
                for h in hierarchy
            )
            
            if not existing:
                if level == 1:
                    hierarchy.append({
                        "type": "chapter",
                        "level": 1,
                        "position": position,
                        "number": "",
                        "title": title,
                    })
                elif level == 2:
                    hierarchy.append({
                        "type": "section",
                        "level": 2,
                        "position": position,
                        "number": "",
                        "title": title,
                    })
                else:  # level >= 3
                    hierarchy.append({
                        "type": "subsection",
                        "level": 3,
                        "position": position,
                        "number": "",
                        "title": title,
                    })
        
        # Sort by position
        hierarchy.sort(key=lambda x: x["position"])
        return hierarchy
    
    @classmethod
    def assign_hierarchy_to_chunks(
        cls,
        chunks: list[str],
        text: str,
    ) -> list[dict[str, Any]]:
        """
        Assign hierarchy information to chunks efficiently.
        
        Args:
            chunks: List of chunk texts.
            text: Full document text.
        
        Returns:
            list[dict[str, Any]]: Chunks with hierarchy metadata.
        """
        hierarchy = cls.extract_hierarchy(text)
        
        if not hierarchy:
            return [
                {"content": chunk, "chapter": None, "section": None, "subsection": None, "title": None}
                for chunk in chunks
            ]
        
        # Track chunk positions
        chunk_positions: list[int] = []
        pos = 0
        for chunk in chunks:
            chunk_positions.append(pos)
            pos += len(chunk)
        
        assigned_chunks: list[dict[str, Any]] = []
        hierarchy_idx = 0
        
        for chunk, chunk_pos in zip(chunks, chunk_positions):
            # Find applicable hierarchy elements
            current_chapter = None
            current_section = None
            current_subsection = None
            
            # Advance hierarchy_idx while elements are before chunk position
            temp_idx = hierarchy_idx
            while temp_idx < len(hierarchy) and hierarchy[temp_idx]["position"] <= chunk_pos:
                element = hierarchy[temp_idx]
                if element["type"] == "chapter":
                    current_chapter = element
                elif element["type"] == "section":
                    current_section = element
                elif element["type"] == "subsection":
                    current_subsection = element
                temp_idx += 1
            
            hierarchy_idx = temp_idx
            
            title = (
                current_subsection["title"] if current_subsection
                else current_section["title"] if current_section
                else current_chapter["title"] if current_chapter
                else None
            )
            
            assigned_chunks.append({
                "content": chunk,
                "chapter": current_chapter["title"] if current_chapter else None,
                "section": current_section["title"] if current_section else None,
                "subsection": current_subsection["title"] if current_subsection else None,
                "title": title,
            })
        
        return assigned_chunks


class SemanticChunker:
    """
    Semantic chunker that splits documents at meaningful boundaries.
    
    Attributes:
        min_tokens: Minimum tokens per chunk.
        max_tokens: Maximum tokens per chunk.
        overlap_ratio: Overlap ratio between adjacent chunks.
        table_processor: TableProcessor instance for table processing.
    """
    
    def __init__(
        self,
        min_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        overlap_ratio: Optional[float] = None,
    ) -> None:
        """
        Initialize semantic chunker.
        
        Args:
            min_tokens: Minimum tokens per chunk.
            max_tokens: Maximum tokens per chunk.
            overlap_ratio: Overlap ratio between chunks.
        """
        self.min_tokens: int = int(min_tokens or settings.CHUNK_MIN_TOKENS)
        self.max_tokens: int = int(max_tokens or settings.CHUNK_MAX_TOKENS)
        self.overlap_ratio: float = float(overlap_ratio or DEFAULT_CHUNK_OVERLAP_PERCENT)
        
        self.boundary_detector = BoundaryDetector()
        self.table_extractor = TableExtractor()
        self.table_processor = TableProcessor()
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from character count."""
        return max(1, len(text) // 4)
    
    def _build_chunks_from_boundaries(
        self,
        text: str,
        boundaries: list[ChunkBoundary],
    ) -> list[str]:
        """
        Build chunks based on detected boundaries.
        
        Args:
            text: Input text.
            boundaries: Detected boundaries.
        
        Returns:
            list[str]: Chunk texts.
        """
        if not text:
            return []
        
        chunks: list[str] = []
        boundary_positions = [b.position for b in boundaries]
        boundary_positions.append(len(text))
        
        current_chunk = ""
        
        for i in range(len(boundary_positions) - 1):
            start_pos = boundary_positions[i]
            end_pos = boundary_positions[i + 1]
            segment = text[start_pos:end_pos]
            
            if not segment.strip():
                continue
            
            current_tokens = self._estimate_tokens(current_chunk)
            segment_tokens = self._estimate_tokens(segment)
            
            if current_tokens + segment_tokens <= self.max_tokens:
                current_chunk += segment
            else:
                if current_chunk:
                    if self._estimate_tokens(current_chunk) >= self.min_tokens:
                        chunks.append(current_chunk)
                    else:
                        # Merge small chunk with previous
                        if chunks:
                            chunks[-1] += current_chunk
                        else:
                            chunks.append(current_chunk)
                
                current_chunk = segment
        
        # Handle final chunk
        if current_chunk and self._estimate_tokens(current_chunk) >= self.min_tokens:
            chunks.append(current_chunk)
        elif current_chunk and chunks:
            chunks[-1] += current_chunk
        
        return chunks
    
    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """
        Apply overlap between adjacent chunks.
        
        Args:
            chunks: List of chunks.
        
        Returns:
            list[str]: Chunks with overlap applied.
        """
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks: list[str] = []
        overlap_tokens = int(self.max_tokens * self.overlap_ratio)
        overlap_chars = overlap_tokens * 4  # Approximate characters
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk
                prev_chunk = chunks[i - 1]
                overlap_start = max(0, len(prev_chunk) - overlap_chars)
                overlap_text = prev_chunk[overlap_start:]
                overlapped_chunks.append(overlap_text + chunk)
        
        return overlapped_chunks
    
    def chunk(self, text: str) -> list[str]:
        """
        Create semantic chunks from text.
        
        Args:
            text: Input text to chunk.
        
        Returns:
            list[str]: List of chunk texts.
        """
        with measure_latency("semantic_chunking") as latency:
            if not text or not text.strip():
                return []
            
            boundaries = self.boundary_detector.find_boundaries(text)
            chunks = self._build_chunks_from_boundaries(text, boundaries)
            chunks = self._apply_overlap(chunks)
            
            latency.stop(
                input_length=len(text),
                chunk_count=len(chunks),
                boundaries=len(boundaries),
            )
            
            return chunks
    
    def chunk_document(self, document: Document) -> list[dict[str, Any]]:
        """
        Create chunks with rich metadata from a document.
        
        Args:
            document: Document to chunk.
        
        Returns:
            list[dict[str, Any]]: Chunks with metadata.
        """
        text = document.raw_text
        
        # Extract tables
        all_tables = self.table_extractor.extract_tables(text)
        
        # Get processed tables from document if available
        processed_tables: list[ProcessedTable] = getattr(document, 'processed_tables', [])
        
        # Create chunks
        chunks = self.chunk(text)
        
        chunks_data: list[dict[str, Any]] = []
        chunk_index = 0
        
        # Create regular chunks
        for chunk in chunks:
            chunks_data.append({
                "content": chunk,
                "page_number": 1,  # Will be refined later
                "ocr": False,
                "has_table": False,
                "source_type": "paragraph",
                "chunk_index": chunk_index,
            })
            chunk_index += 1
        
        # Add processed tables from TableProcessor
        if processed_tables:
            for table in processed_tables:
                # Use the full embedding text as content for better retrieval
                content = table.embedding_text or table.normalized_markdown or table.raw_text
                
                chunks_data.append({
                    "content": content,
                    "page_number": table.page or 1,
                    "ocr": False,
                    "table_id": table.table_id,
                    "headers": table.headers,
                    "rows": table.rows,
                    "row_count": table.row_count,
                    "column_count": table.column_count,
                    "has_table": True,
                    "source_type": "table",
                    "semantic_description": table.semantic_description,
                    "table_type": table.table_type,
                    "keywords": table.keywords,
                    "embedding_text": table.embedding_text,
                    "is_extracted_table": True,
                    "confidence": table.confidence,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1
        
        # Fallback to extracted tables if no processed tables
        if not processed_tables and all_tables:
            for table in all_tables:
                # Generate embedding text for fallback tables
                embedding_text = self._generate_embedding_text_for_table(table)
                
                chunks_data.append({
                    "content": table.text,
                    "page_number": table.page or 1,
                    "ocr": False,
                    "table_id": table.table_id,
                    "headers": table.headers,
                    "rows": table.rows,
                    "row_count": table.row_count,
                    "column_count": table.column_count,
                    "has_table": True,
                    "source_type": "table",
                    "semantic_description": table.semantic_description,
                    "table_type": table.table_type,
                    "keywords": [],
                    "embedding_text": embedding_text,
                    "is_extracted_table": True,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1
        
        # Add legacy tables from document
        for table in document.tables:
            if table.get("has_table", False):
                # Generate embedding text for legacy tables
                headers = table.get("headers", [])
                rows = table.get("rows", [])
                embedding_text = self._generate_embedding_text_from_data(headers, rows)
                
                chunks_data.append({
                    "content": table.get("text", ""),
                    "page_number": table.get("page", 1),
                    "headers": headers,
                    "rows": rows,
                    "row_count": len(rows),
                    "column_count": len(headers),
                    "has_table": True,
                    "source_type": "table",
                    "ocr": False,
                    "embedding_text": embedding_text,
                    "is_extracted_table": True,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1
        
        # Assign page numbers from document pages
        if document.pages:
            for chunk_data in chunks_data:
                if "page_number" not in chunk_data or chunk_data["page_number"] == 1:
                    # Try to find the page for this chunk
                    chunk_text = chunk_data["content"][:200]  # Use first 200 chars
                    for page in document.pages:
                        page_text = page.get("text", "")
                        if chunk_text in page_text:
                            chunk_data["page_number"] = page.get("page_number", 1)
                            break
        
        # Assign chunk indices
        for i, chunk_data in enumerate(chunks_data):
            chunk_data["chunk_index"] = i
        
        return chunks_data
    
    def _generate_embedding_text_for_table(self, table: TableData) -> str:
        """
        Generate embedding text for a table extracted by TableExtractor.
        
        Args:
            table: TableData object.
        
        Returns:
            str: Rich text representation for embedding.
        """
        parts = []
        
        parts.append(f"[TABLE] Table ID: {table.table_id}")
        parts.append(f"Table Type: {table.table_type}")
        parts.append(f"Structure: {table.row_count} rows, {table.column_count} columns")
        
        if table.headers:
            parts.append(f"Columns: {', '.join(h.strip() for h in table.headers)}")
        
        if table.semantic_description:
            parts.append(f"Description: {table.semantic_description}")
        
        # Add all rows for embedding
        if table.rows:
            parts.append("Data:")
            for i, row in enumerate(table.rows, start=1):
                row_text = " | ".join(str(cell).strip() for cell in row)
                parts.append(f"Row {i}: {row_text}")
        
        return "\n".join(parts)
    
    def _generate_embedding_text_from_data(self, headers: list[str], rows: list[list[str]]) -> str:
        """
        Generate embedding text from headers and rows data.
        
        Args:
            headers: Column headers.
            rows: Table rows.
        
        Returns:
            str: Rich text representation for embedding.
        """
        parts = []
        
        parts.append("[TABLE]")
        parts.append(f"Structure: {len(rows)} rows, {len(headers)} columns")
        
        if headers:
            parts.append(f"Columns: {', '.join(h.strip() for h in headers)}")
        
        if rows:
            parts.append("Data:")
            for i, row in enumerate(rows, start=1):
                row_text = " | ".join(str(cell).strip() for cell in row)
                parts.append(f"Row {i}: {row_text}")
        
        return "\n".join(parts)


class HierarchicalChunker(SemanticChunker):
    """
    Hierarchical chunker that preserves document structure.
    
    Extends SemanticChunker to maintain parent-child relationships.
    """
    
    def __init__(
        self,
        min_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        overlap_ratio: Optional[float] = None,
    ) -> None:
        """Initialize hierarchical chunker."""
        super().__init__(min_tokens, max_tokens, overlap_ratio)
        self.hierarchy_extractor = HierarchyExtractor()
    
    def chunk_pages(self, document: Document) -> list[dict[str, Any]]:
        """
        Chunk a document PAGE BY PAGE, preserving accurate page numbers.

        This is the preferred chunking method. It ensures:
        - Every chunk belongs to exactly ONE page (accurate page tracking).
        - No chunk crosses page boundaries.
        - Small pages (< 30 tokens) are skipped.
        - Chunk sizes stay within [min_tokens, max_tokens].

        Returns:
            List of chunk dicts with accurate page_number.
        """
        chunks_data: list[dict[str, Any]] = []
        chunk_index = 0

        # Minimum tokens for a page to be considered
        MIN_PAGE_TOKENS = 30

        for page in document.pages:
            page_num = page.get("page_number")
            page_text = (page.get("text") or "").strip()

            if not page_text:
                continue

            # Skip very small pages (TOC / page numbers only)
            if self._estimate_tokens(page_text) < MIN_PAGE_TOKENS:
                logger.log_ingestion(
                    event=LogEvent.CHUNKING_START,
                    document_id=document.metadata.document_id,
                    document_name=document.metadata.document_name,
                    message=f"Skipping tiny page {page_num} (< {MIN_PAGE_TOKENS} tokens)",
                    details={"page": page_num, "tokens": self._estimate_tokens(page_text)},
                )
                continue

            # Chunk this page (semantic boundaries within the page)
            boundaries = self.boundary_detector.find_boundaries(page_text)
            page_chunks = self._build_chunks_from_boundaries(page_text, boundaries)

            # Apply overlap between chunks in the SAME page only
            page_chunks = self._apply_overlap(page_chunks)

            for chunk_text in page_chunks:
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue

                # Skip chunks that are still too small after chunking
                if self._estimate_tokens(chunk_text) < 5:
                    continue

                chunks_data.append({
                    "content": chunk_text,
                    "page_number": page_num,
                    "ocr": page.get("ocr_applied", False),
                    "has_table": False,
                    "source_type": "paragraph",
                    "chunk_index": chunk_index,
                })
                chunk_index += 1

        logger.log_ingestion(
            event=LogEvent.CHUNKING_COMPLETE,
            document_id=document.metadata.document_id,
            document_name=document.metadata.document_name,
            message=f"Per-page chunking complete: {len(chunks_data)} chunks from {len(document.pages)} pages",
            details={
                "total_pages": len(document.pages),
                "total_chunks": len(chunks_data),
                "avg_chunks_per_page": round(len(chunks_data) / max(1, len(document.pages)), 2),
            },
        )

        return chunks_data

    def chunk_hierarchical_with_tables(self, document: Document) -> list[dict[str, Any]]:
        """
        Preferred method: chunk pages accurately + then process tables.

        Combines:
        - chunk_pages() for accurate page tracking on text chunks.
        - TableProcessor output for structured tables (added at the end).
        """
        # 1. Get per-page text chunks
        text_chunks = self.chunk_pages(document)

        # 2. Get table chunks from processed_tables (if any)
        table_chunks: list[dict[str, Any]] = []
        processed_tables = getattr(document, "processed_tables", []) or []

        for table in processed_tables:
            content = table.embedding_text or table.normalized_markdown or table.raw_text
            table_chunks.append({
                "content": content,
                "page_number": table.page or 1,
                "ocr": False,
                "table_id": table.table_id,
                "headers": table.headers,
                "rows": table.rows,
                "row_count": table.row_count,
                "column_count": table.column_count,
                "has_table": True,
                "source_type": "table",
                "semantic_description": table.semantic_description,
                "table_type": table.table_type,
                "keywords": table.keywords,
                "embedding_text": table.embedding_text,
                "is_extracted_table": True,
                "confidence": table.confidence,
            })

        # 3. Combine (text chunks first, tables after)
        combined = text_chunks + table_chunks

        # 4. Renumber chunk_index
        for i, c in enumerate(combined):
            c["chunk_index"] = i

        logger.log_ingestion(
            event=LogEvent.CHUNKING_COMPLETE,
            document_id=document.metadata.document_id,
            document_name=document.metadata.document_name,
            message=f"Combined chunking: {len(text_chunks)} text + {len(table_chunks)} tables = {len(combined)} total",
            details={
                "text_chunks": len(text_chunks),
                "table_chunks": len(table_chunks),
                "total": len(combined),
            },
        )

        return combined

    def chunk_hierarchical_from_document(self, document: Document) -> list[dict[str, Any]]:
        """
        Create hierarchical chunks with structure preservation from a Document.
        
        Args:
            document: Document to chunk.
        
        Returns:
            list[dict[str, Any]]: Hierarchical chunks with metadata.
        """
        # Step 1: Get chunks with metadata (includes tables)
        chunks_data = self.chunk_document(document)
        
        # Step 2: Extract just the content for hierarchy assignment
        chunk_contents = [c["content"] for c in chunks_data]
        
        # Step 3: Assign hierarchy using the enhanced extractor
        assigned = self.hierarchy_extractor.assign_hierarchy_to_chunks(
            chunk_contents, 
            document.raw_text
        )
        
        # Step 4: Merge hierarchy with existing metadata
        result: list[dict[str, Any]] = []
        parent_ids: dict[int, str] = {}
        
        for i, chunk_data in enumerate(chunks_data):
            # Get hierarchy info for this chunk
            if i < len(assigned):
                hierarchy_info = assigned[i]
                chunk_data["chapter"] = hierarchy_info.get("chapter")
                chunk_data["section"] = hierarchy_info.get("section")
                chunk_data["subsection"] = hierarchy_info.get("subsection")
                chunk_data["title"] = hierarchy_info.get("title")
            
            # Determine hierarchy level
            if chunk_data.get("subsection"):
                level = 3
                parent_level = 2
            elif chunk_data.get("section"):
                level = 2
                parent_level = 1
            elif chunk_data.get("chapter"):
                level = 1
                parent_level = 0
            else:
                level = 4
                parent_level = 3
            
            chunk_data["level"] = level
            
            # Assign parent if exists
            if parent_level in parent_ids:
                chunk_data["parent_chunk_id"] = parent_ids[parent_level]
            
            chunk_id = f"chunk_{i:04d}"
            chunk_data["chunk_id"] = chunk_id
            parent_ids[level] = chunk_id
            
            result.append(chunk_data)
        
        logger.log_ingestion(
            event=LogEvent.CHUNKING_COMPLETE,
            document_id=document.metadata.document_id,
            document_name=document.metadata.document_name,
            message=f"Created {len(result)} hierarchical chunks",
            details={
                "chunk_count": len(result),
                "hierarchy_levels": len(set(c.get("level", 0) for c in result)),
                "table_count": sum(1 for c in result if c.get("has_table", False)),
            },
        )
        
        return result
    
    def chunk_hierarchical(self, text: str, document_metadata: Optional[dict] = None) -> list[dict[str, Any]]:
        """
        Create hierarchical chunks with structure preservation from raw text.
        
        Args:
            text: Raw text to chunk.
            document_metadata: Optional metadata for the document.
        
        Returns:
            list[dict[str, Any]]: Hierarchical chunks with metadata.
        """
        # Create a minimal Document object
        from app.schemas import Document, DocumentMetadata
        
        if document_metadata is None:
            document_metadata = {}
        
        metadata = DocumentMetadata(
            document_id=document_metadata.get("document_id", "doc_temp"),
            document_name=document_metadata.get("document_name", "temp_document"),
            file_path=document_metadata.get("file_path", ""),
            file_size_bytes=document_metadata.get("file_size_bytes", 0),
            file_extension=document_metadata.get("file_extension", "txt"),
            document_hash=document_metadata.get("document_hash", "0" * 64),
        )
        
        # Create a minimal document with raw text
        doc = Document(
            metadata=metadata,
            raw_text=text,
        )
        
        return self.chunk_hierarchical_from_document(doc)