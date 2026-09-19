"""
Document loader module.

This module is responsible for loading documents from various file formats.
It performs validation, format detection, and initial parsing without cleaning,
OCR, chunking, or embedding generation.

Enhanced with table extraction using pdfplumber (primary) and camelot (fallback)
for better table extraction from PDFs, including complex tables with merged cells.
Supports Excel (.xlsx, .xls) with all sheets treated as tables.
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Optional

from app.config.constants import SUPPORTED_EXTENSIONS
from app.config import settings
from app.config.constants import LogEvent, MAX_FILE_SIZE_BYTES
from app.core.exceptions import (
    DocumentNotFoundError,
    DocumentParsingError,
    DocumentValidationError,
)
from app.monitoring import get_logger, measure_latency
from app.schemas import Document, DocumentMetadata, DocumentStructure
from app.ingestion.table_processor import TableProcessor, ProcessedTable


logger = get_logger("food_safety_rag.ingestion.loader")


class DocumentLoader:
    """
    Document loader responsible for loading and validating document files.

    This class handles loading documents from various formats (PDF, DOCX, TXT,
    HTML, MD, XLSX, XLS) and performs initial validation. It does NOT perform
    cleaning, OCR, chunking, or embedding generation - those are handled by
    downstream modules.

    Enhanced with table extraction using pdfplumber (primary) and camelot (fallback)
    for PDF files, and TableProcessor for all table types.
    Enhanced with full Excel workbook support (all sheets extracted as tables).

    Attributes:
        supported_extensions: Set of supported file extensions.
        max_file_size_bytes: Maximum allowed file size.
        table_processor: TableProcessor instance for advanced table handling.
    """

    def __init__(
        self,
        supported_extensions: Optional[set[str]] = None,
        max_file_size_bytes: Optional[int] = None,
    ) -> None:
        """
        Initialize the document loader.

        Args:
            supported_extensions: Set of supported file extensions. Defaults to system constants.
            max_file_size_bytes: Maximum allowed file size. Defaults to system constants.
        """
        self.supported_extensions: set[str] = supported_extensions or SUPPORTED_EXTENSIONS
        self.max_file_size_bytes: int = max_file_size_bytes or MAX_FILE_SIZE_BYTES
        self.table_processor: TableProcessor = TableProcessor()

    def _compute_hash(self, file_path: Path) -> str:
        """
        Compute SHA-256 hash of a file for deduplication.

        Args:
            file_path: Path to the file.

        Returns:
            str: Hexadecimal SHA-256 hash string.
        """
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _validate_file(self, file_path: Path) -> None:
        """
        Validate a file before loading.

        Checks:
        - File exists
        - File is readable
        - File extension is supported
        - File size is within limits
        - File is not empty

        Args:
            file_path: Path to the file to validate.

        Raises:
            DocumentNotFoundError: If the file does not exist or is not readable.
            DocumentValidationError: If validation fails for any reason.
        """
        # Check file exists
        if not file_path.exists():
            logger.log_error(
                event=LogEvent.DOCUMENT_VALIDATION_FAILED,
                message=f"Document not found: {file_path}",
                error_code="DOC_002",
            )
            raise DocumentNotFoundError(str(file_path))

        # Check file is readable
        if not os.access(file_path, os.R_OK):
            raise DocumentValidationError(
                message=f"Document is not readable: {file_path}",
                document_path=str(file_path),
                validation_reason="file_not_readable",
            )

        # Check extension
        file_extension = file_path.suffix.lower()
        if file_extension not in self.supported_extensions:
            raise DocumentValidationError(
                message=f"Unsupported file extension '{file_extension}' for file: {file_path}",
                document_path=str(file_path),
                validation_reason=f"unsupported_extension: {file_extension}",
            )

        # Check file size
        file_size = file_path.stat().st_size
        if file_size == 0:
            raise DocumentValidationError(
                message=f"Document is empty: {file_path}",
                document_path=str(file_path),
                validation_reason="empty_file",
            )

        if file_size > self.max_file_size_bytes:
            raise DocumentValidationError(
                message=f"Document size ({file_size} bytes) exceeds maximum allowed size "
                        f"({self.max_file_size_bytes} bytes): {file_path}",
                document_path=str(file_path),
                validation_reason="file_too_large",
            )

    def _detect_corruption(self, file_path: Path) -> None:
        """
        Detect if a file is corrupted.

        Performs basic corruption checks based on file type.

        Args:
            file_path: Path to the file to check.

        Raises:
            DocumentValidationError: If the file appears corrupted.
        """
        file_size = file_path.stat().st_size
        extension = file_path.suffix.lower()

        # Check for PDF corruption (PDF header)
        if extension == ".pdf":
            with open(file_path, "rb") as f:
                header = f.read(5)
                if header != b"%PDF-":
                    raise DocumentValidationError(
                        message=f"PDF file appears corrupted (invalid header): {file_path}",
                        document_path=str(file_path),
                        validation_reason="corrupted_pdf_header",
                    )

        # Check for DOCX corruption (ZIP header - DOCX is a ZIP archive)
        elif extension == ".docx":
            with open(file_path, "rb") as f:
                header = f.read(4)
                if header != b"PK\x03\x04":
                    raise DocumentValidationError(
                        message=f"DOCX file appears corrupted (invalid ZIP header): {file_path}",
                        document_path=str(file_path),
                        validation_reason="corrupted_docx_header",
                    )

        # Check for XLSX corruption (ZIP header - XLSX is also a ZIP archive)
        elif extension == ".xlsx":
            with open(file_path, "rb") as f:
                header = f.read(4)
                if header != b"PK\x03\x04":
                    raise DocumentValidationError(
                        message=f"XLSX file appears corrupted (invalid ZIP header): {file_path}",
                        document_path=str(file_path),
                        validation_reason="corrupted_xlsx_header",
                    )

        # Check for legacy XLS corruption (OLE2 compound file header)
        elif extension == ".xls":
            with open(file_path, "rb") as f:
                header = f.read(8)
                if header != b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
                    raise DocumentValidationError(
                        message=f"XLS file appears corrupted (invalid OLE2 header): {file_path}",
                        document_path=str(file_path),
                        validation_reason="corrupted_xls_header",
                    )

    def _extract_tables_from_pdf_with_pdfplumber(self, file_path: Path) -> list[ProcessedTable]:
        """
        Extract tables from PDF using pdfplumber (primary method).
        
        Better for complex tables, merged cells, and tables without borders.
        
        Args:
            file_path: Path to the PDF file.
        
        Returns:
            list[ProcessedTable]: List of processed tables.
        """
        processed_tables: list[ProcessedTable] = []
        
        try:
            import pdfplumber
            
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        # Extract tables from page
                        tables = page.extract_tables()
                        
                        if not tables:
                            continue
                        
                        for table_idx, table_data in enumerate(tables, start=1):
                            if not table_data or len(table_data) < 2:
                                continue
                            
                            # Extract headers (first row)
                            headers = []
                            if table_data and len(table_data) > 0:
                                headers = [str(cell).strip() if cell else "" for cell in table_data[0]]
                            
                            # Extract rows (remaining rows)
                            rows = []
                            for row in table_data[1:]:
                                if row and any(cell for cell in row if cell and str(cell).strip()):
                                    cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                                    rows.append(cleaned_row)
                            
                            if not rows:
                                continue
                            
                            # Build raw text representation
                            raw_text = "\n".join([" | ".join(row) for row in [headers] + rows if row])
                            
                            # Process with TableProcessor
                            processed = self.table_processor.process_table_data(
                                raw_headers=headers,
                                raw_rows=rows,
                                page=page_num,
                                raw_text=raw_text,
                            )
                            processed_tables.append(processed)
                            
                    except Exception as exc:
                        logger.log_error(
                            event=LogEvent.WARNING,
                            message=f"Failed to extract tables from page {page_num}: {str(exc)}",
                            exception=exc,
                        )
                        continue
            
            if processed_tables:
                logger.log_ingestion(
                    event=LogEvent.OCR_COMPLETE,
                    document_id="pdf_loader",
                    document_name=file_path.name,
                    message=f"Extracted {len(processed_tables)} tables from PDF using pdfplumber",
                    details={"processed_table_count": len(processed_tables)},
                )
            
            return processed_tables
            
        except ImportError:
            logger.log_ingestion(
                event=LogEvent.WARNING,
                document_id="pdf_loader",
                document_name=file_path.name,
                message="pdfplumber not installed. Falling back to camelot.",
            )
            return []
        except Exception as exc:
            logger.log_error(
                event=LogEvent.WARNING,
                message=f"pdfplumber table extraction failed: {str(exc)}",
                exception=exc,
            )
            return []

    def _extract_tables_from_pdf_with_camelot(self, file_path: Path) -> list[ProcessedTable]:
        """
        Extract tables from PDF using camelot (fallback method).
        
        Better for tables with borders and structured layouts.
        
        Args:
            file_path: Path to the PDF file.
        
        Returns:
            list[ProcessedTable]: List of processed tables.
        """
        processed_tables: list[ProcessedTable] = []
        
        try:
            import camelot

            # Try lattice extraction first (for tables with borders)
            tables = camelot.read_pdf(
                str(file_path),
                pages="all",
                flavor="lattice",
                line_scale=40,
                process_background=True,
            )

            # If no tables found with lattice, try stream (for tables without borders)
            if len(tables) == 0:
                tables = camelot.read_pdf(
                    str(file_path),
                    pages="all",
                    flavor="stream",
                    row_tol=10,
                )

            logger.log_ingestion(
                event=LogEvent.OCR_COMPLETE,
                document_id="pdf_loader",
                document_name=file_path.name,
                message=f"Extracted {len(tables)} raw tables from PDF using camelot",
                details={"raw_table_count": len(tables)},
            )

            # Process each table
            for table in tables:
                try:
                    df = table.df
                    if df is not None and not df.empty:
                        # Extract headers (first row) and data (remaining rows)
                        headers = df.iloc[0].tolist() if len(df) > 0 else []
                        rows = df.iloc[1:].values.tolist() if len(df) > 1 else []
                        raw_text = df.to_string(index=False, header=False)

                        # Process with TableProcessor
                        processed = self.table_processor.process_table_data(
                            raw_headers=headers,
                            raw_rows=rows,
                            page=table.page,
                            raw_text=raw_text,
                        )
                        processed_tables.append(processed)

                except Exception as exc:
                    logger.log_error(
                        event=LogEvent.WARNING,
                        message=f"Failed to process table on page {table.page}: {str(exc)}",
                        exception=exc,
                    )
                    continue

            if processed_tables:
                logger.log_ingestion(
                    event=LogEvent.OCR_COMPLETE,
                    document_id="pdf_loader",
                    document_name=file_path.name,
                    message=f"Successfully processed {len(processed_tables)} tables from PDF using camelot",
                    details={"processed_table_count": len(processed_tables)},
                )

            return processed_tables

        except ImportError:
            logger.log_ingestion(
                event=LogEvent.WARNING,
                document_id="pdf_loader",
                document_name=file_path.name,
                message="camelot-py not installed. Table extraction skipped.",
            )
            return []
        except Exception as exc:
            logger.log_error(
                event=LogEvent.WARNING,
                message=f"camelot table extraction failed: {str(exc)}",
                exception=exc,
            )
            return []

    def _extract_tables_from_pdf(self, file_path: Path) -> list[ProcessedTable]:
        """
        Extract tables from PDF using pdfplumber (primary) with camelot fallback.
        
        Uses pdfplumber first for better complex table support, then camelot as fallback.
        
        Args:
            file_path: Path to the PDF file.
        
        Returns:
            list[ProcessedTable]: List of processed tables.
        """
        # Try pdfplumber first (better for complex tables)
        tables = self._extract_tables_from_pdf_with_pdfplumber(file_path)
        
        # If no tables found, try camelot as fallback
        if not tables:
            tables = self._extract_tables_from_pdf_with_camelot(file_path)
        
        return tables

    def _extract_tables_from_docx(self, file_path: Path) -> list[ProcessedTable]:
        """
        Extract tables from DOCX using python-docx and TableProcessor.

        Args:
            file_path: Path to the DOCX file.

        Returns:
            list[ProcessedTable]: List of processed tables.
        """
        processed_tables: list[ProcessedTable] = []

        try:
            import docx

            doc = docx.Document(file_path)

            for table_idx, table in enumerate(doc.tables):
                try:
                    # Extract headers (first row)
                    headers = []
                    if table.rows and len(table.rows) > 0:
                        headers = [cell.text.strip() for cell in table.rows[0].cells]

                    # Extract data (remaining rows)
                    rows = []
                    for row in table.rows[1:]:
                        row_data = [cell.text.strip() for cell in row.cells]
                        rows.append(row_data)

                    # Build raw text
                    raw_text = "\n".join([" | ".join(row) for row in [headers] + rows if row])

                    processed = self.table_processor.process_table_data(
                        raw_headers=headers,
                        raw_rows=rows,
                        page=1,  # DOCX doesn't have page numbers
                        raw_text=raw_text,
                    )
                    processed_tables.append(processed)

                except Exception as exc:
                    logger.log_error(
                        event=LogEvent.WARNING,
                        message=f"Failed to process DOCX table {table_idx}: {str(exc)}",
                        exception=exc,
                    )
                    continue

            return processed_tables

        except Exception as exc:
            logger.log_error(
                event=LogEvent.WARNING,
                message=f"Failed to extract tables from DOCX: {str(exc)}",
                exception=exc,
            )
            return []

    def _extract_tables_from_excel(self, file_path: Path) -> list[ProcessedTable]:
        """
        Extract tables from an Excel workbook (.xlsx or .xls).

        Every sheet in the workbook is treated as one independent table.
        Uses openpyxl for .xlsx and xlrd (via pandas) for legacy .xls.
        Each sheet is routed through TableProcessor exactly like PDF/DOCX
        tables, so it gets the same classification, keywords, semantic
        description, markdown normalization, and confidence scoring.

        Args:
            file_path: Path to the Excel file.

        Returns:
            list[ProcessedTable]: One ProcessedTable per non-empty sheet.
        """
        processed_tables: list[ProcessedTable] = []
        extension = file_path.suffix.lower()

        try:
            import pandas as pd

            # sheet_name=None returns an ordered dict of {sheet_name: DataFrame}
            engine = "openpyxl" if extension == ".xlsx" else "xlrd"
            all_sheets: dict[str, "pd.DataFrame"] = pd.read_excel(
                file_path,
                sheet_name=None,
                header=None,  # we handle header extraction ourselves
                engine=engine,
            )

            for sheet_index, (sheet_name, df) in enumerate(all_sheets.items(), start=1):
                try:
                    # Drop fully empty rows/columns first
                    df = df.dropna(how="all").dropna(axis=1, how="all")

                    if df is None or df.empty:
                        logger.log_ingestion(
                            event=LogEvent.WARNING,
                            document_id="excel_loader",
                            document_name=file_path.name,
                            message=f"Sheet '{sheet_name}' is empty, skipping",
                        )
                        continue

                    # First remaining row = headers, rest = data rows
                    headers = df.iloc[0].fillna("").astype(str).tolist() if len(df) > 0 else []
                    rows = (
                        df.iloc[1:].fillna("").astype(str).values.tolist()
                        if len(df) > 1
                        else []
                    )

                    raw_text = df.to_string(index=False, header=False)

                    processed = self.table_processor.process_table_data(
                        raw_headers=headers,
                        raw_rows=rows,
                        page=sheet_index,  # sheet order used as "page" number
                        raw_text=raw_text,
                    )
                    # Tag which sheet this table came from for traceability
                    processed.semantic_description = (
                        f"[Sheet: {sheet_name}] {processed.semantic_description}"
                    ).strip()

                    processed_tables.append(processed)

                except Exception as exc:
                    logger.log_error(
                        event=LogEvent.WARNING,
                        message=f"Failed to process Excel sheet '{sheet_name}': {str(exc)}",
                        exception=exc,
                    )
                    continue

            if processed_tables:
                logger.log_ingestion(
                    event=LogEvent.OCR_COMPLETE,
                    document_id="excel_loader",
                    document_name=file_path.name,
                    message=f"Successfully processed {len(processed_tables)} sheet(s) as tables",
                    details={"processed_table_count": len(processed_tables)},
                )

            return processed_tables

        except ImportError as exc:
            missing_lib = "openpyxl" if extension == ".xlsx" else "xlrd"
            logger.log_ingestion(
                event=LogEvent.WARNING,
                document_id="excel_loader",
                document_name=file_path.name,
                message=(
                    f"{missing_lib} not installed. Excel loading failed. "
                    f"Install with: pip install {missing_lib} pandas"
                ),
            )
            raise DocumentParsingError(
                message=f"Missing dependency '{missing_lib}' to read Excel file: {file_path}",
                document_path=str(file_path),
                parser_used="pandas+openpyxl/xlrd",
                original_exception=exc,
            )
        except Exception as exc:
            raise DocumentParsingError(
                message=f"Failed to parse Excel file: {file_path}",
                document_path=str(file_path),
                parser_used="pandas+openpyxl/xlrd",
                original_exception=exc,
            )

    def _load_pdf(self, file_path: Path) -> dict[str, Any]:
        """
        Load and parse a PDF document.

        Enhanced to extract tables using pdfplumber (primary) and camelot (fallback).

        Args:
            file_path: Path to the PDF file.

        Returns:
            dict[str, Any]: Dictionary with parsed content and metadata.

        Raises:
            DocumentParsingError: If parsing fails.
        """
        try:
            import pypdf

            pages: list[dict[str, Any]] = []
            raw_text_parts: list[str] = []
            total_pages = 0
            extracted_tables: list[ProcessedTable] = []

            # Extract text using pypdf
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                total_pages = len(reader.pages)

                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    pages.append({
                        "page_number": page_num,
                        "text": text,
                        "has_text": bool(text.strip()),
                    })
                    if text.strip():
                        raw_text_parts.append(text)

            # Extract tables using pdfplumber (primary) + camelot (fallback)
            try:
                extracted_tables = self._extract_tables_from_pdf(file_path)
            except Exception as exc:
                logger.log_ingestion(
                    event=LogEvent.WARNING,
                    document_id="pdf_loader",
                    document_name=file_path.name,
                    message=f"Failed to extract tables from PDF: {str(exc)}",
                )
                extracted_tables = []

            # Convert ProcessedTable to dict for storage
            table_dicts = [t.to_dict() for t in extracted_tables]

            return {
                "pages": pages,
                "raw_text": "\n\n".join(raw_text_parts),
                "total_pages": total_pages,
                "has_selectable_text": any(p["has_text"] for p in pages),
                "extracted_tables": table_dicts,
                "processed_tables": extracted_tables,
            }

        except Exception as exc:
            raise DocumentParsingError(
                message=f"Failed to parse PDF: {file_path}",
                document_path=str(file_path),
                parser_used="pypdf + pdfplumber/camelot",
                original_exception=exc,
            )

    def _load_docx(self, file_path: Path) -> dict[str, Any]:
        """
        Load and parse a DOCX document.

        Enhanced with table extraction using TableProcessor.

        Args:
            file_path: Path to the DOCX file.

        Returns:
            dict[str, Any]: Dictionary with parsed content and metadata.

        Raises:
            DocumentParsingError: If parsing fails.
        """
        try:
            import docx

            doc = docx.Document(file_path)
            paragraphs: list[str] = []

            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text)

            # Extract tables using TableProcessor
            extracted_tables = self._extract_tables_from_docx(file_path)

            # Convert to dict for storage
            table_dicts = [t.to_dict() for t in extracted_tables]

            raw_text = "\n\n".join(paragraphs)

            # Estimate page count (rough approximation: ~500 words per page)
            word_count = len(raw_text.split())
            estimated_pages = max(1, word_count // 500)

            return {
                "pages": [{"page_number": 1, "text": raw_text, "has_text": True}],
                "raw_text": raw_text,
                "total_pages": estimated_pages,
                "has_selectable_text": True,
                "tables": table_dicts,
                "processed_tables": extracted_tables,
            }

        except Exception as exc:
            raise DocumentParsingError(
                message=f"Failed to parse DOCX: {file_path}",
                document_path=str(file_path),
                parser_used="python-docx",
                original_exception=exc,
            )

    def _load_text(self, file_path: Path) -> dict[str, Any]:
        """
        Load a plain text file.

        Args:
            file_path: Path to the text file.

        Returns:
            dict[str, Any]: Dictionary with parsed content and metadata.

        Raises:
            DocumentParsingError: If reading fails.
        """
        try:
            # Try UTF-8 first, then fallback to other encodings
            content: Optional[str] = None
            encodings = ["utf-8", "utf-16", "iso-8859-1", "cp1252"]

            for encoding in encodings:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                raise DocumentParsingError(
                    message=f"Could not decode text file with any supported encoding: {file_path}",
                    document_path=str(file_path),
                    parser_used="text_reader",
                )

            # Estimate page count (rough approximation: ~3000 chars per page)
            estimated_pages = max(1, len(content) // 3000)

            return {
                "pages": [{"page_number": 1, "text": content, "has_text": True}],
                "raw_text": content,
                "total_pages": estimated_pages,
                "has_selectable_text": True,
                "processed_tables": [],
            }

        except Exception as exc:
            if isinstance(exc, DocumentParsingError):
                raise
            raise DocumentParsingError(
                message=f"Failed to read text file: {file_path}",
                document_path=str(file_path),
                parser_used="text_reader",
                original_exception=exc,
            )

    def _load_html(self, file_path: Path) -> dict[str, Any]:
        """
        Load and parse an HTML document.

        Args:
            file_path: Path to the HTML file.

        Returns:
            dict[str, Any]: Dictionary with parsed content and metadata.

        Raises:
            DocumentParsingError: If parsing fails.
        """
        try:
            from bs4 import BeautifulSoup

            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text(separator="\n", strip=True)

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)

            estimated_pages = max(1, len(text) // 3000)

            return {
                "pages": [{"page_number": 1, "text": text, "has_text": True}],
                "raw_text": text,
                "total_pages": estimated_pages,
                "has_selectable_text": True,
                "processed_tables": [],
            }

        except Exception as exc:
            raise DocumentParsingError(
                message=f"Failed to parse HTML: {file_path}",
                document_path=str(file_path),
                parser_used="beautifulsoup4",
                original_exception=exc,
            )

    def _load_markdown(self, file_path: Path) -> dict[str, Any]:
        """
        Load a Markdown document.

        Args:
            file_path: Path to the Markdown file.

        Returns:
            dict[str, Any]: Dictionary with parsed content and metadata.

        Raises:
            DocumentParsingError: If reading fails.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            estimated_pages = max(1, len(content) // 3000)

            return {
                "pages": [{"page_number": 1, "text": content, "has_text": True}],
                "raw_text": content,
                "total_pages": estimated_pages,
                "has_selectable_text": True,
                "processed_tables": [],
            }

        except Exception as exc:
            raise DocumentParsingError(
                message=f"Failed to read Markdown file: {file_path}",
                document_path=str(file_path),
                parser_used="markdown_reader",
                original_exception=exc,
            )

    def _load_excel(self, file_path: Path) -> dict[str, Any]:
        """
        Load and parse an Excel workbook (.xlsx or .xls).

        Every sheet is extracted as a full table (no truncation, no row
        limit) and routed through TableProcessor so it receives the same
        classification, semantic description, keywords, and markdown
        normalization as PDF/DOCX tables.

        The plain-text representation of all sheets combined becomes the
        document's raw_text, so the chunker/embedding stages can still
        treat the workbook like any other document while the structured
        table data is preserved separately for table-aware retrieval.

        Args:
            file_path: Path to the Excel file (.xlsx or .xls).

        Returns:
            dict[str, Any]: Dictionary with parsed content and metadata.

        Raises:
            DocumentParsingError: If parsing fails or dependencies are missing.
        """
        extracted_tables = self._extract_tables_from_excel(file_path)

        # Build page-like structure: one "page" per sheet
        pages: list[dict[str, Any]] = []
        raw_text_parts: list[str] = []

        for i, table in enumerate(extracted_tables, start=1):
            page_text = table.normalized_markdown or table.raw_text
            pages.append({
                "page_number": i,
                "text": page_text,
                "has_text": bool(page_text.strip()),
            })
            raw_text_parts.append(page_text)

        table_dicts = [t.to_dict() for t in extracted_tables]

        return {
            "pages": pages,
            "raw_text": "\n\n".join(raw_text_parts),
            "total_pages": len(extracted_tables) or 1,
            "has_selectable_text": bool(raw_text_parts),
            "tables": table_dicts,
            "processed_tables": extracted_tables,
        }

    def _parse_document(self, file_path: Path) -> dict[str, Any]:
        """
        Parse a document based on its file extension.

        Args:
            file_path: Path to the document file.

        Returns:
            dict[str, Any]: Parsed document content and metadata.

        Raises:
            DocumentParsingError: If parsing fails.
        """
        extension = file_path.suffix.lower()

        parsers: dict[str, Any] = {
            ".pdf": self._load_pdf,
            ".docx": self._load_docx,
            ".txt": self._load_text,
            ".html": self._load_html,
            ".md": self._load_markdown,
            ".xlsx": self._load_excel,
            ".xls": self._load_excel,
        }

        parser = parsers.get(extension)
        if parser is None:
            raise DocumentParsingError(
                message=f"No parser available for extension: {extension}",
                document_path=str(file_path),
            )

        return parser(file_path)

    def load(self, file_path: str | Path) -> Document:
        """
        Load and validate a document file.

        This is the main entry point for document loading. It validates the file,
        parses its content, and returns a Document object with metadata.

        Args:
            file_path: Path to the document file.

        Returns:
            Document: Loaded document with metadata and content.

        Raises:
            DocumentNotFoundError: If the file does not exist.
            DocumentValidationError: If validation fails.
            DocumentParsingError: If parsing fails.
        """
        path = Path(file_path)

        with measure_latency("document_load") as latency:
            # Validation
            logger.log_ingestion(
                event=LogEvent.DOCUMENT_LOAD_START,
                document_id="pending",
                document_name=path.name,
                message=f"Starting document load: {path.name}",
            )

            self._validate_file(path)
            self._detect_corruption(path)

            # Compute hash
            document_hash = self._compute_hash(path)

            # Parse document
            parsed = self._parse_document(path)

            # Build metadata
            metadata = DocumentMetadata(
                document_id=f"doc_{document_hash[:16]}",
                document_name=path.name,
                file_path=str(path.resolve()),
                file_size_bytes=path.stat().st_size,
                file_extension=path.suffix.lower().lstrip("."),
                total_pages=parsed.get("total_pages"),
                document_hash=document_hash,
                has_tables=bool(parsed.get("processed_tables", [])),
            )

            # Build document structure (minimal at this stage)
            structure = DocumentStructure(
                document_id=metadata.document_id,
                title=path.stem,
            )

            # Build document with extracted tables
            document = Document(
                metadata=metadata,
                structure=structure,
                raw_text=parsed.get("raw_text", ""),
                pages=parsed.get("pages", []),
                tables=parsed.get("tables", []),
            )

            # Add processed tables to document
            processed_tables = parsed.get("processed_tables", [])
            if processed_tables:
                # Store table dicts in document.tables
                document.tables = [t.to_dict() for t in processed_tables]
                # Store processed tables as well (for later use)
                document.processed_tables = processed_tables

            latency.stop(
                document_id=metadata.document_id,
                document_name=metadata.document_name,
                file_size_bytes=metadata.file_size_bytes,
                total_pages=metadata.total_pages,
            )

            logger.log_ingestion(
                event=LogEvent.DOCUMENT_LOAD_COMPLETE,
                document_id=metadata.document_id,
                document_name=metadata.document_name,
                message=f"Document loaded successfully: {metadata.document_name}",
                details={
                    "document_id": metadata.document_id,
                    "total_pages": metadata.total_pages,
                    "has_selectable_text": parsed.get("has_selectable_text", False),
                    "extracted_tables": len(processed_tables),
                    "duration_ms": latency.duration_ms,
                },
            )

            return document

    def load_batch(self, file_paths: list[str | Path]) -> list[Document]:
        """
        Load multiple documents in batch.

        Continues processing remaining documents even if one fails.

        Args:
            file_paths: List of paths to document files.

        Returns:
            list[Document]: List of successfully loaded documents.
        """
        documents: list[Document] = []

        for file_path in file_paths:
            try:
                document = self.load(file_path)
                documents.append(document)
            except (DocumentNotFoundError, DocumentValidationError, DocumentParsingError) as exc:
                logger.log_error(
                    event=LogEvent.DOCUMENT_VALIDATION_FAILED,
                    message=f"Failed to load document: {file_path}",
                    exception=exc,
                    error_code=exc.error_code if hasattr(exc, "error_code") else "DOC_001",
                )
                # Continue with next document

        return documents