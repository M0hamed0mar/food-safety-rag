"""
Text cleaning module for RAG ingestion pipeline.

Handles minimal but essential text normalization:
- Unicode normalization for consistent character representation
- Removal of invisible/control characters that interfere with processing
- Whitespace normalization while preserving document structure
- Repair of line-breaking artifacts from PDF extraction
"""

import re
import unicodedata
from typing import Optional

from app.config.constants import LogEvent, UNICODE_NORMALIZATION_FORM
from app.monitoring import get_logger, measure_latency


logger = get_logger("food_safety_rag.ingestion.cleaner")


class TextCleaner:
    """
    Ultra-minimal text cleaner preserving document content integrity.
    
    Cleaning strategy:
    - Normalize Unicode to consistent form (NFKC)
    - Remove only invisible/control characters (zero-width, soft hyphens, etc.)
    - Light whitespace normalization (collapse spaces, preserve paragraphs)
    - Fix line breaks from PDF extraction (hyphenated words)
    
    Preserves:
    - All semantic content (headers, footers, page numbers)
    - Document structure and formatting hints
    - Original punctuation and dash variants
    """
    
    # =============================================================================
    # CLASS CONSTANTS
    # =============================================================================
    
    # Unicode characters to explicitly remove
    INVISIBLE_CHARS = {
        "\u200b",  # Zero Width Space
        "\u200c",  # Zero Width Non-Joiner
        "\u200d",  # Zero Width Joiner
        "\u2060",  # Word Joiner
        "\ufeff",  # Zero Width No-Break Space (BOM)
        "\u00ad",  # Soft Hyphen
    }
    
    # Control character categories to strip (except newlines, tabs)
    CONTROL_CATEGORIES = {"Cc", "Cf"}
    PRESERVE_CONTROLS = {"\n", "\r", "\t"}
    
    # Regex for hyphenated line breaks
    HYPHENATED_LINE_PATTERN = re.compile(r"(\w+)-\n(\w+)")
    
    # Regex for whitespace normalization
    MULTIPLE_SPACES_PATTERN = re.compile(r" +")
    MULTIPLE_NEWLINES_PATTERN = re.compile(r"\n{3,}")
    
    # Characters to normalize to space
    WHITESPACE_CHARS = re.compile(r"[\xa0\u2000-\u200f\u2028\u2029]")
    
    def __init__(self, normalization_form: Optional[str] = None) -> None:
        """
        Initialize text cleaner.
        
        Args:
            normalization_form: Unicode normalization form (default: NFKC).
        """
        self.normalization_form = normalization_form or UNICODE_NORMALIZATION_FORM
    
    # =============================================================================
    # MAIN CLEANING PIPELINE
    # =============================================================================
    
    def clean(self, text: str) -> str:
        """
        Apply ultra-minimal cleaning pipeline.
        
        Args:
            text: Raw extracted text.
        
        Returns:
            Normalized text with invisible characters and control chars removed.
        """
        if not text or not text.strip():
            logger.log_ingestion(
                event=LogEvent.CLEANING_COMPLETE,
                document_id="in_progress",
                document_name="cleaning",
                message="Skipped empty text",
            )
            return ""
        
        original_length = len(text)
        
        with measure_latency("text_cleaning") as latency:
            logger.log_ingestion(
                event=LogEvent.CLEANING_START,
                document_id="in_progress",
                document_name="cleaning",
                message="Starting text cleaning",
                details={"input_length": original_length},
            )
            
            # Step 1: Unicode normalization
            text = self._normalize_unicode(text)
            
            # Step 2: Remove invisible characters
            text = self._remove_invisible_characters(text)
            
            # Step 3: Normalize whitespace while preserving structure
            text = self._normalize_whitespace(text)
            
            # Step 4: Fix line breaks from PDF extraction
            text = self._fix_hyphenated_lines(text)
            
            cleaned_length = len(text)
            reduction_percent = (
                (original_length - cleaned_length) / original_length * 100
                if original_length > 0
                else 0
            )
            
            latency.stop(
                original_length=original_length,
                cleaned_length=cleaned_length,
                reduction_percent=round(reduction_percent, 2),
            )
            
            logger.log_ingestion(
                event=LogEvent.CLEANING_COMPLETE,
                document_id="in_progress",
                document_name="cleaning",
                message="Text cleaning complete",
                details={
                    "original_length": original_length,
                    "cleaned_length": cleaned_length,
                    "reduction_percent": round(reduction_percent, 2),
                    "duration_ms": latency.duration_ms,
                },
            )
            
            return text
    
    def clean_page(self, page_text: str, page_number: int) -> str:
        """
        Clean text from a single page.
        
        Args:
            page_text: Text extracted from page.
            page_number: Page number for logging.
        
        Returns:
            Cleaned page text.
        """
        cleaned = self.clean(page_text)
        
        logger.log_ingestion(
            event=LogEvent.CLEANING_COMPLETE,
            document_id="in_progress",
            document_name="page_cleaning",
            message=f"Page {page_number} cleaned",
            details={"page_number": page_number, "cleaned_length": len(cleaned)},
        )
        
        return cleaned
    
    # =============================================================================
    # CLEANING STEPS
    # =============================================================================
    
    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters to consistent form.
        
        Args:
            text: Input text.
        
        Returns:
            Unicode-normalized text.
        """
        return unicodedata.normalize(self.normalization_form, text)
    
    def _remove_invisible_characters(self, text: str) -> str:
        """
        Remove zero-width and control characters.
        
        Strips invisible Unicode characters and control chars
        while preserving newlines/tabs needed for structure.
        
        Args:
            text: Input text.
        
        Returns:
            Text with invisible characters removed.
        """
        # Remove explicit invisible characters
        for char in self.INVISIBLE_CHARS:
            text = text.replace(char, "")
        
        # Remove control characters (Cc, Cf) except preserved ones
        filtered = []
        for char in text:
            category = unicodedata.category(char)
            if category not in self.CONTROL_CATEGORIES or char in self.PRESERVE_CONTROLS:
                filtered.append(char)
        
        return "".join(filtered)
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace while preserving document structure.
        
        - Replace tabs/non-breaking spaces with regular spaces
        - Collapse multiple spaces to single space
        - Preserve paragraph breaks (3+ newlines → 2 newlines)
        - Strip trailing whitespace from each line
        
        Args:
            text: Input text.
        
        Returns:
            Text with normalized whitespace.
        """
        # Replace tabs with regular space
        text = text.replace("\t", " ")
        
        # Replace non-breaking spaces and other whitespace variants
        text = self.WHITESPACE_CHARS.sub(" ", text)
        
        # Collapse multiple spaces (but preserve newlines)
        text = self.MULTIPLE_SPACES_PATTERN.sub(" ", text)
        
        # Preserve paragraph structure (3+ newlines → 2)
        text = self.MULTIPLE_NEWLINES_PATTERN.sub("\n\n", text)
        
        # Strip trailing whitespace per line
        lines = [line.rstrip() for line in text.split("\n")]
        
        return "\n".join(lines)
    
    def _fix_hyphenated_lines(self, text: str) -> str:
        """
        Fix word breaks from PDF extraction.
        
        Joins hyphenated words split across lines:
        "word-\nnext" → "wordnext"
        
        Args:
            text: Input text.
        
        Returns:
            Text with hyphenated line breaks fixed.
        """
        return self.HYPHENATED_LINE_PATTERN.sub(r"\1\2", text)