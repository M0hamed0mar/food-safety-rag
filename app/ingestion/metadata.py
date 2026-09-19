"""
Metadata generation module.

This module generates rich, structured metadata for document chunks.
Metadata quality is critical for retrieval, citation, evaluation, and future filtering.

Uses TableProcessor integration for advanced table metadata.
Ensures embedding_text is properly stored for table chunks.
"""

import hashlib
import json
import re
import uuid
from collections import Counter
from datetime import datetime
from typing import Any, Optional

from app.config import settings
from app.config.constants import LogEvent
from app.monitoring import get_logger, measure_latency
from app.schemas import Chunk, ChunkMetadata, Document


logger = get_logger("food_safety_rag.ingestion.metadata")


class MetadataGenerator:
    """
    Generator for rich chunk metadata.
    
    Attributes:
        embedding_model: Name of the embedding model used.
        stop_words: Cached set of stop words for keyword extraction.
        domain_patterns: Cached domain patterns for semantic tagging.
    """
    
    # =============================================================================
    # Class-level constants - computed once for efficiency
    # =============================================================================
    
    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall",
        "can", "need", "dare", "ought", "used", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into",
        "through", "during", "before", "after", "above", "below",
        "between", "under", "and", "but", "or", "yet", "so",
        "if", "because", "although", "though", "while", "where",
        "when", "that", "which", "who", "whom", "whose", "what",
        "this", "these", "those", "i", "you", "he", "she", "it",
        "we", "they", "me", "him", "her", "us", "them", "my",
        "your", "his", "its", "our", "their", "mine", "yours",
        "hers", "ours", "theirs", "all", "each", "every", "both",
        "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "than", "too", "very",
        "just", "also", "now", "then", "here", "there", "up", "down",
        "out", "off", "over", "again", "further", "once", "upon",
    }
    
    # -----------------------------------------------------------------------------
    
    # Domain patterns (EMPTY BY DEFAULT).
    
    #
    
    # This is intentionally empty so the system stays DOMAIN-AGNOSTIC.
    
    # If you want to add domain-specific semantic tagging, populate this dict:
    
    #
    
    #   DOMAIN_PATTERNS = {
    
    #       "payment_terms": ["net 30", "net 60", "invoice"],
    
    #       "warranty": ["warranty", "guarantee", "defect"],
    
    #       ... etc.
    
    #   }
    
    #
    
    # The system will extract these tags automatically and attach them to chunks.
    
    # -----------------------------------------------------------------------------
    
    DOMAIN_PATTERNS: dict[str, list[str]] = {}

    
    # =============================================================================
    # INITIALIZATION
    # =============================================================================
    
    def __init__(self, embedding_model: Optional[str] = None) -> None:
        """
        Initialize the metadata generator.
        
        Args:
            embedding_model: Name of the embedding model. Defaults to settings.
        """
        self.embedding_model: str = embedding_model or settings.EMBEDDING_MODEL
    
    # =============================================================================
    # CHUNK ID GENERATION
    # =============================================================================
    
    def _generate_chunk_id(
        self,
        document_id: str,
        chunk_index: int,
        content: str,
    ) -> str:
        """
        Generate a deterministic, unique chunk identifier.
        
        The ID is based on document ID, chunk index, and content hash
        to ensure uniqueness and determinism.
        
        Args:
            document_id: Parent document ID.
            chunk_index: Index of the chunk within the document.
            content: Chunk content for hashing.
        
        Returns:
            str: Unique chunk identifier.
        """
        formatted_index = f"{chunk_index:04d}" if chunk_index is not None else "0000"
        # Use first 100 chars for hash - more efficient
        hash_input = f"{document_id}:{formatted_index}:{content[:100]}"
        content_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
        return f"chunk_{document_id}_{formatted_index}_{content_hash}"
    
    # =============================================================================
    # KEYWORD EXTRACTION
    # =============================================================================
    
    def _extract_keywords(self, content: str, max_keywords: int = 10) -> list[str]:
        """
        Extract key terms from chunk content using Counter for efficiency.
        
        Args:
            content: Chunk text content.
            max_keywords: Maximum number of keywords to extract.
        
        Returns:
            list[str]: List of extracted keywords.
        """
        words = re.findall(r"\b[a-zA-Z]{3,}\b", content.lower())
        
        # Use Counter for efficient frequency counting
        word_counts = Counter(word for word in words if word not in self.STOP_WORDS)
        
        # Most common returns a list of tuples (word, count)
        return [word for word, _ in word_counts.most_common(max_keywords)]
    
    # =============================================================================
    # SEMANTIC TAGS EXTRACTION
    # =============================================================================
    
    def _extract_semantic_tags(self, content: str) -> list[str]:
        """
        Extract semantic tags based on domain-specific patterns.
        
        Identifies Food Safety domain concepts in the content.
        
        Args:
            content: Chunk text content.
        
        Returns:
            list[str]: List of semantic tags (deduplicated).
        """
        tags: set[str] = set()
        content_lower = content.lower()
        
        # Iterate through patterns and check for matches
        for tag_name, patterns in self.DOMAIN_PATTERNS.items():
            for pattern in patterns:
                if pattern in content_lower:
                    tags.add(tag_name)
                    break  # Found this tag, move to next tag
        
        return list(tags)
    
    # =============================================================================
    # SOURCE TYPE DETECTION
    # =============================================================================
    
    def _detect_source_type(self, content: str) -> str:
        """
        Detect the source type of content based on characteristics.
        
        Args:
            content: Chunk text content.
        
        Returns:
            str: Source type (text, code, formula, etc.).
        """
        content_lower = content.lower()
        
        # Check for code blocks
        if any(lang_indicator in content for lang_indicator in ["```", "def ", "class ", "function"]):
            return "code"
        
        # Check for formulas (math expressions)
        if re.search(r"[+\-*/=()]{5,}", content):
            return "formula"
        
        # Check for lists
        if re.search(r"^\s*[\*\-\+]\s", content, re.MULTILINE):
            return "list"
        
        # Check for headings
        if re.search(r"^(#{1,6}\s|Chapter|Section|الفصل|القسم)", content, re.MULTILINE):
            return "heading"
        
        # Short snippets
        if len(content) < 50:
            return "snippet"
        
        return "text"
    
    # =============================================================================
    # TOKEN COUNT ESTIMATION
    # =============================================================================
    
    def _estimate_token_count(self, content: str) -> int:
        """
        Estimate token count for content (rough approximation).
        
        Args:
            content: Chunk text content.
        
        Returns:
            int: Estimated token count.
        """
        return max(1, len(content.split()) // 4)
    
    # =============================================================================
    # TABLE KEYWORD EXTRACTION
    # =============================================================================
    
    def _extract_table_keywords_for_tags(
        self, headers: list[str], rows: list[list[str]]
    ) -> list[str]:
        """
        Extract keywords from table headers and rows for semantic tagging.
        
        Args:
            headers: Table column headers.
            rows: Table rows containing data.
        
        Returns:
            list[str]: Keywords extracted from table.
        """
        table_content = " ".join(headers)
        # Add sample row data
        if rows:
            table_content += " " + " ".join(str(cell) for row in rows[:3] for cell in row)
        
        return self._extract_keywords(table_content, max_keywords=5)
    
    # =============================================================================
    # TABLE SEMANTIC DESCRIPTION GENERATION
    # =============================================================================
    
    def _generate_table_semantic_description(self, chunk_data: dict[str, Any]) -> str:
        """
        Generate semantic description for table based on its content.
        
        Args:
            chunk_data: Dictionary containing table data.
        
        Returns:
            str: Generated semantic description.
        """
        headers = chunk_data.get("headers", [])
        row_count = chunk_data.get("row_count", 0)
        column_count = chunk_data.get("column_count", 0)
        table_type = chunk_data.get("table_type", "general_table")
        
        if not headers:
            return f"A {table_type} with {row_count} rows and {column_count} columns"
        
        # Limit headers for readability
        headers_str = ", ".join(headers[:5])
        if len(headers) > 5:
            headers_str += f", and {len(headers) - 5} more columns"
        
        # Generate description based on table type
        type_descriptions = {
            "hazard_table": "identifies and categorizes food safety hazards",
            "allergen_table": "lists priority allergens and their sources",
            "process_table": "describes processing steps and parameters",
            "temperature_table": "specifies temperature requirements",
            "pathogen_table": "details pathogen characteristics and control",
            "nutrition_table": "provides nutritional composition data",
            "additive_table": "lists food additives and their functions",
            "control_table": "describes control measures and procedures",
        }
        
        purpose = type_descriptions.get(table_type, "contains structured data")
        
        return f"A {table_type} that {purpose}. Contains {row_count} rows and {column_count} columns: {headers_str}."
    
    # =============================================================================
    # MAIN GENERATION METHOD
    # =============================================================================
    
    def generate(
        self,
        content: str,
        document: Document,
        chunk_index: Optional[int] = None,
        total_chunks: Optional[int] = None,
        page_number: Optional[int] = None,
        chapter: Optional[str] = None,
        section: Optional[str] = None,
        subsection: Optional[str] = None,
        title: Optional[str] = None,
        ocr: bool = False,
        parent_chunk_id: Optional[str] = None,
        child_chunk_ids: Optional[list[str]] = None,
        table_id: Optional[str] = None,
        figure_id: Optional[str] = None,
        headers: Optional[list[str]] = None,
        rows: Optional[list[list[str]]] = None,
        row_count: Optional[int] = None,
        column_count: Optional[int] = None,
        has_table: bool = False,
        semantic_description: Optional[str] = None,
        table_type: Optional[str] = None,
        keywords: Optional[list[str]] = None,
        embedding_text: Optional[str] = None,
        is_extracted_table: bool = False,
        confidence: float = 1.0,
    ) -> ChunkMetadata:
        """
        Generate metadata for a single chunk.
        
        Ensures embedding_text is properly stored for table chunks.
        
        Args:
            content: Chunk text content.
            document: Parent document.
            chunk_index: Index of chunk within document.
            total_chunks: Total number of chunks in document.
            page_number: Page number where chunk appears.
            chapter: Chapter information.
            section: Section information.
            subsection: Subsection information.
            title: Title of the chunk.
            ocr: Whether content was OCR'd.
            parent_chunk_id: Parent chunk ID for hierarchical chunks.
            child_chunk_ids: List of child chunk IDs.
            table_id: ID of associated table.
            figure_id: ID of associated figure.
            headers: Table headers (if applicable).
            rows: Table rows (if applicable).
            row_count: Number of rows.
            column_count: Number of columns.
            has_table: Whether chunk contains table.
            semantic_description: Human-readable table description.
            table_type: Type of table.
            keywords: Pre-extracted keywords.
            embedding_text: Text for embedding generation.
            is_extracted_table: Whether from TableProcessor.
            confidence: Confidence score (0-1).
        
        Returns:
            ChunkMetadata: Generated metadata object.
        """
        with measure_latency("metadata_generation") as latency:
            # Normalize defaults
            chunk_index = chunk_index or 0
            total_chunks = total_chunks or 1
            
            # Generate unique chunk ID
            chunk_id = self._generate_chunk_id(
                document.metadata.document_id,
                chunk_index,
                content,
            )
            
            # Detect source type
            source_type = "table" if has_table else self._detect_source_type(content)
            
            # Extract keywords if not provided
            keywords = keywords or self._extract_keywords(content)
            
            # Extract semantic tags
            semantic_tags = self._extract_semantic_tags(content)
            
            # Add table keywords to semantic tags for better retrieval
            if has_table and headers and rows:
                table_keywords = self._extract_table_keywords_for_tags(headers, rows)
                semantic_tags.extend(table_keywords)
                # Deduplicate efficiently - preserves order
                semantic_tags = list(dict.fromkeys(semantic_tags))
            
            token_count = self._estimate_token_count(content)
            
            # Build table data dict only if table exists
            table_data: Optional[str] = None
            if has_table and headers is not None and rows is not None:
                table_data_dict = {
                    "headers": headers,
                    "rows": rows,
                    "row_count": row_count or len(rows),
                    "column_count": column_count or len(headers),
                    "semantic_description": semantic_description or "",
                    "table_type": table_type or "general_table",
                    "keywords": keywords,
                    "is_extracted_table": is_extracted_table,
                    "confidence": confidence,
                }
                # Only add embedding_text if provided
                if embedding_text:
                    table_data_dict["embedding_text"] = embedding_text
                
                table_data = json.dumps(table_data_dict, ensure_ascii=False)
            
            # Create metadata object
            metadata = ChunkMetadata(
                document_id=document.metadata.document_id,
                document_name=document.metadata.document_name,
                page=page_number,
                chapter=chapter,
                section=section,
                subsection=subsection,
                title=title,
                chunk_id=chunk_id,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                language=document.metadata.language,
                ocr=ocr,
                source_type=source_type,
                table_id=table_id,
                figure_id=figure_id,
                token_count=token_count,
                embedding_model=self.embedding_model,
                parent_chunk_id=parent_chunk_id,
                child_chunk_ids=child_chunk_ids or [],
                semantic_tags=semantic_tags,
                keywords=keywords,
                confidence_score=confidence,
                table_data=table_data,
            )
            
            # Store embedding_text for table chunks
            if is_extracted_table or has_table:
                if embedding_text:
                    metadata.embedding_text = embedding_text
                elif content:
                    # Fallback: use content as embedding text
                    metadata.embedding_text = content
                
                if table_type:
                    metadata.table_type = table_type
                metadata.is_extracted_table = is_extracted_table
            
            # Log latency and details
            latency.stop(
                chunk_id=chunk_id,
                chunk_index=chunk_index,
                source_type=source_type,
                keyword_count=len(keywords),
                semantic_tag_count=len(semantic_tags),
                has_table=has_table,
                is_extracted_table=is_extracted_table,
                has_embedding_text=bool(embedding_text),
            )
            
            logger.log_ingestion(
                event=LogEvent.METADATA_GENERATION,
                document_id=document.metadata.document_id,
                document_name=document.metadata.document_name,
                message=f"Generated metadata for chunk {chunk_index}",
                details={
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "source_type": source_type,
                    "token_count": token_count,
                    "keywords": keywords[:5],
                    "semantic_tags": semantic_tags[:5],
                    "has_table": has_table,
                    "table_type": table_type,
                    "is_extracted_table": is_extracted_table,
                    "has_embedding_text": bool(embedding_text),
                    "duration_ms": latency.duration_ms,
                },
            )
            
            return metadata
    
    # =============================================================================
    # BATCH GENERATION
    # =============================================================================
    
    def generate_batch(
        self,
        chunks_data: list[dict[str, Any]],
        document: Document,
    ) -> list[ChunkMetadata]:
        """
        Generate metadata for multiple chunks in batch.
        
        Ensures embedding_text is properly stored for all table chunks.
        
        Args:
            chunks_data: List of dictionaries containing chunk data.
                Each dict should have 'content', 'chunk_index', and optional fields.
            document: Parent document.
        
        Returns:
            list[ChunkMetadata]: List of generated metadata objects.
        """
        total_chunks = len(chunks_data)
        metadata_list: list[ChunkMetadata] = []
        
        for i, chunk_data in enumerate(chunks_data):
            # Get chunk_index with fallback to position
            chunk_index = chunk_data.get("chunk_index") or i
            
            # Extract table-related fields with defaults
            has_table = chunk_data.get("has_table", False)
            table_type = chunk_data.get("table_type", "general_table")
            is_extracted_table = chunk_data.get("is_extracted_table", False)
            confidence = chunk_data.get("confidence", 1.0)
            
            # Extract embedding_text - CRITICAL for table chunks
            embedding_text = chunk_data.get("embedding_text")
            
            # If embedding_text is missing but it's a table, generate from data
            if has_table and not embedding_text:
                headers = chunk_data.get("headers", [])
                rows = chunk_data.get("rows", [])
                embedding_text = self._generate_embedding_text_from_data(headers, rows)
            
            # Generate semantic description for tables if not provided
            semantic_description = chunk_data.get("semantic_description")
            if has_table and not semantic_description:
                semantic_description = self._generate_table_semantic_description(chunk_data)
            
            # Generate single metadata
            metadata = self.generate(
                content=chunk_data["content"],
                document=document,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                page_number=chunk_data.get("page_number"),
                chapter=chunk_data.get("chapter"),
                section=chunk_data.get("section"),
                subsection=chunk_data.get("subsection"),
                title=chunk_data.get("title"),
                ocr=chunk_data.get("ocr", False),
                parent_chunk_id=chunk_data.get("parent_chunk_id"),
                child_chunk_ids=chunk_data.get("child_chunk_ids"),
                table_id=chunk_data.get("table_id"),
                figure_id=chunk_data.get("figure_id"),
                headers=chunk_data.get("headers"),
                rows=chunk_data.get("rows"),
                row_count=chunk_data.get("row_count"),
                column_count=chunk_data.get("column_count"),
                has_table=has_table,
                semantic_description=semantic_description,
                table_type=table_type,
                keywords=chunk_data.get("keywords"),
                embedding_text=embedding_text,
                is_extracted_table=is_extracted_table,
                confidence=confidence,
            )
            metadata_list.append(metadata)
        
        # Log batch completion
        logger.log_ingestion(
            event=LogEvent.METADATA_GENERATION,
            document_id=document.metadata.document_id,
            document_name=document.metadata.document_name,
            message=f"Generated metadata for {len(metadata_list)} chunks in batch",
            details={
                "chunk_count": len(metadata_list),
                "table_count": sum(1 for m in metadata_list if m.table_data is not None),
                "extracted_table_count": sum(1 for m in metadata_list if getattr(m, 'is_extracted_table', False)),
                "embedding_text_count": sum(1 for m in metadata_list if getattr(m, 'embedding_text', None) is not None),
            },
        )
        
        return metadata_list
    
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