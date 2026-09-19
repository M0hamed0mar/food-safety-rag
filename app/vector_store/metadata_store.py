"""
Metadata store module.

This module provides persistent storage for document and chunk metadata
separate from the vector index. It enables efficient metadata filtering,
document management, and index operations without loading the full vector store.

Stores full_content for complete chunk text to support BM25, Reranker,
and Context Builder with full table data.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from app.config import settings
from app.config.constants import LogEvent
from app.core.exceptions import VectorStoreError
from app.monitoring import get_logger, measure_latency
from app.schemas import Chunk, DocumentMetadata


logger = get_logger("food_safety_rag.vector_store.metadata")


class MetadataStore:
    """
    SQLite-based metadata store for documents and chunks.
    
    This store maintains metadata separately from the vector index to enable:
    - Efficient metadata filtering
    - Document listing and management
    - Chunk lookup by ID
    - Duplicate detection
    - Incremental ingestion tracking
    
    Stores full_content for complete chunk text to support all retrieval components.
    
    Attributes:
        db_path: Path to the SQLite database file.
        connection: SQLite database connection.
    """
    
    def __init__(self, db_path: Optional[Path] = None) -> None:
        """
        Initialize the metadata store.
        
        Args:
            db_path: Path to the SQLite database. Defaults to settings.
        """
        self.db_path: Path = db_path or (settings.FAISS_INDEX_DIR / "metadata.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[sqlite3.Connection] = None
        
        self._initialize_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get or create the SQLite connection.
        
        Returns:
            sqlite3.Connection: Database connection.
        """
        if self._connection is None:
            self._connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
        return self._connection
    
    def _initialize_database(self) -> None:
        """
        Create database tables if they don't exist.
        
        Ensures full_content column exists for complete chunk text.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                document_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size_bytes INTEGER,
                file_extension TEXT,
                document_type TEXT,
                title TEXT,
                author TEXT,
                language TEXT,
                total_pages INTEGER,
                creation_date TEXT,
                ingestion_date TEXT NOT NULL,
                document_hash TEXT NOT NULL UNIQUE,
                ocr_required INTEGER DEFAULT 0,
                has_tables INTEGER DEFAULT 0,
                has_figures INTEGER DEFAULT 0,
                tags TEXT,
                custom_metadata TEXT
            )
        """)
        
        # Chunks table with full_content
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                document_name TEXT NOT NULL,
                page INTEGER,
                chapter TEXT,
                section TEXT,
                subsection TEXT,
                title TEXT,
                chunk_index INTEGER NOT NULL,
                total_chunks INTEGER NOT NULL,
                language TEXT,
                ocr INTEGER DEFAULT 0,
                source_type TEXT,
                table_id TEXT,
                figure_id TEXT,
                token_count INTEGER,
                embedding_model TEXT,
                parent_chunk_id TEXT,
                child_chunk_ids TEXT,
                semantic_tags TEXT,
                keywords TEXT,
                confidence_score REAL,
                extraction_timestamp TEXT,
                content_preview TEXT,
                full_content TEXT,
                faiss_index_id INTEGER,
                deleted INTEGER DEFAULT 0,
                FOREIGN KEY (document_id) REFERENCES documents(document_id)
            )
        """)
        
        # Ensure full_content column exists
        cursor.execute("PRAGMA table_info(chunks)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'full_content' not in columns:
            logger.log_event(
                event=LogEvent.SYSTEM_STARTUP,
                message="Adding full_content column to chunks table",
            )
            cursor.execute("ALTER TABLE chunks ADD COLUMN full_content TEXT")
        
        # Indexes for efficient queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_faiss ON chunks(faiss_index_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(document_hash)")
        
        conn.commit()
        
        logger.log_event(
            event=LogEvent.SYSTEM_STARTUP,
            message="Metadata store initialized",
            details={
                "db_path": str(self.db_path),
                "has_full_content_column": 'full_content' in columns,
            },
        )
    
    def add_document(self, metadata: DocumentMetadata) -> None:
        """
        Add or update a document in the metadata store.
        
        Args:
            metadata: Document metadata to store.
        
        Raises:
            VectorStoreError: If database operation fails.
        """
        with measure_latency("metadata_store_add_document") as latency:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO documents (
                        document_id, document_name, file_path, file_size_bytes,
                        file_extension, document_type, title, author, language,
                        total_pages, creation_date, ingestion_date, document_hash,
                        ocr_required, has_tables, has_figures, tags, custom_metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.document_id,
                    metadata.document_name,
                    metadata.file_path,
                    metadata.file_size_bytes,
                    metadata.file_extension,
                    metadata.document_type,
                    metadata.title,
                    metadata.author,
                    metadata.language,
                    metadata.total_pages,
                    metadata.creation_date.isoformat() if metadata.creation_date else None,
                    metadata.ingestion_date.isoformat(),
                    metadata.document_hash,
                    int(metadata.ocr_required),
                    int(metadata.has_tables),
                    int(metadata.has_figures),
                    json.dumps(metadata.tags),
                    json.dumps(metadata.custom_metadata),
                ))
                
                conn.commit()
                
                latency.stop(document_id=metadata.document_id)
                
                logger.log_ingestion(
                    event=LogEvent.INDEX_UPDATE,
                    document_id=metadata.document_id,
                    document_name=metadata.document_name,
                    message=f"Document metadata stored: {metadata.document_name}",
                )
                
            except Exception as exc:
                raise VectorStoreError(
                    message=f"Failed to store document metadata: {str(exc)}",
                    index_name=str(self.db_path),
                    operation="add_document",
                    original_exception=exc,
                )
    
    def add_chunks(self, chunks: list[Chunk]) -> None:
        """
        Add or update chunks in the metadata store.
        
        Stores both content_preview (for display) and full_content (for retrieval).
        Ensures full_content is always populated with complete chunk text.
        
        Args:
            chunks: List of chunks to store.
        
        Raises:
            VectorStoreError: If database operation fails.
        """
        with measure_latency("metadata_store_add_chunks") as latency:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                for chunk in chunks:
                    meta = chunk.metadata
                    
                    # Ensure content is populated
                    content = chunk.content if chunk.content else ""
                    
                    # For table chunks, use embedding_text if available and larger
                    if hasattr(meta, 'embedding_text') and meta.embedding_text:
                        embedding_text = meta.embedding_text
                        # Use embedding_text as content if it's more comprehensive
                        if len(embedding_text) > len(content):
                            content = embedding_text
                    
                    # If content is still empty, try to get from metadata
                    if not content and meta and hasattr(meta, 'content_preview') and meta.content_preview:
                        content = meta.content_preview
                        chunk.content = content
                    
                    content_preview = content[:500] if content else ""
                    full_content = content if content else ""
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO chunks (
                            chunk_id, document_id, document_name, page, chapter,
                            section, subsection, title, chunk_index, total_chunks,
                            language, ocr, source_type, table_id, figure_id,
                            token_count, embedding_model, parent_chunk_id,
                            child_chunk_ids, semantic_tags, keywords,
                            confidence_score, extraction_timestamp, content_preview,
                            full_content, faiss_index_id, deleted
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        meta.chunk_id,
                        meta.document_id,
                        meta.document_name,
                        meta.page,
                        meta.chapter,
                        meta.section,
                        meta.subsection,
                        meta.title,
                        meta.chunk_index,
                        meta.total_chunks,
                        meta.language,
                        int(meta.ocr),
                        meta.source_type,
                        meta.table_id,
                        meta.figure_id,
                        meta.token_count,
                        meta.embedding_model,
                        meta.parent_chunk_id,
                        json.dumps(meta.child_chunk_ids),
                        json.dumps(meta.semantic_tags),
                        json.dumps(meta.keywords),
                        meta.confidence_score,
                        meta.extraction_timestamp.isoformat(),
                        content_preview,
                        full_content,
                        chunk.embedding_id,
                        0,  # deleted flag (0 = active, 1 = deleted)
                    ))
                
                conn.commit()
                
                # Count table chunks with full_content
                table_chunks = sum(1 for c in chunks if c.metadata.source_type == "table" or c.metadata.table_data)
                full_content_count = sum(1 for c in chunks if c.content)
                
                latency.stop(chunk_count=len(chunks))
                
                logger.log_ingestion(
                    event=LogEvent.INDEX_UPDATE,
                    document_id=chunks[0].metadata.document_id if chunks else "unknown",
                    document_name=chunks[0].metadata.document_name if chunks else "unknown",
                    message=f"Stored {len(chunks)} chunks with full_content",
                    details={
                        "chunk_count": len(chunks),
                        "table_chunks": table_chunks,
                        "full_content_count": full_content_count,
                        "all_chunks_have_full_content": full_content_count == len(chunks),
                    },
                )
                
            except Exception as exc:
                raise VectorStoreError(
                    message=f"Failed to store chunk metadata: {str(exc)}",
                    index_name=str(self.db_path),
                    operation="add_chunks",
                    original_exception=exc,
                )
    
    def get_document_by_hash(self, document_hash: str) -> Optional[dict[str, Any]]:
        """
        Look up a document by its content hash for duplicate detection.
        
        Args:
            document_hash: SHA-256 hash of the document.
        
        Returns:
            Optional[dict[str, Any]]: Document record if found, None otherwise.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM documents WHERE document_hash = ?",
                (document_hash,),
            )
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
            
        except Exception as exc:
            logger.log_error(
                event=LogEvent.ERROR,
                message=f"Failed to lookup document by hash: {str(exc)}",
                exception=exc,
            )
            return None
    
    def get_document_chunks(self, document_id: str) -> list[dict[str, Any]]:
        """
        Get all chunks for a specific document.
        
        Returns full_content when available.
        
        Args:
            document_id: Document identifier.
        
        Returns:
            list[dict[str, Any]]: List of chunk records with full_content.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM chunks WHERE document_id = ? AND deleted = 0 ORDER BY chunk_index",
                (document_id,),
            )
            
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as exc:
            logger.log_error(
                event=LogEvent.ERROR,
                message=f"Failed to get document chunks: {str(exc)}",
                exception=exc,
            )
            return []
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[dict[str, Any]]:
        """
        Get a specific chunk by its ID.
        
        Returns full_content when available.
        
        Args:
            chunk_id: Chunk identifier.
        
        Returns:
            Optional[dict[str, Any]]: Chunk record with full_content if found.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            )
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
            
        except Exception as exc:
            logger.log_error(
                event=LogEvent.ERROR,
                message=f"Failed to get chunk by ID: {str(exc)}",
                exception=exc,
            )
            return None
    
    def list_documents(self) -> list[dict[str, Any]]:
        """
        List all documents in the store.
        
        Returns:
            list[dict[str, Any]]: List of document records.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM documents ORDER BY ingestion_date DESC")
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as exc:
            logger.log_error(
                event=LogEvent.ERROR,
                message=f"Failed to list documents: {str(exc)}",
                exception=exc,
            )
            return []
    
    def delete_document(self, document_id: str) -> int:
        """
        Mark a document and all its chunks as deleted.
        
        Args:
            document_id: Document identifier.
        
        Returns:
            int: Number of chunks marked as deleted.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Mark chunks as deleted
            cursor.execute(
                "UPDATE chunks SET deleted = 1 WHERE document_id = ?",
                (document_id,),
            )
            chunks_deleted = cursor.rowcount
            
            conn.commit()
            
            logger.log_ingestion(
                event=LogEvent.INDEX_UPDATE,
                document_id=document_id,
                document_name="deleted",
                message=f"Marked {chunks_deleted} chunks as deleted for document {document_id}",
            )
            
            return chunks_deleted
            
        except Exception as exc:
            logger.log_error(
                event=LogEvent.ERROR,
                message=f"Failed to delete document: {str(exc)}",
                exception=exc,
            )
            return 0
    
    def delete_chunk(self, chunk_id: str) -> None:
        """
        Delete a specific chunk from the metadata store.
        
        Args:
            chunk_id: Chunk identifier.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "DELETE FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            )
            conn.commit()
            
        except Exception as exc:
            logger.log_error(
                event=LogEvent.ERROR,
                message=f"Failed to delete chunk {chunk_id}: {str(exc)}",
                exception=exc,
            )
    
    def get_all_active_chunks(self) -> list[dict[str, Any]]:
        """
        Get all active (non-deleted) chunks from the metadata store.
        
        Returns full_content when available.
        
        Returns:
            list[dict[str, Any]]: List of active chunk records with full_content.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM chunks WHERE deleted = 0 ORDER BY document_id, chunk_index",
            )
            
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as exc:
            logger.log_error(
                event=LogEvent.ERROR,
                message=f"Failed to get active chunks: {str(exc)}",
                exception=exc,
            )
            return []
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get metadata store statistics.
        
        Includes full_content coverage stats.
        
        Returns:
            dict[str, Any]: Dictionary with store statistics.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) as count FROM documents")
            document_count = cursor.fetchone()["count"]
            
            cursor.execute("SELECT COUNT(*) as count FROM chunks WHERE deleted = 0")
            active_chunk_count = cursor.fetchone()["count"]
            
            cursor.execute("SELECT COUNT(*) as count FROM chunks WHERE deleted = 1")
            deleted_chunk_count = cursor.fetchone()["count"]
            
            # Check full_content coverage
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN full_content IS NOT NULL AND full_content != '' THEN 1 ELSE 0 END) as has_full_content
                FROM chunks WHERE deleted = 0
            """)
            content_stats = cursor.fetchone()
            has_full_content = content_stats["has_full_content"] or 0
            total_chunks = content_stats["total"] or 0
            full_content_coverage = (has_full_content / total_chunks * 100) if total_chunks > 0 else 0
            
            # Check table chunks with full_content
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN full_content IS NOT NULL AND full_content != '' THEN 1 ELSE 0 END) as has_full_content
                FROM chunks WHERE deleted = 0 AND (source_type = 'table' OR table_id IS NOT NULL)
            """)
            table_stats = cursor.fetchone()
            table_total = table_stats["total"] or 0
            table_full_content = table_stats["has_full_content"] or 0
            table_coverage = (table_full_content / table_total * 100) if table_total > 0 else 0
            
            cursor.execute("SELECT document_id, COUNT(*) as chunk_count FROM chunks WHERE deleted = 0 GROUP BY document_id")
            chunks_per_doc = {row["document_id"]: row["chunk_count"] for row in cursor.fetchall()}
            
            return {
                "total_documents": document_count,
                "active_chunks": active_chunk_count,
                "deleted_chunks": deleted_chunk_count,
                "total_chunks": active_chunk_count + deleted_chunk_count,
                "chunks_per_document": chunks_per_doc,
                "db_path": str(self.db_path),
                "full_content_coverage_percent": round(full_content_coverage, 2),
                "chunks_with_full_content": has_full_content,
                "table_chunks_total": table_total,
                "table_chunks_with_full_content": table_full_content,
                "table_coverage_percent": round(table_coverage, 2),
            }
            
        except Exception as exc:
            logger.log_error(
                event=LogEvent.ERROR,
                message=f"Failed to get metadata stats: {str(exc)}",
                exception=exc,
            )
            return {}
    
    def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None