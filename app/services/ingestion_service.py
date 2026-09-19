"""
Ingestion service module.

This module provides a high-level service for document ingestion that
orchestrates the complete pipeline from loading through indexing.
It coordinates all ingestion components and handles error recovery.

PHASE 3: Stores full_content in metadata store for BM25 and Reranker.
FIX: Ensure full_content is populated during ingestion.
FIX: Rebuild BM25 from all chunks after ingestion.
"""

import uuid
from pathlib import Path
from typing import Any, Optional

from app.config import settings
from app.config.constants import LogEvent
from app.core.exceptions import (
    DocumentError,
    DocumentValidationError,
    DuplicateDocumentError,
)
from app.ingestion import (
    DocumentLoader,
    EmbeddingGenerator,
    HierarchicalChunker,
    MetadataGenerator,
    SmartOCR,
    TextCleaner,
)
from app.monitoring import get_logger, measure_latency
from app.retrieval.bm25 import BM25Retriever
from app.schemas import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentMetadata,
    IngestionMetadata,
)
from app.vector_store import FAISSStore, MetadataStore


logger = get_logger("food_safety_rag.services.ingestion")


class IngestionService:
    """
    High-level service for document ingestion.
    
    Orchestrates the complete ingestion pipeline:
    Document → Load → Validate → Parse → Smart OCR → Clean → 
    Structure Extract → Chunk → Metadata → Embed → Index
    
    PHASE 3: Stores full_content in metadata store.
    FIX: Rebuilds BM25 from all chunks after ingestion.
    
    Handles errors gracefully and continues processing remaining documents
    when one fails.
    
    Attributes:
        loader: Document loader.
        ocr: Smart OCR processor.
        cleaner: Text cleaner.
        chunker: Hierarchical chunker.
        metadata_generator: Metadata generator.
        embedding_generator: Embedding generator.
        vector_store: FAISS vector store.
        metadata_store: Metadata store.
        bm25_retriever: BM25 retriever for sparse retrieval.
    """
    
    def __init__(
        self,
        loader: Optional[DocumentLoader] = None,
        ocr: Optional[SmartOCR] = None,
        cleaner: Optional[TextCleaner] = None,
        chunker: Optional[HierarchicalChunker] = None,
        metadata_generator: Optional[MetadataGenerator] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        vector_store: Optional[FAISSStore] = None,
        metadata_store: Optional[MetadataStore] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        index_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
    ) -> None:
        """
        Initialize the ingestion service with all components.
        
        Args:
            loader: Document loader. Creates new if None.
            ocr: Smart OCR. Creates new if None.
            cleaner: Text cleaner. Creates new if None.
            chunker: Hierarchical chunker. Creates new if None.
            metadata_generator: Metadata generator. Creates new if None.
            embedding_generator: Embedding generator. Creates new if None.
            vector_store: FAISS vector store. Creates new if None.
            metadata_store: Metadata store. Creates new if None.
            bm25_retriever: BM25 retriever. Creates new if None.
            index_path: Path for FAISS index (used if vector_store is None).
            db_path: Path for metadata database (used if metadata_store is None).
        """
        self.loader: DocumentLoader = loader or DocumentLoader()
        self.ocr: SmartOCR = ocr or SmartOCR()
        self.cleaner: TextCleaner = cleaner or TextCleaner()
        self.chunker: HierarchicalChunker = chunker or HierarchicalChunker()
        self.metadata_generator: MetadataGenerator = metadata_generator or MetadataGenerator()
        self.embedding_generator: EmbeddingGenerator = embedding_generator or EmbeddingGenerator()
        self.bm25_retriever: BM25Retriever = bm25_retriever or BM25Retriever()
        
        if vector_store:
            self.vector_store: FAISSStore = vector_store
        else:
            self.vector_store = FAISSStore(index_path=index_path)
        
        if metadata_store:
            self.metadata_store: MetadataStore = metadata_store
        else:
            self.metadata_store = MetadataStore(db_path=db_path)
        
        self._last_chunks: list[Chunk] = []
    
    def _check_duplicate(self, document_hash: str) -> bool:
        """
        Check if a document with this hash already exists.
        
        Args:
            document_hash: SHA-256 hash of the document.
        
        Returns:
            bool: True if duplicate exists.
        """
        existing = self.metadata_store.get_document_by_hash(document_hash)
        return existing is not None
    
    def _reconstruct_chunk_from_metadata(self, meta: dict[str, Any]) -> Optional[Chunk]:
        """
        Reconstruct a Chunk from metadata dictionary.
        
        PHASE 3: Uses full_content from metadata.
        
        Args:
            meta: Metadata dictionary from metadata_store.
        
        Returns:
            Optional[Chunk]: Reconstructed chunk, or None if failed.
        """
        try:
            import json
            
            def parse_json_field(value: Any, default: Any = None) -> Any:
                if value is None:
                    return default
                if isinstance(value, str):
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError:
                        return default
                return value
            
            # PHASE 3: Use full_content from metadata
            content = meta.get("full_content", meta.get("content_preview", ""))
            
            chunk_metadata = ChunkMetadata(
                document_id=meta["document_id"],
                document_name=meta["document_name"],
                page=meta.get("page"),
                chapter=meta.get("chapter"),
                section=meta.get("section"),
                subsection=meta.get("subsection"),
                title=meta.get("title"),
                chunk_id=meta["chunk_id"],
                chunk_index=meta["chunk_index"],
                total_chunks=meta["total_chunks"],
                language=meta.get("language"),
                ocr=bool(meta.get("ocr", 0)),
                source_type=meta.get("source_type", "paragraph"),
                table_id=meta.get("table_id"),
                figure_id=meta.get("figure_id"),
                token_count=meta.get("token_count", 0),
                embedding_model=meta.get("embedding_model"),
                parent_chunk_id=meta.get("parent_chunk_id"),
                child_chunk_ids=parse_json_field(meta.get("child_chunk_ids", "[]"), []),
                semantic_tags=parse_json_field(meta.get("semantic_tags", "[]"), []),
                keywords=parse_json_field(meta.get("keywords", "[]"), []),
                confidence_score=meta.get("confidence_score", 1.0),
            )
            
            return Chunk(
                content=content,  # PHASE 3: Full content
                metadata=chunk_metadata,
                embedding_id=meta.get("faiss_index_id"),
            )
        except Exception as exc:
            logger.log_error(
                event=LogEvent.WARNING,
                message=f"Failed to reconstruct chunk from metadata: {str(exc)}",
                exception=exc,
                details={"chunk_id": meta.get("chunk_id")},
            )
            return None

    def _rebuild_bm25_from_all_chunks(self) -> None:
        """
        Rebuild the BM25 index from ALL chunks in the metadata store.
        
        PHASE 3: Uses full_content from metadata store.
        FIX: Called after each ingestion to keep the index up to date.
        """
        try:
            # Get all active chunks from metadata store
            all_chunks_meta = self.metadata_store.get_all_active_chunks()
            
            if not all_chunks_meta:
                logger.log_event(
                    event=LogEvent.WARNING,
                    message="No active chunks found for BM25 rebuild",
                    level=30,
                )
                return
            
            # Reconstruct chunks from metadata (uses full_content)
            reconstructed_chunks: list[Chunk] = []
            for meta in all_chunks_meta:
                chunk = self._reconstruct_chunk_from_metadata(meta)
                if chunk and chunk.content:  # Ensure content exists
                    reconstructed_chunks.append(chunk)
            
            if not reconstructed_chunks:
                logger.log_event(
                    event=LogEvent.WARNING,
                    message="No chunks could be reconstructed for BM25 rebuild",
                    level=30,
                )
                return
            
            # Rebuild BM25 index with ALL chunks
            self.bm25_retriever.index(reconstructed_chunks)
            
            logger.log_event(
                event=LogEvent.INDEX_UPDATE,
                message=f"PHASE 3 FIX: BM25 index rebuilt with {len(reconstructed_chunks)} total chunks (full content)",
                details={
                    "chunk_count": len(reconstructed_chunks),
                    "rebuild_reason": "post_ingestion_sync",
                    "full_content_used": True,
                },
            )
            
        except Exception as exc:
            logger.log_error(
                event=LogEvent.ERROR,
                message=f"Failed to rebuild BM25 index: {str(exc)}",
                exception=exc,
            )
    
    def _index_bm25(self, chunks: list[Chunk]) -> None:
        """
        Index chunks in BM25 for sparse retrieval.
        
        PHASE 3: Rebuilds from ALL chunks with full_content.
        FIX: Called after each ingestion.
        
        Args:
            chunks: List of chunks to index (used for logging).
        """
        if chunks:
            # Rebuild from ALL chunks to ensure consistency
            self._rebuild_bm25_from_all_chunks()
            
            logger.log_ingestion(
                event=LogEvent.INDEX_UPDATE,
                document_id="bm25",
                document_name="bm25_index",
                message=f"PHASE 3 FIX: BM25 index rebuilt with all chunks (including {len(chunks)} new)",
                details={"new_chunks": len(chunks), "full_content_used": True},
            )
    
    def ingest_document(self, file_path: str | Path) -> IngestionMetadata:
        """
        Ingest a single document through the complete pipeline.
        
        PHASE 3: Stores full_content in metadata store.
        FIX: Rebuilds BM25 after ingestion.
        
        Args:
            file_path: Path to the document file.
        
        Returns:
            IngestionMetadata: Complete ingestion metadata and statistics.
        
        Raises:
            DuplicateDocumentError: If document already exists.
            DocumentValidationError: If document is invalid.
        """
        file_path = Path(file_path)
        ingestion_id = f"ing_{uuid.uuid4().hex[:12]}"
        
        ingestion_meta = IngestionMetadata(
            ingestion_id=ingestion_id,
            document_id="pending",
            document_name=file_path.name,
        )
        
        with measure_latency("document_ingestion") as latency:
            try:
                logger.log_ingestion(
                    event=LogEvent.DOCUMENT_LOAD_START,
                    document_id=ingestion_id,
                    document_name=file_path.name,
                    message=f"Starting ingestion: {file_path.name}",
                )
                
                # 1. Load document (includes table extraction via TableProcessor)
                document = self.loader.load(file_path)
                
                # 2. Check for duplicates
                if self._check_duplicate(document.metadata.document_hash):
                    raise DuplicateDocumentError(
                        document_path=str(file_path),
                        document_hash=document.metadata.document_hash,
                    )
                
                ingestion_meta.document_id = document.metadata.document_id
                ingestion_meta.add_stage("load", 0.0)
                
                # 3. OCR processing (if PDF)
                if file_path.suffix.lower() == ".pdf":
                    ocr_results = self.ocr.process_document(
                        pages=document.pages,
                        pdf_path=file_path,
                    )
                    
                    ocr_pages = sum(1 for r in ocr_results if r["ocr_applied"])
                    skipped_pages = sum(1 for r in ocr_results if not r["ocr_applied"])
                    
                    for i, page in enumerate(document.pages):
                        if i < len(ocr_results) and ocr_results[i]["ocr_applied"]:
                            page["text"] = ocr_results[i]["text"]
                            page["ocr_applied"] = True
                    
                    ingestion_meta.ocr_pages = ocr_pages
                    ingestion_meta.skipped_ocr_pages = skipped_pages
                    document.metadata.ocr_required = ocr_pages > 0
                
                ingestion_meta.add_stage("ocr", 0.0)
                
                # 4. Text cleaning
                cleaned_text = self.cleaner.clean(document.raw_text)
                document.raw_text = cleaned_text
                
                for page in document.pages:
                    if "text" in page:
                        page["text"] = self.cleaner.clean(page["text"])
                
                ingestion_meta.add_stage("clean", 0.0)
                
                # 5. Chunking with hierarchical structure and processed tables
                chunks_data = self.chunker.chunk_hierarchical_with_tables(document)
                ingestion_meta.chunk_count = len(chunks_data)
                ingestion_meta.add_stage("chunk", 0.0)
                
                # 6. Generate metadata for all chunks
                chunks: list[Chunk] = []
                for i, chunk_data in enumerate(chunks_data):
                    meta = self.metadata_generator.generate(
                        content=chunk_data["content"],
                        document=document,
                        chunk_index=i,
                        total_chunks=len(chunks_data),
                        page_number=chunk_data.get("page_number"),
                        ocr=chunk_data.get("ocr", False),
                        table_id=chunk_data.get("table_id"),
                        headers=chunk_data.get("headers"),
                        rows=chunk_data.get("rows"),
                        row_count=chunk_data.get("row_count"),
                        column_count=chunk_data.get("column_count"),
                        has_table=chunk_data.get("has_table", False),
                        semantic_description=chunk_data.get("semantic_description"),
                        table_type=chunk_data.get("table_type"),
                        keywords=chunk_data.get("keywords"),
                        embedding_text=chunk_data.get("embedding_text"),
                        is_extracted_table=chunk_data.get("is_extracted_table", False),
                        confidence=chunk_data.get("confidence", 1.0),
                    )
                    
                    chunk = Chunk(
                        content=chunk_data["content"],
                        metadata=meta,
                    )
                    chunks.append(chunk)
                
                ingestion_meta.add_stage("metadata", 0.0)
                
                # 7. Generate embeddings
                chunks = self.embedding_generator.embed_chunks(chunks)
                ingestion_meta.embedding_count = len(chunks)
                ingestion_meta.add_stage("embed", 0.0)
                
                # 8. Store metadata (includes full_content)
                self.metadata_store.add_document(document.metadata)
                
                # PHASE 3 FIX: Ensure chunks have content before storing
                for chunk in chunks:
                    if not chunk.content:
                        chunk.content = chunk.metadata.content_preview or ""
                
                self.metadata_store.add_chunks(chunks)
                ingestion_meta.add_stage("metadata_store", 0.0)
                
                # 9. Index in FAISS
                self.vector_store.add(chunks)
                ingestion_meta.add_stage("index", 0.0)
                
                # 10. Save FAISS index
                self.vector_store.save()
                
                self._last_chunks = chunks
                
                # 11. PHASE 3 FIX: Rebuild BM25 from all chunks
                self._rebuild_bm25_from_all_chunks()
                ingestion_meta.add_stage("bm25_index", 0.0)
                
                ingestion_meta.complete(status="completed")
                
                latency.stop(
                    document_id=document.metadata.document_id,
                    document_name=document.metadata.document_name,
                )
                
                logger.log_ingestion(
                    event=LogEvent.DOCUMENT_LOAD_COMPLETE,
                    document_id=document.metadata.document_id,
                    document_name=document.metadata.document_name,
                    message=f"PHASE 3 FIX: Ingestion complete: {document.metadata.document_name}",
                    details={
                        "chunks": len(chunks),
                        "tables": len(document.get_processed_tables()),
                        "ocr_pages": ingestion_meta.ocr_pages,
                        "duration_ms": latency.duration_ms,
                        "full_content_stored": True,
                        "bm25_rebuilt": True,
                    },
                )
                
                return ingestion_meta
                
            except DuplicateDocumentError:
                raise
            except DocumentValidationError:
                raise
            except Exception as exc:
                ingestion_meta.complete(status="failed")
                ingestion_meta.add_error(str(exc), stage="unknown")
                
                logger.log_error(
                    event=LogEvent.DOCUMENT_VALIDATION_FAILED,
                    message=f"Ingestion failed for {file_path.name}: {str(exc)}",
                    exception=exc,
                )
                
                raise DocumentError(
                    message=f"Ingestion failed: {str(exc)}",
                    document_path=str(file_path),
                    original_exception=exc,
                )
    
    def ingest_batch(self, file_paths: list[str | Path]) -> list[IngestionMetadata]:
        """
        Ingest multiple documents in batch.
        
        PHASE 3: Rebuilds BM25 after all documents are ingested.
        
        Continues with remaining documents if one fails.
        
        Args:
            file_paths: List of document file paths.
        
        Returns:
            list[IngestionMetadata]: Results for each document.
        """
        results: list[IngestionMetadata] = []
        all_chunks: list[Chunk] = []
        
        for file_path in file_paths:
            try:
                result = self.ingest_document(file_path)
                results.append(result)
                if self._last_chunks:
                    all_chunks.extend(self._last_chunks)
            except (DuplicateDocumentError, DocumentValidationError, DocumentError) as exc:
                failed_result = IngestionMetadata(
                    ingestion_id=f"ing_{uuid.uuid4().hex[:12]}",
                    document_id="failed",
                    document_name=Path(file_path).name,
                )
                failed_result.complete(status="failed")
                failed_result.add_error(str(exc), stage="ingestion")
                results.append(failed_result)
                
                logger.log_error(
                    event=LogEvent.DOCUMENT_VALIDATION_FAILED,
                    message=f"Skipping failed document, continuing batch: {file_path}",
                    exception=exc,
                )
                continue
        
        # PHASE 3 FIX: Rebuild BM25 from ALL chunks after batch
        if all_chunks:
            self._rebuild_bm25_from_all_chunks()
            logger.log_ingestion(
                event=LogEvent.INDEX_UPDATE,
                document_id="bm25",
                document_name="bm25_index",
                message=f"PHASE 3 FIX: BM25 index rebuilt with all chunks after batch ingestion",
                details={"total_chunks": len(all_chunks), "full_content_used": True},
            )
        
        logger.log_event(
            event=LogEvent.DOCUMENT_LOAD_COMPLETE,
            message=f"PHASE 3 FIX: Batch ingestion complete: {len(results)} documents processed",
            details={
                "total": len(results),
                "successful": sum(1 for r in results if r.status == "completed"),
                "failed": sum(1 for r in results if r.status == "failed"),
                "full_content_stored": True,
                "bm25_rebuilt": True,
            },
        )
        
        return results
    
    def get_document_stats(self) -> dict[str, Any]:
        """
        Get statistics about ingested documents.
        
        Returns:
            dict[str, Any]: Document and chunk statistics.
        """
        return {
            "metadata_store": self.metadata_store.get_stats(),
            "vector_store": self.vector_store.get_stats(),
            "bm25": self.bm25_retriever.get_stats(),
        }
    
    def get_bm25_retriever(self) -> BM25Retriever:
        """
        Get the BM25 retriever instance.
        
        Returns:
            BM25Retriever: The BM25 retriever.
        """
        return self.bm25_retriever