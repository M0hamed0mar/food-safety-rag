"""
Core pipeline orchestrator module.

This module provides the main pipeline that coordinates ingestion,
retrieval, and generation for the Food Safety RAG System.
It serves as the central entry point for all operations.

PHASE 15.7 FIX: Language detection and preservation.
- Added _is_arabic() method for language detection
- Query.is_arabic is now set during query creation
- Ensures LLM responds in the user's language
"""

from pathlib import Path
from typing import Any, AsyncGenerator, Optional, List, Tuple, Dict

from app.config import settings, settings
from app.config.constants import LogEvent
from app.ingestion import (
    DocumentLoader,
    EmbeddingGenerator,
    HierarchicalChunker,
    MetadataGenerator,
    SmartOCR,
    TextCleaner,
)
from app.monitoring import get_logger, measure_latency, setup_logging
from app.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    QueryExpander,
    ReciprocalRankFusion,
    RetrievalRouter,
    CrossEncoderReranker,
    RetrievalOrchestrator,
    ResultValidator,
)
from app.schemas import Answer, Chunk, ChunkMetadata, Document, IngestionMetadata, Query, StreamingChunk
from app.services import GenerationService, IngestionService, RetrievalService
from app.vector_store import FAISSStore, MetadataStore
from app.vector_store.chat_store import ChatStore
from app.llm.translator import get_translator


logger = get_logger("food_safety_rag.pipeline")


class RAGPipeline:
    """
    Main RAG pipeline orchestrator for the Food Safety RAG System.
    
    PHASE 15.7 FIX: Language detection and preservation.
    """
    
    # ========================================================================
    # LANGUAGE DETECTION - PHASE 15.7 FIX
    # ========================================================================
    
    @staticmethod
    def _is_arabic(text: str) -> bool:
        """
        Detect if text contains Arabic characters.
        
        PHASE 15.7 FIX: Added this method for language detection.
        Used to set Query.is_arabic during query creation.
        
        Args:
            text: Text to check.
        
        Returns:
            bool: True if text is Arabic, False otherwise.
        """
        if not text or not text.strip():
            return False
        
        text = text.strip()
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        
        # If more than 30% of characters are Arabic, consider it Arabic
        return arabic_chars / len(text) >= 0.3
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def __init__(self) -> None:
        """Initialize the RAG pipeline."""
        self.ingestion_service: Optional[IngestionService] = None
        self.retrieval_service: Optional[RetrievalService] = None
        self.generation_service: Optional[GenerationService] = None
        self.evaluator = None
        self.chat_store: Optional[ChatStore] = None
        self.metadata_store: Optional[MetadataStore] = None
        self.initialized: bool = False
        self._reranker: Optional[CrossEncoderReranker] = None
        self._translator = None
        self._validator: Optional[ResultValidator] = None

    def _get_reranker(self) -> CrossEncoderReranker:
        """Get the shared singleton reranker instance."""
        if self._reranker is None:
            from app.retrieval.reranker import get_reranker
            self._reranker = get_reranker()
            logger.log_event(
                event=LogEvent.SYSTEM_STARTUP,
                message="Retrieved shared reranker instance from pipeline",
                details={"instance_id": id(self._reranker)},
            )
        return self._reranker

    def _get_translator(self):
        """Get the shared singleton translator instance."""
        if self._translator is None:
            self._translator = get_translator()
            logger.log_event(
                event=LogEvent.SYSTEM_STARTUP,
                message="Retrieved shared translator instance from pipeline",
                details={"instance_id": id(self._translator) if self._translator else None},
            )
        return self._translator

    def _get_validator(self) -> Optional[ResultValidator]:
        """Get validator instance if enabled."""
        if self._validator is None and settings.ENABLE_VALIDATOR:
            self._validator = ResultValidator(
                enable_semantic_dedup=settings.VALIDATOR_ENABLE_SEMANTIC_DEDUP,
                semantic_dedup_threshold=settings.VALIDATOR_SEMANTIC_DEDUP_THRESHOLD,
                min_quality_score=settings.VALIDATOR_MIN_QUALITY_SCORE,
            )
            logger.log_event(
                event=LogEvent.SYSTEM_STARTUP,
                message="Validator enabled",
            )
        return self._validator

    def initialize(self) -> None:
        """
        Initialize the complete pipeline.
        """
        with measure_latency("pipeline_initialization") as latency:
            logger.log_event(
                event=LogEvent.SYSTEM_STARTUP,
                message="Initializing Food Safety RAG Pipeline",
            )

            setup_logging()
            settings.ensure_directories()

            self.chat_store = ChatStore()

            has_existing_data = self._check_existing_indices()
            if has_existing_data:
                logger.log_event(
                    event=LogEvent.SYSTEM_STARTUP,
                    message="Existing indices detected, loading from disk...",
                )

            vector_store = FAISSStore()
            self.metadata_store = MetadataStore()

            loader = DocumentLoader()
            ocr = SmartOCR()
            cleaner = TextCleaner()
            chunker = HierarchicalChunker()
            metadata_generator = MetadataGenerator()
            embedding_generator = EmbeddingGenerator()

            bm25_retriever = BM25Retriever()
            dense_retriever = DenseRetriever(vector_store=vector_store)
            expander = QueryExpander()
            
            fusion = ReciprocalRankFusion(
                normalize_scores=settings.HYBRID_NORMALIZE_SCORES,
                normalization_method=settings.HYBRID_NORMALIZATION_METHOD,
                adaptive_k=True,
                strategy_weighting=True,
            )
            
            reranker = self._get_reranker() if settings.RERANKER_ENABLED else None
            
            validator = None
            if settings.ENABLE_VALIDATOR:
                validator = ResultValidator(
                    enable_semantic_dedup=settings.VALIDATOR_ENABLE_SEMANTIC_DEDUP,
                    semantic_dedup_threshold=settings.VALIDATOR_SEMANTIC_DEDUP_THRESHOLD,
                    min_quality_score=settings.VALIDATOR_MIN_QUALITY_SCORE,
                )
                logger.log_event(
                    event=LogEvent.SYSTEM_STARTUP,
                    message="Validator enabled",
                )
            else:
                logger.log_event(
                    event=LogEvent.SYSTEM_STARTUP,
                    message="Validator disabled",
                )
            
            hybrid_retriever = HybridRetriever(
                dense_retriever=dense_retriever,
                bm25_retriever=bm25_retriever,
                expander=expander,
                fusion=fusion,
                metadata_store=self.metadata_store,
                normalize_scores=settings.HYBRID_NORMALIZE_SCORES,
                normalization_method=settings.HYBRID_NORMALIZATION_METHOD,
                auto_rebuild_bm25=True,
            )
            
            translator = self._get_translator()
            
            orchestrator = RetrievalOrchestrator(
                hybrid_retriever=hybrid_retriever,
                reranker=reranker,
                validator=validator,
                metadata_store=self.metadata_store,
                expander=expander,
                translator=translator,
                enable_reranking=settings.RERANKER_ENABLED,
                adaptive_pipeline=True,
                trace_enabled=True,
            )
            
            router = RetrievalRouter(
                expander=expander,
                dense_retriever=dense_retriever,
                bm25_retriever=bm25_retriever,
                fusion=fusion,
                reranker=reranker,
                validator=validator,
                orchestrator=orchestrator,
                metadata_store=self.metadata_store,
                trace_enabled=True,
            )

            self.ingestion_service = IngestionService(
                loader=loader,
                ocr=ocr,
                cleaner=cleaner,
                chunker=chunker,
                metadata_generator=metadata_generator,
                embedding_generator=embedding_generator,
                vector_store=vector_store,
                metadata_store=self.metadata_store,
                bm25_retriever=bm25_retriever,
            )

            self.retrieval_service = RetrievalService(
                router=router,
                orchestrator=orchestrator,
                metadata_store=self.metadata_store,
                trace_enabled=True,
                reranker=reranker,
            )

            from app.generation import AnswerGenerator, ContextBuilder, ContextCompressor
            from app.llm import PromptManager

            self.generation_service = GenerationService(
                retrieval_service=self.retrieval_service,
                answer_generator=AnswerGenerator(),
                prompt_manager=PromptManager(),
            )

            self.evaluator = None

            if has_existing_data:
                self._load_existing_indices()

            self._initialize_bm25_from_metadata()
            self._force_bm25_rebuild()

            self.initialized = True

            latency.stop()

            logger.log_event(
                event=LogEvent.SYSTEM_STARTUP,
                message="Food Safety RAG Pipeline initialized successfully",
                details={
                    "initialization_duration_ms": latency.duration_ms,
                    "has_existing_data": has_existing_data,
                    "translation_enabled": getattr(settings, "TRANSLATION_ENABLED", True),
                    "translation_mode": getattr(settings, "TRANSLATION_MODE", "fused"),
                    "phase_15_7_language_fix": True,
                },
            )

    def _force_bm25_rebuild(self) -> None:
        """Force BM25 rebuild from metadata store."""
        try:
            if not self.metadata_store or not self.retrieval_service:
                return
            
            all_chunks_meta = self.metadata_store.get_all_active_chunks()
            if not all_chunks_meta:
                return
            
            reconstructed_chunks: List[Chunk] = []
            for meta in all_chunks_meta:
                content = meta.get("full_content", meta.get("content_preview", ""))
                if not content:
                    continue
                
                chunk = self._reconstruct_chunk_from_metadata(meta)
                if chunk and content:
                    chunk.content = content
                    reconstructed_chunks.append(chunk)
            
            if not reconstructed_chunks:
                return
            
            bm25_retriever = self.retrieval_service.router.bm25_retriever
            bm25_retriever.index(reconstructed_chunks)
            
            logger.log_event(
                event=LogEvent.INDEX_UPDATE,
                message=f"BM25 index rebuilt with {len(reconstructed_chunks)} chunks",
                details={"chunk_count": len(reconstructed_chunks)},
            )
            
        except Exception as exc:
            logger.log_error(
                event=LogEvent.WARNING,
                message=f"Failed to rebuild BM25: {str(exc)}",
                exception=exc,
            )

    def _check_existing_indices(self) -> bool:
        """Check if indices already exist on disk."""
        faiss_file = settings.FAISS_INDEX_DIR / f"{settings.FAISS_INDEX_NAME}.faiss"
        meta_file = settings.FAISS_INDEX_DIR / f"{settings.FAISS_INDEX_NAME}.meta"
        db_file = settings.FAISS_INDEX_DIR / "metadata.db"
        return faiss_file.exists() and meta_file.exists() and db_file.exists()

    def _load_existing_indices(self) -> None:
        """Load existing FAISS and BM25 indices from disk."""
        try:
            logger.log_event(
                event=LogEvent.SYSTEM_STARTUP,
                message="Loading existing indices...",
            )

            if self.retrieval_service and self.retrieval_service.router:
                vector_store = FAISSStore()
                vector_store.load()
                self.retrieval_service.router.dense_retriever.vector_store = vector_store

                logger.log_event(
                    event=LogEvent.SYSTEM_STARTUP,
                    message=f"FAISS index loaded: {vector_store.index.ntotal if vector_store.index else 0} vectors",
                    details={"vectors": vector_store.index.ntotal if vector_store.index else 0},
                )

            metadata_store = MetadataStore()
            documents = metadata_store.list_documents()

            if documents and self.retrieval_service and self.retrieval_service.router:
                all_chunks_data = []
                for doc in documents:
                    chunks_data = metadata_store.get_document_chunks(doc['document_id'])
                    all_chunks_data.extend(chunks_data)

                if all_chunks_data:
                    chunks = []
                    for meta in all_chunks_data:
                        content = meta.get("full_content", meta.get("content_preview", ""))
                        
                        chunk = Chunk(
                            content=content,
                            metadata=ChunkMetadata(
                                document_id=meta["document_id"],
                                document_name=meta["document_name"],
                                chunk_id=meta["chunk_id"],
                                chunk_index=meta["chunk_index"],
                                total_chunks=meta["total_chunks"],
                                page=meta.get("page"),
                                section=meta.get("section"),
                            ),
                        )
                        chunks.append(chunk)

                    if chunks:
                        bm25 = BM25Retriever()
                        bm25.index(chunks)
                        self.retrieval_service.router.bm25_retriever = bm25

                        logger.log_event(
                            event=LogEvent.SYSTEM_STARTUP,
                            message=f"BM25 index built from metadata: {len(chunks)} chunks",
                            details={"chunks": len(chunks)},
                        )

            metadata_store.close()

            logger.log_event(
                event=LogEvent.SYSTEM_STARTUP,
                message="Existing indices loaded successfully",
            )

        except Exception as exc:
            logger.log_error(
                event=LogEvent.WARNING,
                message=f"Failed to load existing indices: {str(exc)}",
                exception=exc,
            )

    def _initialize_bm25_from_metadata(self) -> None:
        """Auto-initialize BM25 from metadata store."""
        try:
            if not self.metadata_store or not self.retrieval_service:
                return
            
            all_chunks_meta = self.metadata_store.get_all_active_chunks()
            if not all_chunks_meta:
                return
            
            reconstructed_chunks: List[Chunk] = []
            for meta in all_chunks_meta:
                chunk = self._reconstruct_chunk_from_metadata(meta)
                if chunk:
                    reconstructed_chunks.append(chunk)
            
            if not reconstructed_chunks:
                return
            
            current_stats = self.retrieval_service.router.bm25_retriever.get_stats()
            current_size = current_stats.get('corpus_size', 0)
            
            if current_size < len(reconstructed_chunks):
                self.retrieval_service.router.bm25_retriever.index(reconstructed_chunks)
                
                logger.log_event(
                    event=LogEvent.INDEX_UPDATE,
                    message=f"BM25 index auto-initialized with {len(reconstructed_chunks)} chunks",
                    details={"chunk_count": len(reconstructed_chunks)},
                )
            
        except Exception as exc:
            logger.log_error(
                event=LogEvent.WARNING,
                message=f"Failed to auto-initialize BM25: {str(exc)}",
                exception=exc,
            )

    def _reconstruct_chunk_from_metadata(self, meta: Dict[str, Any]) -> Optional[Chunk]:
        """Reconstruct a Chunk from metadata dictionary."""
        try:
            from app.schemas import Chunk, ChunkMetadata
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
                content=content,
                metadata=chunk_metadata,
                embedding_id=meta.get("faiss_index_id"),
            )
        except Exception as exc:
            logger.log_error(
                event=LogEvent.WARNING,
                message=f"Failed to reconstruct chunk from metadata: {str(exc)}",
                exception=exc,
            )
            return None

    # ============================================================
    # INGESTION METHODS
    # ============================================================

    def ingest_document(self, file_path: str | Path) -> IngestionMetadata:
        """Ingest a single document into the system."""
        if not self.initialized or self.ingestion_service is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        return self.ingestion_service.ingest_document(file_path)

    def ingest_documents(self, file_paths: List[str | Path]) -> List[IngestionMetadata]:
        """Ingest multiple documents into the system."""
        if not self.initialized or self.ingestion_service is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        return self.ingestion_service.ingest_batch(file_paths)

    # ============================================================
    # CHAT MANAGEMENT METHODS
    # ============================================================

    def create_chat(self, title: str = "New Chat") -> str:
        """Create a new chat session."""
        if not self.initialized or self.chat_store is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        return self.chat_store.create_chat(title)

    def get_chat_history(self, session_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent messages from a chat session."""
        if not self.initialized or self.chat_store is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        return self.chat_store.get_chat_history(session_id, limit)

    def get_chat_session(self, session_id: str) -> Optional[Any]:
        """Get a complete chat session."""
        if not self.initialized or self.chat_store is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        return self.chat_store.get_chat(session_id)

    def list_chats(self) -> List[Dict[str, Any]]:
        """List all chat sessions."""
        if not self.initialized or self.chat_store is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        return self.chat_store.list_chats()

    def delete_chat(self, session_id: str) -> bool:
        """Delete a chat session."""
        if not self.initialized or self.chat_store is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        return self.chat_store.delete_chat(session_id)

    def save_chat_message(self, session_id: str, role: str, content: str) -> None:
        """Save a message to a chat session."""
        if not self.initialized or self.chat_store is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        self.chat_store.add_message(session_id, role, content)

    def update_chat_title(self, session_id: str, title: str) -> bool:
        """Update the title of a chat session."""
        if not self.initialized or self.chat_store is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        return self.chat_store.update_title(session_id, title)

    # ============================================================
    # QUERY METHODS - PHASE 15.7 FIX: Language Detection
    # ============================================================

    def ask(self, query_text: str, query_id: Optional[str] = None, session_id: Optional[str] = None) -> Answer:
        """
        Ask a question and get an answer.
        
        PHASE 15.7 FIX: Detects language and sets is_arabic on Query.
        This ensures the LLM responds in the correct language.
        """
        if not self.initialized or self.generation_service is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        history = None
        if session_id and self.chat_store:
            history = self.chat_store.get_chat_history(session_id, limit=3)

        # ============================================================
        # PHASE 15.7 FIX: Detect language and set is_arabic
        # ============================================================
        is_arabic = self._is_arabic(query_text)
        
        logger.log_event(
            event=LogEvent.QUERY_RECEIVED,
            message=f"Query received: {query_text[:50]}... (is_arabic={is_arabic})",
            details={
                "query_text_preview": query_text[:100],
                "is_arabic": is_arabic,
                "session_id": session_id,
            },
        )

        query = Query(
            query_id=query_id or f"qry_{hash(query_text) & 0xFFFFFFFF:08x}",
            original_text=query_text,
            is_arabic=is_arabic,  # PHASE 15.7 FIX
            session_id=session_id,
            history=history,
        )

        answer = self.generation_service.generate_answer(query)

        if session_id and self.chat_store:
            self.chat_store.add_message(session_id, "user", query_text)
            self.chat_store.add_message(session_id, "assistant", answer.text)

        return answer

    async def ask_streaming(
        self,
        query_text: str,
        query_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AsyncGenerator[StreamingChunk, None]:
        """
        Ask a question and get a streaming answer.
        
        PHASE 15.7 FIX: Detects language and sets is_arabic on Query.
        This ensures the LLM responds in the correct language.
        """
        if not self.initialized or self.generation_service is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        history = None
        if session_id and self.chat_store:
            history = self.chat_store.get_chat_history(session_id, limit=3)

        # ============================================================
        # PHASE 15.7 FIX: Detect language and set is_arabic
        # ============================================================
        is_arabic = self._is_arabic(query_text)
        
        logger.log_event(
            event=LogEvent.QUERY_RECEIVED,
            message=f"Streaming query received: {query_text[:50]}... (is_arabic={is_arabic})",
            details={
                "query_text_preview": query_text[:100],
                "is_arabic": is_arabic,
                "session_id": session_id,
            },
        )

        query = Query(
            query_id=query_id or f"qry_{hash(query_text) & 0xFFFFFFFF:08x}",
            original_text=query_text,
            is_arabic=is_arabic,  # PHASE 15.7 FIX
            session_id=session_id,
            history=history,
        )

        if session_id and self.chat_store:
            self.chat_store.add_message(session_id, "user", query_text)

        full_answer = ""
        async for chunk in self.generation_service.generate_answer_streaming(query):
            full_answer += chunk.token
            yield chunk

        if session_id and self.chat_store and full_answer:
            self.chat_store.add_message(session_id, "assistant", full_answer)

    def retrieve(self, query_text: str, query_id: Optional[str] = None) -> List[Chunk]:
        """
        Retrieve relevant chunks for a query without generating an answer.
        """
        if not self.initialized or self.retrieval_service is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        # PHASE 15.7 FIX: Detect language and set is_arabic
        is_arabic = self._is_arabic(query_text)

        query = Query(
            query_id=query_id or f"qry_{hash(query_text) & 0xFFFFFFFF:08x}",
            original_text=query_text,
            is_arabic=is_arabic,  # PHASE 15.7 FIX
        )

        return self.retrieval_service.retrieve_chunks_only(query)

    def retrieve_with_trace(
        self,
        query_text: str,
        query_id: Optional[str] = None,
        expected_page: Optional[int] = None,
    ) -> Tuple[List[Chunk], Dict[str, Any]]:
        """Retrieve relevant chunks with full trace for debugging."""
        if not self.initialized or self.retrieval_service is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        self.retrieval_service.trace_enabled = True
        self.retrieval_service.clear_traces()
        if hasattr(self.retrieval_service, 'router'):
            self.retrieval_service.router.trace_enabled = True

        # PHASE 15.7 FIX: Detect language and set is_arabic
        is_arabic = self._is_arabic(query_text)

        query = Query(
            query_id=query_id or f"qry_{hash(query_text) & 0xFFFFFFFF:08x}",
            original_text=query_text,
            is_arabic=is_arabic,  # PHASE 15.7 FIX
        )

        chunks = self.retrieval_service.retrieve_chunks_only(
            query,
            expected_page=expected_page,
        )

        traces = self.retrieval_service.get_traces()
        trace = traces[-1] if traces else {}

        return chunks, trace

    # ============================================================
    # EVALUATION METHOD
    # ============================================================

    def evaluate(
        self,
        benchmark_name: str,
        k_values: List[int] = [1, 5, 10, 20],
    ) -> Dict[str, Any]:
        """
        Evaluation is handled by scripts/run_benchmark.py.
        """
        logger.log_event(
            event=LogEvent.WARNING,
            message="pipeline.evaluate() is deprecated. Use scripts/run_benchmark.py instead.",
            level=30,
        )
        return {
            "error": "Evaluation is handled by scripts/run_benchmark.py",
            "status": "deprecated",
            "recommendation": "Run: python scripts/run_benchmark.py",
            "benchmark_file": "benchmarks/benchmark_v2.json",
            "results_file": "benchmarks/benchmark_results.json",
        }

    # ============================================================
    # SYSTEM METHODS
    # ============================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats: Dict[str, Any] = {
            "initialized": self.initialized,
            "translation_enabled": getattr(settings, "TRANSLATION_ENABLED", True),
            "translation_mode": getattr(settings, "TRANSLATION_MODE", "fused"),
            "validator_enabled": settings.ENABLE_VALIDATOR,
            "reranker_enabled": settings.RERANKER_ENABLED,
            "fallback_enabled": settings.RERANKER_FALLBACK_ENABLED,
            "phase_15_7_language_fix": True,
        }

        if self.ingestion_service:
            stats["ingestion"] = self.ingestion_service.get_document_stats()

        if self.retrieval_service:
            stats["retrieval"] = self.retrieval_service.get_stats()

        if self.generation_service:
            stats["generation"] = self.generation_service.get_stats()

        if self.chat_store:
            stats["chats"] = self.chat_store.get_stats()

        stats["features"] = {
            "semantic_dedup": settings.VALIDATOR_ENABLE_SEMANTIC_DEDUP,
            "score_normalization": settings.HYBRID_NORMALIZE_SCORES,
            "adaptive_pipeline": True,
            "fused_translation": getattr(settings, "TRANSLATION_MODE", "fused") == "fused",
            "language_detection": True,
        }

        return stats


# Global pipeline instance
_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Get or create the global pipeline instance."""
    global _pipeline

    if _pipeline is None:
        _pipeline = RAGPipeline()
        _pipeline.initialize()
        
        try:
            _pipeline._force_bm25_rebuild()
        except Exception as exc:
            logger.log_error(
                event=LogEvent.WARNING,
                message=f"Failed to rebuild BM25 in get_pipeline: {str(exc)}",
                exception=exc,
            )

    return _pipeline


def reset_pipeline() -> None:
    """Reset the global pipeline instance."""
    global _pipeline
    _pipeline = None