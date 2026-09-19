"""
Retrieval service module.

This module provides a high-level service for query retrieval that
orchestrates the complete retrieval pipeline from query through context.
Supports metadata filtering and parent/neighbor context expansion.
"""

from typing import Any, Optional, List, Tuple, Dict

from app.config import settings
from app.config.constants import LogEvent
from app.core.exceptions import RetrievalError
from app.generation import ContextBuilder, ContextCompressor
from app.monitoring import get_logger, measure_latency
from app.retrieval import RetrievalRouter, RetrievalOrchestrator
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import ReciprocalRankFusion
from app.retrieval.expander import QueryExpander
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.validator import ResultValidator
from app.schemas import Chunk, Query
from app.vector_store import MetadataStore
from app.llm.translator import get_translator


logger = get_logger("food_safety_rag.services.retrieval")


class RetrievalService:
    """
    High-level service for query retrieval.
    
    Attributes:
        router: Retrieval router component.
        orchestrator: Retrieval orchestrator component.
        compressor: Context compressor component.
        context_builder: Context builder component.
        metadata_store: Metadata store component.
        trace_enabled: Whether tracing is enabled.
        traces: List of traces from the last retrieval.
    """
    
    def __init__(
        self,
        router: Optional[RetrievalRouter] = None,
        orchestrator: Optional[RetrievalOrchestrator] = None,
        compressor: Optional[ContextCompressor] = None,
        context_builder: Optional[ContextBuilder] = None,
        metadata_store: Optional[MetadataStore] = None,
        trace_enabled: bool = False,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> None:
        """
        Initialize the retrieval service.
        """
        self.metadata_store: MetadataStore = metadata_store or MetadataStore()
        
        self._reranker = reranker
        if self._reranker is None and settings.RERANKER_ENABLED:
            from app.retrieval.reranker import get_reranker
            self._reranker = get_reranker()
            logger.log_event(
                event=LogEvent.SYSTEM_STARTUP,
                message="Service using shared reranker singleton",
                details={"instance_id": id(self._reranker) if self._reranker else None},
            )
        
        self._initialize_components()
        
        self.router: RetrievalRouter = router or RetrievalRouter(
            trace_enabled=trace_enabled,
            metadata_store=self.metadata_store,
            dense_retriever=self.dense_retriever,
            bm25_retriever=self.bm25_retriever,
            expander=self.expander,
            fusion=self.fusion,
            reranker=self._reranker,
            validator=self.validator,
            orchestrator=orchestrator,
        )
        
        self.orchestrator: RetrievalOrchestrator = orchestrator or RetrievalOrchestrator(
            metadata_store=self.metadata_store,
            hybrid_retriever=self.hybrid_retriever,
            reranker=self._reranker,
            validator=self.validator,
            expander=self.expander,
            translator=get_translator(),
            enable_reranking=settings.RERANKER_ENABLED,
            adaptive_pipeline=True,
            trace_enabled=trace_enabled,
        )
        
        self.compressor: ContextCompressor = compressor or ContextCompressor()
        self.context_builder: ContextBuilder = context_builder or ContextBuilder()
        self.trace_enabled: bool = trace_enabled
        self.traces: List[Dict[str, Any]] = []
        
        logger.log_event(
            event=LogEvent.SYSTEM_STARTUP,
            message="Retrieval service initialized",
            details={
                "trace_enabled": trace_enabled,
                "context_expansion_enabled": settings.CONTEXT_EXPANSION_ENABLED,
                "adaptive_pipeline": True,
                "reranker_instance_id": id(self._reranker) if self._reranker else None,
            },
        )
    
    def _initialize_components(self) -> None:
        """Initialize all retrieval components."""
        self.bm25_retriever = BM25Retriever()
        self.dense_retriever = DenseRetriever()
        self.expander = QueryExpander()
        
        self.fusion = ReciprocalRankFusion(
            normalize_scores=settings.HYBRID_NORMALIZE_SCORES,
            normalization_method=settings.HYBRID_NORMALIZATION_METHOD,
            adaptive_k=True,
            strategy_weighting=True,
        )
        
        self.validator = ResultValidator(
            enable_semantic_dedup=settings.VALIDATOR_ENABLE_SEMANTIC_DEDUP,
            semantic_dedup_threshold=settings.VALIDATOR_SEMANTIC_DEDUP_THRESHOLD,
            min_quality_score=settings.VALIDATOR_MIN_QUALITY_SCORE,
        )
        
        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            bm25_retriever=self.bm25_retriever,
            expander=self.expander,
            fusion=self.fusion,
            metadata_store=self.metadata_store,
            normalize_scores=settings.HYBRID_NORMALIZE_SCORES,
            normalization_method=settings.HYBRID_NORMALIZATION_METHOD,
            auto_rebuild_bm25=True,
        )
    
    def _expand_with_context(
        self,
        chunks: List[Chunk],
        max_neighbors: int = 1,
        include_parent: bool = True,
        include_children: bool = False,
    ) -> List[Chunk]:
        """
        Expand chunks with parent and neighbor context.
        """
        if not chunks or not settings.CONTEXT_EXPANSION_ENABLED:
            return chunks
        
        expanded: List[Chunk] = []
        seen_ids: set[str] = set()
        
        for chunk in chunks:
            chunk_id = chunk.metadata.chunk_id
            
            if chunk_id not in seen_ids:
                expanded.append(chunk)
                seen_ids.add(chunk_id)
            
            if include_parent and chunk.metadata.parent_chunk_id:
                parent_id = chunk.metadata.parent_chunk_id
                if parent_id not in seen_ids:
                    parent_meta = self.metadata_store.get_chunk_by_id(parent_id)
                    if parent_meta:
                        parent_chunk = self._reconstruct_chunk_from_metadata(parent_meta)
                        if parent_chunk:
                            expanded.append(parent_chunk)
                            seen_ids.add(parent_id)
            
            if max_neighbors > 0:
                doc_id = chunk.metadata.document_id
                chunk_index = chunk.metadata.chunk_index
                
                if doc_id is not None and chunk_index is not None:
                    doc_chunks = self.metadata_store.get_document_chunks(doc_id)
                    
                    current_pos = None
                    for i, doc_chunk in enumerate(doc_chunks):
                        if doc_chunk.get("chunk_index") == chunk_index:
                            current_pos = i
                            break
                    
                    if current_pos is not None:
                        for i in range(max_neighbors):
                            prev_pos = current_pos - (i + 1)
                            if prev_pos >= 0:
                                prev_meta = doc_chunks[prev_pos]
                                prev_id = prev_meta.get("chunk_id")
                                if prev_id and prev_id not in seen_ids:
                                    prev_chunk = self._reconstruct_chunk_from_metadata(prev_meta)
                                    if prev_chunk:
                                        expanded.append(prev_chunk)
                                        seen_ids.add(prev_id)
                        
                        for i in range(max_neighbors):
                            next_pos = current_pos + (i + 1)
                            if next_pos < len(doc_chunks):
                                next_meta = doc_chunks[next_pos]
                                next_id = next_meta.get("chunk_id")
                                if next_id and next_id not in seen_ids:
                                    next_chunk = self._reconstruct_chunk_from_metadata(next_meta)
                                    if next_chunk:
                                        expanded.append(next_chunk)
                                        seen_ids.add(next_id)
        
        if len(expanded) > len(chunks):
            logger.log_event(
                event=LogEvent.CONTEXT_EXPANSION,
                message=f"Expanded {len(chunks)} chunks to {len(expanded)} chunks with context",
                details={
                    "original_count": len(chunks),
                    "expanded_count": len(expanded),
                    "max_neighbors": max_neighbors,
                    "include_parent": include_parent,
                    "include_children": include_children,
                },
            )
        
        return expanded
    
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
            
            content = meta.get("full_content", meta.get("content_preview", ""))
            
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
                details={"chunk_id": meta.get("chunk_id")},
            )
            return None
    
    def _apply_retrieval_fallback(
        self,
        query: Query,
        chunks: List[Chunk],
        metadata_filters: Optional[Dict[str, Any]],
    ) -> List[Chunk]:
        """
        Apply retrieval fallback when results are weak.
        """
        if not settings.RETRIEVAL_FALLBACK_ENABLED:
            return chunks
        
        if not chunks:
            logger.log_event(
                event=LogEvent.WARNING,
                message="No chunks retrieved, trying fallback without filters",
                level=30,
            )
            try:
                fallback_chunks = self.router.retrieve(query, metadata_filters=None)
                if fallback_chunks:
                    logger.log_event(
                        event=LogEvent.CONTEXT_BUILDING,
                        message=f"Fallback retrieval found {len(fallback_chunks)} chunks without filters",
                        details={"fallback_chunks": len(fallback_chunks)},
                    )
                    return fallback_chunks
            except Exception as exc:
                logger.log_error(
                    event=LogEvent.WARNING,
                    message=f"Fallback retrieval failed: {str(exc)}",
                    exception=exc,
                )
            return chunks
        
        max_score = max((c.reranker_score or 0.0) for c in chunks) if chunks else 0.0
        
        if len(chunks) < settings.RETRIEVAL_FALLBACK_MIN_CHUNKS or max_score < settings.RETRIEVAL_FALLBACK_MIN_SCORE:
            logger.log_event(
                event=LogEvent.WARNING,
                message=f"Weak retrieval results ({len(chunks)} chunks, max_score={max_score:.2f}), trying fallback without filters",
                details={
                    "chunks": len(chunks),
                    "max_score": max_score,
                    "min_chunks": settings.RETRIEVAL_FALLBACK_MIN_CHUNKS,
                    "min_score": settings.RETRIEVAL_FALLBACK_MIN_SCORE,
                },
            )
            
            try:
                fallback_chunks = self.router.retrieve(query, metadata_filters=None)
                if len(fallback_chunks) > len(chunks):
                    logger.log_event(
                        event=LogEvent.CONTEXT_BUILDING,
                        message=f"Fallback retrieval improved results: {len(chunks)} → {len(fallback_chunks)} chunks",
                        details={
                            "original": len(chunks),
                            "fallback": len(fallback_chunks),
                        },
                    )
                    return fallback_chunks
            except Exception as exc:
                logger.log_error(
                    event=LogEvent.WARNING,
                    message=f"Fallback retrieval failed: {str(exc)}",
                    exception=exc,
                )
        
        return chunks
    
    def retrieve(
        self,
        query: Query,
        metadata_filters: Optional[Dict[str, Any]] = None,
        expected_page: Optional[int] = None,
    ) -> Tuple[List[Chunk], str]:
        """
        Retrieve relevant chunks and build context for a query.
        
        Args:
            query: The user query (may have translated_text).
            metadata_filters: Optional metadata filters.
            expected_page: Optional expected page for trace validation.
        
        Returns:
            tuple[list[Chunk], str]: (chunks, context_string)
        """
        query_text = query.get_effective_query()
        
        with measure_latency("retrieval_service") as latency:
            logger.log_retrieval(
                event=LogEvent.QUERY_RECEIVED,
                query_id=query.query_id,
                message=f"Retrieval service processing",
                details={
                    "original_text": query.original_text[:100],
                    "translated_text": query.translated_text[:100] if query.translated_text else None,
                    "was_translated": query.has_translation(),
                },
            )
            
            try:
                candidates = self.router.retrieve(
                    query, 
                    metadata_filters=metadata_filters,
                    expected_page=expected_page,
                )
                
                chunks = candidates
                
                if settings.RETRIEVAL_FALLBACK_ENABLED:
                    chunks = self._apply_retrieval_fallback(query, chunks, metadata_filters)
                
                if not chunks:
                    logger.log_retrieval(
                        event=LogEvent.DENSE_RETRIEVAL_COMPLETE,
                        query_id=query.query_id,
                        message="No relevant chunks found",
                    )
                    return [], ""
                
                if settings.CONTEXT_EXPANSION_ENABLED:
                    chunks = self._expand_with_context(
                        chunks,
                        max_neighbors=settings.CONTEXT_EXPANSION_MAX_NEIGHBORS,
                        include_parent=settings.CONTEXT_EXPANSION_INCLUDE_PARENT,
                        include_children=settings.CONTEXT_EXPANSION_INCLUDE_CHILDREN,
                    )
                
                compressed_chunks = self.compressor.compress(chunks)
                
                if len(compressed_chunks) > settings.MAX_CONTEXT_CHUNKS:
                    compressed_chunks = compressed_chunks[:settings.MAX_CONTEXT_CHUNKS]
                
                context = self.context_builder.build(compressed_chunks)
                
                if self.trace_enabled and hasattr(self.router, 'get_traces'):
                    self.traces = self.router.get_traces()
                
                latency.stop(
                    query_id=query.query_id,
                    chunks_found=len(chunks),
                    final_chunks=len(compressed_chunks),
                )
                
                logger.log_retrieval(
                    event=LogEvent.DENSE_RETRIEVAL_COMPLETE,
                    query_id=query.query_id,
                    message=f"Retrieval complete: {len(compressed_chunks)} chunks in context",
                    details={
                        "chunks_found": len(chunks),
                        "final_chunks": len(compressed_chunks),
                        "context_length": len(context),
                        "duration_ms": latency.duration_ms,
                        "was_translated": query.has_translation(),
                    },
                )
                
                return compressed_chunks, context
                
            except RetrievalError:
                raise
            except Exception as exc:
                raise RetrievalError(
                    message=f"Retrieval service failed: {str(exc)}",
                    query=query_text,
                    error_code="SVC_001",
                    original_exception=exc,
                )
    
    def retrieve_chunks_only(
        self,
        query: Query,
        metadata_filters: Optional[Dict[str, Any]] = None,
        expected_page: Optional[int] = None,
    ) -> List[Chunk]:
        """Retrieve chunks without building context."""
        chunks, _ = self.retrieve(
            query, 
            metadata_filters=metadata_filters,
            expected_page=expected_page,
        )
        return chunks
    
    def rebuild_bm25(self) -> None:
        """Force rebuild BM25 index from metadata."""
        if hasattr(self.router, 'rebuild_bm25'):
            self.router.rebuild_bm25()
            logger.log_event(
                event=LogEvent.INDEX_UPDATE,
                message="BM25 index rebuilt from service layer",
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval service statistics."""
        stats = {
            "router": self.router.get_stats(),
            "orchestrator": self.orchestrator.get_pipeline_config(),
            "context_expansion_enabled": settings.CONTEXT_EXPANSION_ENABLED,
            "max_neighbors": settings.CONTEXT_EXPANSION_MAX_NEIGHBORS,
            "include_children": settings.CONTEXT_EXPANSION_INCLUDE_CHILDREN,
            "fallback_enabled": settings.RETRIEVAL_FALLBACK_ENABLED,
            "trace_enabled": self.trace_enabled,
            "traces_count": len(self.traces),
            "semantic_dedup_enabled": settings.VALIDATOR_ENABLE_SEMANTIC_DEDUP,
            "normalize_scores": settings.HYBRID_NORMALIZE_SCORES,
            "normalization_method": settings.HYBRID_NORMALIZATION_METHOD,
            "reranker_instance_id": id(self._reranker) if self._reranker else None,
        }
        
        if self._reranker:
            stats["reranker"] = self._reranker.get_stats()
        
        return stats
    
    def get_traces(self) -> List[Dict[str, Any]]:
        """Get traces from the last retrieval."""
        return self.traces
    
    def clear_traces(self) -> None:
        """Clear stored traces."""
        self.traces = []
        if hasattr(self.router, 'clear_traces'):
            self.router.clear_traces()
    
    def get_trace_summary(self) -> Dict[str, Any]:
        """Get summary of traces."""
        if not self.traces:
            return {"total": 0, "success": 0, "failed": 0}
        
        total = len(self.traces)
        success = sum(1 for t in self.traces if t.get('failure_category') == 'success')
        failed = total - success
        
        categories = {}
        for trace in self.traces:
            cat = trace.get('failure_category', 'unknown')
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1
        
        return {
            "total": total,
            "success": success,
            "failed": failed,
            "success_rate": success / total if total > 0 else 0,
            "categories": categories,
        }