"""
Hybrid retrieval module (Phase 15.6 - Dual Query).

This module combines dense semantic retrieval with BM25 lexical retrieval
in a unified interface. Enhanced with dual query generation:
- DENSE query: Natural, conversational, semantic
- BM25 query: Keyword-focused, lexical, domain-specific

Phase 15.6 Changes:
- BM25 receives ONLY the bm25_query (keyword-focused)
- Dense receives dense_query + synonyms for expansion
- Integrated with dual query generation from translator.py
- PRESERVES ALL SCORES in RetrievalCandidate for tracing

PHASE 3: Uses full chunk content.
PHASE 4: Returns RetrievalCandidate instead of modifying Chunk.
PHASE 14: Supports translated queries.
PHASE 15: Fused translation + isolated BM25.
PHASE 15.6: Dual query generation (separate queries for Dense and BM25).
"""

from typing import Any, Optional, Dict, List, Tuple

from app.config import settings
from app.config.constants import LogEvent
from app.core.exceptions import RetrievalError
from app.monitoring import get_logger, measure_latency
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import ReciprocalRankFusion
from app.retrieval.expander import QueryExpander
from app.schemas import Chunk, Query
from app.schemas.retrieval import RetrievalCandidate
from app.vector_store import MetadataStore
from app.llm.translator import get_translator


logger = get_logger("food_safety_rag.retrieval.hybrid")


class HybridRetriever:
    """
    Hybrid retriever that combines dense and sparse retrieval strategies.
    
    Phase 15.6: Dense and BM25 receive different optimized queries.
    - BM25 gets ONLY the keyword-focused query
    - Dense gets the semantic query + synonyms
    
    This class manages both retrieval approaches independently:
    - Dense retrieval captures semantic meaning and conceptual similarity
    - BM25 retrieval captures exact terminology, codes, and identifiers
    
    Results are normalized to the same scale before being combined using 
    Reciprocal Rank Fusion (RRF) which is robust across different scoring algorithms.
    
    Attributes:
        dense_retriever: Dense semantic retrieval component.
        bm25_retriever: BM25 lexical retrieval component.
        expander: Query expansion component.
        fusion: RRF fusion engine.
        dense_top_k: Number of dense results to retrieve.
        bm25_top_k: Number of BM25 results to retrieve.
        dense_weight: Weight for dense results in fusion (0.0 to 1.0).
        bm25_weight: Weight for BM25 results in fusion (0.0 to 1.0).
        normalize_scores: Whether to normalize scores before fusion.
        normalization_method: Method for normalization ('minmax', 'rank', 'zscore').
        metadata_store: Metadata store for rebuilding BM25 index.
        _dense_candidates: Last dense retrieval results (for tracing).
        _bm25_candidates: Last BM25 retrieval results (for tracing).
        _bm25_initialized: Whether BM25 index is initialized.
    """

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        expander: Optional[QueryExpander] = None,
        fusion: Optional[ReciprocalRankFusion] = None,
        metadata_store: Optional[MetadataStore] = None,
        dense_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
        enable_expansion: Optional[bool] = None,
        dense_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        normalize_scores: bool = True,
        normalization_method: str = "minmax",
        auto_rebuild_bm25: bool = True,
    ) -> None:
        """
        Initialize the hybrid retriever.
        
        Phase 15.6: Removed internal expansion logic (now handled by translator).
        """
        self.dense_retriever: DenseRetriever = dense_retriever or DenseRetriever()
        self.bm25_retriever: BM25Retriever = bm25_retriever or BM25Retriever()
        self.expander: QueryExpander = expander or QueryExpander()
        
        self.fusion: ReciprocalRankFusion = fusion or ReciprocalRankFusion(
            normalize_scores=normalize_scores,
            normalization_method=normalization_method,
        )
        
        self.metadata_store: MetadataStore = metadata_store or MetadataStore()
        
        self.dense_top_k: int = int(dense_top_k or settings.DENSE_TOP_K)
        self.bm25_top_k: int = int(bm25_top_k or settings.BM25_TOP_K)
        
        # Expansion is now handled by translator
        self.enable_expansion: bool = False
        
        self.dense_weight: float = float(dense_weight or getattr(settings, 'HYBRID_DENSE_WEIGHT', 0.55))
        self.bm25_weight: float = float(bm25_weight or getattr(settings, 'HYBRID_BM25_WEIGHT', 0.45))
        self.normalize_scores: bool = normalize_scores
        self.normalization_method: str = normalization_method
        
        total_weight = self.dense_weight + self.bm25_weight
        if total_weight > 0:
            self.dense_weight /= total_weight
            self.bm25_weight /= total_weight
        
        self._dense_candidates: List[RetrievalCandidate] = []
        self._bm25_candidates: List[RetrievalCandidate] = []
        self._fusion_trace: Optional[Dict[str, Any]] = None
        
        self._bm25_initialized = False
        if auto_rebuild_bm25:
            self._rebuild_bm25_if_needed()
        
        logger.log_event(
            event=LogEvent.SYSTEM_STARTUP,
            message="Hybrid retriever initialized (Phase 15.6 - Dual Query)",
            details={
                "dense_top_k": self.dense_top_k,
                "bm25_top_k": self.bm25_top_k,
                "enable_expansion": self.enable_expansion,
                "dense_weight": round(self.dense_weight, 3),
                "bm25_weight": round(self.bm25_weight, 3),
                "normalize_scores": self.normalize_scores,
                "normalization_method": self.normalization_method,
                "bm25_initialized": self._bm25_initialized,
                "phase_15_6_dual_query": True,
                "bm25_isolated": True,
            },
        )

    # ==========================================================================
    # BM25 INDEX MANAGEMENT
    # ==========================================================================

    def _rebuild_bm25_if_needed(self) -> bool:
        """Rebuild BM25 index from metadata store if chunks exist."""
        try:
            all_chunks_meta = self.metadata_store.get_all_active_chunks()
            
            if not all_chunks_meta:
                logger.log_event(
                    event=LogEvent.WARNING,
                    message="No active chunks found for BM25 rebuild",
                    level=30,
                )
                return False
            
            reconstructed_chunks: List[Chunk] = []
            for meta in all_chunks_meta:
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
                    
                    chunk = Chunk(
                        content=meta.get("full_content", meta.get("content_preview", "")),
                        metadata=chunk_metadata,
                        embedding_id=meta.get("faiss_index_id"),
                    )
                    reconstructed_chunks.append(chunk)
                except Exception as exc:
                    logger.log_error(
                        event=LogEvent.WARNING,
                        message=f"Failed to reconstruct chunk: {str(exc)}",
                        exception=exc,
                        details={"chunk_id": meta.get("chunk_id")},
                    )
                    continue
            
            if not reconstructed_chunks:
                logger.log_event(
                    event=LogEvent.WARNING,
                    message="No chunks could be reconstructed for BM25 rebuild",
                    level=30,
                )
                return False
            
            self.bm25_retriever.index(reconstructed_chunks)
            self._bm25_initialized = True
            
            logger.log_event(
                event=LogEvent.INDEX_UPDATE,
                message=f"BM25 index rebuilt with {len(reconstructed_chunks)} chunks",
                details={"chunk_count": len(reconstructed_chunks)},
            )
            return True
            
        except Exception as exc:
            logger.log_error(
                event=LogEvent.WARNING,
                message=f"Failed to rebuild BM25 index: {str(exc)}",
                exception=exc,
            )
            self._bm25_initialized = False
            return False

    def _ensure_bm25_ready(self) -> bool:
        """Ensure BM25 is ready for retrieval."""
        stats = self.bm25_retriever.get_stats()
        if stats.get('corpus_size', 0) > 0:
            self._bm25_initialized = True
            return True
        
        return self._rebuild_bm25_if_needed()

    # ==========================================================================
    # RETRIEVAL METHODS - Phase 15.6: Dual Query
    # ==========================================================================

    def retrieve(
        self,
        query: Query,
        candidate_count: Optional[int] = None,
        use_expansion: Optional[bool] = None,
        expanded_queries: Optional[List[Tuple[str, float, int, int]]] = None,
        return_trace: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform hybrid retrieval combining dense and BM25 results.

        Phase 15.6: Dense and BM25 use different optimized queries.
        - BM25 uses the keyword-focused query
        - Dense uses the semantic query + synonyms
        Uses dual query generation from translator.py.

        RETURNS: Dictionary with:
        - 'dense_candidates': List of RetrievalCandidate from dense retrieval
        - 'bm25_candidates': List of RetrievalCandidate from BM25 retrieval
        - 'fused_candidates': List of RetrievalCandidate after fusion
        
        PRESERVES ALL SCORES in RetrievalCandidate for tracing.
        """
        with measure_latency("hybrid_retrieval") as latency:
            query_text = query.get_effective_query()
            
            # Ensure candidate_count is int
            if candidate_count is None:
                candidate_count = getattr(settings, 'FUSION_CANDIDATES_BEFORE_MMR', 40)
            candidate_count = int(candidate_count)
            
            bm25_ready = self._ensure_bm25_ready()
            
            logger.log_retrieval(
                event=LogEvent.DENSE_RETRIEVAL_START,
                query_id=query.query_id,
                message=f"Phase 15.6: Starting hybrid retrieval with dual queries: {query_text[:100]}...",
                details={
                    "query_text": query_text,
                    "dense_top_k": self.dense_top_k,
                    "bm25_top_k": self.bm25_top_k,
                    "dense_weight": round(self.dense_weight, 3),
                    "bm25_weight": round(self.bm25_weight, 3),
                    "bm25_ready": bm25_ready,
                    "requested_candidates": candidate_count,
                    "normalize_scores": self.normalize_scores,
                    "phase_15_6_dual_query": True,
                    "bm25_isolated": True,
                },
            )

            # ================================================================
            # Phase 15.6: Dual Query Generation
            # ================================================================
            
            translator = get_translator()
            query_result = translator.translate_and_expand(query_text)
            
            dense_query = query_result.get_dense_query()
            bm25_query = query_result.get_bm25_query()
            all_queries = query_result.get_all_queries()
            weighted_queries = query_result.get_weighted_queries()
            
            logger.log_event(
                event=LogEvent.QUERY_EXPANSION,
                message="Phase 15.6: Dual query generation complete",
                details={
                    "dense_query": dense_query[:50],
                    "bm25_query": bm25_query[:50],
                    "total_variants": len(all_queries),
                    "was_translated": query_result.was_translated,
                },
            )
            
            # ================================================================
            # Phase 15.6: BM25 gets ONLY bm25_query (keyword-focused)
            # ================================================================
            
            all_dense_candidates: List[RetrievalCandidate] = []
            all_bm25_candidates: List[RetrievalCandidate] = []
            dense_success = False
            bm25_success = False
            
            # ================================================================
            # BM25 RETRIEVAL - Phase 15.6: bm25_query ONLY
            # ================================================================
            
            if bm25_ready and bm25_query:
                try:
                    logger.log_event(
                        event=LogEvent.BM25_RETRIEVAL_START,
                        message="Phase 15.6: BM25 retrieving keyword-focused query",
                        details={"bm25_query": bm25_query[:50]},
                    )
                    
                    bm25_results = self.bm25_retriever.retrieve(
                        query=bm25_query,
                        top_k=self.bm25_top_k,
                    )
                    
                    all_bm25_candidates = bm25_results
                    bm25_success = len(all_bm25_candidates) > 0
                    
                    logger.log_retrieval(
                        event=LogEvent.BM25_RETRIEVAL_COMPLETE,
                        query_id=query.query_id,
                        message=f"Phase 15.6: BM25 returned {len(all_bm25_candidates)} candidates",
                        details={
                            "bm25_candidates": len(all_bm25_candidates),
                            "bm25_query": bm25_query[:50],
                            "phase_15_6_isolated": True,
                        },
                    )
                except Exception as exc:
                    logger.log_error(
                        event=LogEvent.ERROR,
                        message=f"BM25 retrieval failed: {str(exc)}",
                        exception=exc,
                    )
                    bm25_success = False
            else:
                if not bm25_ready:
                    logger.log_event(
                        event=LogEvent.WARNING,
                        message="BM25 not ready, skipping BM25 retrieval",
                    )
                elif not bm25_query:
                    logger.log_event(
                        event=LogEvent.WARNING,
                        message="BM25 query empty, skipping BM25 retrieval",
                    )
            
            # ================================================================
            # DENSE RETRIEVAL - Phase 15.6: dense_query + synonyms
            # ================================================================
            
            try:
                # Build dense variants from weighted queries
                dense_variants = []
                for variant_text, weight in weighted_queries:
                    dense_variants.append(
                        (variant_text, weight, self.dense_top_k, self.bm25_top_k)
                    )
                
                logger.log_event(
                    event=LogEvent.DENSE_RETRIEVAL_START,
                    message=f"Phase 15.6: Dense retrieving {len(dense_variants)} variants",
                    details={
                        "variants": [q for q, _, _, _ in dense_variants],
                        "weights": [w for _, w, _, _ in dense_variants],
                    },
                )
                
                # Create a temporary query for dense retrieval
                temp_query = Query(
                    query_id=f"{query.query_id}_dense",
                    original_text=dense_query,
                )
                
                dense_multi_results = self.dense_retriever.retrieve_multi_query(
                    temp_query,
                    dense_variants,
                )
                
                # Collect unique dense results
                chunk_scores_dense: Dict[str, RetrievalCandidate] = {}
                for _, _, results in dense_multi_results:
                    for candidate in results:
                        chunk_id = candidate.chunk.metadata.chunk_id
                        if (chunk_id not in chunk_scores_dense or 
                            (candidate.dense_score or 0.0) > (chunk_scores_dense[chunk_id].dense_score or 0.0)):
                            chunk_scores_dense[chunk_id] = candidate
                
                all_dense_candidates = list(chunk_scores_dense.values())
                all_dense_candidates.sort(key=lambda x: x.dense_score or 0.0, reverse=True)
                all_dense_candidates = all_dense_candidates[:self.dense_top_k]
                
                dense_success = len(all_dense_candidates) > 0
                
                logger.log_retrieval(
                    event=LogEvent.DENSE_RETRIEVAL_COMPLETE,
                    query_id=query.query_id,
                    message=f"Phase 15.6: Dense returned {len(all_dense_candidates)} candidates",
                    details={
                        "dense_candidates": len(all_dense_candidates),
                        "variants_used": len(dense_variants),
                        "phase_15_6_dual": True,
                    },
                )
            except Exception as exc:
                logger.log_error(
                    event=LogEvent.ERROR,
                    message=f"Dense retrieval failed: {str(exc)}",
                    exception=exc,
                )
                dense_success = False
            
            # ================================================================
            # Check if any retrieval succeeded
            # ================================================================
            
            if not dense_success and not bm25_success:
                raise RetrievalError(
                    message="Both dense and BM25 retrieval failed",
                    query=query_text,
                    error_code="RET_001",
                )
            
            # ================================================================
            # FUSION - Phase 15.6: Apply strategy weights
            # ================================================================
            
            if dense_success and bm25_success:
                # Apply strategy weights
                weighted_dense = []
                for candidate in all_dense_candidates:
                    new_candidate = RetrievalCandidate(
                        chunk=candidate.chunk,
                        dense_score=(candidate.dense_score or 0.0) * self.dense_weight,
                        bm25_score=candidate.bm25_score,
                    )
                    weighted_dense.append(new_candidate)
                
                weighted_bm25 = []
                for candidate in all_bm25_candidates:
                    new_candidate = RetrievalCandidate(
                        chunk=candidate.chunk,
                        dense_score=candidate.dense_score,
                        bm25_score=(candidate.bm25_score or 0.0) * self.bm25_weight,
                    )
                    weighted_bm25.append(new_candidate)
                
                fused_candidates = self.fusion.fuse(
                    weighted_dense,
                    weighted_bm25,
                    candidate_count=candidate_count,
                )
                
                self._fusion_trace = {
                    "dense_count": len(all_dense_candidates),
                    "bm25_count": len(all_bm25_candidates),
                    "fused_count": len(fused_candidates),
                    "dense_weight": self.dense_weight,
                    "bm25_weight": self.bm25_weight,
                    "phase_15_6_dual_query": True,
                }
                
            elif dense_success:
                fused_candidates = all_dense_candidates[:candidate_count]
                self._fusion_trace = {
                    "dense_count": len(all_dense_candidates),
                    "bm25_count": 0,
                    "fused_count": len(fused_candidates),
                    "strategy": "dense_only",
                    "phase_15_6_dual_query": True,
                }
            else:
                fused_candidates = all_bm25_candidates[:candidate_count]
                self._fusion_trace = {
                    "dense_count": 0,
                    "bm25_count": len(all_bm25_candidates),
                    "fused_count": len(fused_candidates),
                    "strategy": "bm25_only",
                    "phase_15_6_dual_query": True,
                }
            
            self._dense_candidates = all_dense_candidates
            self._bm25_candidates = all_bm25_candidates
            
            latency.stop(
                query_id=query.query_id,
                dense_candidates=len(all_dense_candidates),
                bm25_candidates=len(all_bm25_candidates),
                fused_candidates=len(fused_candidates),
                bm25_ready=bm25_ready,
                requested_candidates=candidate_count,
                phase_15_6_dual_query=True,
            )
            
            logger.log_retrieval(
                event=LogEvent.RRF_FUSION,
                query_id=query.query_id,
                message=f"Phase 15.6: Hybrid retrieval complete: {len(fused_candidates)} fused candidates",
                details={
                    "dense_candidates": len(all_dense_candidates),
                    "bm25_candidates": len(all_bm25_candidates),
                    "fused_candidates": len(fused_candidates),
                    "duration_ms": latency.duration_ms,
                    "bm25_ready": bm25_ready,
                    "normalized": self.normalize_scores,
                    "requested_candidates": candidate_count,
                    "phase_15_6_dual_query": True,
                },
            )
            
            result = {
                'dense_candidates': all_dense_candidates,
                'bm25_candidates': all_bm25_candidates,
                'fused_candidates': fused_candidates,
            }
            
            if return_trace:
                result['trace'] = {
                    'normalization_used': self.normalize_scores,
                    'normalization_method': self.normalization_method,
                    'dense_weight': self.dense_weight,
                    'bm25_weight': self.bm25_weight,
                    'fusion_info': self._fusion_trace,
                    'latency_ms': latency.duration_ms,
                    'phase_15_6_dual_query': True,
                    'dense_query': dense_query,
                    'bm25_query': bm25_query,
                }
            
            return result

    def retrieve_batch(
        self,
        queries: List[Query],
        candidate_count: Optional[int] = None,
        expanded_queries_list: Optional[List[Optional[List[Tuple[str, float, int, int]]]]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid retrieval for multiple queries."""
        results: List[Dict[str, Any]] = []
        
        for idx, query in enumerate(queries):
            try:
                expanded_queries = None
                if expanded_queries_list and idx < len(expanded_queries_list):
                    expanded_queries = expanded_queries_list[idx]
                
                result = self.retrieve(
                    query, 
                    candidate_count=candidate_count,
                    expanded_queries=expanded_queries,
                )
                results.append(result)
            except RetrievalError as exc:
                logger.log_error(
                    event=LogEvent.ERROR,
                    message=f"Hybrid retrieval failed for query {query.query_id}: {str(exc)}",
                    exception=exc,
                )
                results.append({
                    'dense_candidates': [],
                    'bm25_candidates': [],
                    'fused_candidates': [],
                })
        
        return results

    # ==========================================================================
    # INDEX MANAGEMENT
    # ==========================================================================

    def index_bm25(self, chunks: List[Chunk]) -> None:
        """Index chunks for BM25 retrieval."""
        self.bm25_retriever.index(chunks)
        self._bm25_initialized = True
        
        logger.log_event(
            event=LogEvent.INDEX_UPDATE,
            message=f"BM25 index built with {len(chunks)} chunks",
            details={"chunk_count": len(chunks)},
        )

    def add_to_bm25(self, chunks: List[Chunk]) -> None:
        """Add new chunks to the BM25 index."""
        self.bm25_retriever.add(chunks)
        self._bm25_initialized = True

    def add_to_dense(self, chunks: List[Chunk]) -> None:
        """Add new chunks to the dense vector store."""
        self.dense_retriever.vector_store.add(chunks)

    def save(self) -> None:
        """Save both dense and BM25 indexes."""
        self.dense_retriever.vector_store.save()

    def load(self) -> None:
        """Load dense index."""
        self.dense_retriever.vector_store.load()

    def rebuild_bm25(self) -> None:
        """Force rebuild BM25 from metadata store."""
        self._rebuild_bm25_if_needed()

    # ==========================================================================
    # STATISTICS
    # ==========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get hybrid retriever statistics."""
        return {
            "dense": self.dense_retriever.get_stats(),
            "bm25": self.bm25_retriever.get_stats(),
            "fusion": self.fusion.get_config(),
            "expansion_enabled": self.enable_expansion,
            "dense_weight": round(self.dense_weight, 3),
            "bm25_weight": round(self.bm25_weight, 3),
            "normalize_scores": self.normalize_scores,
            "normalization_method": self.normalization_method,
            "bm25_initialized": self._bm25_initialized,
            "last_dense_candidates": len(self._dense_candidates),
            "last_bm25_candidates": len(self._bm25_candidates),
            "fusion_trace": self._fusion_trace,
            "phase_15_6_dual_query": True,
            "candidate_count_always_int": True,
        }

    def get_last_results(self) -> Dict[str, Any]:
        """Get the last retrieval results for tracing."""
        return {
            'dense_candidates': self._dense_candidates,
            'bm25_candidates': self._bm25_candidates,
            'fusion_trace': self._fusion_trace,
        }