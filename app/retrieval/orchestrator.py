"""
Retrieval Orchestrator module.

Central coordinator for the retrieval pipeline.
Manages the orchestration of all retrieval strategies in a unified pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, List, Tuple, Dict

from app.config import settings
from app.config.constants import LogEvent
from app.core.exceptions import RetrievalError
from app.monitoring import get_logger, measure_latency
from app.schemas import Chunk, Query
from app.schemas.retrieval import RetrievalCandidate
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.validator import ResultValidator
from app.retrieval.expander import QueryExpander
from app.llm.translator import get_translator
from app.vector_store import MetadataStore


logger = get_logger("food_safety_rag.retrieval.orchestrator")


class QueryComplexity(str, Enum):
    """Query complexity levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class FallbackLevel(str, Enum):
    """Fallback levels."""
    NONE = "none"
    SKIP_RERANKER = "skip_reranker"
    HYBRID_ONLY = "hybrid_only"
    BM25_ONLY = "bm25_only"
    EMERGENCY = "emergency"


@dataclass
class PipelineTrace:
    """
    Unified trace object for all pipeline stages.
    """
    query_id: str
    query_text: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    query_complexity: QueryComplexity = QueryComplexity.MEDIUM
    language: str = "english"
    detected_domain: Optional[str] = None
    word_count: int = 0
    
    translated_text: Optional[str] = None
    was_translated: bool = False
    dense_query: Optional[str] = None
    bm25_query: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    
    dense_pages: List[Tuple[int, float]] = field(default_factory=list)
    bm25_pages: List[Tuple[int, float]] = field(default_factory=list)
    fusion_pages: List[Tuple[int, float]] = field(default_factory=list)
    reranker_pages: List[Tuple[int, float]] = field(default_factory=list)
    final_pages: List[Tuple[int, float]] = field(default_factory=list)
    
    expansion_results: Optional[Dict[str, Any]] = None
    hybrid_retrieval_results: Optional[Dict[str, Any]] = None
    fusion_results: Optional[Dict[str, Any]] = None
    reranking_results: Optional[Dict[str, Any]] = None
    final_results: Optional[List[RetrievalCandidate]] = None
    
    timings: Dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    pipeline_config: Dict[str, Any] = field(default_factory=dict)
    
    fallback_used: bool = False
    fallback_level: str = "none"
    error_occurred: bool = False
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for JSON export."""
        return {
            'query_id': self.query_id,
            'query_text': self.query_text,
            'timestamp': self.timestamp.isoformat(),
            'query_complexity': self.query_complexity.value,
            'language': self.language,
            'detected_domain': self.detected_domain,
            'word_count': self.word_count,
            'translated_text': self.translated_text,
            'was_translated': self.was_translated,
            'dense_query': self.dense_query,
            'bm25_query': self.bm25_query,
            'synonyms': self.synonyms,
            'dense_pages': self.dense_pages[:100],
            'bm25_pages': self.bm25_pages[:100],
            'fusion_pages': self.fusion_pages[:100],
            'reranker_pages': self.reranker_pages[:100],
            'final_pages': self.final_pages[:100],
            'expansion_results': self.expansion_results,
            'hybrid_retrieval_results': self.hybrid_retrieval_results,
            'fusion_results': self.fusion_results,
            'reranking_results': self.reranking_results,
            'final_results_count': len(self.final_results) if self.final_results else 0,
            'timings': {k: round(v, 2) for k, v in self.timings.items() if v is not None},
            'total_latency_ms': round(self.total_latency_ms, 2) if self.total_latency_ms is not None else 0.0,
            'pipeline_config': self.pipeline_config,
            'fallback_used': self.fallback_used,
            'fallback_level': self.fallback_level,
            'error_occurred': self.error_occurred,
            'error_message': self.error_message,
        }


class RetrievalOrchestrator:
    """
    Central orchestrator for the retrieval pipeline.
    
    Pipeline Flow:
    1. Detect language and translate (dual query generation)
    2. Analyze query
    3. Hybrid retrieval (dense + BM25 + fusion)
    4. Cross-encoder reranking
    5. Result validation
    6. Return results
    """

    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        validator: Optional[ResultValidator] = None,
        metadata_store: Optional[MetadataStore] = None,
        expander: Optional[QueryExpander] = None,
        translator: Optional[Any] = None,
        enable_reranking: Optional[bool] = None,
        fusion_candidates: Optional[int] = None,
        reranker_top_k: Optional[int] = None,
        adaptive_pipeline: bool = True,
        trace_enabled: bool = True,
    ) -> None:
        """
        Initialize the retrieval orchestrator.
        """
        self.metadata_store = metadata_store or MetadataStore()
        
        self.hybrid_retriever = hybrid_retriever or HybridRetriever(
            metadata_store=self.metadata_store,
            auto_rebuild_bm25=True,
        )
        self.reranker = reranker or CrossEncoderReranker()
        self.validator = validator or ResultValidator()
        self.translator = translator or get_translator()
        
        self.enable_reranking = enable_reranking if enable_reranking is not None else True
        self.adaptive_pipeline = adaptive_pipeline
        
        self.fusion_candidates = int(fusion_candidates or settings.FUSION_CANDIDATES_BEFORE_MMR)
        self.reranker_top_k = int(reranker_top_k or settings.RERANKER_TOP_K)
        
        self._trace_enabled: bool = trace_enabled
        self._last_trace: Dict[str, Any] = {}
        self._traces: List[PipelineTrace] = []
        self._max_traces: int = 1000
        
        logger.log_event(
            event=LogEvent.SYSTEM_STARTUP,
            message="Retrieval orchestrator initialized",
            details={
                "enable_reranking": self.enable_reranking,
                "adaptive_pipeline": self.adaptive_pipeline,
                "fusion_candidates": self.fusion_candidates,
                "reranker_top_k": self.reranker_top_k,
                "trace_enabled": self._trace_enabled,
            },
        )

    def _detect_language(self, query_text: str) -> str:
        """Detect the language of the query text."""
        if not query_text:
            return 'unknown'
        
        arabic_chars = sum(1 for c in query_text if '\u0600' <= c <= '\u06FF')
        ratio = arabic_chars / len(query_text)
        
        if ratio >= 0.3:
            return 'arabic'
        elif ratio == 0:
            return 'english'
        return 'unknown'

    def _analyze_query(self, query: Query) -> Dict[str, Any]:
        """Analyze query characteristics."""
        query_text = query.get_effective_query()
        word_count = len(query_text.split())
        
        if word_count <= 3:
            complexity = QueryComplexity.SIMPLE
        elif word_count < 10:
            complexity = QueryComplexity.MEDIUM
        else:
            complexity = QueryComplexity.COMPLEX
        
        language = self._detect_language(query.original_text)
        domain = self._detect_domain(query_text)
        
        return {
            'complexity': complexity,
            'language': language,
            'domain': domain,
            'word_count': word_count,
            'has_numbers': any(c.isdigit() for c in query_text),
            'has_question_words': any(w in query_text.lower() for w in ['what', 'how', 'why', 'when', 'where', 'which', 'who']),
        }

    def _detect_domain(self, query_text: str) -> Optional[str]:
        """Detect if query is about specific domain."""
        query_lower = query_text.lower()
        
        domain_keywords = {
            'biological_hazards': ['pathogen', 'bacteria', 'virus', 'microorganism', 'biological', 'salmonella', 'e. coli', 'listeria'],
            'chemical_hazards': ['chemical', 'toxin', 'pesticide', 'additive', 'preservative', 'allergen', 'toxic'],
            'physical_hazards': ['physical', 'extraneous', 'glass', 'metal', 'plastic', 'foreign', 'material'],
            'processing': ['process', 'step', 'procedure', 'cooling', 'heating', 'mixing', 'packaging', 'storage'],
            'nutrition': ['nutrition', 'vitamin', 'mineral', 'nutrient', 'diet', 'supplement'],
            'haccp': ['haccp', 'ccp', 'critical control point', 'hazard analysis'],
            'allergen': ['allergen', 'allergy', 'hypersensitivity', 'reaction'],
        }
        
        for domain, keywords in domain_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return domain
        
        return None

    def _get_adaptive_pipeline_config(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get pipeline config based on query analysis."""
        complexity = analysis['complexity']
        
        config = {
            'enable_expansion': True,
            'fusion_candidates': self.fusion_candidates,
            'reranker_top_k': self.reranker_top_k,
            'enable_reranking': self.enable_reranking,
        }
        
        if not self.adaptive_pipeline:
            return config
        
        if complexity == QueryComplexity.SIMPLE:
            config['fusion_candidates'] = min(50, self.fusion_candidates)
        elif complexity == QueryComplexity.COMPLEX:
            config['fusion_candidates'] = min(100, self.fusion_candidates * 1.5)
        
        return config

    def retrieve(
        self,
        query: Query,
        return_trace: bool = False,
    ) -> List[RetrievalCandidate]:
        """
        Execute the complete retrieval pipeline.

        Pipeline Flow:
        1. Detect language and translate (dual query generation)
        2. Analyze query
        3. Hybrid retrieval (dense + BM25 + fusion)
        4. Cross-encoder reranking
        5. Result validation
        6. Return results
        """
        query_text = query.get_original_query()
        query_id = query.query_id
        
        trace = PipelineTrace(
            query_id=query_id,
            query_text=query_text,
        )
        
        with measure_latency("full_retrieval_pipeline") as pipeline_latency:
            logger.log_retrieval(
                event=LogEvent.DENSE_RETRIEVAL_START,
                query_id=query_id,
                message=f"Starting retrieval pipeline: {query_text[:100]}...",
                details={
                    "query_text": query_text[:100],
                    "enable_reranking": self.enable_reranking,
                    "adaptive_pipeline": self.adaptive_pipeline,
                },
            )

            try:
                # Stage 1: Dual Query Generation (Phase 15.6)
                translation_result = self.translator.translate_and_expand(query_text)
                
                # Get separate queries for Dense and BM25
                dense_query = translation_result.get_dense_query()
                bm25_query = translation_result.get_bm25_query()
                
                trace.dense_query = dense_query
                trace.bm25_query = bm25_query
                trace.synonyms = []
                trace.was_translated = translation_result.was_translated
                trace.translated_text = dense_query if translation_result.was_translated else None
                
                logger.log_event(
                    event=LogEvent.QUERY_EXPANSION,
                    message="Phase 15.6: Dual query generation complete",
                    details={
                        "dense_query": dense_query[:50],
                        "bm25_query": bm25_query[:50],
                        "was_translated": translation_result.was_translated,
                    },
                )

                # Stage 2: Query Analysis
                analysis = self._analyze_query(query)
                trace.query_complexity = analysis['complexity']
                trace.language = analysis['language']
                trace.detected_domain = analysis['domain']
                trace.word_count = analysis['word_count']

                # Stage 3: Adaptive Pipeline Configuration
                pipeline_config = self._get_adaptive_pipeline_config(analysis)
                trace.pipeline_config = pipeline_config
                
                effective_fusion_candidates = pipeline_config.get('fusion_candidates', self.fusion_candidates)
                effective_reranker_top_k = pipeline_config.get('reranker_top_k', self.reranker_top_k)

                # Stage 4: Hybrid Retrieval
                with measure_latency("hybrid_retrieval_stage") as hybrid_latency:
                    try:
                        hybrid_result = self.hybrid_retriever.retrieve(
                            query,
                            candidate_count=effective_fusion_candidates,
                            use_expansion=False,
                            expanded_queries=None,
                            return_trace=True,
                        )
                        
                        fusion_candidates = hybrid_result.get('fused_candidates', [])
                        dense_candidates = hybrid_result.get('dense_candidates', [])
                        bm25_candidates = hybrid_result.get('bm25_candidates', [])
                        
                        trace.dense_pages = self._extract_pages_from_candidates(dense_candidates)
                        trace.bm25_pages = self._extract_pages_from_candidates(bm25_candidates)
                        
                        trace.hybrid_retrieval_results = {
                            'dense_count': len(dense_candidates),
                            'bm25_count': len(bm25_candidates),
                            'fusion_count': len(fusion_candidates),
                            'normalization_used': self.hybrid_retriever.normalize_scores,
                            'phase_15_6_dual_query': True,
                        }
                        trace.fusion_results = {
                            'candidates_after_fusion': len(fusion_candidates),
                            'fusion_method': 'rrf_weighted',
                        }
                        trace.fusion_pages = self._extract_pages_from_candidates(fusion_candidates)
                    except Exception as exc:
                        logger.log_error(
                            event=LogEvent.ERROR,
                            message=f"Hybrid retrieval failed: {str(exc)}",
                            exception=exc,
                        )
                        trace.fallback_used = True
                        trace.fallback_level = FallbackLevel.HYBRID_ONLY.value
                        fusion_candidates = []
                
                trace.timings['hybrid_retrieval'] = hybrid_latency.duration_ms if hybrid_latency.duration_ms is not None else 0.0

                if not fusion_candidates:
                    trace.fallback_used = True
                    trace.fallback_level = FallbackLevel.BM25_ONLY.value
                    try:
                        bm25_fallback = self.hybrid_retriever.bm25_retriever.retrieve(
                            query=bm25_query if bm25_query else dense_query,
                            top_k=effective_reranker_top_k,
                        )
                        fusion_candidates = bm25_fallback
                        trace.fusion_pages = self._extract_pages_from_candidates(fusion_candidates)
                    except Exception as exc:
                        logger.log_error(
                            event=LogEvent.ERROR,
                            message=f"BM25 fallback failed: {str(exc)}",
                            exception=exc,
                        )
                        fusion_candidates = []

                # Stage 5: Reranking
                reranked_candidates = fusion_candidates
                
                if self.enable_reranking and fusion_candidates:
                    with measure_latency("reranking_stage") as rerank_latency:
                        try:
                            reranked_candidates = self.reranker.rerank(
                                query,
                                fusion_candidates,
                                top_k=effective_reranker_top_k,
                            )
                            trace.reranking_results = {
                                'candidates_before': len(fusion_candidates),
                                'candidates_after': len(reranked_candidates),
                            }
                            trace.reranker_pages = self._extract_pages_from_candidates(reranked_candidates)
                        except Exception as exc:
                            logger.log_error(
                                event=LogEvent.ERROR,
                                message=f"Reranking failed: {str(exc)}",
                                exception=exc,
                            )
                            reranked_candidates = fusion_candidates[:effective_reranker_top_k]
                            trace.fallback_used = True
                            trace.fallback_level = FallbackLevel.SKIP_RERANKER.value
                            trace.reranker_pages = self._extract_pages_from_candidates(reranked_candidates)
                    
                    trace.timings['reranking'] = rerank_latency.duration_ms if rerank_latency.duration_ms is not None else 0.0

                # Stage 6: Validation
                validated_candidates = reranked_candidates
                try:
                    validated_candidates = self.validator.validate(
                        query,
                        reranked_candidates,
                    )
                except Exception as exc:
                    logger.log_error(
                        event=LogEvent.WARNING,
                        message=f"Validation failed: {str(exc)}",
                        exception=exc,
                    )
                    validated_candidates = reranked_candidates
                
                trace.final_pages = self._extract_pages_from_candidates(validated_candidates)

                trace.final_results = validated_candidates
                trace.total_latency_ms = pipeline_latency.duration_ms if pipeline_latency.duration_ms is not None else 0.0
                trace.error_occurred = False

                self._store_trace(trace)

                pipeline_latency.stop(
                    query_id=query_id,
                    fusion_results=len(fusion_candidates),
                    reranked_results=len(reranked_candidates),
                    validated_results=len(validated_candidates),
                    use_reranking=self.enable_reranking,
                    adaptive_pipeline=self.adaptive_pipeline,
                )

                logger.log_retrieval(
                    event=LogEvent.DENSE_RETRIEVAL_COMPLETE,
                    query_id=query_id,
                    message=f"Retrieval pipeline complete: {len(validated_candidates)} final candidates",
                    details={
                        "fusion_results": len(fusion_candidates),
                        "reranked_results": len(reranked_candidates),
                        "final_results": len(validated_candidates),
                        "duration_ms": round(pipeline_latency.duration_ms, 2) if pipeline_latency.duration_ms is not None else 0.0,
                        "adaptive_pipeline": self.adaptive_pipeline,
                        "trace_enabled": self._trace_enabled,
                        "phase_15_6_dual_query": True,
                    },
                )

                if return_trace:
                    return validated_candidates
                return validated_candidates

            except Exception as exc:
                trace.error_occurred = True
                trace.error_message = str(exc)
                trace.total_latency_ms = pipeline_latency.duration_ms if pipeline_latency.duration_ms is not None else 0.0
                self._store_trace(trace)

                raise RetrievalError(
                    message=f"Retrieval orchestration failed: {str(exc)}",
                    query=query_text,
                    error_code="ORCH_001",
                    original_exception=exc,
                )

    def _extract_pages_from_candidates(
        self,
        candidates: List[RetrievalCandidate],
        max_items: int = 100,
    ) -> List[Tuple[int, float]]:
        """Extract page numbers and scores from RetrievalCandidate list."""
        pages = []
        if not candidates:
            return pages
        
        for candidate in candidates[:max_items]:
            if candidate is None or candidate.chunk is None:
                continue
            if candidate.chunk.metadata and candidate.chunk.metadata.page is not None:
                try:
                    page = int(candidate.chunk.metadata.page)
                    score = candidate.get_best_score()
                    pages.append((page, float(score)))
                except (ValueError, TypeError):
                    continue
        return pages

    def _store_trace(self, trace: PipelineTrace) -> None:
        """Store trace for later analysis."""
        self._traces.append(trace)
        self._last_trace = trace.to_dict()
        
        if len(self._traces) > self._max_traces:
            self._traces = self._traces[-self._max_traces:]

    def get_traces(self) -> List[Dict[str, Any]]:
        """Get all stored traces."""
        return [t.to_dict() for t in self._traces]

    def get_last_trace(self) -> Dict[str, Any]:
        """Get the last trace."""
        return self._last_trace

    def clear_traces(self) -> None:
        """Clear all stored traces."""
        self._traces = []
        self._last_trace = {}

    def set_trace_enabled(self, enabled: bool) -> None:
        """Enable or disable trace collection."""
        self._trace_enabled = enabled
        if not enabled:
            self.clear_traces()

    def get_trace_summary(self) -> Dict[str, Any]:
        """Get summary of traces."""
        if not self._traces:
            return {"total": 0, "success": 0, "failed": 0}
        
        total = len(self._traces)
        success = sum(1 for t in self._traces if not t.error_occurred)
        failed = total - success
        
        return {
            "total": total,
            "success": success,
            "failed": failed,
            "success_rate": success / total if total > 0 else 0,
        }

    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get complete pipeline configuration."""
        return {
            "enable_reranking": self.enable_reranking,
            "adaptive_pipeline": self.adaptive_pipeline,
            "fusion_candidates": self.fusion_candidates,
            "reranker_top_k": self.reranker_top_k,
            "trace_enabled": self._trace_enabled,
            "translation_enabled": self.translator.enabled,
            "translation_mode": "dual_query",
        }

    def rebuild_bm25(self) -> None:
        """Force rebuild BM25 index from metadata."""
        if hasattr(self.hybrid_retriever, 'rebuild_bm25'):
            self.hybrid_retriever.rebuild_bm25()
            logger.log_event(
                event=LogEvent.INDEX_UPDATE,
                message="BM25 index rebuilt on demand",
            )