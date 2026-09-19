"""
Retrieval router module.

This module routes queries to appropriate retrieval strategies and
orchestrates the complete retrieval pipeline including expansion,
dense search, BM25 search, fusion, and reranking.
"""

import re
import unicodedata
from typing import Any, Optional, List, Dict, Tuple

from app.config import settings
from app.config.constants import LogEvent
from app.core.exceptions import RetrievalError, QueryValidationError
from app.monitoring import get_logger, measure_latency
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.expander import QueryExpander
from app.retrieval.fusion import ReciprocalRankFusion
from app.retrieval.orchestrator import RetrievalOrchestrator
from app.retrieval.validator import ResultValidator
from app.retrieval.reranker import CrossEncoderReranker
from app.vector_store import MetadataStore
from app.schemas import Chunk, Query
from app.llm.translator import get_translator


logger = get_logger("food_safety_rag.retrieval.router")


class RetrievalRouter:
    """
    Central retrieval router that orchestrates the complete retrieval pipeline.
    
    Attributes:
        expander: Query expansion component.
        dense_retriever: Dense retriever component.
        bm25_retriever: BM25 retriever component.
        fusion: RRF fusion component.
        reranker: Cross-encoder reranker component.
        validator: Result validator component.
        orchestrator: Retrieval orchestrator component.
        metadata_store: Metadata store component.
        trace_enabled: Whether trace collection is enabled.
    """
    
    MIN_CONTENT_WORDS: int = 1
    MIN_MEANINGFUL_CHARS: int = 5
    MIN_UNIQUE_CHARS: int = 3
    MAX_QUERY_LENGTH: int = 10000
    MIN_QUERY_LENGTH: int = 2
    
    STOP_WORDS: set[str] = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall",
        "can", "need", "to", "of", "in", "for", "on", "with", "at",
        "by", "from", "as", "into", "through", "during", "before",
        "after", "above", "below", "between", "under", "and", "but",
        "or", "yet", "so", "if", "because", "although", "though",
        "while", "where", "when", "that", "which", "who", "whom",
        "whose", "what", "this", "these", "those", "i", "you", "he",
        "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "its", "our", "their", "mine", "yours",
        "hers", "ours", "theirs",
    }
    
    def __init__(
        self,
        expander: Optional[QueryExpander] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        fusion: Optional[ReciprocalRankFusion] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        validator: Optional[ResultValidator] = None,
        orchestrator: Optional[RetrievalOrchestrator] = None,
        metadata_store: Optional[MetadataStore] = None,
        trace_enabled: bool = False,
    ) -> None:
        """
        Initialize the retrieval router.
        """
        self.expander: QueryExpander = expander or QueryExpander()
        self.dense_retriever: DenseRetriever = dense_retriever or DenseRetriever()
        self.bm25_retriever: BM25Retriever = bm25_retriever or BM25Retriever()
        self.fusion: ReciprocalRankFusion = fusion or ReciprocalRankFusion()
        self.metadata_store: MetadataStore = metadata_store or MetadataStore()
        self.trace_enabled: bool = trace_enabled
        
        self.reranker = reranker
        self.validator = validator
        if self.reranker is None and settings.RERANKER_ENABLED:
            from app.retrieval.reranker import get_reranker
            self.reranker = get_reranker()
            logger.log_event(
                event=LogEvent.SYSTEM_STARTUP,
                message="Router using shared reranker singleton",
                details={"instance_id": id(self.reranker) if self.reranker else None},
            )
        
        self.orchestrator: RetrievalOrchestrator = orchestrator or RetrievalOrchestrator(
            hybrid_retriever=None,
            reranker=self.reranker,
            validator=validator,
            metadata_store=self.metadata_store,
            expander=self.expander,
            translator=get_translator(),
        )
        self.orchestrator.set_trace_enabled(trace_enabled)
        
        self._enable_query_quality_check: bool = getattr(settings, "ENABLE_QUERY_QUALITY_CHECK", True)
        self._min_content_words: int = getattr(settings, "MIN_CONTENT_WORDS", self.MIN_CONTENT_WORDS)
        self._min_meaningful_chars: int = getattr(settings, "MIN_MEANINGFUL_CHARS", self.MIN_MEANINGFUL_CHARS)
        
        logger.log_event(
            event=LogEvent.SYSTEM_STARTUP,
            message="Retrieval router initialized",
            details={
                "trace_enabled": trace_enabled,
                "reranker_enabled": settings.RERANKER_ENABLED,
                "query_quality_check_enabled": self._enable_query_quality_check,
                "min_content_words": self._min_content_words,
                "min_meaningful_chars": self._min_meaningful_chars,
            },
        )
    
    def _validate_query_length(self, query_text: str) -> Tuple[bool, str]:
        """
        Validate query length.
        
        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        if not query_text or not query_text.strip():
            return False, "Query is empty"
        
        stripped = query_text.strip()
        
        if len(stripped) < self.MIN_QUERY_LENGTH:
            return False, f"Query too short: {len(stripped)} chars (minimum {self.MIN_QUERY_LENGTH})"
        
        if len(stripped) > self.MAX_QUERY_LENGTH:
            return False, f"Query too long: {len(stripped)} chars (maximum {self.MAX_QUERY_LENGTH})"
        
        return True, "Valid length"
    
    def _validate_query_content(self, query_text: str) -> Tuple[bool, str]:
        """
        Validate query content quality.
        
        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        stripped = query_text.strip()
        
        if len(set(stripped)) < self.MIN_UNIQUE_CHARS:
            return False, f"Too few unique characters: {len(set(stripped))} (minimum {self.MIN_UNIQUE_CHARS})"
        
        meaningful_chars = sum(1 for c in stripped if c.isalnum())
        if meaningful_chars < self.MIN_MEANINGFUL_CHARS:
            return False, f"Too few meaningful characters: {meaningful_chars} (minimum {self.MIN_MEANINGFUL_CHARS})"
        
        words = stripped.split()
        content_words = [
            w for w in words 
            if w.lower() not in self.STOP_WORDS 
            and len(w) > 2
            and w.isalnum()
        ]
        
        if len(content_words) < self._min_content_words:
            # Fallback: accept query if it has at least 1 real content word
            # (i.e. not a stop word) AND is at least 3 chars long.
            real_words = [
                w for w in words
                if w.lower() not in self.STOP_WORDS
                and len(w) > 2
                and w.isalnum()
            ]
            if len(real_words) >= 1:
                return True, "Valid content (fallback)"
            return False, f"Too few content words: {len(content_words)} (minimum {self._min_content_words})"
        
        return True, "Valid content"
    
    def _validate_query_normalization(self, query_text: str) -> Tuple[bool, str]:
        """
        Check if query is properly normalized.
        
        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        if "  " in query_text:
            return False, "Query contains excessive whitespace (multiple spaces)"
        
        if query_text != query_text.strip():
            return False, "Query has leading or trailing whitespace"
        
        return True, "Valid normalization"
    
    def _validate_query_quality(self, query_text: str) -> Tuple[bool, str]:
        """
        Validate query quality before processing.
        
        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        if not self._enable_query_quality_check:
            return True, "Quality check disabled"
        
        normalized = unicodedata.normalize("NFKC", query_text)
        normalized = " ".join(normalized.split())
        
        is_valid, reason = self._validate_query_length(normalized)
        if not is_valid:
            return False, reason
        
        is_valid, reason = self._validate_query_content(normalized)
        if not is_valid:
            return False, reason
        
        is_valid, reason = self._validate_query_normalization(normalized)
        if not is_valid:
            return False, reason
        
        return True, "Valid query"
    
    def _extract_metadata_filters(self, query_text: str) -> Dict[str, Any]:
        """
        Extract metadata filters from query text — SAFELY.

        RULES (fixed):
            - Only extract a document name when the query EXPLICITLY
              references a filename with an extension
              (e.g. "in X.pdf", "from Y.docx", "the Z.md file").
            - The extracted name MUST match an existing document
              in the metadata store; otherwise no filter is applied.
            - Only apply 'page' filters when the query explicitly says
              "page N" or "p. N" — never guess.

        This prevents false filters from phrases like
        "hazards identified in the document" which would otherwise
        be misinterpreted as a document name.
        """
        filters: Dict[str, Any] = {}

        if not query_text:
            return filters

        # --------------------------------------------------------
        # 1) Look for an explicit filename (with extension)
        # --------------------------------------------------------
        # Match things like: "in X.pdf", "from the Y.docx file",
        #                    "according to Z.md", "based on report_v2.pdf"
        filename_pattern = re.compile(
            r"(?:in|from|per|according to|based on|see|within|inside|read|use)"
            r"\s+(?:the\s+)?"
            r"([A-Za-z0-9_\-\s\.]+\.(?:pdf|docx|doc|txt|md|html|xlsx|xls))",
            re.IGNORECASE,
        )

        doc_name = None
        for match in filename_pattern.finditer(query_text):
            candidate = match.group(1).strip()
            # Trim leading/trailing whitespace and quotes
            candidate = candidate.strip().strip("'\"")
            # Reject empty / absurdly long candidates
            if 3 <= len(candidate) <= 120:
                doc_name = candidate
                break

        # --------------------------------------------------------
        # 2) Validate against actual documents in the store
        # --------------------------------------------------------
        if doc_name:
            if self._document_exists(doc_name):
                filters["document_name_contains"] = doc_name
                logger.log_event(
                    event=LogEvent.METADATA_FILTER_APPLIED,
                    message=f"Applied document filter: {doc_name}",
                )
            else:
                logger.log_event(
                    event=LogEvent.WARNING,
                    message=(
                        f"Extracted document name '{doc_name}' "
                        f"does not match any indexed document — "
                        f"ignoring filter."
                    ),
                    level=30,
                )

        # --------------------------------------------------------
        # 3) Explicit page reference only (page N / p. N)
        # --------------------------------------------------------
        page_pattern = re.compile(r"\b(?:page|p\.?)\s*(\d{1,4})\b", re.IGNORECASE)
        page_match = page_pattern.search(query_text)
        if page_match:
            try:
                page_num = int(page_match.group(1))
                if 1 <= page_num <= 2000:
                    filters["page"] = page_num
            except ValueError:
                pass

        return filters

    def _document_exists(self, doc_name: str) -> bool:
        """
        Check whether a document name is present in the metadata store.

        Uses a case-insensitive substring match against every
        document_name in the store.
        """
        try:
            documents = self.metadata_store.list_documents()
        except Exception:
            return False

        if not documents:
            return False

        target = doc_name.lower().strip()
        for doc in documents:
            name = (doc.get("document_name") or "").lower()
            if not name:
                continue
            # Substring match either way
            if target in name or name in target:
                return True
        return False

    def retrieve(
        self,
        query: Query,
        metadata_filters: Optional[Dict[str, Any]] = None,
        expected_page: Optional[int] = None,
    ) -> List[Chunk]:
        """
        Execute the complete retrieval pipeline.

        Pipeline Flow:
        1. Query validation and normalization
        2. Query quality check
        3. Fused translation
        4. Smart metadata filters
        5. Delegates to Orchestrator
        6. Result validation
        """
        with measure_latency("total_retrieval") as total_latency:
            query_text = query.get_effective_query()
            
            logger.log_retrieval(
                event=LogEvent.QUERY_RECEIVED,
                query_id=query.query_id,
                message=f"Retrieval pipeline started",
                details={"query_text": query_text[:100]},
            )
            
            try:
                self._validate_query(query)
                query = self._normalize_query(query)
                
                is_valid, reason = self._validate_query_quality(query_text)
                if not is_valid:
                    logger.log_event(
                        event=LogEvent.WARNING,
                        message=f"Query rejected by quality check: {reason}",
                        details={"query": query_text[:100], "reason": reason},
                    )
                    raise QueryValidationError(
                        message=f"Query validation failed: {reason}",
                        query=query_text,
                        validation_reason=reason,
                    )
                
                logger.log_event(
                    event=LogEvent.SYSTEM_STARTUP,
                    message="Query quality check passed",
                    details={"query": query_text[:50]},
                )
                
                effective_filters = metadata_filters
                if effective_filters is None:
                    extracted_filters = self._extract_metadata_filters(query_text)
                    meaningful_filters = {}
                    for k, v in extracted_filters.items():
                        if v and str(v).strip() and len(str(v).strip()) > 1:
                            meaningful_filters[k] = v
                    if meaningful_filters:
                        effective_filters = meaningful_filters
                        logger.log_event(
                            event=LogEvent.METADATA_FILTER_APPLIED,
                            message=f"Applied metadata filters: {list(meaningful_filters.keys())}",
                        )
                
                results = self.orchestrator.retrieve(query)
                final_chunks = [candidate.chunk for candidate in results]
                
                seen_ids: set[str] = set()
                deduplicated: List[Chunk] = []
                for chunk in final_chunks:
                    chunk_id = chunk.metadata.chunk_id
                    if chunk_id not in seen_ids:
                        seen_ids.add(chunk_id)
                        deduplicated.append(chunk)
                
                total_latency.stop(
                    query_id=query.query_id,
                    final_chunks=len(deduplicated),
                    reranker_enabled=settings.RERANKER_ENABLED,
                    filters_applied=bool(effective_filters),
                )
                
                logger.log_retrieval(
                    event=LogEvent.DENSE_RETRIEVAL_COMPLETE,
                    query_id=query.query_id,
                    message=f"Retrieval pipeline complete: {len(deduplicated)} final chunks",
                    details={
                        "final_chunks": len(deduplicated),
                        "total_duration_ms": total_latency.duration_ms,
                    },
                )
                
                return deduplicated
                
            except QueryValidationError:
                raise
            except RetrievalError:
                raise
            except Exception as exc:
                raise RetrievalError(
                    message=f"Retrieval pipeline failed: {str(exc)}",
                    query=query_text,
                    error_code="RET_001",
                    original_exception=exc,
                )
    
    def _validate_query(self, query: Query) -> None:
        """Basic query validation."""
        query_text = query.get_effective_query()
        if not query_text or not query_text.strip():
            raise RetrievalError(
                message="Query is empty or whitespace-only",
                query=query_text,
                error_code="RET_006",
            )
        if len(query_text) < 2:
            raise RetrievalError(
                message=f"Query too short ({len(query_text)} chars, minimum 2)",
                query=query_text,
                error_code="RET_006",
            )
        if len(query_text) > 10000:
            raise RetrievalError(
                message=f"Query too long ({len(query_text)} chars, maximum 10000)",
                query=query_text,
                error_code="RET_006",
            )
    
    def _normalize_query(self, query: Query) -> Query:
        """Normalize query text."""
        text = query.get_effective_query()
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", text)
        text = " ".join(text.split())
        text = re.sub(r"[!?]{2,}", "!", text)
        text = re.sub(r"\.{2,}", ".", text)
        query.normalized_text = text
        return query
    
    def get_traces(self) -> List[Dict[str, Any]]:
        """Get traces from orchestrator."""
        return self.orchestrator.get_traces()
    
    def clear_traces(self) -> None:
        """Clear traces from orchestrator."""
        self.orchestrator.clear_traces()
    
    def rebuild_bm25(self) -> None:
        """Rebuild BM25 index."""
        self.orchestrator.rebuild_bm25()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        return {
            "expander": {"num_variants": self.expander.num_variants},
            "adaptive_retrieval_enabled": settings.ADAPTIVE_RETRIEVAL_ENABLED,
            "trace_enabled": self.trace_enabled,
            "query_quality_check_enabled": self._enable_query_quality_check,
            "min_content_words": self._min_content_words,
            "min_meaningful_chars": self._min_meaningful_chars,
            "orchestrator": self.orchestrator.get_pipeline_config(),
            "validator": self.validator.get_config() if hasattr(self.validator, 'get_config') else {},
            "reranker_enabled": settings.RERANKER_ENABLED,
        }