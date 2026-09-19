# ============================================================================
# FILE: core/retrieval/bm25.py
# ============================================================================
"""
BM25 sparse retrieval module - REVISED.

Key fixes vs previous version:
  - Removed domain stop-words that stripped critical food-safety terms
    (haccp, temperature, contamination, pathogen, etc). BM25's own IDF
    weighting already handles high-frequency low-information terms;
    manually deleting domain-content words destroys recall for exactly
    the queries this system exists to answer.
  - Numeric tokens are no longer discarded (critical limits, temperatures,
    durations are almost always numeric in this domain).
  - Minimum token length lowered + an explicit allow-list protects short
    but meaningful acronyms (pH, GMP, CCP, SOP...).
  - Restored optional Arabic tokenization/normalization.
  - Restored true multi-query retrieval (weighted merge across variants)
    instead of silently dropping all but the first variant.
  - Real per-call query_id instead of a hardcoded constant.
  - Bounded tokenization cache (simple LRU) to avoid unbounded memory growth.
  - Basic thread-safety around shared mutable state.
"""

import hashlib
import re
import threading
from collections import OrderedDict
from typing import Any, Optional, List, Tuple, Dict, Set

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import settings
from app.config.constants import BM25_B, BM25_K1, LogEvent
from app.core.exceptions import BM25Error
from app.monitoring import get_logger, measure_latency
from app.schemas import Chunk, Query
from app.schemas.retrieval import RetrievalCandidate


logger = get_logger("food_safety_rag.retrieval.bm25")


class BM25Retriever:
    """
    BM25 sparse retriever - REVISED.

    Attributes:
        corpus: List of chunk texts in the index.
        chunks: Parallel list of Chunk objects.
        bm25: BM25Okapi index instance.
        tokenized_corpus: Tokenized corpus for BM25.
        top_k: Number of results to retrieve.
        _last_results: Last retrieved results for tracing.
        _last_query_id: Last query ID for tracing.
        _tokenization_cache: Bounded cache for tokenization results.
    """

    # ========================================================================
    # BASE ENGLISH STOP WORDS (deduplicated, genuinely low-information words)
    # ========================================================================
    BASE_STOP_WORDS: Set[str] = {
        'a', 'an', 'the', 'and', 'or', 'but', 'for', 'nor', 'on', 'at',
        'to', 'by', 'in', 'with', 'without', 'of', 'from', 'up', 'down',
        'off', 'over', 'under', 'above', 'below', 'between', 'among',
        'through', 'during', 'within', 'about', 'against',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
        'used', 'as', 'into', 'before', 'after',
        'yet', 'so', 'if', 'because', 'although', 'though', 'while', 'where',
        'when', 'that', 'which', 'who', 'whom', 'whose', 'what', 'this',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its',
        'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'not', 'only', 'own', 'same',
        'than', 'too', 'very', 'just', 'also', 'now', 'then', 'here',
        'there', 'out', 'again', 'further',
        'once', 'upon', 'across', 'along',
        'around', 'behind', 'beneath',
        'beside', 'beyond', 'despite',
        'except', 'inside', 'like', 'near',
        'onto', 'outside', 'past',
        'per', 'plus', 'since', 'throughout', 'toward',
        'until',
    }

    # ========================================================================
    # SAFE STRUCTURAL STOP WORDS
    # These are pure document-boilerplate words (page/section/figure refs)
    # that carry no food-safety content. Unlike the old DOMAIN_STOP_WORDS,
    # this list deliberately EXCLUDES any substantive domain term
    # (haccp, temperature, contamination, allergen, pathogen, etc.) because
    # those are exactly what users search for. BM25's IDF already down-weights
    # terms that appear in almost every document - we don't need to also
    # delete them and lose the ability to match on them at all.
    # ========================================================================
    STRUCTURAL_STOP_WORDS: Set[str] = {
        'page', 'pages', 'section', 'sections', 'chapter', 'chapters',
        'appendix', 'appendices', 'annex', 'annexes', 'clause', 'clauses',
        'figure', 'figures', 'chart', 'charts', 'diagram', 'diagrams',
        'illustration', 'illustrations', 'refer', 'referring', 'reference',
        'references', 'see', 'note', 'notes', 'following', 'herein',
        'hereinafter', 'thereof', 'thereby', 'wherein', 'whereas',
        'pursuant', 'hereby', 'aforementioned',
    }

    # Combined English stop words
    STOP_WORDS: Set[str] = BASE_STOP_WORDS | STRUCTURAL_STOP_WORDS

    # ========================================================================
    # ALWAYS-KEEP ALLOW-LIST
    # Short but critical domain acronyms/units that must survive the
    # minimum-length filter regardless of length.
    # ========================================================================
    ALWAYS_KEEP: Set[str] = {
        'ph', 'aw', 'cip', 'atp', 'rte', 'gmp', 'gap', 'ccp', 'sop', 'ssop',
        'haccp', 'iso', 'usda', 'fda', 'osha', 'epa', 'who', 'eu', 'uk',
        'us', 'cfu', 'ppm', 'ppb', 'uht', 'pcr', 'gmo', 'bpa', 'coli',
    }

    # Minimum token length for non-allow-listed tokens
    MIN_TOKEN_LENGTH = 2

    # ========================================================================
    # OPTIONAL ARABIC SUPPORT
    # ========================================================================
    ARABIC_STOP_WORDS: Set[str] = {
        'من', 'إلى', 'الى', 'على', 'عن', 'في', 'مع', 'هذا', 'هذه', 'ذلك',
        'تلك', 'التي', 'الذي', 'الذين', 'أن', 'ان', 'إن', 'كان', 'يكون',
        'لم', 'لن', 'لا', 'ما', 'بين', 'عند', 'حتى', 'أو', 'او', 'ثم',
        'قد', 'لقد', 'كل', 'بعض', 'غير', 'دون', 'بدون', 'كما', 'أيضا',
        'ايضا', 'هو', 'هي', 'هم', 'نحن', 'أنت', 'انت', 'أنتم', 'انتم',
        'الا', 'إلا', 'لكن', 'لكل', 'حول', 'خلال', 'بعد', 'قبل', 'فوق',
        'تحت', 'أمام', 'امام', 'خلف',
    }

    # Arabic diacritics (tashkeel) range to strip
    _ARABIC_DIACRITICS = re.compile(r'[\u064B-\u0652\u0670\u0640]')

    def __init__(
        self,
        top_k: Optional[int] = None,
        enable_arabic: bool = True,
        enable_light_stemming: bool = True,
        max_cache_size: int = 50_000,
        # Legacy parameters (kept for backward compatibility, now honored)
        domain_aware_tokenization: Optional[bool] = None,
        arabic_normalization: Optional[bool] = None,
    ) -> None:
        """
        Initialize the BM25 retriever.

        Args:
            top_k: Number of results to retrieve. Defaults to settings.
            enable_arabic: Enable Arabic tokenization/normalization/stopwords.
            enable_light_stemming: Enable light, conservative English suffix
                stripping (plurals / -ing / -ed) to improve recall.
            max_cache_size: Max number of entries kept in the tokenization
                cache before oldest entries are evicted (bounded memory).
            domain_aware_tokenization: LEGACY - accepted, no longer needed
                (domain-content stop words were removed; see module docstring).
            arabic_normalization: LEGACY - if explicitly passed, overrides
                enable_arabic for backward compatibility.
        """
        self.corpus: List[str] = []
        self.chunks: List[Chunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []
        self.top_k: int = int(top_k or settings.BM25_TOP_K)

        self.enable_arabic: bool = (
            arabic_normalization if arabic_normalization is not None else enable_arabic
        )
        self.enable_light_stemming: bool = enable_light_stemming

        # Bounded LRU-style cache + lock for thread-safety
        self._max_cache_size = max_cache_size
        self._tokenization_cache: "OrderedDict[str, List[str]]" = OrderedDict()
        self._chunk_id_map: Dict[str, int] = {}
        self._lock = threading.RLock()

        # For tracing
        self._last_results: Optional[List[RetrievalCandidate]] = None
        self._last_query_id: Optional[str] = None

        logger.log_event(
            event=LogEvent.SYSTEM_STARTUP,
            message="BM25 retriever initialized (REVISED)",
            details={
                "top_k": self.top_k,
                "base_stop_words": len(self.BASE_STOP_WORDS),
                "structural_stop_words": len(self.STRUCTURAL_STOP_WORDS),
                "total_english_stop_words": len(self.STOP_WORDS),
                "arabic_support": self.enable_arabic,
                "light_stemming": self.enable_light_stemming,
                "always_keep_terms": len(self.ALWAYS_KEEP),
                "multi_query_enabled": True,
            },
        )

    # ==========================================================================
    # NORMALIZATION HELPERS
    # ==========================================================================

    def _normalize_arabic(self, text: str) -> str:
        """
        Normalize Arabic text: strip diacritics/tatweel, unify letter forms.
        """
        text = self._ARABIC_DIACRITICS.sub('', text)
        text = re.sub(r'[إأآا]', 'ا', text)
        text = text.replace('ى', 'ي')
        text = text.replace('ة', 'ه')
        text = text.replace('ؤ', 'و')
        text = text.replace('ئ', 'ي')
        return text

    def _light_stem(self, token: str) -> str:
        """
        Conservative English suffix stripping to catch simple morphological
        variants (plurals, -ing, -ed) without being aggressive enough to
        cause false matches. Only applied to tokens made of pure letters.
        """
        if not token.isalpha() or len(token) <= 4:
            return token
        if token in self.ALWAYS_KEEP:
            return token
        for suffix, replacement in (
            ('ies', 'y'), ('ing', ''), ('ed', ''), ('es', ''), ('s', ''),
        ):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                return token[: -len(suffix)] + replacement
        return token

    # ==========================================================================
    # TOKENIZATION
    # ==========================================================================

    def _cache_get(self, text: str) -> Optional[List[str]]:
        with self._lock:
            if text in self._tokenization_cache:
                self._tokenization_cache.move_to_end(text)
                return self._tokenization_cache[text]
        return None

    def _cache_put(self, text: str, tokens: List[str]) -> None:
        with self._lock:
            self._tokenization_cache[text] = tokens
            self._tokenization_cache.move_to_end(text)
            while len(self._tokenization_cache) > self._max_cache_size:
                self._tokenization_cache.popitem(last=False)

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing/querying.

        Fixes vs previous version:
          - No domain-content stop words removed (see STRUCTURAL_STOP_WORDS).
          - Numeric tokens are kept (critical for temperatures/limits/durations).
          - Short but meaningful acronyms are preserved via ALWAYS_KEEP.
          - Optional Arabic normalization + stop words.
          - Optional light stemming for better recall on morphological variants.
        """
        cached = self._cache_get(text)
        if cached is not None:
            return cached

        if not text:
            return []

        text_lower = text.lower()

        if self.enable_arabic:
            text_lower = self._normalize_arabic(text_lower)
            raw_tokens = re.findall(r'[a-z0-9]+|[\u0600-\u06FF]+', text_lower)
        else:
            raw_tokens = re.findall(r'[a-z0-9]+', text_lower)

        filtered_tokens: List[str] = []
        for token in raw_tokens:
            is_arabic = bool(re.match(r'^[\u0600-\u06FF]+$', token))

            if token in self.ALWAYS_KEEP:
                filtered_tokens.append(token)
                continue

            if is_arabic:
                if token in self.ARABIC_STOP_WORDS:
                    continue
                if len(token) < 2:
                    continue
                filtered_tokens.append(token)
                continue

            # English / numeric path
            if token in self.STOP_WORDS:
                continue
            # Numeric tokens are ALWAYS kept (even single-digit) because they
            # carry critical values: temperatures, pH, concentrations, durations.
            if not token.isdigit() and len(token) < self.MIN_TOKEN_LENGTH:
                continue
            # NOTE: numeric tokens are intentionally KEPT - critical limits,
            # temperatures, and durations in food safety are numeric and
            # dropping them silently breaks retrieval for exactly the
            # queries that matter most (e.g. "165 degrees", "4 hours").

            if self.enable_light_stemming:
                token = self._light_stem(token)

            filtered_tokens.append(token)

        self._cache_put(text, filtered_tokens)
        return filtered_tokens

    # ==========================================================================
    # INDEX MANAGEMENT
    # ==========================================================================

    def index(self, chunks: List[Chunk]) -> None:
        """
        Build BM25 index from chunks.

        Raises:
            BM25Error: If indexing fails.
        """
        with measure_latency("bm25_indexing") as latency:
            logger.log_event(
                event=LogEvent.INDEX_START,
                message=f"Building BM25 index for {len(chunks)} chunks",
                details={
                    "chunk_count": len(chunks),
                    "stop_words_count": len(self.STOP_WORDS),
                },
            )

            try:
                with self._lock:
                    self.corpus = []
                    self.chunks = []
                    self.tokenized_corpus = []
                    self._chunk_id_map = {}
                    self._tokenization_cache.clear()

                for idx, chunk in enumerate(chunks):
                    text = chunk.content
                    if not text:
                        continue

                    self.corpus.append(text)
                    self.chunks.append(chunk)

                    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'chunk_id'):
                        self._chunk_id_map[chunk.metadata.chunk_id] = idx

                    tokens = self._tokenize(text)
                    self.tokenized_corpus.append(tokens)

                if self.tokenized_corpus:
                    with self._lock:
                        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=BM25_K1, b=BM25_B)

                latency.stop(
                    chunk_count=len(chunks),
                    corpus_size=len(self.corpus),
                )

                logger.log_event(
                    event=LogEvent.INDEX_COMPLETE,
                    message=f"BM25 index built with {len(self.corpus)} documents",
                    details={
                        "corpus_size": len(self.corpus),
                        "tokenized_corpus_size": len(self.tokenized_corpus),
                        "duration_ms": latency.duration_ms,
                        "avg_tokens_per_doc": (
                            sum(len(t) for t in self.tokenized_corpus) // len(self.tokenized_corpus)
                            if self.tokenized_corpus else 0
                        ),
                    },
                )

            except Exception as exc:
                raise BM25Error(
                    message=f"BM25 indexing failed: {str(exc)}",
                    corpus_size=len(self.corpus),
                    original_exception=exc,
                )

    def add(self, chunks: List[Chunk]) -> None:
        """
        Add new chunks to the existing BM25 index.

        Note: rank_bm25's BM25Okapi has no true incremental-update API, so
        this rebuilds the underlying index object after appending - it is
        "incremental" only from the caller's perspective (you don't have to
        resupply the whole corpus), not in terms of computational cost.
        """
        logger.log_event(
            event=LogEvent.INDEX_UPDATE,
            message=f"Adding {len(chunks)} chunks to BM25 index",
            details={"current_corpus_size": len(self.corpus)},
        )

        try:
            for chunk in chunks:
                text = chunk.content
                if not text:
                    continue

                idx = len(self.corpus)
                self.corpus.append(text)
                self.chunks.append(chunk)

                if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'chunk_id'):
                    self._chunk_id_map[chunk.metadata.chunk_id] = idx

                tokens = self._tokenize(text)
                self.tokenized_corpus.append(tokens)

            if self.tokenized_corpus:
                with self._lock:
                    self.bm25 = BM25Okapi(self.tokenized_corpus, k1=BM25_K1, b=BM25_B)

                logger.log_event(
                    event=LogEvent.INDEX_UPDATE,
                    message=f"BM25 index updated: {len(self.corpus)} total documents",
                    details={"corpus_size": len(self.corpus)},
                )

        except Exception as exc:
            logger.log_error(
                event=LogEvent.ERROR,
                message="Failed to add chunks to BM25 index",
                exception=exc,
            )

    # ==========================================================================
    # RETRIEVAL METHODS
    # ==========================================================================

    @staticmethod
    def _make_query_id(query_text: str) -> str:
        """Build a real, stable, short query id instead of a hardcoded constant."""
        digest = hashlib.sha1(query_text.encode("utf-8")).hexdigest()[:10]
        return f"bm25_q_{digest}"

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        query_id: Optional[str] = None,
    ) -> List[RetrievalCandidate]:
        """
        Perform BM25 retrieval for a query string.

        Args:
            query: The query string.
            top_k: Override for number of results.
            query_id: Optional caller-supplied id for tracing; if omitted,
                a stable id is derived from the query text so different
                queries are actually distinguishable in logs/traces.

        Returns:
            list[RetrievalCandidate]: Results sorted by score descending.

        Raises:
            BM25Error: If retrieval fails.
        """
        with measure_latency("bm25_retrieval") as latency:
            effective_top_k = int(top_k or self.top_k)
            query_text = query.strip() if query else ""
            effective_query_id = query_id or self._make_query_id(query_text)

            logger.log_retrieval(
                event=LogEvent.BM25_RETRIEVAL_START,
                query_id=effective_query_id,
                message=f"Starting BM25 retrieval: {query_text[:100]}...",
                details={
                    "query_text": query_text,
                    "top_k": effective_top_k,
                    "corpus_size": len(self.corpus),
                    "stop_words_count": len(self.STOP_WORDS),
                },
            )

            try:
                if not self.bm25 or len(self.corpus) == 0:
                    logger.log_event(
                        event=LogEvent.WARNING,
                        message="BM25 index is empty or uninitialized",
                    )
                    self._last_query_id = effective_query_id
                    self._last_results = []
                    return []

                tokenized_query = self._tokenize(query_text)

                if not tokenized_query:
                    logger.log_event(
                        event=LogEvent.WARNING,
                        message="Query tokenization resulted in empty tokens",
                        details={
                            "query": query_text[:100],
                            "query_length": len(query_text),
                        },
                    )
                    self._last_query_id = effective_query_id
                    self._last_results = []
                    return []

                scores = self.bm25.get_scores(tokenized_query)

                if scores is None or len(scores) == 0:
                    self._last_query_id = effective_query_id
                    self._last_results = []
                    return []

                if not isinstance(scores, np.ndarray):
                    scores = np.array(scores)

                max_score = float(np.max(scores)) if scores.size > 0 else 0.0

                if max_score < 1e-6:
                    logger.log_event(
                        event=LogEvent.WARNING,
                        message="BM25 scores are zero, no results found",
                        details={"query": query_text[:100]},
                    )
                    self._last_query_id = effective_query_id
                    self._last_results = []
                    return []

                scored_results: List[Tuple[Chunk, float]] = []
                for i, score in enumerate(scores):
                    if i < len(self.chunks):
                        self.chunks[i].bm25_score = float(score)
                        scored_results.append((self.chunks[i], float(score)))

                scored_results.sort(key=lambda x: x[1], reverse=True)
                top_results = scored_results[:effective_top_k]

                candidates: List[RetrievalCandidate] = []
                for chunk, score in top_results:
                    candidates.append(RetrievalCandidate(chunk=chunk, bm25_score=score))

                for rank, candidate in enumerate(candidates, start=1):
                    candidate.rank = rank

                self._last_query_id = effective_query_id
                self._last_results = candidates

                latency.stop(
                    query_id=effective_query_id,
                    results_count=len(candidates),
                    top_k=effective_top_k,
                    max_score=float(max_score),
                )

                logger.log_retrieval(
                    event=LogEvent.BM25_RETRIEVAL_COMPLETE,
                    query_id=effective_query_id,
                    message=f"BM25 retrieval complete: {len(candidates)} candidates",
                    details={
                        "results_count": len(candidates),
                        "top_k": effective_top_k,
                        "duration_ms": latency.duration_ms,
                        "max_score": float(max_score),
                        "top_results": [
                            {
                                "chunk_id": c.chunk.metadata.chunk_id if c.chunk.metadata else None,
                                "page": c.chunk.metadata.page if c.chunk.metadata else None,
                                "score": round(c.bm25_score or 0.0, 4),
                            }
                            for c in candidates[:10]
                        ],
                    },
                )

                return candidates

            except Exception as exc:
                self._last_query_id = effective_query_id
                self._last_results = []
                raise BM25Error(
                    message=f"BM25 retrieval failed: {str(exc)}",
                    corpus_size=len(self.corpus),
                    original_exception=exc,
                )

    # ==========================================================================
    # MULTI-QUERY RETRIEVAL - RESTORED
    # ==========================================================================

    def retrieve_multi_query(
        self,
        query: Query,
        query_variants: List[Tuple[str, float, int, int]],
    ) -> List[Tuple[str, float, List[RetrievalCandidate]]]:
        """
        Perform BM25 retrieval for multiple query variants and return a
        weighted merge across all of them (not just the first variant).

        Rationale: query expansion (synonyms, rephrasings) exists to
        improve recall. Silently discarding all variants but the first
        throws away that recall benefit for no measured gain in precision.
        If the expansion generates true noise you don't want, that should
        be fixed by improving the expansion step or tuning per-variant
        weights - not by ignoring the mechanism wholesale.

        Args:
            query: Original query (for logging).
            query_variants: List of (variant_text, weight, dense_top_k, bm25_top_k).

        Returns:
            List of (variant_text, weight, results) tuples, one entry per
            variant PLUS the results reflect a weighted merge available via
            `get_merged_multi_query_results`.
        """
        results: List[Tuple[str, float, List[RetrievalCandidate]]] = []

        if not query_variants:
            return results

        logger.log_event(
            event=LogEvent.BM25_RETRIEVAL_START,
            message="Multi-query BM25 retrieval across all variants",
            details={
                "query_id": query.query_id,
                "variant_count": len(query_variants),
            },
        )

        merged_scores: Dict[str, float] = {}
        merged_chunks: Dict[str, Chunk] = {}

        for variant_text, weight, _dense_top_k, bm25_top_k in query_variants:
            try:
                variant_results = self.retrieve(
                    query=variant_text,
                    top_k=int(bm25_top_k),
                    query_id=f"{query.query_id}:{self._make_query_id(variant_text)}",
                )

                weighted_results = []
                for candidate in variant_results:
                    weighted_candidate = RetrievalCandidate(
                        chunk=candidate.chunk,
                        bm25_score=(candidate.bm25_score or 0.0) * weight,
                    )
                    weighted_results.append(weighted_candidate)

                    cid = (
                        candidate.chunk.metadata.chunk_id
                        if candidate.chunk.metadata else str(id(candidate.chunk))
                    )
                    merged_scores[cid] = merged_scores.get(cid, 0.0) + weighted_candidate.bm25_score
                    merged_chunks[cid] = candidate.chunk

                results.append((variant_text, weight, weighted_results))

            except BM25Error as exc:
                logger.log_error(
                    event=LogEvent.ERROR,
                    message=f"BM25 retrieval failed for variant: {variant_text[:50]}...",
                    exception=exc,
                    details={"query_id": query.query_id},
                )
                results.append((variant_text, weight, []))

        # Store merged, deduplicated ranking for convenience
        merged_sorted = sorted(merged_scores.items(), key=lambda kv: kv[1], reverse=True)
        merged_candidates = [
            RetrievalCandidate(chunk=merged_chunks[cid], bm25_score=score)
            for cid, score in merged_sorted
        ]
        for rank, candidate in enumerate(merged_candidates, start=1):
            candidate.rank = rank

        self._last_results = merged_candidates
        self._last_query_id = query.query_id
        self._last_merged_multi_query_results = merged_candidates

        logger.log_event(
            event=LogEvent.BM25_RETRIEVAL_COMPLETE,
            message="Multi-query BM25 retrieval complete (weighted merge)",
            details={
                "query_id": query.query_id,
                "variant_count": len(query_variants),
                "merged_unique_chunks": len(merged_candidates),
            },
        )

        return results

    def get_merged_multi_query_results(self) -> Optional[List[RetrievalCandidate]]:
        """Get the deduplicated, weight-merged ranking from the last multi-query call."""
        return getattr(self, "_last_merged_multi_query_results", None)

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
    ) -> List[List[RetrievalCandidate]]:
        """
        Perform BM25 retrieval for multiple independent queries.

        Args:
            queries: List of query strings.
            top_k: Number of results per query.

        Returns:
            List of result lists per query.
        """
        results: List[List[RetrievalCandidate]] = []

        for query in queries:
            try:
                result = self.retrieve(query, top_k=top_k)
                results.append(result)
            except BM25Error as exc:
                logger.log_error(
                    event=LogEvent.ERROR,
                    message=f"BM25 retrieval failed for query: {query[:50]}...",
                    exception=exc,
                )
                results.append([])

        return results

    # ==========================================================================
    # STATISTICS AND UTILITY METHODS
    # ==========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get BM25 index statistics."""
        avg_doc_length = 0
        if self.tokenized_corpus:
            total_tokens = sum(len(t) for t in self.tokenized_corpus)
            avg_doc_length = total_tokens // len(self.tokenized_corpus) if self.tokenized_corpus else 0

        return {
            "corpus_size": len(self.corpus),
            "indexed": self.bm25 is not None,
            "avg_doc_length": avg_doc_length,
            "k1": BM25_K1,
            "b": BM25_B,
            "tokenization_cache_size": len(self._tokenization_cache),
            "tokenization_cache_max_size": self._max_cache_size,
            "arabic_support": self.enable_arabic,
            "light_stemming": self.enable_light_stemming,
            "multi_query_enabled": True,
            "base_stop_words": len(self.BASE_STOP_WORDS),
            "structural_stop_words": len(self.STRUCTURAL_STOP_WORDS),
            "total_english_stop_words": len(self.STOP_WORDS),
            "always_keep_terms": len(self.ALWAYS_KEEP),
        }

    def get_last_results(self) -> Optional[List[RetrievalCandidate]]:
        """Get the last retrieved results for tracing."""
        return self._last_results

    def get_last_query_id(self) -> Optional[str]:
        """Get the last query ID for tracing."""
        return self._last_query_id

    def clear_cache(self) -> None:
        """Clear the tokenization cache."""
        with self._lock:
            self._tokenization_cache.clear()
        logger.log_event(
            event=LogEvent.CACHE_CLEAR,
            message="BM25 retriever tokenization cache cleared",
        )

    def get_config(self) -> Dict[str, Any]:
        """Get BM25 configuration."""
        return {
            "top_k": self.top_k,
            "arabic_support": self.enable_arabic,
            "light_stemming": self.enable_light_stemming,
            "multi_query_enabled": True,
            "base_stop_words": len(self.BASE_STOP_WORDS),
            "structural_stop_words": len(self.STRUCTURAL_STOP_WORDS),
            "total_english_stop_words": len(self.STOP_WORDS),
            "always_keep_terms": len(self.ALWAYS_KEEP),
            "max_cache_size": self._max_cache_size,
        }