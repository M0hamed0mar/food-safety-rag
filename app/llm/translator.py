"""
Query translator: generate optimized queries for Dense + BM25 retrieval.

Preserves the original dual-query API but delegates the LLM call to
`app.core.llm_client.LLMClient` (Groq).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from app.config import settings
from app.config.constants import LogEvent
from app.config.prompts import get_prompt, PromptTemplate
from app.core.llm_client import LLMClient, llm_client
from app.monitoring import get_logger


logger = get_logger("app.llm.translator")


class QueryRewriterResult:
    """Result of dual query generation."""

    def __init__(
        self,
        dense_query: str,
        bm25_query: str,
        original_query: str = "",
        was_translated: bool = False,
        is_arabic: bool = False,
    ) -> None:
        self.dense_query = (dense_query or "").strip()
        self.bm25_query = (bm25_query or "").strip()
        self.original_query = original_query
        self.was_translated = was_translated
        self.is_arabic = is_arabic

    def get_dense_query(self) -> str:
        return self.dense_query

    def get_bm25_query(self) -> str:
        return self.bm25_query

    def get_all_queries(self) -> List[str]:
        out: List[str] = []
        if self.dense_query:
            out.append(self.dense_query)
        if self.bm25_query and self.bm25_query != self.dense_query:
            out.append(self.bm25_query)
        return out

    def get_weighted_queries(self) -> List[tuple]:
        out: List[tuple] = []
        if self.dense_query:
            out.append((self.dense_query, 1.0))
        if self.bm25_query and self.bm25_query != self.dense_query:
            out.append((self.bm25_query, 0.7))
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dense_query": self.dense_query,
            "bm25_query": self.bm25_query,
            "original_query": self.original_query,
            "was_translated": self.was_translated,
            "is_arabic": self.is_arabic,
        }

    def is_empty(self) -> bool:
        return not self.dense_query and not self.bm25_query


class QueryTranslator:
    """Dual-query generation using Groq (via LLMClient)."""

    _JSON_PATTERN = re.compile(
        r"```(?:json)?\s*([\s\S]*?)\s*```|"
        r'(\{[^{}]*"(?:dense_query|bm25_query)"\s*:\s*"[^"]*"\s*,\s*"(?:dense_query|bm25_query)"\s*:\s*"[^"]*"\s*\})',
        re.IGNORECASE,
    )

    def __init__(self, client: Optional[LLMClient] = None) -> None:
        self.client: LLMClient = client or llm_client
        self.enabled: bool = bool(getattr(settings, "TRANSLATION_ENABLED", False))
        self._cache: Dict[str, QueryRewriterResult] = {}
        self._stats: Dict[str, int] = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "translation_success": 0,
            "translation_fallback": 0,
        }
        try:
            self._prompt: PromptTemplate = get_prompt("query_rewriter")
        except Exception:
            self._prompt = None

        logger.log_event(
            event=LogEvent.SYSTEM_STARTUP,
            message="Query translator initialized",
            details={
                "enabled": self.enabled,
                "model": self.client._model,
                "has_prompt": self._prompt is not None,
            },
        )

    def _is_arabic(self, text: str) -> bool:
        if not text:
            return False
        arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
        return len(text) > 0 and (arabic_chars / len(text)) >= 0.3

    def _is_english(self, text: str) -> bool:
        if not text:
            return True
        return not self._is_arabic(text)

    def _extract_json(self, response: str) -> Optional[Dict[str, Any]]:
        if not response:
            return None

        match = self._JSON_PATTERN.search(response)
        if match:
            json_str = match.group(1) or match.group(2)
            if json_str:
                try:
                    return json.loads(json_str.strip())
                except json.JSONDecodeError:
                    pass

        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        cleaned = response.strip()
        cleaned = re.sub(r",\s*}", "}", cleaned)
        cleaned = re.sub(r",\s*\]", "]", cleaned)
        cleaned = re.sub(r"'", '"', cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        dense_match = re.search(r'"dense_query"\s*:\s*"([^"]*)"', response)
        bm25_match = re.search(r'"bm25_query"\s*:\s*"([^"]*)"', response)
        if dense_match or bm25_match:
            return {
                "dense_query": dense_match.group(1) if dense_match else "",
                "bm25_query": bm25_match.group(1) if bm25_match else "",
            }

        return None

    def _parse_response(self, response: str) -> Dict[str, str]:
        if not response:
            return {"dense_query": "", "bm25_query": ""}

        data = self._extract_json(response)
        if data:
            dense_query = str(data.get("dense_query", "")).strip()
            bm25_query = str(data.get("bm25_query", "")).strip()
            if dense_query and not bm25_query:
                bm25_query = dense_query
            return {"dense_query": dense_query, "bm25_query": bm25_query}

        cleaned = response.strip()
        cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```", "", cleaned)
        if len(cleaned) > 500:
            cleaned = cleaned[:500]
        return {"dense_query": cleaned, "bm25_query": cleaned}

    def translate_and_expand(self, query: str) -> QueryRewriterResult:
        self._stats["total_requests"] += 1

        if not query or not query.strip():
            return QueryRewriterResult(
                dense_query="", bm25_query="",
                original_query=query, was_translated=False, is_arabic=False,
            )

        query = " ".join(query.split())
        is_arabic = self._is_arabic(query)

        if not self.enabled:
            logger.log_event(
                event=LogEvent.SYSTEM_STARTUP,
                message="Translation disabled, returning original query",
                details={"query": query[:50], "is_arabic": is_arabic},
            )
            return QueryRewriterResult(
                dense_query=query, bm25_query=query,
                original_query=query, was_translated=False, is_arabic=is_arabic,
            )

        cache_key = query.lower().strip()
        if cache_key in self._cache:
            self._stats["cache_hits"] += 1
            cached = self._cache[cache_key]
            cached.is_arabic = is_arabic
            return cached

        self._stats["cache_misses"] += 1

        try:
            if self._prompt is None:
                raise RuntimeError("Prompt 'query_rewriter' not available")

            prompt_text = self._prompt.template.format(query=query)

            response = self.client.generate_text_sync(
                prompt=prompt_text,
                instructions=(
                    "You are a bilingual search-query optimizer. Generate separate "
                    "optimized queries for Dense (semantic) and BM25 (lexical) retrieval. "
                    "Return ONLY valid JSON matching the requested schema."
                ),
                max_tokens=256,
                temperature=0.0,
            )

            data = self._parse_response(response)
            dense_query = data.get("dense_query", "") or query
            bm25_query = data.get("bm25_query", "") or dense_query

            if len(dense_query) < 2:
                dense_query = query
            if len(bm25_query) < 2:
                bm25_query = dense_query

            result = QueryRewriterResult(
                dense_query=dense_query,
                bm25_query=bm25_query,
                original_query=query,
                was_translated=True,
                is_arabic=is_arabic,
            )
            self._cache[cache_key] = result
            self._stats["translation_success"] += 1
            return result

        except Exception as exc:
            logger.log_error(
                event=LogEvent.ERROR,
                message=f"Dual query generation failed: {exc}",
                exception=exc,
                details={"query": query[:100]},
            )
            self._stats["translation_fallback"] += 1
            return QueryRewriterResult(
                dense_query=query, bm25_query=query,
                original_query=query, was_translated=False, is_arabic=is_arabic,
            )

    def translate_batch(self, queries: List[str]) -> List[QueryRewriterResult]:
        return [self.translate_and_expand(q) for q in queries]

    def clear_cache(self) -> None:
        n = len(self._cache)
        self._cache.clear()
        logger.log_event(LogEvent.CACHE_CLEAR, f"Translation cache cleared ({n} entries)")

    def get_cache_stats(self) -> Dict[str, int]:
        return {
            "cache_size": len(self._cache),
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["cache_misses"],
            "total_requests": self._stats["total_requests"],
            "translation_success": self._stats["translation_success"],
            "translation_fallback": self._stats["translation_fallback"],
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "cache": self.get_cache_stats(),
            "model": self.client._model,
        }

    def get_config(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model": self.client._model,
            "max_tokens": 256,
            "temperature": 0.0,
        }


_translator: Optional[QueryTranslator] = None


def get_translator() -> QueryTranslator:
    global _translator
    if _translator is None:
        _translator = QueryTranslator()
        logger.log_event(
            event=LogEvent.SYSTEM_STARTUP,
            message="Created shared translator singleton",
            details={"instance_id": id(_translator)},
        )
    return _translator


def reset_translator() -> None:
    global _translator
    _translator = None
    logger.log_event(LogEvent.SYSTEM_STARTUP, "Translator singleton reset")


__all__ = [
    "QueryTranslator",
    "QueryRewriterResult",
    "get_translator",
    "reset_translator",
]
