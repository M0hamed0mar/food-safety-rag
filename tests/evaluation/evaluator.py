"""
Benchmark evaluator: runs questions through the pipeline and computes metrics.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from app.pipeline import get_pipeline
from tests.evaluation.benchmark_data import BenchmarkQuestion
from tests.evaluation.cache import EvalCache
from tests.evaluation.metrics import (
    answer_relevance,
    exact_match,
    hit_rate_at_k,
    mrr,
    precision_at_k,
    recall_at_k,
    substring_match,
    token_f1,
)


@dataclass
class EvaluationResult:
    """Result of evaluating a single benchmark question."""

    question_id: str
    question: str
    category: str
    difficulty: str

    predicted_answer: str = ""
    expected_answer: str = ""

    citations: list[dict[str, Any]] = field(default_factory=list)
    retrieved_pages: list[int] = field(default_factory=list)

    # Generation metrics
    exact_match: float = 0.0
    substring_match: float = 0.0
    token_f1: float = 0.0
    answer_relevance: float = 0.0

    # Retrieval metrics
    mrr: float = 0.0
    hit_rate_at_5: float = 0.0
    hit_rate_at_10: float = 0.0
    precision_at_5: float = 0.0
    recall_at_10: float = 0.0

    # Timing
    retrieval_duration_ms: float = 0.0
    total_duration_ms: float = 0.0

    error: str | None = None
    from_cache: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkEvaluator:
    """
    Runs the RAG pipeline on benchmark questions and computes metrics.

    Features:
        - Persistent cache (avoids rerunning same query)
        - Rich per-question result with all metrics
        - Summary aggregation
    """

    def __init__(
        self,
        pipeline: Any | None = None,
        cache: EvalCache | None = None,
        use_cache: bool = True,
    ) -> None:
        self.pipeline = pipeline  # lazy load
        self.cache = cache or EvalCache(enabled=use_cache)
        self.use_cache = use_cache

    # --------------------------------------------------------
    # Pipeline access
    # --------------------------------------------------------

    def _get_pipeline(self) -> Any:
        if self.pipeline is None:
            self.pipeline = get_pipeline()
        return self.pipeline

    # --------------------------------------------------------
    # Single question
    # --------------------------------------------------------

    def evaluate_question(self, q: BenchmarkQuestion) -> EvaluationResult:
        """Evaluate a single benchmark question."""
        result = EvaluationResult(
            question_id=q.id,
            question=q.question,
            category=q.category,
            difficulty=q.difficulty,
            expected_answer=q.expected_answer,
        )

        # --------------------------------------------------------
        # Check cache
        # --------------------------------------------------------
        if self.use_cache:
            cached = self.cache.get(q.question)
            if cached:
                result.predicted_answer = cached.get("answer", "")
                result.citations = cached.get("citations", [])
                result.retrieved_pages = cached.get("retrieved_pages", [])
                result.retrieval_duration_ms = cached.get("retrieval_duration_ms", 0.0)
                result.total_duration_ms = cached.get("total_duration_ms", 0.0)
                result.from_cache = True

        # --------------------------------------------------------
        # Run pipeline if not cached
        # --------------------------------------------------------
        if not result.from_cache:
            try:
                pipeline = self._get_pipeline()

                t0 = time.perf_counter()
                answer = pipeline.ask(q.question)
                total_ms = (time.perf_counter() - t0) * 1000

                result.predicted_answer = answer.text or ""
                result.citations = [
                    c.model_dump() if hasattr(c, "model_dump") else c
                    for c in (answer.citations or [])
                ]
                result.retrieved_pages = sorted(set(
                    c.get("page") for c in result.citations
                    if isinstance(c, dict) and c.get("page")
                ))
                result.total_duration_ms = total_ms
                result.retrieval_duration_ms = getattr(answer.metadata, "retrieval_duration_ms", 0.0) or 0.0

                # Save to cache
                self.cache.set(
                    q.question,
                    {
                        "answer": result.predicted_answer,
                        "citations": result.citations,
                        "retrieved_pages": result.retrieved_pages,
                        "retrieval_duration_ms": result.retrieval_duration_ms,
                        "total_duration_ms": result.total_duration_ms,
                    },
                )
                self.cache.flush()

            except Exception as exc:
                result.error = f"{type(exc).__name__}: {exc}"
                return result

        # --------------------------------------------------------
        # Compute metrics
        # --------------------------------------------------------
        result.exact_match = exact_match(result.predicted_answer, q.expected_answer)
        result.substring_match = substring_match(result.predicted_answer, q.expected_answer)
        result.token_f1 = token_f1(result.predicted_answer, q.expected_answer)
        result.answer_relevance = answer_relevance(result.predicted_answer, q.expected_keywords)

        if q.expected_pages:
            result.mrr = mrr(result.retrieved_pages, q.expected_pages)
            result.hit_rate_at_5 = hit_rate_at_k(result.retrieved_pages, q.expected_pages, k=5)
            result.hit_rate_at_10 = hit_rate_at_k(result.retrieved_pages, q.expected_pages, k=10)
            result.precision_at_5 = precision_at_k(result.retrieved_pages, q.expected_pages, k=5)
            result.recall_at_10 = recall_at_k(result.retrieved_pages, q.expected_pages, k=10)

        return result

    # --------------------------------------------------------
    # Batch
    # --------------------------------------------------------

    def evaluate_batch(self, questions: list[BenchmarkQuestion]) -> list[EvaluationResult]:
        """Evaluate a list of questions."""
        results = []
        for i, q in enumerate(questions, start=1):
            print(f"[{i}/{len(questions)}] {q.id}: {q.question[:70]}...")
            r = self.evaluate_question(q)
            if r.error:
                print(f"     ERROR: {r.error}")
            else:
                print(
                    f"     F1={r.token_f1:.3f}  "
                    f"Relevance={r.answer_relevance:.3f}  "
                    f"MRR={r.mrr:.3f}  "
                    f"{'[cached]' if r.from_cache else ''}"
                )
            results.append(r)
        return results

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    @staticmethod
    def summarize(results: list[EvaluationResult]) -> dict[str, Any]:
        """Aggregate metrics across all results."""
        if not results:
            return {
                "total": 0,
                "errors": 0,
                "avg_exact_match": 0.0,
                "avg_substring_match": 0.0,
                "avg_token_f1": 0.0,
                "avg_answer_relevance": 0.0,
                "avg_mrr": 0.0,
                "avg_hit_rate_at_5": 0.0,
                "avg_hit_rate_at_10": 0.0,
                "avg_precision_at_5": 0.0,
                "avg_recall_at_10": 0.0,
                "total_duration_s": 0.0,
            }

        def _avg(attr: str) -> float:
            vals = [getattr(r, attr) for r in results if r.error is None]
            return sum(vals) / len(vals) if vals else 0.0

        errors = sum(1 for r in results if r.error)
        total_ms = sum(r.total_duration_ms for r in results)

        return {
            "total": len(results),
            "errors": errors,
            "avg_exact_match": round(_avg("exact_match"), 4),
            "avg_substring_match": round(_avg("substring_match"), 4),
            "avg_token_f1": round(_avg("token_f1"), 4),
            "avg_answer_relevance": round(_avg("answer_relevance"), 4),
            "avg_mrr": round(_avg("mrr"), 4),
            "avg_hit_rate_at_5": round(_avg("hit_rate_at_5"), 4),
            "avg_hit_rate_at_10": round(_avg("hit_rate_at_10"), 4),
            "avg_precision_at_5": round(_avg("precision_at_5"), 4),
            "avg_recall_at_10": round(_avg("recall_at_10"), 4),
            "total_duration_s": round(total_ms / 1000, 2),
        }


__all__ = ["BenchmarkEvaluator", "EvaluationResult"]
