"""
RAG Quality Evaluation Suite.

Tests:
    1. test_01_factual_accuracy:    Direct Q&A accuracy
    2. test_02_retrieval_quality:   Retrieval metrics (Hit Rate, MRR)
    3. test_03_citation_accuracy:   Citation page/document accuracy
    4. test_04_edge_cases:          Multi-hop, negative, ambiguous
    5. test_05_arabic_support:      Arabic language support

Reports:
    All results saved to data/reports/
        - JSON: full data (machine-readable)
        - Markdown: human-readable (README-ready)
"""

from tests.evaluation.benchmark_data import (
    BenchmarkQuestion,
    BENCHMARK_QUESTIONS,
    get_questions_by_category,
)
from tests.evaluation.metrics import (
    exact_match,
    token_f1,
    answer_relevance,
    mrr,
    hit_rate_at_k,
    precision_at_k,
)
from tests.evaluation.evaluator import BenchmarkEvaluator, EvaluationResult
from tests.evaluation.cache import EvalCache
from tests.evaluation.report_generator import (
    ConsoleReporter,
    ReportGenerator,
    REPORTS_DIR,
)

__all__ = [
    # Data
    "BenchmarkQuestion",
    "BENCHMARK_QUESTIONS",
    "get_questions_by_category",
    # Metrics
    "exact_match",
    "token_f1",
    "answer_relevance",
    "mrr",
    "hit_rate_at_k",
    "precision_at_k",
    # Runner
    "BenchmarkEvaluator",
    "EvaluationResult",
    "EvalCache",
    # Reporting
    "ConsoleReporter",
    "ReportGenerator",
    "REPORTS_DIR",
]
