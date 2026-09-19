"""
Test Suite 1: Factual Accuracy.

Evaluates the RAG system on factual questions.

Console output: concise (one line per question + summary).
Full report: data/reports/test_01_factual_accuracy_*.{json,md}

Usage:
    python -m tests.test_01_factual_accuracy
"""

from __future__ import annotations

import sys

from tests.evaluation.benchmark_data import get_questions_by_category
from tests.evaluation.evaluator import BenchmarkEvaluator
from tests.evaluation.report_generator import ConsoleReporter, ReportGenerator


SUITE_NAME = "test_01_factual_accuracy"


def main() -> int:
    ConsoleReporter.header("Test Suite 1: FACTUAL ACCURACY")

    questions = get_questions_by_category("factual")
    print(f"Running {len(questions)} factual questions...\n")

    evaluator = BenchmarkEvaluator(use_cache=True)

    # ---- Run with progress ----
    results = []
    for i, q in enumerate(questions, start=1):
        ConsoleReporter.progress(i, len(questions), q.id, q.question)
        r = evaluator.evaluate_question(q)
        ConsoleReporter.question_result(r)
        results.append(r)

    # ---- Summary ----
    summary = evaluator.summarize(results)
    ConsoleReporter.summary(summary)

    # ---- Save reports ----
    gen = ReportGenerator()
    json_path, md_path = gen.save(results, suite_name=SUITE_NAME, summary=summary)

    print()
    print(f"📄 JSON report:     {json_path}")
    print(f"📄 Markdown report: {md_path}")
    print(f"📄 Latest JSON:     {json_path.parent / f'latest_{SUITE_NAME}.json'}")
    print()

    # ---- Pass/Fail ----
    pass_f1 = summary["avg_token_f1"] >= 0.50
    pass_rel = summary["avg_answer_relevance"] >= 0.60
    pass_err = summary["errors"] == 0

    if pass_f1 and pass_rel and pass_err:
        print("✅ PASSED")
        return 0

    print("❌ FAILED")
    if not pass_err:
        print(f"   - {summary['errors']} errors")
    if not pass_f1:
        print(f"   - F1 {summary['avg_token_f1']:.3f} < 0.50 target")
    if not pass_rel:
        print(f"   - Relevance {summary['avg_answer_relevance']:.3f} < 0.60 target")
    return 1


if __name__ == "__main__":
    sys.exit(main())
