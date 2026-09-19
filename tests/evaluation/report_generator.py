"""
Report generator for RAG benchmark results.

Produces:
    - JSON report: full details, machine-readable
    - Markdown report: human-readable, README-ready
    - Console output: concise progress + summary

Usage:
    from tests.evaluation.report_generator import ReportGenerator
    gen = ReportGenerator()
    gen.save(results, suite_name="test_01_factual_accuracy")
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import settings
from tests.evaluation.evaluator import EvaluationResult


REPORTS_DIR = settings.DATA_DIR / "reports"


# ============================================================
# Console formatting
# ============================================================

class ConsoleReporter:
    """Minimal, colorized console output."""

    # ANSI colors (work on Windows 10+ / modern terminals)
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @classmethod
    def header(cls, title: str, width: int = 72) -> None:
        print()
        print(cls.CYAN + "=" * width + cls.RESET)
        print(cls.BOLD + cls.CYAN + f"  {title}" + cls.RESET)
        print(cls.CYAN + "=" * width + cls.RESET)
        print()

    @classmethod
    def progress(cls, current: int, total: int, qid: str, question: str) -> None:
        """Print single-line progress."""
        prefix = f"[{current:>2}/{total}]"
        q = question[:55] + "..." if len(question) > 55 else question
        print(f"{cls.GRAY}{prefix}{cls.RESET} {cls.BOLD}{qid}{cls.RESET}  {q}")

    @classmethod
    def question_result(cls, r: EvaluationResult) -> None:
        """Print compact one-line result for a question."""
        if r.error:
            print(f"      {cls.RED}[X] ERROR: {r.error[:60]}{cls.RESET}")
            return

        # Color F1 by quality
        f1 = r.token_f1
        if f1 >= 0.7:
            f1_color = cls.GREEN
        elif f1 >= 0.4:
            f1_color = cls.YELLOW
        else:
            f1_color = cls.RED

        cache_flag = f" {cls.GRAY}[cached]{cls.RESET}" if r.from_cache else ""

        print(
            f"      F1={f1_color}{f1:.3f}{cls.RESET}  "
            f"Rel={r.answer_relevance:.3f}  "
            f"MRR={r.mrr:.3f}  "
            f"{cls.GRAY}{r.total_duration_ms/1000:.1f}s{cls.RESET}"
            f"{cache_flag}"
        )

    @classmethod
    def summary(cls, summary: dict[str, Any]) -> None:
        """Print final summary block."""
        print()
        print(cls.CYAN + "=" * 72 + cls.RESET)
        print(cls.BOLD + cls.CYAN + "  SUMMARY" + cls.RESET)
        print(cls.CYAN + "=" * 72 + cls.RESET)

        def _fmt(key: str, label: str, target: float | None = None) -> None:
            v = summary.get(key, 0.0)
            if isinstance(v, float):
                color = ""
                if target is not None:
                    if v >= target:
                        color = cls.GREEN
                    elif v >= target * 0.75:
                        color = cls.YELLOW
                    else:
                        color = cls.RED
                target_str = f"  {cls.GRAY}(target >= {target}){cls.RESET}" if target else ""
                print(f"  {label:<28} {color}{v:.4f}{cls.RESET}{target_str}")
            else:
                print(f"  {label:<28} {v}")

        _fmt("total", "Total questions")
        _fmt("errors", "Errors")
        _fmt("avg_exact_match", "Avg Exact Match")
        _fmt("avg_substring_match", "Avg Substring Match")
        _fmt("avg_token_f1", "Avg Token F1", target=0.50)
        _fmt("avg_answer_relevance", "Avg Answer Relevance", target=0.60)
        _fmt("avg_mrr", "Avg MRR", target=0.50)
        _fmt("avg_hit_rate_at_5", "Avg Hit Rate @ 5", target=0.70)
        _fmt("avg_precision_at_5", "Avg Precision @ 5", target=0.40)
        _fmt("avg_recall_at_10", "Avg Recall @ 10", target=0.70)
        _fmt("total_duration_s", "Total duration (s)")

        print(cls.CYAN + "=" * 72 + cls.RESET)


# ============================================================
# Report Generator
# ============================================================

class ReportGenerator:
    """
    Generates JSON + Markdown reports for a benchmark run.

    Files:
        data/reports/{suite}_{timestamp}.json
        data/reports/{suite}_{timestamp}.md
        data/reports/latest_{suite}.json  (symlink-like: latest)
    """

    def __init__(self, reports_dir: Path | None = None) -> None:
        self.reports_dir: Path = reports_dir or REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        results: list[EvaluationResult],
        suite_name: str,
        summary: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        """
        Save results as JSON + Markdown.

        Returns:
            (json_path, md_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if summary is None:
            from tests.evaluation.evaluator import BenchmarkEvaluator
            summary = BenchmarkEvaluator.summarize(results)

        payload = {
            "suite": suite_name,
            "timestamp": timestamp,
            "generated_at": datetime.now().isoformat(),
            "summary": summary,
            "metadata": metadata or {},
            "results": [r.to_dict() for r in results],
        }

        # ---- JSON ----
        json_path = self.reports_dir / f"{suite_name}_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # ---- Latest JSON ----
        latest_json = self.reports_dir / f"latest_{suite_name}.json"
        with open(latest_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # ---- Markdown ----
        md_path = self.reports_dir / f"{suite_name}_{timestamp}.md"
        md_content = self._build_markdown(payload)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        # ---- Latest Markdown ----
        latest_md = self.reports_dir / f"latest_{suite_name}.md"
        with open(latest_md, "w", encoding="utf-8") as f:
            f.write(md_content)

        return json_path, md_path

    # --------------------------------------------------------
    # Markdown builder
    # --------------------------------------------------------

    @staticmethod
    def _build_markdown(payload: dict[str, Any]) -> str:
        """Build a human-readable Markdown report."""
        lines: list[str] = []

        suite = payload["suite"]
        ts = payload["timestamp"]
        summary = payload["summary"]
        results = payload["results"]

        # ---- Title ----
        lines.append(f"# Benchmark Report: `{suite}`")
        lines.append("")
        lines.append(f"**Generated**: {ts}")
        lines.append("")

        # ---- Summary table ----
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value | Target | Status |")
        lines.append("|--------|-------|--------|--------|")

        def row(label: str, key: str, target: float | None = None) -> None:
            v = summary.get(key, 0.0)
            if isinstance(v, float):
                val_str = f"{v:.4f}"
            else:
                val_str = str(v)

            if target is not None and isinstance(v, float):
                status = "OK" if v >= target else ("WARN" if v >= target * 0.75 else "FAIL")
                tgt_str = f"{target}"
            else:
                status = "-"
                tgt_str = "-"

            lines.append(f"| {label} | {val_str} | {tgt_str} | {status} |")

        row("Total questions", "total")
        row("Errors", "errors")
        row("Avg Exact Match", "avg_exact_match")
        row("Avg Substring Match", "avg_substring_match")
        row("Avg Token F1", "avg_token_f1", 0.50)
        row("Avg Answer Relevance", "avg_answer_relevance", 0.60)
        row("Avg MRR", "avg_mrr", 0.50)
        row("Avg Hit Rate @ 5", "avg_hit_rate_at_5", 0.70)
        row("Avg Precision @ 5", "avg_precision_at_5", 0.40)
        row("Avg Recall @ 10", "avg_recall_at_10", 0.70)
        row("Total duration (s)", "total_duration_s")
        lines.append("")

        # ---- Per-question table ----
        lines.append("## Per-Question Results")
        lines.append("")
        lines.append("| ID | Category | Difficulty | F1 | Rel | MRR | Duration | Status |")
        lines.append("|----|----------|-----------|-----|-----|-----|----------|--------|")

        for r in results:
            if r.get("error"):
                status = "[X] ERROR"
            elif r.get("token_f1", 0) >= 0.5:
                status = "OK"
            else:
                status = "WARN"

            lines.append(
                f"| {r['question_id']} | {r['category']} | {r['difficulty']} | "
                f"{r.get('token_f1', 0):.3f} | "
                f"{r.get('answer_relevance', 0):.3f} | "
                f"{r.get('mrr', 0):.3f} | "
                f"{r.get('total_duration_ms', 0)/1000:.1f}s | "
                f"{status} |"
            )
        lines.append("")

        # ---- Detailed answers ----
        lines.append("## Detailed Answers")
        lines.append("")
        for r in results:
            lines.append(f"### `{r['question_id']}` - {r['difficulty']}")
            lines.append("")
            lines.append(f"**Question**: {r['question']}")
            lines.append("")
            if r.get("error"):
                lines.append(f"**[X] Error**: `{r['error']}`")
                lines.append("")
                continue
            lines.append(f"**Expected**: {r.get('expected_answer', '')[:400]}")
            lines.append("")
            lines.append(f"**Predicted**: {r.get('predicted_answer', '')[:400]}")
            lines.append("")
            pages = r.get("retrieved_pages") or []
            if pages:
                lines.append(f"**Cited pages**: {', '.join(str(p) for p in pages)}")
                lines.append("")
            lines.append(
                f"**Metrics**: F1=`{r.get('token_f1', 0):.3f}` | "
                f"Rel=`{r.get('answer_relevance', 0):.3f}` | "
                f"MRR=`{r.get('mrr', 0):.3f}`"
            )
            lines.append("")

        return "\n".join(lines)


__all__ = ["ReportGenerator", "ConsoleReporter", "REPORTS_DIR"]
