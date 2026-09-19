"""
Evaluation metrics for RAG quality assessment.

Implements:
    - exact_match: normalized string equality
    - token_f1: harmonic mean of precision/recall on tokens
    - answer_relevance: keyword-based relevance score (0-1)
    - mrr: mean reciprocal rank
    - hit_rate_at_k: whether relevant item is in top-k
    - precision_at_k: fraction of relevant items in top-k
"""

from __future__ import annotations

import re
import string
from typing import Sequence


# ============================================================
# Text normalization
# ============================================================

def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation, collapse whitespace."""
    if not text:
        return ""
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenization after normalization."""
    return _normalize_text(text).split()


# ============================================================
# Exact Match
# ============================================================

def exact_match(predicted: str, expected: str) -> float:
    """
    Normalized exact match: 1.0 if identical after normalization, else 0.0.
    """
    return 1.0 if _normalize_text(predicted) == _normalize_text(expected) else 0.0


def substring_match(predicted: str, expected: str) -> float:
    """
    Substring match: 1.0 if normalized expected is contained in normalized predicted.
    """
    return 1.0 if _normalize_text(expected) in _normalize_text(predicted) else 0.0


# ============================================================
# Token F1
# ============================================================

def token_f1(predicted: str, expected: str) -> float:
    """
    Token-level F1 score between predicted and expected answers.
    """
    pred_tokens = _tokenize(predicted)
    exp_tokens = _tokenize(expected)

    if not pred_tokens and not exp_tokens:
        return 1.0
    if not pred_tokens or not exp_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    exp_set = set(exp_tokens)

    # Count of overlapping tokens
    common = pred_set & exp_set

    if not common:
        return 0.0

    precision = len(common) / len(pred_set)
    recall = len(common) / len(exp_set)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


# ============================================================
# Keyword-based relevance
# ============================================================

def answer_relevance(predicted: str, expected_keywords: Sequence[str]) -> float:
    """
    Fraction of expected keywords that appear in the predicted answer.
    Returns 0.0 to 1.0.
    """
    if not expected_keywords:
        return 0.0

    pred_lower = _normalize_text(predicted)
    if not pred_lower:
        return 0.0

    matches = 0
    for kw in expected_keywords:
        kw_norm = _normalize_text(kw)
        if kw_norm and kw_norm in pred_lower:
            matches += 1

    return matches / len(expected_keywords)


# ============================================================
# Retrieval metrics
# ============================================================

def mrr(retrieved_pages: Sequence[int], expected_pages: Sequence[int]) -> float:
    """
    Mean Reciprocal Rank for a single query.
    Returns 1/rank of the first relevant item, or 0.0 if none found.
    """
    if not expected_pages:
        return 0.0

    expected_set = set(expected_pages)
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in expected_set:
            return 1.0 / rank

    return 0.0


def hit_rate_at_k(
    retrieved_pages: Sequence[int],
    expected_pages: Sequence[int],
    k: int = 5,
) -> float:
    """
    Hit Rate @ K: 1.0 if any relevant item is in top-k, else 0.0.
    """
    if not expected_pages:
        return 0.0

    expected_set = set(expected_pages)
    top_k = retrieved_pages[:k]

    return 1.0 if any(p in expected_set for p in top_k) else 0.0


def precision_at_k(
    retrieved_pages: Sequence[int],
    expected_pages: Sequence[int],
    k: int = 5,
) -> float:
    """
    Precision @ K: fraction of top-k retrieved pages that are relevant.
    """
    if not retrieved_pages or not expected_pages:
        return 0.0

    expected_set = set(expected_pages)
    top_k = retrieved_pages[:k]

    if not top_k:
        return 0.0

    hits = sum(1 for p in top_k if p in expected_set)
    return hits / len(top_k)


def recall_at_k(
    retrieved_pages: Sequence[int],
    expected_pages: Sequence[int],
    k: int = 5,
) -> float:
    """
    Recall @ K: fraction of relevant items found in top-k.
    """
    if not expected_pages:
        return 0.0

    expected_set = set(expected_pages)
    top_k = retrieved_pages[:k]

    hits = sum(1 for p in expected_set if p in top_k)
    return hits / len(expected_set)


# ============================================================
# Aggregation helpers
# ============================================================

def average(values: Sequence[float]) -> float:
    """Safe average."""
    return sum(values) / len(values) if values else 0.0


__all__ = [
    "exact_match",
    "substring_match",
    "token_f1",
    "answer_relevance",
    "mrr",
    "hit_rate_at_k",
    "precision_at_k",
    "recall_at_k",
    "average",
]
