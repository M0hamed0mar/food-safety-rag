"""
Retrieval module for the Food Safety RAG System.

Exports retrieval components.
"""

from food_safety_rag.core.retrieval.dense import DenseRetriever
from food_safety_rag.core.retrieval.bm25 import BM25Retriever
from food_safety_rag.core.retrieval.fusion import RRFFusion
from food_safety_rag.core.retrieval.hybrid import HybridRetriever

__all__ = [
    "DenseRetriever",
    "BM25Retriever",
    "RRFFusion",
    "HybridRetriever",
]
