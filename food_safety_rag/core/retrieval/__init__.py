"""
Retrieval module for the Food Safety RAG System.

Exports all retrieval components.
"""

from food_safety_rag.core.retrieval.dense import DenseRetriever
from food_safety_rag.core.retrieval.bm25 import BM25Retriever
from food_safety_rag.core.retrieval.fusion import RRFFusion
from food_safety_rag.core.retrieval.hybrid import HybridRetriever
from food_safety_rag.core.retrieval.reranker import Reranker
from food_safety_rag.core.retrieval.query_expander import QueryExpander
from food_safety_rag.core.retrieval.query_normalizer import QueryNormalizer
from food_safety_rag.core.retrieval.router import RetrievalRouter, RetrievalStrategy

__all__ = [
    "DenseRetriever",
    "BM25Retriever",
    "RRFFusion",
    "HybridRetriever",
    "Reranker",
    "QueryExpander",
    "QueryNormalizer",
    "RetrievalRouter",
    "RetrievalStrategy",
]
