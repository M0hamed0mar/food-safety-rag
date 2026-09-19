"""
Retrieval package.

Exposes the complete retrieval stack:
    - BM25Retriever, DenseRetriever
    - QueryExpander
    - ReciprocalRankFusion
    - CrossEncoderReranker
    - ResultValidator
    - RetrievalRouter
    - RetrievalOrchestrator
    - HybridRetriever
"""

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.expander import QueryExpander
from app.retrieval.fusion import ReciprocalRankFusion
from app.retrieval.validator import ResultValidator
from app.retrieval.reranker import CrossEncoderReranker, get_reranker
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.router import RetrievalRouter
from app.retrieval.orchestrator import RetrievalOrchestrator

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "QueryExpander",
    "ReciprocalRankFusion",
    "ResultValidator",
    "CrossEncoderReranker",
    "get_reranker",
    "HybridRetriever",
    "RetrievalRouter",
    "RetrievalOrchestrator",
]
