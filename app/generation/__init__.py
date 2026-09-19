"""
Generation package.

Exposes:
    - AnswerGenerator: LLM answer generation (Groq)
    - CitationEngine: deterministic citation generation
    - ContextBuilder: build context from retrieved chunks
    - ContextCompressor: deduplicate & compress context
"""

from app.generation.citation import CitationEngine
from app.generation.compressor import ContextCompressor
from app.generation.context_builder import ContextBuilder
from app.generation.generator import AnswerGenerator

__all__ = [
    "AnswerGenerator",
    "CitationEngine",
    "ContextBuilder",
    "ContextCompressor",
]
