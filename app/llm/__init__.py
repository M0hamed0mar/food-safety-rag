"""
LLM package.

Exposes:
    - PromptManager: template management
    - QueryTranslator: query translation + expansion
    - get_translator: shared singleton
"""

from app.llm.prompt_manager import PromptManager
from app.llm.translator import (
    QueryTranslator,
    QueryRewriterResult,
    get_translator,
    reset_translator,
)

__all__ = [
    "PromptManager",
    "QueryTranslator",
    "QueryRewriterResult",
    "get_translator",
    "reset_translator",
]
