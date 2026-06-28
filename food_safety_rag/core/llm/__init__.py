"""
LLM module for the Food Safety RAG System.

Exports LLM client and prompt management.
"""

from food_safety_rag.core.llm.openai_client import OpenAIClient
from food_safety_rag.core.llm.prompt_manager import PromptManager

__all__ = [
    "OpenAIClient",
    "PromptManager",
]
