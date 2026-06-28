"""
Configuration module for the Food Safety RAG System.

Exports settings, constants, and prompts.
"""

from food_safety_rag.core.config.settings import Settings, get_settings
from food_safety_rag.core.config.prompts import PromptManager, PromptTemplates

__all__ = [
    "Settings",
    "get_settings",
    "PromptManager",
    "PromptTemplates",
]
