"""
Prompt manager module.

This module centralizes prompt management for the LLM layer.
All prompts are loaded from the configuration system and can be
versioned, customized, and easily modified without changing source code.
"""

from typing import Any, Optional

from app.config.constants import LogEvent
from app.config.prompts import (
    ANSWER_GENERATION_PROMPT,
    ANSWER_WITH_CITATION_PROMPT,
    CONTEXT_COMPRESSION_PROMPT,
    DOCUMENT_SUMMARY_PROMPT,
    PROMPT_REGISTRY,
    QUERY_EXPANSION_PROMPT,
    SYSTEM_PROMPT,
    UNSUPPORTED_QUESTION_PROMPT,
    format_prompt,
    get_prompt,
    list_prompts,
    register_prompt,
)
from app.monitoring import get_logger


logger = get_logger("food_safety_rag.llm.prompt_manager")


class PromptManager:
    """
    Centralized prompt manager for the LLM layer.
    
    Provides a clean interface for accessing, formatting, and managing
    prompt templates. Supports versioning and dynamic registration.
    
    Attributes:
        system_prompt: Base system prompt.
        answer_prompt: Standard answer generation prompt.
        citation_prompt: Answer prompt with strict citation requirements.
        query_expansion_prompt: Query expansion prompt.
        context_compression_prompt: Context compression prompt.
        unsupported_prompt: Unsupported question prompt.
        document_summary_prompt: Document summary prompt.
    """
    
    def __init__(self) -> None:
        """Initialize the prompt manager with default prompts."""
        self.system_prompt = SYSTEM_PROMPT
        self.answer_prompt = ANSWER_GENERATION_PROMPT
        self.citation_prompt = ANSWER_WITH_CITATION_PROMPT
        self.query_expansion_prompt = QUERY_EXPANSION_PROMPT
        self.context_compression_prompt = CONTEXT_COMPRESSION_PROMPT
        self.unsupported_prompt = UNSUPPORTED_QUESTION_PROMPT
        self.document_summary_prompt = DOCUMENT_SUMMARY_PROMPT
        
        logger.log_event(
            event=LogEvent.SYSTEM_STARTUP,
            message="Prompt manager initialized with default prompts",
            details={"available_prompts": list_prompts()},
        )
    
    def get_system_prompt(self) -> str:
        """
        Get the base system prompt.
        
        Returns:
            str: System prompt text.
        """
        return self.system_prompt.template
    
    def format_answer_prompt(
        self,
        context: str,
        query: str,
        use_citations: bool = False,
    ) -> str:
        """
        Format the answer generation prompt.
        
        Args:
            context: Retrieved document context.
            query: User query.
            use_citations: Whether to use strict citation prompt.
        
        Returns:
            str: Formatted prompt.
        """
        prompt_name = "answer_with_citation" if use_citations else "answer_generation"
        
        try:
            return format_prompt(prompt_name, context=context, query=query)
        except Exception as exc:
            logger.log_error(
                event=LogEvent.ERROR,
                message=f"Failed to format answer prompt: {str(exc)}",
                exception=exc,
            )
            # Fallback to manual formatting
            template = self.citation_prompt.template if use_citations else self.answer_prompt.template
            return template.format(context=context, query=query)
    
    def format_query_expansion_prompt(
        self,
        query: str,
        num_variants: int = 3,
    ) -> str:
        """
        Format the query expansion prompt.
        
        Args:
            query: Original query.
            num_variants: Number of variants to generate.
        
        Returns:
            str: Formatted prompt.
        """
        try:
            return format_prompt("query_expansion", query=query, num_variants=num_variants)
        except Exception as exc:
            logger.log_error(
                event=LogEvent.ERROR,
                message=f"Failed to format query expansion prompt: {str(exc)}",
                exception=exc,
            )
            return self.query_expansion_prompt.template.format(query=query, num_variants=num_variants)
    
    def format_unsupported_prompt(self, query: str) -> str:
        """
        Format the unsupported question prompt.
        
        Args:
            query: User query.
        
        Returns:
            str: Formatted prompt.
        """
        try:
            return format_prompt("unsupported_question", query=query)
        except Exception as exc:
            logger.log_error(
                event=LogEvent.ERROR,
                message=f"Failed to format unsupported prompt: {str(exc)}",
                exception=exc,
            )
            return self.unsupported_prompt.template.format(query=query)
    
    def format_document_summary_prompt(self, content: str) -> str:
        """
        Format the document summary prompt.
        
        Args:
            content: Document content.
        
        Returns:
            str: Formatted prompt.
        """
        try:
            return format_prompt("document_summary", content=content)
        except Exception as exc:
            logger.log_error(
                event=LogEvent.ERROR,
                message=f"Failed to format document summary prompt: {str(exc)}",
                exception=exc,
            )
            return self.document_summary_prompt.template.format(content=content)
    
    def get_prompt_version(self, prompt_name: str) -> Optional[str]:
        """
        Get the version of a prompt.
        
        Args:
            prompt_name: Name of the prompt.
        
        Returns:
            Optional[str]: Version string, or None if not found.
        """
        try:
            prompt = get_prompt(prompt_name)
            return prompt.version
        except KeyError:
            return None
    
    def list_available_prompts(self) -> list[str]:
        """
        List all available prompt names.
        
        Returns:
            list[str]: Sorted list of prompt names.
        """
        return list_prompts()
    
    def get_prompt_info(self, prompt_name: str) -> Optional[dict[str, Any]]:
        """
        Get information about a prompt.
        
        Args:
            prompt_name: Name of the prompt.
        
        Returns:
            Optional[dict[str, Any]]: Prompt information, or None if not found.
        """
        try:
            prompt = get_prompt(prompt_name)
            return {
                "name": prompt.name,
                "version": prompt.version,
                "description": prompt.description,
                "template_length": len(prompt.template),
            }
        except KeyError:
            return None
    
    def register_custom_prompt(
        self,
        name: str,
        template: str,
        version: str,
        description: str,
    ) -> None:
        """
        Register a custom prompt at runtime.
        
        Args:
            name: Prompt name.
            template: Prompt template string.
            version: Version string.
            description: Human-readable description.
        
        Raises:
            ValueError: If prompt name already exists.
        """
        from app.config.prompts import PromptTemplate
        
        custom_prompt = PromptTemplate(
            name=name,
            version=version,
            template=template,
            description=description,
        )
        
        register_prompt(custom_prompt)
        
        logger.log_event(
            event=LogEvent.SYSTEM_STARTUP,
            message=f"Registered custom prompt: {name} v{version}",
            details={"prompt_name": name, "version": version},
        )