"""
Prompt manager for dynamic prompt loading.

Separates prompts from code for easy modification.
"""

from typing import Optional

from food_safety_rag.core.config.prompts import PromptManager as ConfigPromptManager
from food_safety_rag.core.monitoring.logger import get_logger

logger = get_logger(__name__)


class PromptManager:
    """
    Unified prompt manager for LLM interactions.
    
    Wraps the configuration prompt manager with additional LLM-specific logic.
    """

    def __init__(self) -> None:
        """
        Initialize prompt manager.
        """
        self.config_manager = ConfigPromptManager()
        logger.debug("Prompt manager initialized")

    def get_system_prompt(self) -> str:
        """
        Get the system prompt for answer generation.
        
        Returns:
            str: System prompt.
        """
        return self.config_manager.get_system_prompt()

    def get_answer_generation_prompt(
        self, query: str, context: str
    ) -> str:
        """
        Get formatted prompt for answer generation.
        
        Args:
            query: The user query.
            context: Retrieved context chunks.
            
        Returns:
            str: Formatted prompt ready for LLM.
        """
        return self.config_manager.get_answer_prompt(query, context)

    def get_query_expansion_prompt(self, query: str) -> str:
        """
        Get prompt for query expansion.
        
        Args:
            query: Original user query.
            
        Returns:
            str: Formatted expansion prompt.
        """
        return self.config_manager.get_query_expansion_prompt(query)

    def format_context_for_llm(
        self, context_chunks: list, max_tokens: Optional[int] = None
    ) -> str:
        """
        Format context chunks for LLM consumption.
        
        Args:
            context_chunks: List of context chunks.
            max_tokens: Optional token limit for context.
            
        Returns:
            str: Formatted context string.
        """
        if not context_chunks:
            return "No relevant context found."

        formatted_chunks = []
        for i, chunk in enumerate(context_chunks, 1):
            # Assuming chunks have text and metadata attributes
            text = getattr(chunk, "text", str(chunk))
            metadata = getattr(chunk, "metadata", None)

            formatted_chunk = f"[Source {i}]\n{text}"

            if metadata:
                doc_name = getattr(metadata, "document_name", "Unknown")
                page = getattr(metadata, "page", None)
                section = getattr(metadata, "section", None)

                source_info = f"\nDocument: {doc_name}"
                if page:
                    source_info += f", Page: {page}"
                if section:
                    source_info += f", Section: {section}"
                formatted_chunk += source_info

            formatted_chunks.append(formatted_chunk)

        return "\n\n".join(formatted_chunks)

    def format_citations(
        self, citations: list
    ) -> str:
        """
        Format citations for inclusion in response.
        
        Args:
            citations: List of citation objects.
            
        Returns:
            str: Formatted citations.
        """
        if not citations:
            return ""

        formatted_citations = ["\n\nCitations:"]
        for i, citation in enumerate(citations, 1):
            doc_name = getattr(citation, "document_name", "Unknown")
            page = getattr(citation, "page", None)
            section = getattr(citation, "section", None)
            chunk_id = getattr(citation, "chunk_id", None)

            citation_text = f"[{i}] {doc_name}"
            if page:
                citation_text += f", p. {page}"
            if section:
                citation_text += f" - {section}"
            if chunk_id:
                citation_text += f" (ID: {chunk_id})"

            formatted_citations.append(citation_text)

        return "\n".join(formatted_citations)
