"""
Prompt templates for the Food Safety RAG System.

All prompts are stored externally to allow easy modification without code changes.
This module centralizes all LLM prompt management.
"""

from typing import Dict


class PromptTemplates:
    """
    Container for all LLM prompt templates.
    
    Prompts are versioned and can be updated independently from code.
    """

    @staticmethod
    def system_prompt() -> str:
        """
        System prompt for the answer generation stage.
        
        Returns:
            str: System prompt that instructs the LLM behavior.
        """
        return """You are an expert Food Safety assistant. Your role is to answer questions 
about food safety, regulations, procedures, and best practices based exclusively on the provided 
documents.

CRITICAL RULES:
1. ONLY use information from the provided context. Never use external knowledge.
2. If the context does not contain sufficient information to answer the question, explicitly state: 
   "Based on the available documents, I cannot provide a definitive answer to this question."
3. Always cite your sources using the provided document references.
4. Be precise and technically accurate.
5. If you identify conflicting information in the documents, acknowledge it.
6. Never fabricate specific values, regulations, or procedures.
7. For procedures, always preserve the exact sequence and details from the documents.

Your goal is to help users understand food safety regulations and best practices 
with maximum accuracy and full traceability."""

    @staticmethod
    def answer_prompt_template() -> str:
        """
        Template for answer generation prompts.
        
        Returns:
            str: Prompt template with placeholders for context and query.
        """
        return """Based on the following context from food safety documents, answer the user's question accurately and completely.

CONTEXT:
{context}

USER QUESTION:
{query}

Please provide a comprehensive answer that:
1. Directly addresses the question
2. Cites specific documents and sections
3. Includes relevant details and procedures
4. Acknowledges any gaps in available documentation

ANSWER:"""

    @staticmethod
    def query_expansion_template() -> str:
        """
        Template for query expansion.
        
        Returns:
            str: Template for generating alternative search queries.
        """
        return """Given the following food safety query, generate 3-5 alternative search queries that 
capture different aspects and terminology related to the original question.

ORIGINAL QUERY: {query}

Generate alternative queries that:
1. Use different terminology or synonyms
2. Expand acronyms or abbreviations
3. Add domain-specific keywords
4. Rephrase for different search strategies

Alternative queries (one per line):"""

    @staticmethod
    def citation_format_template() -> str:
        """
        Template for formatting citations.
        
        Returns:
            str: Template for consistent citation formatting.
        """
        return """Citation Format:
- Document: {document_name}
- Page: {page}
- Section: {section}
- Reference ID: {chunk_id}"""

    @staticmethod
    def context_builder_prompt() -> str:
        """
        Template for context compression instructions.
        
        Returns:
            str: Instructions for optimizing context.
        """
        return """The following context will be sent to an LLM. Review and optimize it by:
1. Removing duplicate information (keep the highest quality version)
2. Removing irrelevant metadata
3. Preserving all citations
4. Maintaining semantic coherence
5. Keeping the most relevant content

Current context:
{context}

Optimized context:"""


class PromptManager:
    """
    Manager for dynamic prompt loading and caching.
    
    Allows prompts to be versioned and managed externally.
    """

    def __init__(self) -> None:
        """Initialize prompt manager with template references."""
        self._prompts: Dict[str, str] = {}
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load all prompts from templates."""
        self._prompts = {
            "system": PromptTemplates.system_prompt(),
            "answer": PromptTemplates.answer_prompt_template(),
            "query_expansion": PromptTemplates.query_expansion_template(),
            "citation": PromptTemplates.citation_format_template(),
            "context_builder": PromptTemplates.context_builder_prompt(),
        }

    def get_system_prompt(self) -> str:
        """
        Get the system prompt.
        
        Returns:
            str: System prompt for LLM initialization.
        """
        return self._prompts["system"]

    def get_answer_prompt(self, query: str, context: str) -> str:
        """
        Get formatted answer prompt with context and query.
        
        Args:
            query: User question.
            context: Retrieved context chunks.
            
        Returns:
            str: Formatted prompt ready for LLM.
        """
        template = self._prompts["answer"]
        return template.format(query=query, context=context)

    def get_query_expansion_prompt(self, query: str) -> str:
        """
        Get formatted query expansion prompt.
        
        Args:
            query: Original user query.
            
        Returns:
            str: Formatted prompt for query expansion.
        """
        template = self._prompts["query_expansion"]
        return template.format(query=query)

    def get_citation_format(
        self,
        document_name: str,
        page: int | None,
        section: str | None,
        chunk_id: str,
    ) -> str:
        """
        Get formatted citation string.
        
        Args:
            document_name: Name of the source document.
            page: Page number (optional).
            section: Section name (optional).
            chunk_id: Unique chunk identifier.
            
        Returns:
            str: Formatted citation.
        """
        template = self._prompts["citation"]
        return template.format(
            document_name=document_name,
            page=page or "N/A",
            section=section or "N/A",
            chunk_id=chunk_id,
        )
