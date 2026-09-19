"""
Answer schema definitions.

This module defines Pydantic models for generated answers, citations,
and streaming response chunks.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """
    A single citation referencing a source document chunk.
    
    Citations provide traceability from generated answers back to
    the original retrieved evidence.
    
    Attributes:
        document_name: Name of the source document.
        document_id: Unique identifier of the source document.
        page: Page number in the source document.
        chapter: Chapter containing the cited content.
        section: Section containing the cited content.
        subsection: Subsection containing the cited content.
        chunk_id: Unique identifier of the cited chunk.
        chunk_index: Index of the cited chunk within the document.
        relevance_score: Reranker relevance score for this citation.
        quoted_text: The exact text from the source that supports the claim.
        confidence: Confidence that this citation supports the claim (0.0 to 1.0).
    """
    
    document_name: Optional[str] = Field(
        default=None,
        description="Name of the source document.",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Unique identifier of the source document.",
    )
    page: Optional[int] = Field(
        default=None,
        description="Page number in the source document.",
        ge=1,
    )
    chapter: Optional[str] = Field(
        default=None,
        description="Chapter containing the cited content.",
    )
    section: Optional[str] = Field(
        default=None,
        description="Section containing the cited content.",
    )
    subsection: Optional[str] = Field(
        default=None,
        description="Subsection containing the cited content.",
    )
    chunk_id: str = Field(
        ...,
        description="Unique identifier of the cited chunk.",
        min_length=1,
    )
    chunk_index: Optional[int] = Field(
        default=None,
        description="Index of the cited chunk within the document.",
        ge=0,
    )
    relevance_score: Optional[float] = Field(
        default=None,
        description="Reranker relevance score for this citation.",
    )
    quoted_text: Optional[str] = Field(
        default=None,
        description="The exact text from the source that supports the claim.",
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence that this citation supports the claim (0.0 to 1.0).",
        ge=0.0,
        le=1.0,
    )
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "document_name": "HACCP_Manual_2024.pdf",
                "document_id": "doc_abc123",
                "page": 17,
                "chapter": "Hazard Analysis",
                "section": "Biological Hazards",
                "chunk_id": "chunk_abc123_042",
                "chunk_index": 42,
                "relevance_score": 0.92,
                "quoted_text": "Salmonella prevention requires maintaining cooking temperatures above 74°C.",
                "confidence": 0.95,
            }
        }
    
    def to_string(self) -> str:
        """
        Format this citation as a human-readable string.
        
        Returns:
            str: Formatted citation string.
        """
        parts: list[str] = []
        
        if self.document_name:
            parts.append(self.document_name)
        if self.page:
            parts.append(f"p.{self.page}")
        if self.section:
            parts.append(self.section)
        
        if not parts:
            return f"[{self.chunk_id}]"
        
        return f"[{', '.join(parts)}]"
    
    def to_dict_for_display(self) -> dict[str, Any]:
        """
        Convert citation to a display-friendly dictionary.
        
        Returns:
            dict[str, Any]: Dictionary with non-None values only.
        """
        result: dict[str, Any] = {"chunk_id": self.chunk_id}
        
        for field_name, value in self.model_dump().items():
            if value is not None and field_name != "chunk_id":
                result[field_name] = value
        
        return result


class AnswerMetadata(BaseModel):
    """
    Metadata about the answer generation process.
    
    This model captures timing, model information, and pipeline
    statistics for monitoring and evaluation.
    
    Attributes:
        answer_id: Unique identifier for this answer.
        query_id: Reference to the original query.
        model_name: Name of the LLM used for generation.
        generation_timestamp: When the answer was generated.
        total_tokens_generated: Number of tokens in the generated answer.
        total_tokens_prompt: Number of tokens in the prompt/context.
        generation_duration_ms: Time taken to generate the answer in milliseconds.
        retrieval_duration_ms: Time taken for the retrieval pipeline in milliseconds.
        total_duration_ms: Total end-to-end duration in milliseconds.
        num_retrieved_chunks: Number of chunks retrieved.
        num_reranked_chunks: Number of chunks sent to reranker.
        num_context_chunks: Number of chunks in final context.
        num_citations: Number of citations in the answer.
        cache_hit: Whether the answer was served from cache.
        streaming: Whether the response was streamed.
    """
    
    answer_id: str = Field(
        ...,
        description="Unique identifier for this answer.",
        min_length=1,
    )
    query_id: str = Field(
        ...,
        description="Reference to the original query.",
        min_length=1,
    )
    model_name: str = Field(
        ...,
        description="Name of the LLM used for generation.",
    )
    generation_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the answer was generated.",
    )
    total_tokens_generated: int = Field(
        default=0,
        description="Number of tokens in the generated answer.",
        ge=0,
    )
    total_tokens_prompt: int = Field(
        default=0,
        description="Number of tokens in the prompt/context.",
        ge=0,
    )
    generation_duration_ms: Optional[float] = Field(
        default=None,
        description="Time taken to generate the answer in milliseconds.",
        ge=0.0,
    )
    retrieval_duration_ms: Optional[float] = Field(
        default=None,
        description="Time taken for the retrieval pipeline in milliseconds.",
        ge=0.0,
    )
    total_duration_ms: Optional[float] = Field(
        default=None,
        description="Total end-to-end duration in milliseconds.",
        ge=0.0,
    )
    num_retrieved_chunks: int = Field(
        default=0,
        description="Number of chunks retrieved.",
        ge=0,
    )
    num_reranked_chunks: int = Field(
        default=0,
        description="Number of chunks sent to reranker.",
        ge=0,
    )
    num_context_chunks: int = Field(
        default=0,
        description="Number of chunks in final context.",
        ge=0,
    )
    num_citations: int = Field(
        default=0,
        description="Number of citations in the answer.",
        ge=0,
    )
    cache_hit: bool = Field(
        default=False,
        description="Whether the answer was served from cache.",
    )
    streaming: bool = Field(
        default=False,
        description="Whether the response was streamed.",
    )
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "answer_id": "ans_def456",
                "query_id": "qry_xyz789",
                "model_name": "gpt-4o-mini",
                "total_tokens_generated": 245,
                "total_tokens_prompt": 1847,
                "generation_duration_ms": 1200.0,
                "retrieval_duration_ms": 350.0,
                "total_duration_ms": 1550.0,
                "num_retrieved_chunks": 50,
                "num_reranked_chunks": 30,
                "num_context_chunks": 6,
                "num_citations": 4,
                "cache_hit": False,
                "streaming": True,
            }
        }


class Answer(BaseModel):
    """
    A complete generated answer with citations and metadata.
    
    This is the primary output model of the generation pipeline.
    
    Attributes:
        answer_id: Unique identifier for this answer.
        query_id: Reference to the original query.
        text: The generated answer text.
        citations: List of citations supporting the answer.
        metadata: Generation metadata and statistics.
        is_supported: Whether the answer is supported by retrieved documents.
        confidence: Overall confidence in the answer (0.0 to 1.0).
        streaming_complete: Whether streaming has finished.
    """
    
    answer_id: str = Field(
        ...,
        description="Unique identifier for this answer.",
        min_length=1,
    )
    query_id: str = Field(
        ...,
        description="Reference to the original query.",
        min_length=1,
    )
    text: str = Field(
        default="",
        description="The generated answer text.",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="List of citations supporting the answer.",
    )
    metadata: AnswerMetadata = Field(
        ...,
        description="Generation metadata and statistics.",
    )
    is_supported: bool = Field(
        default=True,
        description="Whether the answer is supported by retrieved documents.",
    )
    confidence: float = Field(
        default=1.0,
        description="Overall confidence in the answer (0.0 to 1.0).",
        ge=0.0,
        le=1.0,
    )
    streaming_complete: bool = Field(
        default=True,
        description="Whether streaming has finished.",
    )
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "answer_id": "ans_def456",
                "query_id": "qry_xyz789",
                "text": "Critical control points for Salmonella prevention include...",
                "citations": [
                    {
                        "document_name": "HACCP_Manual_2024.pdf",
                        "page": 17,
                        "chunk_id": "chunk_abc123_042",
                    }
                ],
                "is_supported": True,
                "confidence": 0.92,
            }
        }
    
    def get_citation_text(self) -> str:
        """
        Get the answer text with inline citations.
        
        Returns:
            str: Answer text with citation markers.
        """
        if not self.citations:
            return self.text
        
        return self.text
    
    def get_citations_as_strings(self) -> list[str]:
        """
        Get all citations as formatted strings.
        
        Returns:
            list[str]: List of formatted citation strings.
        """
        return [citation.to_string() for citation in self.citations]
    
    def to_display_dict(self) -> dict[str, Any]:
        """
        Convert answer to a display-friendly dictionary.
        
        Returns:
            dict[str, Any]: Dictionary with answer data formatted for UI display.
        """
        return {
            "answer_id": self.answer_id,
            "text": self.text,
            "citations": [c.to_dict_for_display() for c in self.citations],
            "is_supported": self.is_supported,
            "confidence": self.confidence,
            "metadata": {
                "model_name": self.metadata.model_name,
                "generation_timestamp": self.metadata.generation_timestamp.isoformat(),
                "total_tokens_generated": self.metadata.total_tokens_generated,
                "total_duration_ms": self.metadata.total_duration_ms,
                "num_citations": self.metadata.num_citations,
                "cache_hit": self.metadata.cache_hit,
            },
        }


class StreamingChunk(BaseModel):
    """
    A single chunk in a streaming response.
    
    Used for real-time token-by-token delivery to the client.
    
    Attributes:
        chunk_id: Unique identifier for this chunk within the stream.
        answer_id: Reference to the parent answer.
        query_id: Reference to the original query.
        token: The text token/content of this chunk.
        is_final: Whether this is the last chunk in the stream.
        citations: Citations associated with this chunk (usually empty until final).
        timestamp: When this chunk was generated.
    """
    
    chunk_id: int = Field(
        ...,
        description="Unique identifier for this chunk within the stream.",
        ge=0,
    )
    answer_id: str = Field(
        ...,
        description="Reference to the parent answer.",
        min_length=1,
    )
    query_id: str = Field(
        ...,
        description="Reference to the original query.",
        min_length=1,
    )
    token: str = Field(
        ...,
        description="The text token/content of this chunk.",
    )
    is_final: bool = Field(
        default=False,
        description="Whether this is the last chunk in the stream.",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Citations associated with this chunk.",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this chunk was generated.",
    )
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "chunk_id": 42,
                "answer_id": "ans_def456",
                "query_id": "qry_xyz789",
                "token": "prevention",
                "is_final": False,
            }
        }
    
    def to_sse_format(self) -> str:
        """
        Format this chunk as a Server-Sent Events (SSE) message.
        
        Returns:
            str: SSE-formatted string for streaming.
        """
        import json
        
        data = {
            "chunk_id": self.chunk_id,
            "token": self.token,
            "is_final": self.is_final,
        }
        
        if self.is_final and self.citations:
            data["citations"] = [c.to_dict_for_display() for c in self.citations]
        
        return f"data: {json.dumps(data)}\n\n"


class UnsupportedAnswer(BaseModel):
    """
    A response generated when no relevant documents support the query.
    
    This model provides a structured way to communicate that the system
    cannot answer based on available documents.
    
    Attributes:
        answer_id: Unique identifier for this response.
        query_id: Reference to the original query.
        message: The polite unsupported message.
        suggestion: Suggestion for the user.
        metadata: Generation metadata.
    """
    
    answer_id: str = Field(
        ...,
        description="Unique identifier for this response.",
        min_length=1,
    )
    query_id: str = Field(
        ...,
        description="Reference to the original query.",
        min_length=1,
    )
    message: str = Field(
        default="The available documents do not contain sufficient information to answer this question.",
        description="The polite unsupported message.",
    )
    suggestion: Optional[str] = Field(
        default="Please try uploading additional relevant documents or rephrasing your question.",
        description="Suggestion for the user.",
    )
    metadata: AnswerMetadata = Field(
        ...,
        description="Generation metadata.",
    )
    
    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "answer_id": "ans_unsupported_001",
                "query_id": "qry_xyz789",
                "message": "The available documents do not contain sufficient information to answer this question.",
                "suggestion": "Please try uploading additional relevant documents or rephrasing your question.",
            }
        }
    
    def to_answer(self) -> Answer:
        """
        Convert this unsupported answer to a standard Answer model.
        
        Returns:
            Answer: Standard answer with is_supported=False.
        """
        return Answer(
            answer_id=self.answer_id,
            query_id=self.query_id,
            text=self.message,
            citations=[],
            metadata=self.metadata,
            is_supported=False,
            confidence=0.0,
        )