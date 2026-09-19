"""
Generation service module.

This module provides a high-level service for answer generation that
orchestrates context retrieval, prompt construction, LLM generation,
and citation attachment.

Supports chat history for multi-turn conversations.
"""

from typing import Any, AsyncGenerator, Optional

from app.config import settings
from app.config.constants import LogEvent
from app.core.exceptions import GenerationError
from app.generation import AnswerGenerator, CitationEngine, ContextBuilder, ContextCompressor
from app.llm import PromptManager
from app.monitoring import get_logger, measure_latency
from app.schemas import Answer, Chunk, Query, StreamingChunk
from app.services.retrieval_service import RetrievalService


logger = get_logger("food_safety_rag.services.generation")


class GenerationService:
    """
    High-level service for answer generation.
    
    Orchestrates the complete generation pipeline:
    Query → Retrieve Context → Build Prompt → Generate Answer → Attach Citations
    
    Supports caching and streaming responses.
    Supports chat history for multi-turn conversations.
    
    Attributes:
        retrieval_service: Retrieval service for context.
        answer_generator: Answer generator using LLM.
        prompt_manager: Prompt manager for template management.
    """
    
    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        prompt_manager: Optional[PromptManager] = None,
    ) -> None:
        """
        Initialize the generation service.
        
        Args:
            retrieval_service: Retrieval service. Creates new if None.
            answer_generator: Answer generator. Creates new if None.
            prompt_manager: Prompt manager. Creates new if None.
        """
        self.retrieval_service: RetrievalService = retrieval_service or RetrievalService()
        self.answer_generator: AnswerGenerator = answer_generator or AnswerGenerator()
        self.prompt_manager: PromptManager = prompt_manager or PromptManager()
    
    def _format_history_for_prompt(self, history: Optional[list[dict[str, str]]]) -> str:
        """
        Format chat history for inclusion in the prompt.
        
        Args:
            history: List of previous messages with role and content.
        
        Returns:
            str: Formatted history text.
        """
        if not history:
            return ""
        
        lines = ["Previous conversation:"]
        for msg in history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    def generate_answer(
        self,
        query: Query,
    ) -> Answer:
        """
        Generate an answer for a user query.
        
        Retrieves context, generates answer using LLM, and attaches citations.
        Supports chat history for multi-turn conversations.
        
        Args:
            query: User query with optional history.
        
        Returns:
            Answer: Generated answer with citations.
        
        Raises:
            GenerationError: If generation fails.
        """
        query_text = query.get_effective_query()
        history_text = self._format_history_for_prompt(query.history)
        
        with measure_latency("generation_service") as latency:
            logger.log_generation(
                event=LogEvent.GENERATION_START,
                answer_id="pending",
                query_id=query.query_id,
                message=f"Generation service processing: {query_text[:100]}...",
                details={
                    "has_history": bool(query.history),
                    "history_length": len(query.history) if query.history else 0,
                    "session_id": query.session_id,
                },
            )
            
            # Retrieve context
            chunks, context = self.retrieval_service.retrieve(query)
            
            if not chunks:
                # No relevant documents found
                answer = self.answer_generator.generate_unsupported(
                    query=query,
                    retrieval_duration_ms=latency.duration_ms,
                )
                
                logger.log_generation(
                    event=LogEvent.GENERATION_COMPLETE,
                    answer_id=answer.answer_id,
                    query_id=query.query_id,
                    message="Generated unsupported question response",
                )
                
                return answer
            
            # Generate answer with history
            try:
                answer = self.answer_generator.generate(
                    query=query,
                    context=context,
                    context_chunks=chunks,
                    retrieval_duration_ms=latency.duration_ms,
                    history_text=history_text,
                )
                
                latency.stop(
                    answer_id=answer.answer_id,
                    query_id=query.query_id,
                )
                
                logger.log_generation(
                    event=LogEvent.GENERATION_COMPLETE,
                    answer_id=answer.answer_id,
                    query_id=query.query_id,
                    message="Answer generation complete",
                    details={
                        "citations": len(answer.citations),
                        "supported": answer.is_supported,
                        "duration_ms": latency.duration_ms,
                        "used_history": bool(query.history),
                    },
                )
                
                return answer
                
            except GenerationError:
                raise
            except Exception as exc:
                raise GenerationError(
                    message=f"Answer generation failed: {str(exc)}",
                    model_name=settings.GROQ_MODEL,
                    original_exception=exc,
                )
    
    async def generate_answer_streaming(
        self,
        query: Query,
    ) -> AsyncGenerator[StreamingChunk, None]:
        """
        Generate an answer with streaming token-by-token delivery.
        
        Supports chat history for multi-turn conversations.
        
        Args:
            query: User query with optional history.
        
        Yields:
            StreamingChunk: Individual tokens of the response.
        
        Raises:
            GenerationError: If generation fails.
        """
        query_text = query.get_effective_query()
        history_text = self._format_history_for_prompt(query.history)
        
        logger.log_generation(
            event=LogEvent.STREAMING_START,
            answer_id="pending",
            query_id=query.query_id,
            message=f"Streaming generation started: {query_text[:100]}...",
            details={
                "has_history": bool(query.history),
                "history_length": len(query.history) if query.history else 0,
                "session_id": query.session_id,
            },
        )
        
        # Retrieve context
        chunks, context = self.retrieval_service.retrieve(query)
        
        if not chunks:
            # No relevant documents - stream unsupported message
            unsupported_answer = self.answer_generator.generate_unsupported(query)
            
            for word in unsupported_answer.text.split():
                yield StreamingChunk(
                    chunk_id=0,
                    answer_id=unsupported_answer.answer_id,
                    query_id=query.query_id,
                    token=word + " ",
                    is_final=False,
                )
            
            yield StreamingChunk(
                chunk_id=1,
                answer_id=unsupported_answer.answer_id,
                query_id=query.query_id,
                token="",
                is_final=True,
                citations=[],
            )
            
            return
        
        # Generate streaming response with history
        try:
            async for chunk in self.answer_generator.generate_streaming(
                query=query,
                context=context,
                context_chunks=chunks,
                history_text=history_text,
            ):
                yield chunk
                
        except Exception as exc:
            raise GenerationError(
                message=f"Streaming generation failed: {str(exc)}",
                model_name=settings.GROQ_MODEL,
                original_exception=exc,
            )
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get generation service statistics.
        
        Returns:
            dict[str, Any]: Component statistics.
        """
        return {
            "retrieval": self.retrieval_service.get_stats(),
        }