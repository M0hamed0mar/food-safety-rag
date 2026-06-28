"""
OpenAI client for the Food Safety RAG System.

Handles all interactions with OpenAI APIs (embeddings and LLM).
"""

import time
from typing import Optional, List

import openai
from openai import OpenAI, APIError, RateLimitError

from food_safety_rag.core.config.settings import get_settings
from food_safety_rag.core.exceptions import (
    EmbeddingAPIException,
    GenerationAPIException,
)
from food_safety_rag.core.monitoring.logger import get_logger

logger = get_logger(__name__)


class OpenAIClient:
    """
    Client for OpenAI API interactions.
    
    Handles embedding generation and LLM calls with retry logic,
    rate limit handling, and error management.
    """

    def __init__(self) -> None:
        """
        Initialize OpenAI client with configuration.
        
        Raises:
            ValueError: If OpenAI API key is not configured.
        """
        try:
            settings = get_settings()
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            raise ValueError("Settings configuration failed") from e

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=settings.openai_api_key)
        self.embedding_model = settings.openai_embedding_model
        self.llm_model = settings.openai_llm_model
        self.embedding_batch_size = settings.openai_embedding_batch_size
        self.request_timeout = settings.openai_request_timeout
        self.max_retries = settings.openai_max_retries
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens

        logger.info(
            "OpenAI client initialized",
            embedding_model=self.embedding_model,
            llm_model=self.llm_model,
        )

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI API.
        
        Args:
            text: Text to embed.
            
        Returns:
            List[float]: 384-dimensional embedding vector.
            
        Raises:
            EmbeddingAPIException: If embedding generation fails.
        """
        if not text or not text.strip():
            raise EmbeddingAPIException("Cannot embed empty text")

        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text.strip(),
                    encoding_format="float",
                )
                embedding = response.data[0].embedding
                logger.debug(
                    f"Generated embedding for text",
                    text_length=len(text),
                    embedding_dimension=len(embedding),
                )
                return embedding
            except RateLimitError as e:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(
                    f"Rate limit hit, retrying in {wait_time}s",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                else:
                    raise EmbeddingAPIException(
                        f"Rate limit exceeded after {self.max_retries} retries",
                        status_code=429,
                    ) from e
            except APIError as e:
                logger.error(f"OpenAI API error: {e}", attempt=attempt + 1)
                if attempt == self.max_retries - 1:
                    raise EmbeddingAPIException(
                        f"Failed to generate embedding: {str(e)}",
                        status_code=getattr(e, "status_code", None),
                    ) from e
                time.sleep(2 ** attempt)

        raise EmbeddingAPIException("Embedding generation failed after all retries")

    def generate_embeddings_batch(
        self, texts: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List[List[float]]: List of embedding vectors.
            
        Raises:
            EmbeddingAPIException: If batch embedding fails.
        """
        if not texts:
            return []

        embeddings = []

        for i in range(0, len(texts), self.embedding_batch_size):
            batch = texts[i : i + self.embedding_batch_size]
            logger.debug(
                f"Processing embedding batch",
                batch_start=i,
                batch_size=len(batch),
                total=len(texts),
            )

            for attempt in range(self.max_retries):
                try:
                    response = self.client.embeddings.create(
                        model=self.embedding_model,
                        input=batch,
                        encoding_format="float",
                    )

                    batch_embeddings = [
                        item.embedding for item in response.data
                    ]
                    embeddings.extend(batch_embeddings)
                    break
                except RateLimitError as e:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Rate limit in batch embedding, retrying",
                        wait_time=wait_time,
                        attempt=attempt + 1,
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(wait_time)
                    else:
                        raise EmbeddingAPIException(
                            f"Batch embedding rate limited after {self.max_retries} retries",
                            status_code=429,
                        ) from e
                except APIError as e:
                    logger.error(f"Batch embedding API error: {e}")
                    if attempt == self.max_retries - 1:
                        raise EmbeddingAPIException(
                            f"Batch embedding failed: {str(e)}"
                        ) from e
                    time.sleep(2 ** attempt)

        logger.info(
            f"Generated embeddings for batch",
            total_texts=len(texts),
            total_embeddings=len(embeddings),
        )
        return embeddings

    def generate_completion(
        self,
        system_prompt: str,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate LLM completion.
        
        Args:
            system_prompt: System prompt to set behavior.
            user_message: User message/query.
            temperature: Optional temperature override.
            max_tokens: Optional max tokens override.
            
        Returns:
            str: Generated completion text.
            
        Raises:
            GenerationAPIException: If generation fails.
        """
        if not user_message or not user_message.strip():
            raise GenerationAPIException("Cannot generate completion for empty message")

        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=temp,
                    max_tokens=max_tok,
                )

                completion = response.choices[0].message.content
                logger.debug(
                    f"Generated completion",
                    model=self.llm_model,
                    completion_length=len(completion),
                )
                return completion
            except RateLimitError as e:
                wait_time = 2 ** attempt
                logger.warning(
                    f"Rate limit in completion generation, retrying",
                    wait_time=wait_time,
                    attempt=attempt + 1,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                else:
                    raise GenerationAPIException(
                        f"Completion generation rate limited after {self.max_retries} retries",
                        status_code=429,
                        model=self.llm_model,
                    ) from e
            except APIError as e:
                logger.error(f"Completion generation API error: {e}")
                if attempt == self.max_retries - 1:
                    raise GenerationAPIException(
                        f"Completion generation failed: {str(e)}",
                        model=self.llm_model,
                    ) from e
                time.sleep(2 ** attempt)

        raise GenerationAPIException(
            "Completion generation failed after all retries",
            model=self.llm_model,
        )

    def generate_completion_stream(
        self,
        system_prompt: str,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Generate LLM completion with streaming.
        
        Args:
            system_prompt: System prompt to set behavior.
            user_message: User message/query.
            temperature: Optional temperature override.
            max_tokens: Optional max tokens override.
            
        Yields:
            str: Token chunks as they arrive.
            
        Raises:
            GenerationAPIException: If streaming fails.
        """
        if not user_message or not user_message.strip():
            raise GenerationAPIException("Cannot generate completion for empty message")

        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        for attempt in range(self.max_retries):
            try:
                stream = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=temp,
                    max_tokens=max_tok,
                    stream=True,
                )

                logger.debug(f"Started streaming completion")

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

                logger.debug(f"Completed streaming completion")
                break
            except RateLimitError as e:
                wait_time = 2 ** attempt
                logger.warning(
                    f"Rate limit in streaming, retrying",
                    wait_time=wait_time,
                    attempt=attempt + 1,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                else:
                    raise GenerationAPIException(
                        f"Streaming rate limited after {self.max_retries} retries",
                        status_code=429,
                        model=self.llm_model,
                    ) from e
            except APIError as e:
                logger.error(f"Streaming API error: {e}")
                if attempt == self.max_retries - 1:
                    raise GenerationAPIException(
                        f"Streaming failed: {str(e)}",
                        model=self.llm_model,
                    ) from e
                time.sleep(2 ** attempt)
