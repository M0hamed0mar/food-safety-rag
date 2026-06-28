"""
Cache management system for the Food Safety RAG System.

Provides multi-layer caching for embeddings, retrieval, and LLM responses.
"""

import hashlib
import pickle
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Generic, TypeVar

from food_safety_rag.core.config.settings import get_settings
from food_safety_rag.core.exceptions import CacheException
from food_safety_rag.core.monitoring.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CacheEntry(Generic[T]):
    """
    A cached entry with expiration tracking.
    
    Generic class that can cache any type of data.
    """

    def __init__(self, data: T, ttl_seconds: int) -> None:
        """
        Initialize cache entry.
        
        Args:
            data: The data to cache.
            ttl_seconds: Time-to-live in seconds.
        """
        self.data = data
        self.created_at = datetime.utcnow()
        self.ttl_seconds = ttl_seconds

    def is_expired(self) -> bool:
        """
        Check if entry has expired.
        
        Returns:
            bool: True if entry has exceeded TTL.
        """
        age = datetime.utcnow() - self.created_at
        return age.total_seconds() > self.ttl_seconds

    def get(self) -> Optional[T]:
        """
        Get cached data if not expired.
        
        Returns:
            Optional[T]: Cached data or None if expired.
        """
        if self.is_expired():
            return None
        return self.data


class Cache(ABC, Generic[T]):
    """
    Abstract base class for cache implementations.
    
    Defines the interface for caching systems.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """
        Retrieve value from cache.
        
        Args:
            key: Cache key.
            
        Returns:
            Optional[T]: Cached value or None.
        """
        pass

    @abstractmethod
    def set(self, key: str, value: T, ttl_seconds: int) -> None:
        """
        Store value in cache.
        
        Args:
            key: Cache key.
            value: Value to cache.
            ttl_seconds: Time-to-live in seconds.
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """
        Remove value from cache.
        
        Args:
            key: Cache key.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all cache entries.
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key.
            
        Returns:
            bool: True if key exists and not expired.
        """
        pass


class InMemoryCache(Cache[T]):
    """
    In-memory cache implementation.
    
    Stores data in memory with TTL-based expiration.
    Suitable for single-instance deployments.
    """

    def __init__(self) -> None:
        """
        Initialize in-memory cache.
        """
        self.store: Dict[str, CacheEntry[T]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[T]:
        """
        Retrieve value from cache.
        
        Args:
            key: Cache key.
            
        Returns:
            Optional[T]: Cached value or None.
        """
        if key not in self.store:
            self.misses += 1
            return None

        entry = self.store[key]
        if entry.is_expired():
            del self.store[key]
            self.misses += 1
            return None

        self.hits += 1
        return entry.get()

    def set(self, key: str, value: T, ttl_seconds: int) -> None:
        """
        Store value in cache.
        
        Args:
            key: Cache key.
            value: Value to cache.
            ttl_seconds: Time-to-live in seconds.
        """
        self.store[key] = CacheEntry(value, ttl_seconds)

    def delete(self, key: str) -> None:
        """
        Remove value from cache.
        
        Args:
            key: Cache key.
        """
        if key in self.store:
            del self.store[key]

    def clear(self) -> None:
        """
        Clear all cache entries.
        """
        self.store.clear()
        self.hits = 0
        self.misses = 0

    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key.
            
        Returns:
            bool: True if key exists and not expired.
        """
        if key not in self.store:
            return False
        return not self.store[key].is_expired()

    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dict[str, int]: Cache hit/miss statistics.
        """
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate_percent": hit_rate,
            "size": len(self.store),
        }


class FileSystemCache(Cache[T]):
    """
    File system-based cache implementation.
    
    Stores cache entries as files with TTL-based expiration.
    Useful for persistent caching across sessions.
    """

    def __init__(self, cache_dir: Path) -> None:
        """
        Initialize file system cache.
        
        Args:
            cache_dir: Directory for cache files.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _get_key_path(self, key: str) -> Path:
        """
        Get file path for a cache key.
        
        Args:
            key: Cache key.
            
        Returns:
            Path: File path for the key.
        """
        # Hash key to create valid filename
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hashed_key}.cache"

    def get(self, key: str) -> Optional[T]:
        """
        Retrieve value from cache.
        
        Args:
            key: Cache key.
            
        Returns:
            Optional[T]: Cached value or None.
        """
        try:
            path = self._get_key_path(key)
            if not path.exists():
                self.misses += 1
                return None

            with open(path, "rb") as f:
                entry = pickle.load(f)

            if entry.is_expired():
                path.unlink()
                self.misses += 1
                return None

            self.hits += 1
            return entry.get()
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}", key=key)
            self.misses += 1
            return None

    def set(self, key: str, value: T, ttl_seconds: int) -> None:
        """
        Store value in cache.
        
        Args:
            key: Cache key.
            value: Value to cache.
            ttl_seconds: Time-to-live in seconds.
        """
        try:
            path = self._get_key_path(key)
            entry = CacheEntry(value, ttl_seconds)
            with open(path, "wb") as f:
                pickle.dump(entry, f)
        except Exception as e:
            logger.error(f"Cache write error: {e}", key=key)

    def delete(self, key: str) -> None:
        """
        Remove value from cache.
        
        Args:
            key: Cache key.
        """
        try:
            path = self._get_key_path(key)
            if path.exists():
                path.unlink()
        except Exception as e:
            logger.error(f"Cache delete error: {e}", key=key)

    def clear(self) -> None:
        """
        Clear all cache entries.
        """
        try:
            for file in self.cache_dir.glob("*.cache"):
                file.unlink()
            self.hits = 0
            self.misses = 0
        except Exception as e:
            logger.error(f"Cache clear error: {e}")

    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key.
            
        Returns:
            bool: True if key exists and not expired.
        """
        try:
            path = self._get_key_path(key)
            if not path.exists():
                return False

            with open(path, "rb") as f:
                entry = pickle.load(f)

            if entry.is_expired():
                path.unlink()
                return False

            return True
        except Exception:
            return False


class CacheManager:
    """
    Centralized cache manager for the RAG system.
    
    Manages multiple cache layers (embedding, retrieval, context, LLM).
    """

    def __init__(self) -> None:
        """
        Initialize cache manager with all cache layers.
        """
        try:
            settings = get_settings()
        except Exception:
            logger.warning("Failed to load settings, using default cache configuration")
            settings = None

        # Initialize caches based on settings
        self.embedding_cache: Optional[Cache] = None
        self.retrieval_cache: Optional[Cache] = None
        self.context_cache: Optional[Cache] = None
        self.llm_response_cache: Optional[Cache] = None

        if settings:
            if getattr(settings, "enable_embedding_cache", True):
                self.embedding_cache = InMemoryCache()
                logger.info("Embedding cache enabled")

            if getattr(settings, "enable_retrieval_cache", True):
                self.retrieval_cache = InMemoryCache()
                logger.info("Retrieval cache enabled")

            if getattr(settings, "enable_context_cache", True):
                self.context_cache = InMemoryCache()
                logger.info("Context cache enabled")

            if getattr(settings, "enable_llm_response_cache", True):
                self.llm_response_cache = InMemoryCache()
                logger.info("LLM response cache enabled")
        else:
            # Default: enable all caches
            self.embedding_cache = InMemoryCache()
            self.retrieval_cache = InMemoryCache()
            self.context_cache = InMemoryCache()
            self.llm_response_cache = InMemoryCache()

    def get_embedding_cache_ttl(self) -> int:
        """
        Get TTL for embedding cache.
        
        Returns:
            int: TTL in seconds.
        """
        try:
            settings = get_settings()
            return getattr(settings, "cache_embedding_ttl", 604800)
        except Exception:
            return 604800  # 7 days default

    def get_retrieval_cache_ttl(self) -> int:
        """
        Get TTL for retrieval cache.
        
        Returns:
            int: TTL in seconds.
        """
        try:
            settings = get_settings()
            return getattr(settings, "cache_retrieval_ttl", 3600)
        except Exception:
            return 3600  # 1 hour default

    def get_context_cache_ttl(self) -> int:
        """
        Get TTL for context cache.
        
        Returns:
            int: TTL in seconds.
        """
        try:
            settings = get_settings()
            return getattr(settings, "cache_context_ttl", 1800)
        except Exception:
            return 1800  # 30 minutes default

    def get_llm_response_cache_ttl(self) -> int:
        """
        Get TTL for LLM response cache.
        
        Returns:
            int: TTL in seconds.
        """
        try:
            settings = get_settings()
            return getattr(settings, "cache_llm_response_ttl", 3600)
        except Exception:
            return 3600  # 1 hour default

    def cache_embedding(self, text: str, embedding: list) -> None:
        """
        Cache an embedding.
        
        Args:
            text: The text that was embedded.
            embedding: The embedding vector.
        """
        if not self.embedding_cache:
            return

        key = hashlib.sha256(text.encode()).hexdigest()
        ttl = self.get_embedding_cache_ttl()
        self.embedding_cache.set(key, embedding, ttl)

    def get_cached_embedding(self, text: str) -> Optional[list]:
        """
        Retrieve a cached embedding.
        
        Args:
            text: The text to retrieve embedding for.
            
        Returns:
            Optional[list]: Cached embedding or None.
        """
        if not self.embedding_cache:
            return None

        key = hashlib.sha256(text.encode()).hexdigest()
        return self.embedding_cache.get(key)

    def clear_all(self) -> None:
        """
        Clear all caches.
        """
        if self.embedding_cache:
            self.embedding_cache.clear()
        if self.retrieval_cache:
            self.retrieval_cache.clear()
        if self.context_cache:
            self.context_cache.clear()
        if self.llm_response_cache:
            self.llm_response_cache.clear()
        logger.info("All caches cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all caches.
        
        Returns:
            Dict[str, Any]: Statistics for each cache layer.
        """
        stats = {}
        if self.embedding_cache and hasattr(self.embedding_cache, "get_stats"):
            stats["embedding"] = self.embedding_cache.get_stats()
        if self.retrieval_cache and hasattr(self.retrieval_cache, "get_stats"):
            stats["retrieval"] = self.retrieval_cache.get_stats()
        if self.context_cache and hasattr(self.context_cache, "get_stats"):
            stats["context"] = self.context_cache.get_stats()
        if self.llm_response_cache and hasattr(self.llm_response_cache, "get_stats"):
            stats["llm_response"] = self.llm_response_cache.get_stats()
        return stats
