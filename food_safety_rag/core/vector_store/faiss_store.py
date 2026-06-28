"""
FAISS-based vector store implementation.

Provides semantic search and index management for document embeddings.
"""

import os
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import faiss
import numpy as np

from food_safety_rag.core.config.constants import EMBEDDING_DIMENSION
from food_safety_rag.core.exceptions import (
    VectorStoreException,
    IndexException,
)
from food_safety_rag.core.monitoring.logger import get_logger

logger = get_logger(__name__)


class FAISSStore:
    """
    FAISS-based vector store for semantic search.
    
    Manages dense embeddings and provides fast similarity search.
    Supports index persistence and incremental updates.
    """

    def __init__(self, index_dir: Path, dimension: int = EMBEDDING_DIMENSION) -> None:
        """
        Initialize FAISS store.
        
        Args:
            index_dir: Directory for storing index files.
            dimension: Embedding dimension (default: 384 for text-embedding-3-small).
            
        Raises:
            VectorStoreException: If initialization fails.
        """
        self.index_dir = Path(index_dir)
        self.dimension = dimension
        self.index_path = self.index_dir / "index.faiss"
        self.id_map_path = self.index_dir / "id_map.pkl"
        self.metadata_path = self.index_dir / "metadata.pkl"

        # Create directory if it doesn't exist
        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise VectorStoreException(
                f"Failed to create index directory: {e}"
            ) from e

        # Initialize or load index
        self.index: Optional[faiss.IndexIDMap] = None
        self.chunk_id_to_idx: Dict[str, int] = {}
        self.idx_to_chunk_id: Dict[int, str] = {}
        self.metadata_store: Dict[str, Dict[str, Any]] = {}
        self._next_idx = 0

        self._load_or_create_index()
        logger.info(
            "FAISS store initialized",
            index_dir=str(self.index_dir),
            dimension=dimension,
        )

    def _load_or_create_index(self) -> None:
        """
        Load existing index or create new one.
        
        Raises:
            IndexException: If index operations fail.
        """
        try:
            if self.index_path.exists():
                self._load_index()
                logger.info(
                    "Loaded existing FAISS index",
                    index_size=self.index.index.ntotal,
                )
            else:
                self._create_index()
                logger.info("Created new FAISS index")
        except Exception as e:
            raise IndexException(f"Failed to load or create index: {e}") from e

    def _create_index(self) -> None:
        """
        Create new FAISS index.
        
        Uses IDMap wrapper to support arbitrary chunk IDs.
        """
        # Create base index with inner product (equivalent to cosine for normalized vectors)
        base_index = faiss.IndexFlatIP(self.dimension)
        # Wrap with IDMap to support custom IDs
        self.index = faiss.IndexIDMap(base_index)

    def _load_index(self) -> None:
        """
        Load FAISS index from disk.
        
        Raises:
            IndexException: If loading fails.
        """
        try:
            # Load index
            self.index = faiss.read_index(str(self.index_path))

            # Load ID mappings
            if self.id_map_path.exists():
                with open(self.id_map_path, "rb") as f:
                    mappings = pickle.load(f)
                    self.chunk_id_to_idx = mappings.get(
                        "chunk_id_to_idx", {}
                    )
                    self.idx_to_chunk_id = mappings.get(
                        "idx_to_chunk_id", {}
                    )
                    self._next_idx = mappings.get("next_idx", 0)

            # Load metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, "rb") as f:
                    self.metadata_store = pickle.load(f)

            logger.debug(
                "Index loaded successfully",
                chunk_count=len(self.chunk_id_to_idx),
            )
        except Exception as e:
            raise IndexException(f"Failed to load index: {e}") from e

    def add_embeddings(
        self,
        embeddings: List[List[float]],
        chunk_ids: List[str],
        metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Add embeddings to index.
        
        Args:
            embeddings: List of embedding vectors.
            chunk_ids: List of corresponding chunk IDs.
            metadata: Optional metadata for each chunk.
            
        Raises:
            VectorStoreException: If adding embeddings fails.
        """
        if len(embeddings) != len(chunk_ids):
            raise VectorStoreException(
                "Embeddings and chunk_ids must have same length"
            )

        try:
            # Convert to numpy array and normalize for cosine similarity
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)

            # Create IDs for FAISS
            ids = np.arange(
                self._next_idx, self._next_idx + len(embeddings), dtype=np.int64
            )

            # Add to index
            self.index.add_with_ids(embeddings_array, ids)

            # Update mappings
            for i, chunk_id in enumerate(chunk_ids):
                idx = ids[i]
                self.chunk_id_to_idx[chunk_id] = idx
                self.idx_to_chunk_id[idx] = chunk_id

                # Store metadata
                if metadata and chunk_id in metadata:
                    self.metadata_store[chunk_id] = metadata[chunk_id]

            self._next_idx += len(embeddings)
            logger.info(
                "Added embeddings to index",
                count=len(embeddings),
                total_index_size=self.index.index.ntotal,
            )
        except Exception as e:
            raise VectorStoreException(f"Failed to add embeddings: {e}") from e

    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        threshold: Optional[float] = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector.
            k: Number of results to return.
            threshold: Optional similarity threshold for filtering.
            
        Returns:
            Tuple[List[str], List[float]]: Chunk IDs and similarity scores.
            
        Raises:
            VectorStoreException: If search fails.
        """
        if self.index is None or self.index.index.ntotal == 0:
            return [], []

        try:
            # Normalize query embedding
            query_array = np.array(
                [query_embedding], dtype=np.float32
            )
            faiss.normalize_L2(query_array)

            # Search
            distances, ids = self.index.search(query_array, k)
            distances = distances[0]  # Get first (only) result
            ids = ids[0]

            # Convert IDs back to chunk IDs
            chunk_ids = []
            scores = []
            for idx, distance in zip(ids, distances):
                # FAISS returns -1 for invalid IDs when using IDMap
                if idx == -1:
                    continue

                chunk_id = self.idx_to_chunk_id.get(int(idx))
                if chunk_id:
                    # Filter by threshold if provided
                    if threshold is None or distance >= threshold:
                        chunk_ids.append(chunk_id)
                        scores.append(float(distance))

            logger.debug(
                "Search completed",
                query_embedding_dimension=len(query_embedding),
                results_returned=len(chunk_ids),
                k_requested=k,
            )
            return chunk_ids, scores
        except Exception as e:
            raise VectorStoreException(f"Search failed: {e}") from e

    def get_metadata(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a chunk.
        
        Args:
            chunk_id: The chunk ID.
            
        Returns:
            Optional[Dict[str, Any]]: Metadata dictionary or None.
        """
        return self.metadata_store.get(chunk_id)

    def delete_chunk(self, chunk_id: str) -> None:
        """
        Delete a chunk from the index.
        
        Note: FAISS doesn't support true deletion, but we can mark as removed.
        
        Args:
            chunk_id: The chunk ID to delete.
        """
        if chunk_id in self.chunk_id_to_idx:
            idx = self.chunk_id_to_idx[chunk_id]
            del self.chunk_id_to_idx[chunk_id]
            del self.idx_to_chunk_id[idx]
            if chunk_id in self.metadata_store:
                del self.metadata_store[chunk_id]
            logger.info(f"Marked chunk as deleted", chunk_id=chunk_id)

    def save(self) -> None:
        """
        Persist index to disk.
        
        Raises:
            IndexException: If saving fails.
        """
        try:
            # Save FAISS index
            faiss.write_index(self.index.index, str(self.index_path))

            # Save ID mappings
            mappings = {
                "chunk_id_to_idx": self.chunk_id_to_idx,
                "idx_to_chunk_id": self.idx_to_chunk_id,
                "next_idx": self._next_idx,
            }
            with open(self.id_map_path, "wb") as f:
                pickle.dump(mappings, f)

            # Save metadata
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.metadata_store, f)

            logger.info(
                "Index saved to disk",
                index_path=str(self.index_path),
                chunk_count=len(self.chunk_id_to_idx),
            )
        except Exception as e:
            raise IndexException(f"Failed to save index: {e}") from e

    def get_index_size(self) -> int:
        """
        Get number of embeddings in index.
        
        Returns:
            int: Number of chunks indexed.
        """
        return self.index.index.ntotal if self.index else 0

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dict[str, Any]: Index statistics.
        """
        return {
            "total_chunks": self.get_index_size(),
            "unique_chunks": len(self.chunk_id_to_idx),
            "metadata_entries": len(self.metadata_store),
            "dimension": self.dimension,
            "index_path": str(self.index_path),
        }
