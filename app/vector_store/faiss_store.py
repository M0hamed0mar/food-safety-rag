"""
FAISS vector store implementation.

This module provides a FAISS-based vector store with an abstraction layer
that isolates FAISS details from other modules. Future migration to
other vector databases (Qdrant, Weaviate, etc.) should require minimal changes.

Supports multiple index types:
- Flat: Brute-force search (default, good for < 10,000 vectors)
- IVF: Inverted File Index (good for 10,000 - 1M vectors)
- HNSW: Hierarchical Navigable Small World (good for large datasets)
- IVFPQ: IVF with Product Quantization (good for very large datasets)

Implements metadata filtering by pre-filtering chunks before vector search.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np

from app.config import settings
from app.config.constants import (
    FAISS_INDEX_DEFAULT,
    FAISS_INDEX_FLAT,
    FAISS_INDEX_HNSW,
    FAISS_INDEX_IVFFLAT,
    FAISS_INDEX_IVFPQ,
    FAISS_NLIST_DEFAULT,
    FAISS_NPROBE_DEFAULT,
    FAISS_HNSW_M_DEFAULT,
    FAISS_HNSW_EF_CONSTRUCTION_DEFAULT,
    FAISS_HNSW_EF_SEARCH_DEFAULT,
    LogEvent,
)
from app.core.exceptions import VectorStoreError
from app.monitoring import get_logger, measure_latency
from app.schemas import Chunk, ChunkMetadata
from app.vector_store.metadata_store import MetadataStore


logger = get_logger("food_safety_rag.vector_store.faiss")


class VectorStore:
    """
    Abstract base class for vector store implementations.

    Defines the interface that all vector store implementations must follow.
    This abstraction enables easy swapping of underlying vector databases.

    Methods:
        add: Add chunks with embeddings to the store.
        search: Search for similar vectors.
        delete: Remove chunks from the store.
        save: Persist the store to disk.
        load: Load the store from disk.
        get_stats: Get store statistics.
        compact: Rebuild the index excluding deleted chunks.
    """

    def add(self, chunks: list[Chunk]) -> None:
        """
        Add chunks with embeddings to the vector store.

        Args:
            chunks: List of chunks with embeddings.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 50,
        filter_dict: Optional[dict[str, Any]] = None,
    ) -> list[tuple[Chunk, float]]:
        """
        Search for chunks similar to the query embedding.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.
            filter_dict: Optional metadata filters.

        Returns:
            list[tuple[Chunk, float]]: List of (chunk, similarity_score) tuples.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def delete(self, chunk_ids: list[str]) -> None:
        """
        Delete chunks from the vector store.

        Args:
            chunk_ids: List of chunk IDs to delete.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def save(self, path: Optional[Path] = None) -> None:
        """
        Save the vector store to disk.

        Args:
            path: Path to save to. Uses default if not provided.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def load(self, path: Optional[Path] = None) -> None:
        """
        Load the vector store from disk.

        Args:
            path: Path to load from. Uses default if not provided.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_stats(self) -> dict[str, Any]:
        """
        Get vector store statistics.

        Returns:
            dict[str, Any]: Dictionary with store statistics.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def compact(self) -> int:
        """
        Rebuild the index excluding deleted chunks.

        Returns:
            int: Number of vectors remaining after compaction.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError


class FAISSStore(VectorStore):
    """
    FAISS-based vector store implementation.

    Uses FAISS for efficient similarity search with cosine similarity.
    Stores only chunk IDs in the index mapping, reconstructing chunks
    from metadata store on search to minimize memory usage.

    Supports multiple index types:
    - Flat: Brute-force search (default, good for < 10,000 vectors)
    - IVF: Inverted File Index (good for 10,000 - 1M vectors)
    - HNSW: Hierarchical Navigable Small World (good for large datasets)
    - IVFPQ: IVF with Product Quantization (good for very large datasets)

    Implements metadata filtering by pre-filtering chunks before vector search.

    Attributes:
        dimension: Dimension of embedding vectors.
        index: FAISS index instance.
        index_to_chunk_id: Mapping from FAISS index position to chunk ID.
        index_path: Path for persisting the index.
        metadata_store: Metadata store for chunk metadata.
        index_type: Type of FAISS index (flat, ivf, hnsw, ivfpq).
        nlist: Number of centroids for IVF.
        nprobe: Number of centroids to probe during search.
        is_trained: Whether the index has been trained.
    """

    def __init__(
        self,
        dimension: Optional[int] = None,
        index_path: Optional[Path] = None,
        metric: str = "cosine",
        metadata_store: Optional[MetadataStore] = None,
        index_type: Optional[str] = None,
        nlist: Optional[int] = None,
        nprobe: Optional[int] = None,
        hnsw_m: Optional[int] = None,
        hnsw_ef_construction: Optional[int] = None,
        hnsw_ef_search: Optional[int] = None,
        ivf_threshold: Optional[int] = None,
    ) -> None:
        """
        Initialize the FAISS vector store.

        Args:
            dimension: Vector dimension. Defaults to settings.
            index_path: Path for persistence. Defaults to settings.
            metric: Distance metric (cosine or inner_product).
            metadata_store: Metadata store instance. Creates new if None.
            index_type: Type of FAISS index (flat, ivf, hnsw, ivfpq).
            nlist: Number of centroids for IVF.
            nprobe: Number of centroids to probe during search.
            hnsw_m: HNSW number of neighbors.
            hnsw_ef_construction: HNSW efConstruction parameter.
            hnsw_ef_search: HNSW efSearch parameter.
            ivf_threshold: Minimum vectors to switch from flat to IVF.
        """
        self.dimension: int = dimension or settings.EMBEDDING_DIM
        self.index_path: Path = index_path or (settings.FAISS_INDEX_DIR / settings.FAISS_INDEX_NAME)
        self.metric: str = metric
        self.metadata_store: MetadataStore = metadata_store or MetadataStore()

        self.index_type: str = index_type or getattr(settings, "FAISS_INDEX_TYPE", FAISS_INDEX_DEFAULT)
        self.nlist: int = nlist or getattr(settings, "FAISS_NLIST", FAISS_NLIST_DEFAULT)
        self.nprobe: int = nprobe or getattr(settings, "FAISS_NPROBE", FAISS_NPROBE_DEFAULT)
        self.hnsw_m: int = hnsw_m or getattr(settings, "FAISS_HNSW_M", FAISS_HNSW_M_DEFAULT)
        self.hnsw_ef_construction: int = hnsw_ef_construction or getattr(
            settings, "FAISS_HNSW_EF_CONSTRUCTION", FAISS_HNSW_EF_CONSTRUCTION_DEFAULT
        )
        self.hnsw_ef_search: int = hnsw_ef_search or getattr(
            settings, "FAISS_HNSW_EF_SEARCH", FAISS_HNSW_EF_SEARCH_DEFAULT
        )
        self.ivf_threshold: int = ivf_threshold or getattr(settings, "FAISS_IVF_THRESHOLD", 10000)

        self.index: Optional[faiss.Index] = None
        self.index_to_chunk_id: dict[int, str] = {}
        self._next_id: int = 0
        self.is_trained: bool = False

        self._initialize_index()

    def _get_effective_index_type(self, current_size: int = 0) -> str:
        """
        Determine the appropriate index type based on current size.

        Args:
            current_size: Current number of vectors in the index.

        Returns:
            str: Effective index type to use.
        """
        if self.index_type != FAISS_INDEX_DEFAULT:
            return self.index_type

        if current_size < self.ivf_threshold:
            return FAISS_INDEX_FLAT
        else:
            return FAISS_INDEX_IVFFLAT

    def _initialize_index(self) -> None:
        """
        Initialize the FAISS index.

        Creates a new index or loads an existing one from disk.
        """
        if self._index_exists():
            self.load()
            return

        effective_type = self._get_effective_index_type(0)
        self._create_new_index(effective_type, 0)

    def _create_new_index(self, index_type: str, size: int) -> None:
        """
        Create a new FAISS index of the specified type.

        Args:
            index_type: Type of index to create (flat, ivf, hnsw, ivfpq).
            size: Current number of vectors (for auto-selection).
        """
        if self.metric == "cosine":
            metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            metric_type = faiss.METRIC_L2

        if index_type == FAISS_INDEX_FLAT:
            self.index = faiss.IndexFlatIP(self.dimension) if self.metric == "cosine" else faiss.IndexFlatL2(self.dimension)
            self.is_trained = True

        elif index_type == FAISS_INDEX_IVFFLAT:
            quantizer = faiss.IndexFlatIP(self.dimension) if self.metric == "cosine" else faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, metric_type)
            self.is_trained = False

        elif index_type == FAISS_INDEX_HNSW:
            self.index = faiss.IndexHNSWFlat(self.dimension, self.hnsw_m, metric_type)
            if hasattr(self.index.hnsw, "efConstruction"):
                self.index.hnsw.efConstruction = self.hnsw_ef_construction
            if hasattr(self.index.hnsw, "efSearch"):
                self.index.hnsw.efSearch = self.hnsw_ef_search
            self.is_trained = True

        elif index_type == FAISS_INDEX_IVFPQ:
            quantizer = faiss.IndexFlatIP(self.dimension) if self.metric == "cosine" else faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFPQ(quantizer, self.dimension, self.nlist, 8, 8, metric_type)
            self.is_trained = False

        else:
            self.index = faiss.IndexFlatIP(self.dimension) if self.metric == "cosine" else faiss.IndexFlatL2(self.dimension)
            self.is_trained = True

        logger.log_event(
            event=LogEvent.INDEX_UPDATE,
            message=f"Created new FAISS index: {index_type}, metric={self.metric}, dim={self.dimension}",
            details={
                "index_type": index_type,
                "metric": self.metric,
                "dimension": self.dimension,
                "is_trained": self.is_trained,
            },
        )

    def _index_exists(self) -> bool:
        """
        Check if a persisted index exists.

        Returns:
            bool: True if index files exist.
        """
        faiss_file = self.index_path.with_suffix(".faiss")
        metadata_file = self.index_path.with_suffix(".meta")
        return faiss_file.exists() and metadata_file.exists()

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize a vector for cosine similarity.

        Args:
            vector: Input vector.

        Returns:
            np.ndarray: L2-normalized vector.
        """
        if self.metric != "cosine":
            return vector

        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector

    def _vectors_to_numpy(self, vectors: list[list[float]]) -> np.ndarray:
        """
        Convert list of vectors to numpy array.

        Args:
            vectors: List of float vectors.

        Returns:
            np.ndarray: 2D numpy array of vectors.
        """
        arr = np.array(vectors, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if self.metric == "cosine":
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1
            arr = arr / norms

        return arr

    def _train_index_if_needed(self, vectors: np.ndarray) -> None:
        """
        Train the index if it requires training.

        Args:
            vectors: Training vectors.

        Raises:
            VectorStoreError: If training fails.
        """
        if self.index is None:
            raise VectorStoreError(
                message="FAISS index is not initialized",
                index_name=str(self.index_path),
                operation="train",
            )

        if not self.is_trained and hasattr(self.index, "train"):
            try:
                if vectors.shape[0] < self.nlist:
                    logger.log_event(
                        event=LogEvent.WARNING,
                        message=f"Not enough vectors for IVF training (need {self.nlist}, have {vectors.shape[0]}), using flat",
                        level=30,
                    )
                    self._create_new_index(FAISS_INDEX_FLAT, vectors.shape[0])
                    return

                logger.log_event(
                    event=LogEvent.INDEX_UPDATE,
                    message="Training FAISS index",
                    details={"vectors": vectors.shape[0], "nlist": self.nlist},
                )

                self.index.train(vectors)
                self.is_trained = True

                logger.log_event(
                    event=LogEvent.INDEX_UPDATE,
                    message="FAISS index training complete",
                )

            except Exception as exc:
                raise VectorStoreError(
                    message=f"Failed to train FAISS index: {str(exc)}",
                    index_name=str(self.index_path),
                    operation="train",
                    original_exception=exc,
                )

    def _reconstruct_chunk_from_metadata(self, chunk_id: str) -> Optional[Chunk]:
        """
        Reconstruct a Chunk object from metadata store.

        Args:
            chunk_id: The chunk ID to reconstruct.

        Returns:
            Optional[Chunk]: Reconstructed chunk, or None if not found.
        """
        metadata_dict = self.metadata_store.get_chunk_by_id(chunk_id)
        if metadata_dict is None:
            return None

        def parse_json_field(value: Any, default: Any = None) -> Any:
            if value is None:
                return default
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return default
            return value

        chunk_metadata = ChunkMetadata(
            document_id=metadata_dict["document_id"],
            document_name=metadata_dict["document_name"],
            page=metadata_dict.get("page"),
            chapter=metadata_dict.get("chapter"),
            section=metadata_dict.get("section"),
            subsection=metadata_dict.get("subsection"),
            title=metadata_dict.get("title"),
            chunk_id=metadata_dict["chunk_id"],
            chunk_index=metadata_dict["chunk_index"],
            total_chunks=metadata_dict["total_chunks"],
            language=metadata_dict.get("language"),
            ocr=bool(metadata_dict.get("ocr", 0)),
            source_type=metadata_dict.get("source_type", "paragraph"),
            table_id=metadata_dict.get("table_id"),
            figure_id=metadata_dict.get("figure_id"),
            token_count=metadata_dict.get("token_count", 0),
            embedding_model=metadata_dict.get("embedding_model"),
            parent_chunk_id=metadata_dict.get("parent_chunk_id"),
            child_chunk_ids=parse_json_field(metadata_dict.get("child_chunk_ids", "[]"), []),
            semantic_tags=parse_json_field(metadata_dict.get("semantic_tags", "[]"), []),
            keywords=parse_json_field(metadata_dict.get("keywords", "[]"), []),
            confidence_score=metadata_dict.get("confidence_score", 1.0),
        )

        return Chunk(
            content=metadata_dict.get("content_preview", ""),
            metadata=chunk_metadata,
            embedding_id=metadata_dict.get("faiss_index_id"),
        )

    def _set_nprobe_if_supported(self) -> None:
        """Set nprobe on the index if it supports it."""
        if self.index is not None and hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe

    def _apply_metadata_filters(
        self,
        chunk_ids: list[str],
        filter_dict: Optional[dict[str, Any]],
    ) -> list[str]:
        """
        Apply metadata filters to a list of chunk IDs.

        Args:
            chunk_ids: List of chunk IDs to filter.
            filter_dict: Dictionary of metadata filters.

        Returns:
            list[str]: Filtered list of chunk IDs.
        """
        if not filter_dict or not chunk_ids:
            return chunk_ids

        filtered_ids: list[str] = []

        for chunk_id in chunk_ids:
            metadata = self.metadata_store.get_chunk_by_id(chunk_id)
            if metadata is None:
                continue

            include = True
            for key, value in filter_dict.items():
                if key == "document_name":
                    # Case-insensitive substring match
                    doc_name = metadata.get("document_name", "").lower()
                    if value.lower() not in doc_name:
                        include = False
                        break
                elif key == "document_id":
                    if metadata.get("document_id") != value:
                        include = False
                        break
                elif key == "section":
                    section = metadata.get("section", "").lower()
                    if value.lower() not in section:
                        include = False
                        break
                elif key == "chapter":
                    chapter = metadata.get("chapter", "").lower()
                    if value.lower() not in chapter:
                        include = False
                        break
                elif key == "source_type":
                    if metadata.get("source_type") != value:
                        include = False
                        break
                elif key == "language":
                    if metadata.get("language") != value:
                        include = False
                        break
                elif key == "page":
                    if metadata.get("page") != value:
                        include = False
                        break
                elif key == "tags":
                    # Check if any tag matches
                    tags = metadata.get("semantic_tags", "[]")
                    if isinstance(tags, str):
                        try:
                            tags = json.loads(tags)
                        except json.JSONDecodeError:
                            tags = []
                    if not isinstance(tags, list):
                        tags = []
                    if value not in tags:
                        include = False
                        break
                elif key.startswith("metadata."):
                    # Custom metadata field
                    field = key.replace("metadata.", "")
                    custom_meta = metadata.get("custom_metadata", "{}")
                    if isinstance(custom_meta, str):
                        try:
                            custom_meta = json.loads(custom_meta)
                        except json.JSONDecodeError:
                            custom_meta = {}
                    if custom_meta.get(field) != value:
                        include = False
                        break
                else:
                    # Default: match in any metadata field (string contains)
                    found = False
                    for meta_key, meta_value in metadata.items():
                        if meta_value is not None and isinstance(meta_value, str):
                            if value.lower() in meta_value.lower():
                                found = True
                                break
                    if not found:
                        include = False
                        break

            if include:
                filtered_ids.append(chunk_id)

        return filtered_ids

    def add(self, chunks: list[Chunk]) -> None:
        """
        Add chunks with embeddings to the FAISS index.

        Stores only chunk IDs in the index mapping, not the full chunk objects.

        Args:
            chunks: List of chunks with embeddings.

        Raises:
            VectorStoreError: If adding fails.
        """
        with measure_latency("vector_index_add") as latency:
            if not chunks:
                return

            valid_chunks = [c for c in chunks if c.embedding is not None]

            if not valid_chunks:
                logger.log_event(
                    event=LogEvent.WARNING,
                    message="No chunks with embeddings to add",
                    level=30,
                )
                return

            vectors = [c.embedding for c in valid_chunks]
            vectors_np = self._vectors_to_numpy(vectors)

            try:
                if self.index is None:
                    raise VectorStoreError(
                        message="FAISS index is not initialized",
                        index_name=str(self.index_path),
                        operation="add",
                    )

                current_size = self.index.ntotal
                new_size = current_size + len(valid_chunks)
                effective_type = self._get_effective_index_type(new_size)

                current_type = self._get_current_index_type()
                if effective_type != current_type and new_size > 0:
                    logger.log_event(
                        event=LogEvent.INDEX_UPDATE,
                        message=f"Switching index type from {current_type} to {effective_type}",
                        details={
                            "from": current_type,
                            "to": effective_type,
                            "size": new_size,
                        },
                    )
                    self._rebuild_with_type(effective_type)

                self._train_index_if_needed(vectors_np)

                start_id = self._next_id
                self.index.add(vectors_np)
                end_id = self._next_id + len(valid_chunks)

                for i, chunk in enumerate(valid_chunks):
                    idx = start_id + i
                    self.index_to_chunk_id[idx] = chunk.metadata.chunk_id
                    chunk.embedding_id = idx

                self._next_id = end_id

                self._set_nprobe_if_supported()

                latency.stop(
                    chunk_count=len(valid_chunks),
                    index_size=self.index.ntotal,
                    index_type=self._get_current_index_type(),
                )

                logger.log_event(
                    event=LogEvent.INDEX_UPDATE,
                    message=f"Added {len(valid_chunks)} chunks to FAISS index",
                    details={
                        "added_count": len(valid_chunks),
                        "total_index_size": self.index.ntotal,
                        "dimension": self.dimension,
                        "index_type": self._get_current_index_type(),
                    },
                )

            except VectorStoreError:
                raise
            except Exception as exc:
                raise VectorStoreError(
                    message=f"Failed to add chunks to FAISS index: {str(exc)}",
                    index_name=str(self.index_path),
                    operation="add",
                    original_exception=exc,
                )

    def _get_current_index_type(self) -> str:
        """
        Get the current index type as a string.

        Returns:
            str: Index type identifier.
        """
        if self.index is None:
            return "none"

        if isinstance(self.index, faiss.IndexFlatIP) or isinstance(self.index, faiss.IndexFlatL2):
            return FAISS_INDEX_FLAT
        elif isinstance(self.index, faiss.IndexIVFFlat):
            return FAISS_INDEX_IVFFLAT
        elif isinstance(self.index, faiss.IndexHNSWFlat):
            return FAISS_INDEX_HNSW
        elif isinstance(self.index, faiss.IndexIVFPQ):
            return FAISS_INDEX_IVFPQ

        return "unknown"

    def _rebuild_with_type(self, new_type: str) -> None:
        """
        Rebuild the index with a new type using existing data.

        Args:
            new_type: The new index type.

        Raises:
            VectorStoreError: If rebuild fails.
        """
        if self.index is None or self.index.ntotal == 0:
            self._create_new_index(new_type, 0)
            return

        vectors = []
        chunk_ids = []

        for idx, chunk_id in list(self.index_to_chunk_id.items()):
            chunk = self._reconstruct_chunk_from_metadata(chunk_id)
            if chunk and chunk.embedding is not None:
                vectors.append(chunk.embedding)
                chunk_ids.append(chunk_id)

        if not vectors:
            logger.log_event(
                event=LogEvent.WARNING,
                message="No embeddings found for rebuild",
                level=30,
            )
            return

        self._create_new_index(new_type, len(vectors))

        vectors_np = self._vectors_to_numpy(vectors)
        self._train_index_if_needed(vectors_np)

        start_id = 0
        self.index.add(vectors_np)

        self.index_to_chunk_id = {start_id + i: cid for i, cid in enumerate(chunk_ids)}
        self._next_id = len(chunk_ids)

        self._set_nprobe_if_supported()

        logger.log_event(
            event=LogEvent.INDEX_UPDATE,
            message=f"Index rebuilt with type {new_type}",
            details={
                "type": new_type,
                "vectors": len(vectors),
            },
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 50,
        filter_dict: Optional[dict[str, Any]] = None,
    ) -> list[tuple[Chunk, float]]:
        """
        Search for chunks similar to the query embedding.

        Reconstructs chunks from metadata store on demand.
        Applies metadata filters before vector search when filters are provided.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.
            filter_dict: Optional metadata filters.

        Returns:
            list[tuple[Chunk, float]]: List of (chunk, similarity_score) tuples.

        Raises:
            VectorStoreError: If search fails.
        """
        with measure_latency("dense_retrieval") as latency:
            if self.index is None or self.index.ntotal == 0:
                return []

            try:
                self._set_nprobe_if_supported()

                query_np = self._vectors_to_numpy([query_embedding])

                # If filters are provided, we need to search more aggressively
                effective_top_k = top_k
                if filter_dict:
                    # Search for more candidates first, then filter
                    effective_top_k = min(top_k * 2, self.index.ntotal)

                distances, indices = self.index.search(query_np, min(effective_top_k, self.index.ntotal))

                # Collect results
                raw_results: list[tuple[str, float]] = []

                for i in range(len(indices[0])):
                    idx = int(indices[0][i])
                    distance = float(distances[0][i])

                    if idx < 0 or idx >= self._next_id:
                        continue

                    chunk_id = self.index_to_chunk_id.get(idx)
                    if chunk_id is None:
                        continue

                    raw_results.append((chunk_id, distance))

                # Apply metadata filters if provided
                if filter_dict:
                    raw_results = self._filter_results(raw_results, filter_dict)

                # Reconstruct chunks
                results: list[tuple[Chunk, float]] = []

                for chunk_id, distance in raw_results[:top_k]:
                    chunk = self._reconstruct_chunk_from_metadata(chunk_id)
                    if chunk is None:
                        continue

                    if self.metric == "cosine":
                        score = distance
                    else:
                        score = 1.0 / (1.0 + distance)

                    chunk.dense_score = score
                    results.append((chunk, score))

                latency.stop(
                    query_dimension=len(query_embedding),
                    top_k_requested=top_k,
                    results_returned=len(results),
                    index_size=self.index.ntotal,
                    index_type=self._get_current_index_type(),
                    filtered=filter_dict is not None,
                )

                logger.log_event(
                    event=LogEvent.DENSE_RETRIEVAL_COMPLETE,
                    message=f"Dense retrieval returned {len(results)} results",
                    details={
                        "top_k": top_k,
                        "results": len(results),
                        "index_size": self.index.ntotal,
                        "index_type": self._get_current_index_type(),
                        "filters_applied": filter_dict is not None,
                    },
                )

                return results

            except Exception as exc:
                raise VectorStoreError(
                    message=f"FAISS search failed: {str(exc)}",
                    index_name=str(self.index_path),
                    operation="search",
                    original_exception=exc,
                )

    def _filter_results(
        self,
        results: list[tuple[str, float]],
        filter_dict: dict[str, Any],
    ) -> list[tuple[str, float]]:
        """
        Filter search results by metadata.

        Args:
            results: List of (chunk_id, distance) tuples.
            filter_dict: Metadata filter dictionary.

        Returns:
            list[tuple[str, float]]: Filtered results.
        """
        if not filter_dict:
            return results

        filtered: list[tuple[str, float]] = []

        for chunk_id, distance in results:
            metadata = self.metadata_store.get_chunk_by_id(chunk_id)
            if metadata is None:
                continue

            include = True
            for key, value in filter_dict.items():
                if key == "document_name":
                    if metadata.get("document_name", "").lower() != value.lower():
                        include = False
                        break
                elif key == "document_id":
                    if metadata.get("document_id") != value:
                        include = False
                        break
                elif key == "section":
                    if metadata.get("section", "").lower() != value.lower():
                        include = False
                        break
                elif key == "chapter":
                    if metadata.get("chapter", "").lower() != value.lower():
                        include = False
                        break
                elif key == "source_type":
                    if metadata.get("source_type") != value:
                        include = False
                        break
                elif key == "language":
                    if metadata.get("language") != value:
                        include = False
                        break
                elif key == "page":
                    if metadata.get("page") != value:
                        include = False
                        break
                elif key == "tags":
                    tags = metadata.get("semantic_tags", "[]")
                    if isinstance(tags, str):
                        try:
                            tags = json.loads(tags)
                        except json.JSONDecodeError:
                            tags = []
                    if value not in tags:
                        include = False
                        break
                elif key == "document_name_contains":
                    if value.lower() not in metadata.get("document_name", "").lower():
                        include = False
                        break
                elif key == "section_contains":
                    if value.lower() not in metadata.get("section", "").lower():
                        include = False
                        break
                elif key == "chapter_contains":
                    if value.lower() not in metadata.get("chapter", "").lower():
                        include = False
                        break
                elif key == "title_contains":
                    if value.lower() not in metadata.get("title", "").lower():
                        include = False
                        break
                else:
                    # Custom filter: check all string fields
                    found = False
                    for meta_key, meta_value in metadata.items():
                        if meta_value is not None and isinstance(meta_value, str):
                            if value.lower() in meta_value.lower():
                                found = True
                                break
                    if not found:
                        include = False
                        break

            if include:
                filtered.append((chunk_id, distance))

        return filtered

    def delete(self, chunk_ids: list[str]) -> None:
        """
        Delete chunks from the vector store.

        Note: FAISS indexes do not support deletion of individual vectors.
        This marks chunks as deleted in metadata but does not remove vectors.
        For true deletion, use the compact() method to rebuild the index.

        Args:
            chunk_ids: List of chunk IDs to delete.

        Raises:
            VectorStoreError: If deletion fails.
        """
        deleted_count = 0

        for chunk_id in chunk_ids:
            metadata_dict = self.metadata_store.get_chunk_by_id(chunk_id)
            if metadata_dict:
                self.metadata_store.delete_chunk(chunk_id)
                deleted_count += 1

            for idx, cid in list(self.index_to_chunk_id.items()):
                if cid == chunk_id:
                    del self.index_to_chunk_id[idx]

        logger.log_event(
            event=LogEvent.INDEX_UPDATE,
            message=f"Marked {deleted_count} chunks as deleted (run compact() to rebuild index)",
            details={
                "deleted_count": deleted_count,
                "chunk_ids": chunk_ids,
                "note": "FAISS does not support true deletion. Use compact() to rebuild the index.",
            },
        )

    def save(self, path: Optional[Path] = None) -> None:
        """
        Save the FAISS index and metadata to disk.

        Args:
            path: Path to save to. Uses default if not provided.

        Raises:
            VectorStoreError: If save fails.
        """
        save_path = path or self.index_path

        with measure_latency("vector_index_save") as latency:
            try:
                save_path.parent.mkdir(parents=True, exist_ok=True)

                faiss_file = save_path.with_suffix(".faiss")
                if self.index is not None:
                    faiss.write_index(self.index, str(faiss_file))

                metadata = {
                    "index_to_chunk_id": self.index_to_chunk_id,
                    "next_id": self._next_id,
                    "dimension": self.dimension,
                    "metric": self.metric,
                    "index_type": self._get_current_index_type(),
                    "nlist": self.nlist,
                    "nprobe": self.nprobe,
                    "is_trained": self.is_trained,
                }

                metadata_file = save_path.with_suffix(".meta")
                with open(metadata_file, "wb") as f:
                    pickle.dump(metadata, f)

                latency.stop(
                    index_size=self.index.ntotal if self.index else 0,
                    file_path=str(save_path),
                    index_type=self._get_current_index_type(),
                )

                logger.log_event(
                    event=LogEvent.INDEX_UPDATE,
                    message=f"FAISS index saved to {save_path}",
                    details={
                        "index_size": self.index.ntotal if self.index else 0,
                        "path": str(save_path),
                        "index_type": self._get_current_index_type(),
                    },
                )

            except Exception as exc:
                raise VectorStoreError(
                    message=f"Failed to save FAISS index: {str(exc)}",
                    index_name=str(save_path),
                    operation="save",
                    original_exception=exc,
                )

    def load(self, path: Optional[Path] = None) -> None:
        """
        Load the FAISS index and metadata from disk.

        Args:
            path: Path to load from. Uses default if not provided.

        Raises:
            VectorStoreError: If load fails.
        """
        load_path = path or self.index_path

        with measure_latency("vector_index_load") as latency:
            try:
                faiss_file = load_path.with_suffix(".faiss")
                metadata_file = load_path.with_suffix(".meta")

                if not faiss_file.exists() or not metadata_file.exists():
                    raise VectorStoreError(
                        message=f"Index files not found at {load_path}",
                        index_name=str(load_path),
                        operation="load",
                    )

                self.index = faiss.read_index(str(faiss_file))

                with open(metadata_file, "rb") as f:
                    metadata = pickle.load(f)

                self.index_to_chunk_id = metadata.get("index_to_chunk_id", {})
                self._next_id = metadata.get("next_id", len(self.index_to_chunk_id))
                self.dimension = metadata.get("dimension", self.dimension)
                self.metric = metadata.get("metric", self.metric)
                self.nlist = metadata.get("nlist", self.nlist)
                self.nprobe = metadata.get("nprobe", self.nprobe)
                self.is_trained = metadata.get("is_trained", True)

                self._set_nprobe_if_supported()

                latency.stop(
                    index_size=self.index.ntotal,
                    file_path=str(load_path),
                    index_type=self._get_current_index_type(),
                )

                logger.log_event(
                    event=LogEvent.INDEX_UPDATE,
                    message=f"FAISS index loaded from {load_path}",
                    details={
                        "index_size": self.index.ntotal,
                        "mapped_chunks": len(self.index_to_chunk_id),
                        "path": str(load_path),
                        "index_type": self._get_current_index_type(),
                    },
                )

            except VectorStoreError:
                raise
            except Exception as exc:
                raise VectorStoreError(
                    message=f"Failed to load FAISS index: {str(exc)}",
                    index_name=str(load_path),
                    operation="load",
                    original_exception=exc,
                )

    def compact(self) -> int:
        """
        Rebuild the FAISS index excluding deleted chunks.

        This method removes deleted chunks from the index by rebuilding
        from metadata store. It should be called periodically to free memory.

        Returns:
            int: Number of vectors remaining after compaction.

        Raises:
            VectorStoreError: If compaction fails.
        """
        with measure_latency("vector_index_compact") as latency:
            try:
                active_chunks = self.metadata_store.get_all_active_chunks()

                if not active_chunks:
                    self._create_new_index(FAISS_INDEX_FLAT, 0)
                    self.index_to_chunk_id = {}
                    self._next_id = 0
                    self.save()
                    logger.log_event(
                        event=LogEvent.INDEX_UPDATE,
                        message="FAISS index compacted: no active chunks remaining",
                    )
                    return 0

                vectors = []
                chunk_ids = []

                for chunk_meta in active_chunks:
                    chunk = self._reconstruct_chunk_from_metadata(chunk_meta["chunk_id"])
                    if chunk and chunk.embedding is not None:
                        vectors.append(chunk.embedding)
                        chunk_ids.append(chunk_meta["chunk_id"])

                if not vectors:
                    logger.log_event(
                        event=LogEvent.WARNING,
                        message="No embeddings found for active chunks during compaction",
                        level=30,
                    )
                    return 0

                effective_type = self._get_effective_index_type(len(vectors))
                self._create_new_index(effective_type, len(vectors))

                vectors_np = self._vectors_to_numpy(vectors)
                self._train_index_if_needed(vectors_np)

                self.index.add(vectors_np)

                self.index_to_chunk_id = {i: cid for i, cid in enumerate(chunk_ids)}
                self._next_id = len(chunk_ids)

                self._set_nprobe_if_supported()

                self.save()

                latency.stop(
                    old_size=len(self.index_to_chunk_id) + len(vectors),
                    new_size=len(chunk_ids),
                    index_type=self._get_current_index_type(),
                )

                logger.log_event(
                    event=LogEvent.INDEX_UPDATE,
                    message=f"FAISS index compacted: {len(chunk_ids)} chunks remaining",
                    details={
                        "remaining_chunks": len(chunk_ids),
                        "duration_ms": latency.duration_ms,
                        "index_type": self._get_current_index_type(),
                    },
                )

                return len(chunk_ids)

            except Exception as exc:
                raise VectorStoreError(
                    message=f"FAISS compaction failed: {str(exc)}",
                    index_name=str(self.index_path),
                    operation="compact",
                    original_exception=exc,
                )

    def get_stats(self) -> dict[str, Any]:
        """
        Get FAISS index statistics.

        Returns:
            dict[str, Any]: Dictionary with index statistics.
        """
        return {
            "index_type": self._get_current_index_type(),
            "metric": self.metric,
            "dimension": self.dimension,
            "total_vectors": self.index.ntotal if self.index else 0,
            "mapped_chunks": len(self.index_to_chunk_id),
            "index_path": str(self.index_path),
            "is_trained": self.is_trained,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "ivf_threshold": self.ivf_threshold,
        }

    def rebuild_index(self, chunks: Optional[list[Chunk]] = None) -> None:
        """
        Rebuild the FAISS index from scratch.

        Args:
            chunks: Optional list of chunks to rebuild with. Uses current chunks if None.
        """
        with measure_latency("vector_index_rebuild") as latency:
            self.index_to_chunk_id = {}
            self._next_id = 0

            if chunks:
                effective_type = self._get_effective_index_type(len(chunks))
            else:
                effective_type = self._get_effective_index_type(0)

            self._create_new_index(effective_type, len(chunks) if chunks else 0)

            if chunks:
                self.add(chunks)
            else:
                self.compact()

            self.save()

            latency.stop(
                new_size=self.index.ntotal if self.index else 0,
                index_type=self._get_current_index_type(),
            )

            logger.log_event(
                event=LogEvent.INDEX_UPDATE,
                message=f"FAISS index rebuilt with {self.index.ntotal if self.index else 0} vectors",
                details={
                    "new_size": self.index.ntotal if self.index else 0,
                    "index_type": self._get_current_index_type(),
                },
            )