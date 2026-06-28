"""
Metadata store for chunk information.

Manages structured metadata separate from embeddings.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import json

from food_safety_rag.core.exceptions import VectorStoreException
from food_safety_rag.core.monitoring.logger import get_logger

logger = get_logger(__name__)


class MetadataStore:
    """
    Store for chunk metadata.
    
    Maintains mapping from chunk IDs to rich metadata for filtering,
    citation, and evaluation.
    """

    def __init__(self, store_dir: Path) -> None:
        """
        Initialize metadata store.
        
        Args:
            store_dir: Directory for metadata storage.
        """
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """
        Load metadata from disk.
        """
        try:
            metadata_file = self.store_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    self.metadata = json.load(f)
                logger.info(
                    "Metadata loaded",
                    count=len(self.metadata),
                )
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            self.metadata = {}

    def store_metadata(
        self, chunk_id: str, metadata: Dict[str, Any]
    ) -> None:
        """
        Store metadata for a chunk.
        
        Args:
            chunk_id: The chunk ID.
            metadata: Metadata dictionary.
        """
        self.metadata[chunk_id] = metadata

    def store_batch(
        self, chunk_metadata: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Store metadata for multiple chunks.
        
        Args:
            chunk_metadata: Dictionary of chunk_id -> metadata.
        """
        self.metadata.update(chunk_metadata)

    def get_metadata(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata for a chunk.
        
        Args:
            chunk_id: The chunk ID.
            
        Returns:
            Optional[Dict[str, Any]]: Metadata or None.
        """
        return self.metadata.get(chunk_id)

    def filter_by_document(
        self, document_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get all metadata for chunks from a specific document.
        
        Args:
            document_id: The document ID.
            
        Returns:
            Dict[str, Dict[str, Any]]: Filtered metadata.
        """
        return {
            chunk_id: meta
            for chunk_id, meta in self.metadata.items()
            if meta.get("document_id") == document_id
        }

    def filter_by_language(
        self, language: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get all metadata for chunks in a specific language.
        
        Args:
            language: The language code (e.g., 'en').
            
        Returns:
            Dict[str, Dict[str, Any]]: Filtered metadata.
        """
        return {
            chunk_id: meta
            for chunk_id, meta in self.metadata.items()
            if meta.get("language") == language
        }

    def delete_metadata(self, chunk_id: str) -> None:
        """
        Delete metadata for a chunk.
        
        Args:
            chunk_id: The chunk ID.
        """
        if chunk_id in self.metadata:
            del self.metadata[chunk_id]

    def delete_document_metadata(self, document_id: str) -> None:
        """
        Delete all metadata for chunks from a document.
        
        Args:
            document_id: The document ID.
        """
        chunk_ids = [
            chunk_id
            for chunk_id, meta in self.metadata.items()
            if meta.get("document_id") == document_id
        ]
        for chunk_id in chunk_ids:
            del self.metadata[chunk_id]
        logger.info(
            "Deleted document metadata",
            document_id=document_id,
            chunks_deleted=len(chunk_ids),
        )

    def save(self) -> None:
        """
        Persist metadata to disk.
        
        Raises:
            VectorStoreException: If save fails.
        """
        try:
            metadata_file = self.store_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2, default=str)
            logger.info(
                "Metadata saved",
                count=len(self.metadata),
                path=str(metadata_file),
            )
        except Exception as e:
            raise VectorStoreException(f"Failed to save metadata: {e}") from e

    def get_stats(self) -> Dict[str, Any]:
        """
        Get metadata store statistics.
        
        Returns:
            Dict[str, Any]: Store statistics.
        """
        documents = set()
        languages = set()
        for meta in self.metadata.values():
            documents.add(meta.get("document_id"))
            languages.add(meta.get("language"))

        return {
            "total_chunks": len(self.metadata),
            "unique_documents": len(documents),
            "languages": list(languages),
        }
