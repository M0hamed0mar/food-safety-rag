"""
Utility functions for the Food Safety RAG System.

Common helper functions used across modules.
"""

import hashlib
import uuid
from pathlib import Path
from typing import Any, List


def generate_document_id() -> str:
    """
    Generate a unique document identifier.
    
    Returns:
        str: UUID-based document ID.
    """
    return f"doc_{uuid.uuid4().hex[:12]}"


def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """
    Generate a unique chunk identifier.
    
    Args:
        document_id: The document the chunk belongs to.
        chunk_index: Sequential index of chunk.
        
    Returns:
        str: Formatted chunk ID.
    """
    return f"{document_id}_chunk_{chunk_index:04d}"


def generate_query_id() -> str:
    """
    Generate a unique query identifier.
    
    Returns:
        str: UUID-based query ID.
    """
    return f"query_{uuid.uuid4().hex[:12]}"


def generate_answer_id() -> str:
    """
    Generate a unique answer identifier.
    
    Returns:
        str: UUID-based answer ID.
    """
    return f"answer_{uuid.uuid4().hex[:12]}"


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file.
    
    Used for duplicate detection.
    
    Args:
        file_path: Path to file.
        
    Returns:
        str: Hex-encoded hash.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compute_content_hash(content: str) -> str:
    """
    Compute SHA256 hash of text content.
    
    Args:
        content: Text content.
        
    Returns:
        str: Hex-encoded hash.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    Removes extra spaces, tabs, and normalizes newlines.
    
    Args:
        text: Input text.
        
    Returns:
        str: Normalized text.
    """
    # Replace multiple spaces with single space
    text = " ".join(text.split())
    # Normalize newlines
    text = text.replace("\r\n", "\n")
    return text.strip()


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size.
    
    Args:
        items: List to chunk.
        chunk_size: Size of each chunk.
        
    Returns:
        List[List[Any]]: List of chunks.
    """
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    
    Uses approximation of ~1 token per 4 characters.
    This is a rough estimate; actual token count depends on tokenizer.
    
    Args:
        text: Input text.
        
    Returns:
        int: Estimated token count.
    """
    return len(text) // 4


def truncate_text(text: str, max_length: int) -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text.
        max_length: Maximum length.
        
    Returns:
        str: Truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."
