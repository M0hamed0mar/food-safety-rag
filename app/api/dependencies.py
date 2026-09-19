"""
Shared FastAPI dependencies.

Provides lazy access to the global RAGPipeline singleton.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from app.pipeline import RAGPipeline, get_pipeline


def get_pipeline_dep() -> RAGPipeline:
    """Return the global RAGPipeline (initialized lazily)."""
    try:
        pipeline = get_pipeline()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline initialization failed: {exc}",
        ) from exc
    return pipeline


__all__ = ["get_pipeline_dep"]
