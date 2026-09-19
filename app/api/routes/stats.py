"""System statistics route."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.dependencies import get_pipeline_dep


router = APIRouter(tags=["stats"])


@router.get("/stats")
async def get_statistics() -> dict[str, Any]:
    pipeline = get_pipeline_dep()
    try:
        return pipeline.get_stats()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {exc}")
