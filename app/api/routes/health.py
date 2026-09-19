"""Health + root routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from app.api.dependencies import get_pipeline_dep


router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Liveness + component status."""
    try:
        pipeline = get_pipeline_dep()
        stats = pipeline.get_stats()
        return {
            "status": "healthy",
            "initialized": stats.get("initialized", False),
            "components": {
                "ingestion": "available" if "ingestion" in stats else "unavailable",
                "retrieval": "available" if "retrieval" in stats else "unavailable",
                "generation": "available" if "generation" in stats else "unavailable",
                "chats": "available" if "chats" in stats else "unavailable",
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {exc}")


@router.get("/api")
async def api_root() -> dict[str, str]:
    """API metadata."""
    return {
        "name": "RAG System",
        "version": "0.2.0",
        "status": "operational",
    }


@router.get("/")
async def root() -> Any:
    """Serve the UI HTML (if present) or fall back to JSON."""
    # Try UI template
    ui_path = Path(__file__).resolve().parents[2] / "web" / "templates" / "index.html"
    if ui_path.exists():
        with open(ui_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return {
        "name": "RAG System",
        "version": "0.2.0",
        "status": "operational",
    }
