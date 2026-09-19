"""Chat management routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query as FastAPIQuery

from app.api.dependencies import get_pipeline_dep


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/new")
async def create_chat(title: str = "New Chat") -> dict[str, Any]:
    pipeline = get_pipeline_dep()
    try:
        session_id = pipeline.create_chat(title)
        return {"session_id": session_id, "title": title}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create chat: {exc}")


@router.get("/list")
async def list_chats() -> list[dict[str, Any]]:
    pipeline = get_pipeline_dep()
    try:
        return pipeline.list_chats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list chats: {exc}")


@router.get("/{session_id}/history")
async def get_chat_history(
    session_id: str,
    limit: int = FastAPIQuery(default=10, ge=1, le=50),
) -> list[dict[str, str]]:
    pipeline = get_pipeline_dep()
    try:
        return pipeline.get_chat_history(session_id, limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to get chat history: {exc}")


@router.get("/{session_id}")
async def get_chat_session(session_id: str) -> dict[str, Any]:
    pipeline = get_pipeline_dep()
    try:
        session = pipeline.get_chat_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Chat session '{session_id}' not found")
        return session.model_dump() if hasattr(session, "model_dump") else session
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to get chat session: {exc}")


@router.put("/{session_id}/title")
async def update_chat_title(session_id: str, title: str) -> dict[str, Any]:
    pipeline = get_pipeline_dep()
    try:
        updated = pipeline.update_chat_title(session_id, title)
        if not updated:
            raise HTTPException(status_code=404, detail=f"Chat session '{session_id}' not found")
        return {"session_id": session_id, "title": title, "updated": True}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to update chat title: {exc}")


@router.delete("/{session_id}")
async def delete_chat(session_id: str) -> dict[str, Any]:
    pipeline = get_pipeline_dep()
    try:
        deleted = pipeline.delete_chat(session_id)
        return {"session_id": session_id, "deleted": deleted}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete chat: {exc}")
