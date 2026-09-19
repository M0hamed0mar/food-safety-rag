"""Q&A + retrieval routes."""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException, Query as FastAPIQuery
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_pipeline_dep


router = APIRouter(tags=["qa"])


@router.post("/ask", response_model=dict[str, Any])
async def ask_question(
    query_text: str = FastAPIQuery(..., min_length=2, max_length=10000),
    session_id: Optional[str] = FastAPIQuery(default=None),
) -> dict[str, Any]:
    """Non-streaming question answering."""
    pipeline = get_pipeline_dep()
    try:
        if not session_id:
            session_id = pipeline.create_chat()

        answer = pipeline.ask(query_text, session_id=session_id)

        citations: list[dict[str, Any]] = []
        has_valid = False
        seen: set[str] = set()

        if answer.is_supported and answer.citations:
            for c in answer.citations:
                doc_name = c.document_name or "Unknown Document"
                page = c.page
                key = f"{doc_name}_{page}" if page else doc_name
                if key in seen:
                    continue
                seen.add(key)
                entry = {
                    "document_name": doc_name,
                    "page": page,
                    "section": c.section,
                    "chunk_id": c.chunk_id,
                    "quoted_text": c.quoted_text,
                }
                entry = {k: v for k, v in entry.items() if v is not None}
                citations.append(entry)
                has_valid = True

        if not has_valid and answer.text.strip():
            citations = [{"document_name": "Knowledge Base"}]

        return {
            "answer_id": answer.answer_id,
            "query_id": answer.query_id,
            "session_id": session_id,
            "text": answer.text,
            "citations": citations,
            "is_supported": answer.is_supported,
            "confidence": answer.confidence,
            "metadata": {
                "model": answer.metadata.model_name,
                "tokens_generated": answer.metadata.total_tokens_generated,
                "tokens_prompt": answer.metadata.total_tokens_prompt,
                "generation_duration_ms": answer.metadata.generation_duration_ms,
                "retrieval_duration_ms": answer.metadata.retrieval_duration_ms,
                "total_duration_ms": answer.metadata.total_duration_ms,
                "num_citations": len([c for c in citations if c.get("document_name") != "Knowledge Base"]),
                "from_knowledge_base": not has_valid and bool(answer.text.strip()),
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {exc}")


@router.get("/ask/stream")
async def ask_question_streaming(
    query_text: str = FastAPIQuery(..., min_length=2, max_length=10000),
    session_id: Optional[str] = FastAPIQuery(default=None),
) -> StreamingResponse:
    """Streaming (SSE) question answering."""
    pipeline = get_pipeline_dep()

    async def event_generator() -> AsyncGenerator[str, None]:
        current_session_id = session_id
        if not current_session_id:
            current_session_id = pipeline.create_chat()

        try:
            yield f"data: {json.dumps({'session_id': current_session_id})}\n\n"

            async for chunk in pipeline.ask_streaming(query_text, session_id=current_session_id):
                if chunk.is_final:
                    citations: list[dict[str, Any]] = []
                    has_valid = False
                    seen: set[str] = set()

                    for c in chunk.citations:
                        doc_name = c.document_name or "Unknown Document"
                        page = c.page
                        key = f"{doc_name}_{page}" if page else doc_name
                        if key in seen:
                            continue
                        seen.add(key)
                        entry = {
                            "document_name": doc_name,
                            "page": page,
                            "section": c.section,
                            "chunk_id": c.chunk_id,
                        }
                        entry = {k: v for k, v in entry.items() if v is not None}
                        citations.append(entry)
                        has_valid = True

                    if not has_valid:
                        citations = [{"document_name": "Knowledge Base"}]

                    data = {
                        "done": True,
                        "citations": citations,
                        "from_knowledge_base": not has_valid,
                        "session_id": current_session_id,
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {chunk.token}\n\n"

        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/retrieve")
async def retrieve_chunks(
    query_text: str = FastAPIQuery(..., min_length=2, max_length=10000),
) -> dict[str, Any]:
    """Retrieve chunks without generating an answer."""
    pipeline = get_pipeline_dep()
    try:
        chunks = pipeline.retrieve(query_text)
        return {
            "query": query_text,
            "chunks": [
                {
                    "chunk_id": c.metadata.chunk_id,
                    "document_name": c.metadata.document_name,
                    "page": c.metadata.page,
                    "section": c.metadata.section,
                    "content": c.content[:500],
                    "score": c.reranker_score,
                }
                for c in chunks
            ],
            "total_chunks": len(chunks),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}")
