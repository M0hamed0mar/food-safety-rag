"""Document ingestion route."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.dependencies import get_pipeline_dep
from app.config.constants import SUPPORTED_EXTENSIONS


router = APIRouter(tags=["ingestion"])


@router.post("/ingest", response_model=dict[str, Any])
async def ingest_document(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload + ingest a document into the RAG system."""
    pipeline = get_pipeline_dep()

    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {file_ext}. "
                f"Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            result = pipeline.ingest_document(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        return {
            "status": result.status,
            "document_id": result.document_id,
            "document_name": result.document_name,
            "chunks_created": result.chunk_count,
            "embeddings_generated": result.embedding_count,
            "ocr_pages": result.ocr_pages,
            "duration_ms": result.get_total_duration_ms(),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
