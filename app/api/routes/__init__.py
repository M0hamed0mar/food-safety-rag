"""
API routes.

Each module exposes an `APIRouter` that is mounted in app.api.main:
    - health:  /health, /
    - ingest:  /ingest
    - chat:    /chat/*
    - ask:     /ask, /ask/stream, /retrieve
    - stats:   /stats
"""

from app.api.routes.health import router as health_router
from app.api.routes.ingest import router as ingest_router
from app.api.routes.chat import router as chat_router
from app.api.routes.ask import router as ask_router
from app.api.routes.stats import router as stats_router

__all__ = [
    "health_router",
    "ingest_router",
    "chat_router",
    "ask_router",
    "stats_router",
]
