"""
FastAPI application factory.

Mounts all routers and static files, and configures middleware.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import (
    ask_router,
    chat_router,
    health_router,
    ingest_router,
    stats_router,
)
from app.config import APP_NAME, APP_VERSION
from app.config.constants import LogEvent
from app.monitoring import get_logger, setup_logging


logger = get_logger("app.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup + shutdown hooks."""
    setup_logging()
    logger.log_event(
        event=LogEvent.SYSTEM_STARTUP,
        message=f"Starting {APP_NAME} API v{APP_VERSION}",
    )
    yield
    logger.log_event(
        event=LogEvent.SYSTEM_SHUTDOWN,
        message=f"Shutting down {APP_NAME} API",
    )


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=f"{APP_NAME} API",
        description="Retrieval-Augmented Generation system with hybrid retrieval",
        version=APP_VERSION,
        lifespan=lifespan,
    )

    # CORS (open for demo; tighten per deployment)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files (CSS/JS) — mount only if directory exists
    static_dir = Path(__file__).resolve().parents[1] / "web" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Routers
    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(chat_router)
    app.include_router(ask_router)
    app.include_router(stats_router)

    return app


app = create_app()


__all__ = ["create_app", "app"]
