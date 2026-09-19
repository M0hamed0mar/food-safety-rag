"""
Structured logging module.

Wraps the standard library `logging` and provides:
    - A `LoggerAdapter` with domain helpers:
        log_event, log_ingestion, log_retrieval,
        log_generation, log_error, log_latency, log_cache
    - A JSON formatter (via the standard lib) for machine-readable logs
    - A global `setup_logging()` used at app startup

We intentionally keep this on top of `logging` (not structlog) to
preserve the API that the rest of the pipeline already expects
(e.g. `logger.log_event(...)`).
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.config import LogEvent, settings
from app.config.constants import LogEvent as _LogEvent  # noqa: F401  (re-export safety)


# ============================================================
# JSON formatter
# ============================================================

class StructuredLogFormatter(logging.Formatter):
    """Render every log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "thread_name": record.threadName,
        }

        if hasattr(record, "event"):
            entry["event"] = record.event
        if hasattr(record, "error_code"):
            entry["error_code"] = record.error_code
        if hasattr(record, "details") and record.details:
            entry["details"] = record.details

        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            entry["exception"] = {
                "type": exc_type.__name__ if exc_type else None,
                "message": str(exc_value) if exc_value else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(entry, default=str, ensure_ascii=False)


# ============================================================
# LoggerAdapter
# ============================================================

class LoggerAdapter(logging.LoggerAdapter):
    """Logger wrapper with domain-specific helpers."""

    def __init__(self, logger: logging.Logger, extra: Optional[dict[str, Any]] = None) -> None:
        super().__init__(logger, extra or {})

    # --------------------------------------------------------
    # Core event helper
    # --------------------------------------------------------

    def log_event(
        self,
        event: LogEvent,
        message: str,
        level: int = logging.INFO,
        details: Optional[dict[str, Any]] = None,
        error_code: Optional[str] = None,
    ) -> None:
        extra: dict[str, Any] = {"event": event.value}
        if details:
            extra["details"] = details
        if error_code:
            extra["error_code"] = error_code
        self.log(level, message, extra=extra)

    # --------------------------------------------------------
    # Domain helpers
    # --------------------------------------------------------

    def log_ingestion(
        self,
        event: LogEvent,
        document_id: str,
        document_name: str,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        d: dict[str, Any] = {
            "document_id": document_id,
            "document_name": document_name,
        }
        if details:
            d.update(details)
        self.log_event(event, message, details=d)

    def log_retrieval(
        self,
        event: LogEvent,
        query_id: str,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        d: dict[str, Any] = {"query_id": query_id}
        if details:
            d.update(details)
        self.log_event(event, message, details=d)

    def log_generation(
        self,
        event: LogEvent,
        answer_id: str,
        query_id: str,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        d: dict[str, Any] = {
            "answer_id": answer_id,
            "query_id": query_id,
        }
        if details:
            d.update(details)
        self.log_event(event, message, details=d)

    def log_error(
        self,
        event: LogEvent,
        message: str,
        exception: Optional[Exception] = None,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        if exception:
            self.log_event(
                event, message, level=logging.ERROR,
                error_code=error_code, details=details,
            )
            self.error(
                message, exc_info=True,
                extra={"event": event.value, "error_code": error_code},
            )
        else:
            self.log_event(
                event, message, level=logging.ERROR,
                error_code=error_code, details=details,
            )

    def log_latency(
        self,
        operation: str,
        duration_ms: float,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        d: dict[str, Any] = {
            "operation": operation,
            "duration_ms": duration_ms,
        }
        if details:
            d.update(details)
        self.log_event(
            LogEvent.TOTAL_REQUEST_LATENCY_MS,
            f"Latency: {operation} completed in {duration_ms:.2f}ms",
            level=logging.DEBUG,
            details=d,
        )

    def log_cache(
        self,
        event: LogEvent,
        cache_key: str,
        cache_type: str,
        hit: bool,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        d: dict[str, Any] = {
            "cache_key": cache_key,
            "cache_type": cache_type,
            "hit": hit,
        }
        if details:
            d.update(details)
        self.log_event(
            event,
            f"Cache {'hit' if hit else 'miss'}: {cache_type}/{cache_key}",
            details=d,
        )


# ============================================================
# Setup
# ============================================================

def setup_logging(
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> None:
    """Configure the root logger (idempotent)."""
    level = (log_level or settings.LOG_LEVEL).upper()
    fmt = (log_format or "json").lower()

    formatter: logging.Formatter
    if fmt == "json":
        formatter = StructuredLogFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
        )

    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))

    # Remove existing handlers (avoid duplicates)
    for h in root.handlers[:]:
        root.removeHandler(h)

    # Console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level, logging.INFO))
    console.setFormatter(formatter)
    root.addHandler(console)

    # File
    try:
        log_path = log_file or (settings.LOGS_DIR / "app.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(getattr(logging, level, logging.INFO))
        fh.setFormatter(formatter)
        root.addHandler(fh)
    except Exception:
        # File logging is best-effort; if it fails, keep console only
        pass

    # Quiet noisy libs
    for noisy in ("urllib3", "requests", "httpx", "openai", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> LoggerAdapter:
    """Get a structured logger adapter."""
    return LoggerAdapter(logging.getLogger(name))


# System logger
system_logger: LoggerAdapter = get_logger("app.system")


__all__ = [
    "StructuredLogFormatter",
    "LoggerAdapter",
    "get_logger",
    "setup_logging",
    "system_logger",
]
