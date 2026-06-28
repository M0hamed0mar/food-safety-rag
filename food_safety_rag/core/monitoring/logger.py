"""
Structured logging system for the Food Safety RAG System.

Provides JSON and text-based structured logging with context preservation.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict
import sys

from food_safety_rag.core.config.settings import get_settings


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured JSON logs.
    
    Converts log records to JSON for easy parsing and analytics.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format.
            
        Returns:
            str: JSON-formatted log entry.
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields if available
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)


class Logger:
    """
    Structured logger for the Food Safety RAG system.
    
    Provides consistent logging across all modules with optional
    context information.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize logger.
        
        Args:
            name: Logger name (typically __name__).
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """
        Setup logging handlers based on configuration.
        
        Creates console and file handlers with appropriate formatters.
        """
        try:
            settings = get_settings()
        except Exception:
            # Fallback if settings fail to load
            settings = None

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Determine log level
        log_level = logging.INFO
        if settings:
            log_level_str = getattr(settings, "log_level", "INFO").upper()
            log_level = getattr(logging, log_level_str, logging.INFO)

        self.logger.setLevel(log_level)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        # Determine formatter
        if settings and getattr(settings, "structured_logs", True):
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler (if logs directory exists)
        if settings:
            try:
                logs_dir = Path(getattr(settings, "logs_dir", "logs"))
                logs_dir.mkdir(parents=True, exist_ok=True)
                file_path = logs_dir / f"{self.name.replace('.', '_')}.log"
                file_handler = logging.FileHandler(file_path)
                file_handler.setLevel(log_level)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception:
                # Silently fail if file handler can't be created
                pass

    def info(self, message: str, **kwargs: Any) -> None:
        """
        Log info level message.
        
        Args:
            message: Log message.
            **kwargs: Additional context fields.
        """
        self._log_with_context(logging.INFO, message, kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """
        Log debug level message.
        
        Args:
            message: Log message.
            **kwargs: Additional context fields.
        """
        self._log_with_context(logging.DEBUG, message, kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """
        Log warning level message.
        
        Args:
            message: Log message.
            **kwargs: Additional context fields.
        """
        self._log_with_context(logging.WARNING, message, kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """
        Log error level message.
        
        Args:
            message: Log message.
            **kwargs: Additional context fields.
        """
        self._log_with_context(logging.ERROR, message, kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """
        Log critical level message.
        
        Args:
            message: Log message.
            **kwargs: Additional context fields.
        """
        self._log_with_context(logging.CRITICAL, message, kwargs)

    def exception(self, message: str, exc_info: bool = True, **kwargs: Any) -> None:
        """
        Log exception with traceback.
        
        Args:
            message: Log message.
            exc_info: Whether to include exception traceback.
            **kwargs: Additional context fields.
        """
        self.logger.error(message, exc_info=exc_info, extra={"extra": kwargs})

    def _log_with_context(self, level: int, message: str, context: Dict[str, Any]) -> None:
        """
        Log message with context fields.
        
        Args:
            level: Logging level.
            message: Log message.
            context: Additional context dictionary.
        """
        record = self.logger.makeRecord(
            self.logger.name,
            level,
            None,
            None,
            message,
            (),
            None,
        )
        if context:
            record.extra = context
        self.logger.handle(record)


def get_logger(name: str) -> Logger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__).
        
    Returns:
        Logger: Configured logger instance.
    """
    return Logger(name)
