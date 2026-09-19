"""
Simple file-based cache for evaluation results.

Avoids re-running the same query against the (slow) pipeline
during iterative development.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from app.config import settings


CACHE_FILE = settings.DATA_DIR / "eval_cache.json"


class EvalCache:
    """
    Persistent JSON cache keyed by a hash of the query text.

    Stores the serialized answer + retrieved pages so tests can
    run offline / repeatable.
    """

    def __init__(self, cache_file: Path | None = None, enabled: bool = True) -> None:
        self.cache_file: Path = cache_file or CACHE_FILE
        self.enabled: bool = enabled
        self._data: dict[str, Any] = {}
        self._dirty: bool = False
        self._load()

    # --------------------------------------------------------
    # Internal
    # --------------------------------------------------------

    def _load(self) -> None:
        if not self.cache_file.exists():
            self._data = {}
            return
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        except Exception:
            self._data = {}

    def _save(self) -> None:
        if not self._dirty:
            return
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            self._dirty = False
        except Exception:
            pass

    @staticmethod
    def _make_key(query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()[:24]

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def get(self, query: str) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        return self._data.get(self._make_key(query))

    def set(self, query: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._data[self._make_key(query)] = value
        self._dirty = True

    def flush(self) -> None:
        self._save()

    def clear(self) -> None:
        self._data = {}
        self._dirty = True
        self.flush()

    def size(self) -> int:
        return len(self._data)


__all__ = ["EvalCache"]
