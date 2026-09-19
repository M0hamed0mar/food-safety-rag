"""
Latency measurement module.

Provides:
    - `LatencyMeasurement` dataclass
    - `LatencyTracker` for collecting + aggregating measurements
    - `measure_latency(...)` context manager (uses the global tracker)
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

from app.config import LogEvent
from app.monitoring.logger import LoggerAdapter, get_logger


logger: LoggerAdapter = get_logger("app.monitoring.latency")


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""

    operation: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def stop(self, **metadata: Any) -> "LatencyMeasurement":
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.metadata.update(metadata)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "duration_ms": round(self.duration_ms, 2) if self.duration_ms is not None else None,
            "metadata": self.metadata,
        }


class LatencyTracker:
    """Tracks and aggregates latency measurements."""

    def __init__(self, max_history: int = 10_000) -> None:
        self.max_history = max_history
        self._measurements: list[LatencyMeasurement] = []
        self._active: dict[str, LatencyMeasurement] = {}

    def start(self, operation: str, **metadata: Any) -> LatencyMeasurement:
        m = LatencyMeasurement(operation=operation, metadata=dict(metadata))
        self._active[operation] = m
        return m

    def stop(self, operation: str, **metadata: Any) -> Optional[LatencyMeasurement]:
        m = self._active.pop(operation, None)
        if m is None:
            logger.log_event(
                LogEvent.WARNING,
                f"No active measurement found for operation: {operation}",
                level=30,
            )
            return None

        m.stop(**metadata)
        self._record(m)

        logger.log_latency(
            operation=m.operation,
            duration_ms=m.duration_ms or 0.0,
            details=m.metadata,
        )
        return m

    def _record(self, m: LatencyMeasurement) -> None:
        self._measurements.append(m)
        if len(self._measurements) > self.max_history:
            self._measurements = self._measurements[-self.max_history:]

    def get_measurements(self, operation: str, limit: Optional[int] = None) -> list[LatencyMeasurement]:
        matching = [m for m in self._measurements if m.operation == operation]
        if limit:
            matching = matching[-limit:]
        return matching

    def get_average(self, operation: str, last_n: Optional[int] = None) -> Optional[float]:
        ms = self.get_measurements(operation, last_n)
        ds = [m.duration_ms for m in ms if m.duration_ms is not None]
        return sum(ds) / len(ds) if ds else None

    def get_percentile(self, operation: str, percentile: float, last_n: Optional[int] = None) -> Optional[float]:
        ms = self.get_measurements(operation, last_n)
        ds = sorted([m.duration_ms for m in ms if m.duration_ms is not None])
        if not ds:
            return None
        idx = min(int(len(ds) * (percentile / 100.0)), len(ds) - 1)
        return ds[idx]

    def get_statistics(self, operation: str, last_n: Optional[int] = None) -> dict[str, Any]:
        ms = self.get_measurements(operation, last_n)
        ds = [m.duration_ms for m in ms if m.duration_ms is not None]
        if not ds:
            return {
                "operation": operation, "count": 0,
                "avg_ms": None, "min_ms": None, "max_ms": None,
                "p50_ms": None, "p95_ms": None, "p99_ms": None,
            }
        ds_sorted = sorted(ds)
        n = len(ds)
        def pct(p: float) -> float:
            return ds_sorted[min(int(n * (p / 100.0)), n - 1)]
        return {
            "operation": operation,
            "count": n,
            "avg_ms": round(sum(ds) / n, 2),
            "min_ms": round(min(ds), 2),
            "max_ms": round(max(ds), 2),
            "p50_ms": round(pct(50.0), 2),
            "p95_ms": round(pct(95.0), 2),
            "p99_ms": round(pct(99.0), 2),
        }

    def get_all_statistics(self) -> dict[str, dict[str, Any]]:
        ops = {m.operation for m in self._measurements}
        return {op: self.get_statistics(op) for op in ops}

    def clear(self) -> None:
        self._measurements.clear()
        self._active.clear()

    @contextmanager
    def measure(self, operation: str, **metadata: Any) -> Generator[LatencyMeasurement, None, None]:
        m = self.start(operation, **metadata)
        try:
            yield m
        finally:
            self.stop(operation, **metadata)


# ============================================================
# Global tracker + convenience context manager
# ============================================================

_global_tracker: LatencyTracker = LatencyTracker()


def get_tracker() -> LatencyTracker:
    return _global_tracker


@contextmanager
def measure_latency(operation: str, **metadata: Any) -> Generator[LatencyMeasurement, None, None]:
    tracker = get_tracker()
    m = tracker.start(operation, **metadata)
    try:
        yield m
    finally:
        tracker.stop(operation, **metadata)


__all__ = [
    "LatencyMeasurement",
    "LatencyTracker",
    "get_tracker",
    "measure_latency",
]
