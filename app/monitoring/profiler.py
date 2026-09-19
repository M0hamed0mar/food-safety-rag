"""
Profiling module.

Exposes:
    - MemoryProfiler (via tracemalloc)
    - CPUProfiler (via cProfile)
    - PipelineProfiler that combines both
    - `profile_operation(...)` context manager (global profiler)
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional

from app.config import LogEvent
from app.monitoring.logger import LoggerAdapter, get_logger


logger: LoggerAdapter = get_logger("app.monitoring.profiler")


@dataclass
class MemorySnapshot:
    timestamp: float
    current_mb: float
    peak_mb: float
    top_allocations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "current_mb": round(self.current_mb, 2),
            "peak_mb": round(self.peak_mb, 2),
            "top_allocations": self.top_allocations,
        }


@dataclass
class ProfileResult:
    operation: str
    duration_ms: float
    memory_delta_mb: float
    cpu_profile: Optional[str] = None
    call_count: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "duration_ms": round(self.duration_ms, 2),
            "memory_delta_mb": round(self.memory_delta_mb, 2),
            "call_count": self.call_count,
            "timestamp": self.timestamp,
        }


class MemoryProfiler:
    def __init__(self) -> None:
        self._tracing: bool = False
        self._snapshots: list[MemorySnapshot] = []

    def start_tracing(self) -> None:
        if not self._tracing:
            tracemalloc.start()
            self._tracing = True

    def stop_tracing(self) -> None:
        if self._tracing:
            tracemalloc.stop()
            self._tracing = False

    def take_snapshot(self, top_n: int = 10) -> MemorySnapshot:
        if not self._tracing:
            self.start_tracing()
        current, peak = tracemalloc.get_traced_memory()
        snap = tracemalloc.take_snapshot()
        top = snap.statistics("lineno")[:top_n]
        allocations = [
            {
                "file": s.traceback.format()[-1] if s.traceback else "unknown",
                "size_bytes": s.size,
                "count": s.count,
            }
            for s in top
        ]
        m = MemorySnapshot(
            timestamp=time.time(),
            current_mb=current / (1024 * 1024),
            peak_mb=peak / (1024 * 1024),
            top_allocations=allocations,
        )
        self._snapshots.append(m)
        return m

    def get_snapshots(self) -> list[MemorySnapshot]:
        return self._snapshots.copy()

    def reset(self) -> None:
        self._snapshots.clear()


class CPUProfiler:
    def __init__(self) -> None:
        self._profiler: Optional[cProfile.Profile] = None

    def start(self) -> None:
        self._profiler = cProfile.Profile()
        self._profiler.enable()

    def stop(self) -> Optional[pstats.Stats]:
        if self._profiler is None:
            return None
        self._profiler.disable()
        stream = io.StringIO()
        stats = pstats.Stats(self._profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats(50)
        return stats


class PipelineProfiler:
    def __init__(self) -> None:
        self.memory_profiler = MemoryProfiler()
        self.cpu_profiler = CPUProfiler()
        self._results: list[ProfileResult] = []

    @contextmanager
    def profile(
        self,
        operation: str,
        profile_cpu: bool = False,
        profile_memory: bool = True,
    ) -> Generator[ProfileResult, None, None]:
        start = time.perf_counter()
        mem_before: Optional[MemorySnapshot] = None

        if profile_memory:
            self.memory_profiler.start_tracing()
            mem_before = self.memory_profiler.take_snapshot(top_n=5)

        if profile_cpu:
            self.cpu_profiler.start()

        result = ProfileResult(operation=operation, duration_ms=0.0, memory_delta_mb=0.0)

        try:
            yield result
        finally:
            result.duration_ms = (time.perf_counter() - start) * 1000

            if profile_memory and mem_before:
                mem_after = self.memory_profiler.take_snapshot(top_n=5)
                result.memory_delta_mb = mem_after.current_mb - mem_before.current_mb

            if profile_cpu:
                cpu_stats = self.cpu_profiler.stop()
                if cpu_stats:
                    stream = io.StringIO()
                    cpu_stats.stream = stream
                    cpu_stats.print_stats(30)
                    result.cpu_profile = stream.getvalue()
                    result.call_count = cpu_stats.total_calls

            result.timestamp = time.time()
            self._results.append(result)

            logger.log_event(
                LogEvent.LATENCY_RECORDED,
                f"Profile complete: {operation} took {result.duration_ms:.2f}ms, "
                f"memory delta: {result.memory_delta_mb:.2f}MB",
                level=20,
                details=result.to_dict(),
            )

    def get_results(self) -> list[ProfileResult]:
        return self._results.copy()

    def get_slowest_operations(self, n: int = 10) -> list[ProfileResult]:
        return sorted(self._results, key=lambda r: r.duration_ms, reverse=True)[:n]

    def export_results(self, file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("=== Pipeline Profiling Results ===\n\n")
            for r in self._results:
                f.write(f"Operation: {r.operation}\n")
                f.write(f"Duration: {r.duration_ms:.2f}ms\n")
                f.write(f"Memory Delta: {r.memory_delta_mb:.2f}MB\n")
                f.write(f"Call Count: {r.call_count}\n")
                f.write(f"Timestamp: {r.timestamp}\n")
                if r.cpu_profile:
                    f.write("\n--- CPU Profile ---\n")
                    f.write(r.cpu_profile)
                f.write("\n" + "=" * 50 + "\n\n")

    def reset(self) -> None:
        self._results.clear()
        self.memory_profiler.reset()


_global_profiler: PipelineProfiler = PipelineProfiler()


def get_profiler() -> PipelineProfiler:
    return _global_profiler


@contextmanager
def profile_operation(
    operation: str,
    profile_cpu: bool = False,
    profile_memory: bool = True,
) -> Generator[ProfileResult, None, None]:
    with _global_profiler.profile(operation, profile_cpu, profile_memory) as result:
        yield result


__all__ = [
    "MemoryProfiler",
    "CPUProfiler",
    "PipelineProfiler",
    "ProfileResult",
    "MemorySnapshot",
    "get_profiler",
    "profile_operation",
]
