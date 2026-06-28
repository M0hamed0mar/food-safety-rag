"""
Latency tracking and performance monitoring.

Tracks execution time for critical pipeline stages.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator, Dict, List, Optional

from food_safety_rag.core.monitoring.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LatencyMetric:
    """
    A single latency measurement.
    
    Records the time taken for a specific operation.
    """

    operation_name: str
    """Name of the operation."""

    duration_ms: float
    """Duration in milliseconds."""

    timestamp: datetime = field(default_factory=datetime.utcnow)
    """When the operation completed."""

    metadata: Dict[str, str] = field(default_factory=dict)
    """Additional metadata about the operation."""

    def __repr__(self) -> str:
        """
        Get string representation.
        
        Returns:
            str: Readable representation.
        """
        return f"{self.operation_name}: {self.duration_ms:.2f}ms"


class LatencyTracker:
    """
    Tracks latency of operations throughout the RAG pipeline.
    
    Provides context managers for easy latency measurement.
    """

    def __init__(self) -> None:
        """
        Initialize latency tracker.
        """
        self.metrics: List[LatencyMetric] = []
        self._start_times: Dict[str, float] = {}

    @contextmanager
    def track(self, operation_name: str, **metadata: str) -> Iterator[None]:
        """
        Context manager for tracking operation latency.
        
        Usage:
            with tracker.track("dense_retrieval", query_id="123"):
                # perform operation
        
        Args:
            operation_name: Name of the operation being tracked.
            **metadata: Additional metadata to record.
            
        Yields:
            None
        """
        start_time = time.time()

        try:
            yield
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            metric = LatencyMetric(
                operation_name=operation_name,
                duration_ms=duration_ms,
                metadata=metadata,
            )
            self.metrics.append(metric)

            # Log long-running operations
            if duration_ms > 1000:  # More than 1 second
                logger.warning(
                    f"Long-running operation: {operation_name}",
                    duration_ms=duration_ms,
                    **metadata,
                )

    def record(self, operation_name: str, duration_ms: float, **metadata: str) -> None:
        """
        Manually record a latency metric.
        
        Args:
            operation_name: Name of the operation.
            duration_ms: Duration in milliseconds.
            **metadata: Additional metadata.
        """
        metric = LatencyMetric(
            operation_name=operation_name,
            duration_ms=duration_ms,
            metadata=metadata,
        )
        self.metrics.append(metric)

    def get_metrics_by_operation(self, operation_name: str) -> List[LatencyMetric]:
        """
        Get all metrics for a specific operation.
        
        Args:
            operation_name: The operation name to filter by.
            
        Returns:
            List[LatencyMetric]: Matching metrics.
        """
        return [m for m in self.metrics if m.operation_name == operation_name]

    def get_average_latency(self, operation_name: str) -> Optional[float]:
        """
        Get average latency for an operation.
        
        Args:
            operation_name: The operation name.
            
        Returns:
            Optional[float]: Average duration in ms, or None if no metrics.
        """
        metrics = self.get_metrics_by_operation(operation_name)
        if not metrics:
            return None
        return sum(m.duration_ms for m in metrics) / len(metrics)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary of all tracked latencies.
        
        Returns:
            Dict: Summary statistics by operation.
        """
        operations = set(m.operation_name for m in self.metrics)
        summary = {}

        for op in operations:
            metrics = self.get_metrics_by_operation(op)
            durations = [m.duration_ms for m in metrics]
            summary[op] = {
                "count": len(durations),
                "total_ms": sum(durations),
                "average_ms": sum(durations) / len(durations) if durations else 0,
                "min_ms": min(durations) if durations else 0,
                "max_ms": max(durations) if durations else 0,
            }

        return summary

    def clear(self) -> None:
        """
        Clear all recorded metrics.
        """
        self.metrics.clear()
        self._start_times.clear()

    def log_summary(self) -> None:
        """
        Log a summary of all latency metrics.
        """
        summary = self.get_summary()
        logger.info(
            "Latency Summary",
            latency_stats=summary,
        )
