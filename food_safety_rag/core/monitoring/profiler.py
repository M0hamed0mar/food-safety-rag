"""
Performance profiling utilities.

Provides memory and CPU profiling capabilities.
"""

import psutil
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

from food_safety_rag.core.monitoring.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SystemMetrics:
    """
    System performance metrics.
    
    Captures CPU, memory, and process statistics.
    """

    timestamp: datetime = field(default_factory=datetime.utcnow)
    """When metrics were captured."""

    cpu_percent: float = 0.0
    """CPU usage percentage."""

    memory_percent: float = 0.0
    """Memory usage percentage."""

    memory_mb: float = 0.0
    """Memory usage in megabytes."""

    process_cpu_num: int = 0
    """Number of CPUs used by process."""

    open_files: int = 0
    """Number of open file descriptors."""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_mb": self.memory_mb,
            "process_cpu_num": self.process_cpu_num,
            "open_files": self.open_files,
        }


class Profiler:
    """
    Performance profiler for the RAG system.
    
    Monitors system resources and application performance.
    """

    def __init__(self) -> None:
        """
        Initialize profiler.
        """
        self.process = psutil.Process(os.getpid())
        self.initial_metrics: Optional[SystemMetrics] = None
        self.capture_initial_metrics()

    def capture_initial_metrics(self) -> None:
        """
        Capture initial system metrics.
        
        Used as baseline for comparison.
        """
        try:
            self.initial_metrics = self.get_current_metrics()
        except Exception as e:
            logger.warning(f"Failed to capture initial metrics: {e}")

    def get_current_metrics(self) -> SystemMetrics:
        """
        Get current system metrics.
        
        Returns:
            SystemMetrics: Current system state.
        """
        try:
            cpu_percent = self.process.cpu_percent(interval=0.1)
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            memory_percent = self.process.memory_percent()
            cpu_num = len(self.process.cpu_num())
            open_files = len(self.process.open_files())

            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_mb=memory_mb,
                process_cpu_num=cpu_num,
                open_files=open_files,
            )
        except Exception as e:
            logger.error(f"Failed to get current metrics: {e}")
            return SystemMetrics()

    def get_memory_delta_mb(self) -> float:
        """
        Get memory usage change since initialization.
        
        Returns:
            float: Memory delta in megabytes (positive = increased).
        """
        if not self.initial_metrics:
            return 0.0

        current = self.get_current_metrics()
        return current.memory_mb - self.initial_metrics.memory_mb

    def log_metrics(self, operation_name: str) -> None:
        """
        Log current metrics for an operation.
        
        Args:
            operation_name: Name of operation being profiled.
        """
        metrics = self.get_current_metrics()
        logger.info(
            f"Metrics for {operation_name}",
            cpu_percent=metrics.cpu_percent,
            memory_percent=metrics.memory_percent,
            memory_mb=metrics.memory_mb,
            open_files=metrics.open_files,
        )

    def reset(self) -> None:
        """
        Reset profiler baseline.
        """
        self.capture_initial_metrics()
