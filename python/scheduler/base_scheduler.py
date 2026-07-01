"""Base scheduler interface."""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class SchedulerType(str, Enum):
    FCFS = "fcfs"
    SJF = "sjf"
    ORACLE_SJF = "oracle_sjf"


@dataclass
class ScheduledRequest:
    """A request in the scheduling pipeline."""

    request_id: str
    prompt: str
    messages: list[dict[str, Any]]
    max_tokens: int
    predicted_tokens: float
    priority: float
    enqueue_time: float = field(default_factory=time.monotonic)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def wait_time_sec(self) -> float:
        return time.monotonic() - self.enqueue_time


@dataclass
class SchedulerStats:
    """Scheduler-level metrics."""

    scheduler_type: str
    queue_depth: int
    total_scheduled: int
    total_completed: int
    total_errors: int
    avg_feature_ms: float
    avg_predict_ms: float
    avg_schedule_ms: float


class BaseScheduler(abc.ABC):
    """Abstract scheduler interface."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._total_scheduled = 0
        self._total_completed = 0
        self._total_errors = 0
        self._feature_latencies: list[float] = []
        self._predict_latencies: list[float] = []
        self._schedule_latencies: list[float] = []

    @abc.abstractmethod
    async def submit(self, request: ScheduledRequest) -> None:
        """Enqueue a request for scheduling."""

    @abc.abstractmethod
    async def acquire(self, timeout: Optional[float] = None) -> ScheduledRequest:
        """Get next request to dispatch to vLLM."""

    @abc.abstractmethod
    async def complete(self, request_id: str, success: bool = True) -> None:
        """Mark request as completed."""

    @abc.abstractmethod
    async def cancel(self, request_id: str) -> bool:
        """Cancel a pending request."""

    def record_timing(self, feature_ms: float, predict_ms: float, schedule_ms: float) -> None:
        self._feature_latencies.append(feature_ms)
        self._predict_latencies.append(predict_ms)
        self._schedule_latencies.append(schedule_ms)

    def stats(self) -> SchedulerStats:
        def _avg(xs: list[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        return SchedulerStats(
            scheduler_type=self.name,
            queue_depth=self.queue_depth,
            total_scheduled=self._total_scheduled,
            total_completed=self._total_completed,
            total_errors=self._total_errors,
            avg_feature_ms=_avg(self._feature_latencies),
            avg_predict_ms=_avg(self._predict_latencies),
            avg_schedule_ms=_avg(self._schedule_latencies),
        )

    @property
    @abc.abstractmethod
    def queue_depth(self) -> int: ...

    async def on_reload(self, config: dict[str, Any]) -> None:
        """Hot-reload configuration."""
