"""Shortest-Job-First scheduler with ML-based length prediction."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from predictor.predictor import OutputLengthPredictor

from scheduler.aging import AgingConfig, create_aging
from scheduler.base_scheduler import BaseScheduler, ScheduledRequest, SchedulerType
from scheduler.fcfs_scheduler import FCFSScheduler
from scheduler.priority_queue import PriorityQueue

logger = logging.getLogger(__name__)


class SJFScheduler(BaseScheduler):
    """ML-predicted SJF scheduler with aging and FCFS fallback."""

    def __init__(
        self,
        predictor: OutputLengthPredictor,
        aging_config: Optional[AgingConfig] = None,
        fallback_to_fcfs: bool = True,
        aging_interval_sec: float = 0.5,
        max_retries: int = 3,
        request_timeout_sec: float = 300.0,
    ) -> None:
        super().__init__("sjf")
        self.predictor = predictor
        self.queue = PriorityQueue()
        self.aging = create_aging(aging_config or AgingConfig())
        self.aging_config = aging_config or AgingConfig()
        self.fallback_to_fcfs = fallback_to_fcfs
        self._fcfs_fallback = FCFSScheduler() if fallback_to_fcfs else None
        self._aging_interval = aging_interval_sec
        self._max_retries = max_retries
        self._timeout = request_timeout_sec
        self._pending: dict[str, ScheduledRequest] = {}
        self._use_fallback = False
        self._aging_task: Optional[asyncio.Task[None]] = None
        self._dispatcher_ready = asyncio.Event()

    async def start(self) -> None:
        if self._aging_task is None:
            self._aging_task = asyncio.create_task(self._aging_loop())
            self._dispatcher_ready.set()

    async def stop(self) -> None:
        if self._aging_task:
            self._aging_task.cancel()
            try:
                await self._aging_task
            except asyncio.CancelledError:
                pass
            self._aging_task = None

    async def submit(self, request: ScheduledRequest) -> None:
        t0 = time.perf_counter()
        try:
            if not request.predicted_tokens:
                pred = self.predictor.predict(request.prompt)
                request.predicted_tokens = pred.predicted_tokens
                self.record_timing(pred.feature_latency_ms, pred.predict_latency_ms, 0.0)
        except Exception as e:
            logger.warning("Prediction failed, using max_tokens: %s", e)
            request.predicted_tokens = float(request.max_tokens)
            if self._fcfs_fallback:
                self._use_fallback = True

        priority = request.predicted_tokens
        schedule_ms = (time.perf_counter() - t0) * 1000.0
        request.priority = priority
        request.metadata["schedule_ms"] = schedule_ms

        if self._use_fallback and self._fcfs_fallback:
            await self._fcfs_fallback.submit(request)
        else:
            self._pending[request.request_id] = request
            await self.queue.enqueue(request.request_id, priority, request)
        self._total_scheduled += 1

    async def acquire(self, timeout: Optional[float] = None) -> ScheduledRequest:
        if self._use_fallback and self._fcfs_fallback:
            return await self._fcfs_fallback.acquire(timeout=timeout)
        item = await self.queue.dequeue(timeout=timeout)
        return item.payload

    async def complete(self, request_id: str, success: bool = True) -> None:
        self._pending.pop(request_id, None)
        if self._fcfs_fallback:
            await self._fcfs_fallback.complete(request_id, success)
        if success:
            self._total_completed += 1
        else:
            self._total_errors += 1

    async def cancel(self, request_id: str) -> bool:
        self._pending.pop(request_id, None)
        if self._fcfs_fallback:
            await self._fcfs_fallback.cancel(request_id)
        return await self.queue.cancel(request_id)

    @property
    def queue_depth(self) -> int:
        if self._use_fallback and self._fcfs_fallback:
            return self._fcfs_fallback.queue_depth
        return self.queue.depth

    async def on_reload(self, config: dict[str, Any]) -> None:
        if "aging" in config:
            self.aging_config = AgingConfig(**config["aging"])
            self.aging = create_aging(self.aging_config)
        if "fallback_to_fcfs" in config:
            self.fallback_to_fcfs = bool(config["fallback_to_fcfs"])

    async def _aging_loop(self) -> None:
        while True:
            await asyncio.sleep(self._aging_interval)
            now = time.monotonic()
            for req_id, req in list(self._pending.items()):
                wait = now - req.enqueue_time
                if wait > self._timeout:
                    await self.cancel(req_id)
                    continue
                new_priority = self.aging.boost(wait, req.priority)
                if new_priority < req.priority - 0.01:
                    await self.queue.update_priority(req_id, new_priority)
                    req.priority = new_priority


class OracleSJFScheduler(BaseScheduler):
    """Oracle SJF using actual max_tokens as perfect prediction (benchmark upper bound)."""

    def __init__(self) -> None:
        super().__init__("oracle_sjf")
        self.queue = PriorityQueue()

    async def submit(self, request: ScheduledRequest) -> None:
        request.predicted_tokens = float(request.max_tokens)
        request.priority = request.predicted_tokens
        await self.queue.enqueue(request.request_id, request.priority, request)
        self._total_scheduled += 1

    async def acquire(self, timeout: Optional[float] = None) -> ScheduledRequest:
        item = await self.queue.dequeue(timeout=timeout)
        return item.payload

    async def complete(self, request_id: str, success: bool = True) -> None:
        if success:
            self._total_completed += 1
        else:
            self._total_errors += 1

    async def cancel(self, request_id: str) -> bool:
        return await self.queue.cancel(request_id)

    @property
    def queue_depth(self) -> int:
        return self.queue.depth


def build_scheduler(
    scheduler_type: SchedulerType,
    predictor: Optional[OutputLengthPredictor] = None,
    config: Optional[dict[str, Any]] = None,
) -> BaseScheduler:
    config = config or {}
    if scheduler_type == SchedulerType.FCFS:
        return FCFSScheduler()
    if scheduler_type == SchedulerType.ORACLE_SJF:
        return OracleSJFScheduler()
    if predictor is None:
        raise ValueError("SJF scheduler requires a predictor")
    aging_cfg = AgingConfig(**config.get("aging", {}))
    return SJFScheduler(
        predictor=predictor,
        aging_config=aging_cfg,
        fallback_to_fcfs=config.get("fallback_to_fcfs", True),
        aging_interval_sec=config.get("aging_interval_sec", 0.5),
        request_timeout_sec=config.get("request_timeout_sec", 300.0),
    )
