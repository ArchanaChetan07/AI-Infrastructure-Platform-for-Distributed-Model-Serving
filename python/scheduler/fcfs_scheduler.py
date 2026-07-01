"""First-Come-First-Served scheduler."""

from __future__ import annotations

import asyncio
from typing import Optional

from scheduler.base_scheduler import BaseScheduler, ScheduledRequest


class FCFSScheduler(BaseScheduler):
    """FIFO queue scheduler — baseline for benchmarking."""

    def __init__(self) -> None:
        super().__init__("fcfs")
        self._queue: asyncio.Queue[ScheduledRequest] = asyncio.Queue()
        self._pending: dict[str, ScheduledRequest] = {}

    async def submit(self, request: ScheduledRequest) -> None:
        self._pending[request.request_id] = request
        await self._queue.put(request)
        self._total_scheduled += 1

    async def acquire(self, timeout: Optional[float] = None) -> ScheduledRequest:
        if timeout:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        return await self._queue.get()

    async def complete(self, request_id: str, success: bool = True) -> None:
        self._pending.pop(request_id, None)
        if success:
            self._total_completed += 1
        else:
            self._total_errors += 1

    async def cancel(self, request_id: str) -> bool:
        if request_id in self._pending:
            self._pending.pop(request_id)
            self._total_cancelled = getattr(self, "_total_cancelled", 0) + 1
            return True
        return False

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()
