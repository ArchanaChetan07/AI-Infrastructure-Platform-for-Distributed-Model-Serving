"""Thread-safe async priority queue for SJF scheduling."""

from __future__ import annotations

import asyncio
import heapq
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass(order=True)
class PrioritizedItem(Generic[T]):
    """Heap item — lower priority value = higher precedence (SJF)."""

    priority: float
    enqueue_time: float
    request_id: str = field(compare=False)
    payload: Any = field(compare=False)


@dataclass
class QueueStats:
    """Queue statistics snapshot."""

    depth: int
    total_enqueued: int
    total_dequeued: int
    total_cancelled: int
    avg_wait_ms: float
    max_wait_ms: float


class PriorityQueue:
    """Async priority queue with cancellation and statistics."""

    def __init__(self) -> None:
        self._heap: list[PrioritizedItem] = []
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._cancelled: set[str] = set()
        self._pending: dict[str, PrioritizedItem] = {}
        self._total_enqueued = 0
        self._total_dequeued = 0
        self._total_cancelled = 0
        self._wait_times: list[float] = []

    async def enqueue(self, request_id: str, priority: float, payload: T) -> None:
        async with self._not_empty:
            item = PrioritizedItem(
                priority=priority,
                enqueue_time=time.monotonic(),
                request_id=request_id,
                payload=payload,
            )
            heapq.heappush(self._heap, item)
            self._pending[request_id] = item
            self._total_enqueued += 1
            self._not_empty.notify()

    async def dequeue(self, timeout: Optional[float] = None) -> PrioritizedItem[T]:
        async with self._not_empty:
            deadline = time.monotonic() + timeout if timeout else None
            while True:
                while self._heap:
                    item = heapq.heappop(self._heap)
                    if item.request_id in self._cancelled:
                        self._cancelled.discard(item.request_id)
                        self._pending.pop(item.request_id, None)
                        self._total_cancelled += 1
                        continue
                    wait_ms = (time.monotonic() - item.enqueue_time) * 1000.0
                    self._wait_times.append(wait_ms)
                    if len(self._wait_times) > 10000:
                        self._wait_times = self._wait_times[-5000:]
                    self._pending.pop(item.request_id, None)
                    self._total_dequeued += 1
                    return item
                if deadline and time.monotonic() >= deadline:
                    raise asyncio.TimeoutError("Queue dequeue timeout")
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise asyncio.TimeoutError("Queue dequeue timeout")
                    try:
                        await asyncio.wait_for(self._not_empty.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        raise asyncio.TimeoutError("Queue dequeue timeout") from None
                else:
                    await self._not_empty.wait()

    async def cancel(self, request_id: str) -> bool:
        async with self._lock:
            if request_id in self._pending:
                self._cancelled.add(request_id)
                return True
            return False

    async def update_priority(self, request_id: str, new_priority: float) -> bool:
        """Re-enqueue with updated priority (used by aging)."""
        async with self._not_empty:
            item = self._pending.get(request_id)
            if item is None:
                return False
            self._cancelled.add(request_id)
            new_item = PrioritizedItem(
                priority=new_priority,
                enqueue_time=item.enqueue_time,
                request_id=request_id,
                payload=item.payload,
            )
            heapq.heappush(self._heap, new_item)
            self._pending[request_id] = new_item
            self._cancelled.discard(request_id)
            self._not_empty.notify()
            return True

    def stats(self) -> QueueStats:
        waits = self._wait_times
        return QueueStats(
            depth=len(self._heap),
            total_enqueued=self._total_enqueued,
            total_dequeued=self._total_dequeued,
            total_cancelled=self._total_cancelled,
            avg_wait_ms=sum(waits) / len(waits) if waits else 0.0,
            max_wait_ms=max(waits) if waits else 0.0,
        )

    @property
    def depth(self) -> int:
        return len(self._heap)


def new_request_id() -> str:
    return str(uuid.uuid4())
