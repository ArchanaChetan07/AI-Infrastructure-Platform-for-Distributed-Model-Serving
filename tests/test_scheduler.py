"""Tests for schedulers."""

from __future__ import annotations

import pytest

from predictor.model import OutputLengthMLP
from predictor.predictor import OutputLengthPredictor
from scheduler.aging import AgingConfig, AgingPolicy, create_aging
from scheduler.base_scheduler import ScheduledRequest
from scheduler.fcfs_scheduler import FCFSScheduler
from scheduler.priority_queue import PriorityQueue, new_request_id
from scheduler.sjf_scheduler import OracleSJFScheduler, SJFScheduler


def _make_request(max_tokens: int = 50, prompt: str = "test") -> ScheduledRequest:
    return ScheduledRequest(
        request_id=new_request_id(),
        prompt=prompt,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        predicted_tokens=0.0,
        priority=0.0,
    )


@pytest.mark.unit
class TestPriorityQueue:
    @pytest.mark.asyncio
    async def test_sjf_ordering(self) -> None:
        q = PriorityQueue()
        await q.enqueue("a", 100.0, "long")
        await q.enqueue("b", 10.0, "short")
        item = await q.dequeue()
        assert item.payload == "short"

    @pytest.mark.asyncio
    async def test_cancel(self) -> None:
        q = PriorityQueue()
        await q.enqueue("x", 1.0, "data")
        assert await q.cancel("x")
        await q.enqueue("y", 2.0, "other")
        item = await q.dequeue()
        assert item.request_id == "y"


@pytest.mark.unit
class TestFCFS:
    @pytest.mark.asyncio
    async def test_fifo_order(self) -> None:
        sched = FCFSScheduler()
        r1 = _make_request(10, "first")
        r2 = _make_request(20, "second")
        await sched.submit(r1)
        await sched.submit(r2)
        got1 = await sched.acquire()
        got2 = await sched.acquire()
        assert got1.request_id == r1.request_id
        assert got2.request_id == r2.request_id


@pytest.mark.unit
class TestSJF:
    @pytest.mark.asyncio
    async def test_shortest_first(self) -> None:
        predictor = OutputLengthPredictor(OutputLengthMLP())
        sched = SJFScheduler(predictor)
        await sched.start()
        short = _make_request(10, "Hi")
        long = _make_request(500, "Write a detailed essay about " * 20)
        await sched.submit(long)
        await sched.submit(short)
        first = await sched.acquire(timeout=5.0)
        await sched.stop()
        assert first.max_tokens <= long.max_tokens

    @pytest.mark.asyncio
    async def test_oracle_sjf(self) -> None:
        sched = OracleSJFScheduler()
        r1 = _make_request(200)
        r2 = _make_request(20)
        await sched.submit(r1)
        await sched.submit(r2)
        first = await sched.acquire()
        assert first.max_tokens == 20


@pytest.mark.unit
class TestAging:
    def test_linear_aging_reduces_priority(self) -> None:
        aging = create_aging(AgingConfig(policy=AgingPolicy.LINEAR, factor=1.0))
        p0 = aging.boost(0.0, 100.0)
        p1 = aging.boost(10.0, 100.0)
        assert p1 < p0

    def test_logarithmic_aging(self) -> None:
        aging = create_aging(AgingConfig(policy=AgingPolicy.LOGARITHMIC))
        assert aging.boost(5.0, 50.0) < 50.0

    def test_exponential_aging(self) -> None:
        aging = create_aging(AgingConfig(policy=AgingPolicy.EXPONENTIAL))
        assert aging.boost(3.0, 80.0) < 80.0
