"""Simulated and live FCFS benchmark runner."""

from __future__ import annotations

import asyncio
import random
import time
from typing import List

from scheduler.base_scheduler import ScheduledRequest
from scheduler.fcfs_scheduler import FCFSScheduler
from scheduler.priority_queue import new_request_id

from benchmark.metrics import RequestMetric, aggregate_metrics

PROMPTS = [
    "What is machine learning?",
    "Write a Python sort function.",
    "Explain quantum computing briefly.",
    "List 5 database optimization tips.",
    "How does attention work in transformers?",
    "Debug this SQL query.",
    "Summarize the history of AI.",
    "Compare REST and GraphQL.",
]


async def simulate_fcfs(
    concurrency: int,
    n_requests: int,
    service_time_fn,
) -> tuple[list[RequestMetric], float]:
    """Discrete-event simulation of FCFS scheduling."""
    scheduler = FCFSScheduler()
    metrics: List[RequestMetric] = []
    sem = asyncio.Semaphore(concurrency)
    t_start = time.monotonic()

    async def worker(i: int) -> None:
        async with sem:
            prompt = PROMPTS[i % len(PROMPTS)]
            max_tokens = random.randint(20, 200)
            req = ScheduledRequest(
                request_id=new_request_id(),
                prompt=prompt,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                predicted_tokens=float(max_tokens),
                priority=float(max_tokens),
            )
            t_enqueue = time.monotonic()
            await scheduler.submit(req)
            acquired = await scheduler.acquire()
            wait_ms = (time.monotonic() - t_enqueue) * 1000.0
            svc_ms = await service_time_fn(acquired.max_tokens)
            await scheduler.complete(acquired.request_id)
            metrics.append(
                RequestMetric(
                    success=True,
                    e2e_ms=wait_ms + svc_ms,
                    queue_wait_ms=wait_ms,
                    feature_ms=0.0,
                    predict_ms=0.0,
                    schedule_ms=0.0,
                    tokens=acquired.max_tokens,
                    concurrency=concurrency,
                )
            )

    await asyncio.gather(*[worker(i) for i in range(n_requests)])
    return metrics, time.monotonic() - t_start


async def run_fcfs_benchmark(concurrency: int, n_requests: int) -> dict:
    async def service_time(tokens: int) -> float:
        await asyncio.sleep(tokens * 0.002 + random.uniform(0.01, 0.05))
        return tokens * 2.0

    metrics, elapsed = await simulate_fcfs(concurrency, n_requests, service_time)
    result = aggregate_metrics("fcfs", concurrency, metrics, elapsed)
    return result.__dict__
