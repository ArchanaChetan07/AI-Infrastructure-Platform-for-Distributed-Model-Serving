"""SJF and Oracle SJF benchmark runners."""

from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path
from typing import List, Optional

from predictor.predictor import OutputLengthPredictor
from scheduler.base_scheduler import ScheduledRequest
from scheduler.priority_queue import new_request_id
from scheduler.sjf_scheduler import OracleSJFScheduler, SJFScheduler

from benchmark.fcfs import PROMPTS
from benchmark.metrics import RequestMetric, aggregate_metrics


async def simulate_sjf(
    scheduler,
    predictor: Optional[OutputLengthPredictor],
    concurrency: int,
    n_requests: int,
    service_time_fn,
    use_oracle: bool = False,
) -> tuple[list[RequestMetric], float]:
    if isinstance(scheduler, SJFScheduler):
        await scheduler.start()

    metrics: List[RequestMetric] = []
    sem = asyncio.Semaphore(concurrency)
    t_start = time.monotonic()

    async def worker(i: int) -> None:
        async with sem:
            prompt = PROMPTS[i % len(PROMPTS)]
            max_tokens = random.randint(20, 200)
            feature_ms = predict_ms = 0.0
            predicted = float(max_tokens) if use_oracle else 0.0

            if predictor and not use_oracle:
                pred = predictor.predict(prompt)
                predicted = pred.predicted_tokens
                feature_ms = pred.feature_latency_ms
                predict_ms = pred.predict_latency_ms

            req = ScheduledRequest(
                request_id=new_request_id(),
                prompt=prompt,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                predicted_tokens=predicted,
                priority=predicted,
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
                    feature_ms=feature_ms,
                    predict_ms=predict_ms,
                    schedule_ms=0.1,
                    tokens=acquired.max_tokens,
                    concurrency=concurrency,
                )
            )

    await asyncio.gather(*[worker(i) for i in range(n_requests)])
    if isinstance(scheduler, SJFScheduler):
        await scheduler.stop()
    return metrics, time.monotonic() - t_start


async def run_sjf_benchmark(
    concurrency: int,
    n_requests: int,
    models_dir: str = "shared/models",
) -> dict:
    predictor = OutputLengthPredictor.from_checkpoint(Path(models_dir) / "output_length_mlp.pt")
    scheduler = SJFScheduler(predictor)

    async def service_time(tokens: int) -> float:
        await asyncio.sleep(tokens * 0.002 + random.uniform(0.01, 0.05))
        return tokens * 2.0

    metrics, elapsed = await simulate_sjf(
        scheduler, predictor, concurrency, n_requests, service_time
    )
    return aggregate_metrics("sjf", concurrency, metrics, elapsed).__dict__


async def run_oracle_benchmark(concurrency: int, n_requests: int) -> dict:
    scheduler = OracleSJFScheduler()

    async def service_time(tokens: int) -> float:
        await asyncio.sleep(tokens * 0.002 + random.uniform(0.01, 0.05))
        return tokens * 2.0

    metrics, elapsed = await simulate_sjf(
        scheduler, None, concurrency, n_requests, service_time, use_oracle=True
    )
    return aggregate_metrics("oracle_sjf", concurrency, metrics, elapsed).__dict__
