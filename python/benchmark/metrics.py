"""Benchmark metrics collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List


@dataclass
class RequestMetric:
    success: bool
    e2e_ms: float
    queue_wait_ms: float
    feature_ms: float
    predict_ms: float
    schedule_ms: float
    tokens: int
    concurrency: int


@dataclass
class BenchmarkResult:
    scheduler: str
    concurrency: int
    n_requests: int
    n_success: int
    throughput_rps: float
    tokens_per_sec: float
    e2e_p50_ms: float
    e2e_p95_ms: float
    e2e_p99_ms: float
    queue_wait_p50_ms: float
    queue_wait_p99_ms: float
    feature_p50_ms: float
    predict_p50_ms: float
    schedule_p50_ms: float
    error_rate: float
    elapsed_sec: float
    # Resource metrics (optional)
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = max(0, int(len(sorted_data) * p) - 1)
    return sorted_data[min(idx, len(sorted_data) - 1)]


def aggregate_metrics(
    scheduler: str,
    concurrency: int,
    metrics: List[RequestMetric],
    elapsed_sec: float,
) -> BenchmarkResult:
    successes = [m for m in metrics if m.success]
    e2es = [m.e2e_ms for m in metrics]
    waits = [m.queue_wait_ms for m in metrics]
    features = [m.feature_ms for m in metrics]
    predicts = [m.predict_ms for m in metrics]
    schedules = [m.schedule_ms for m in metrics]
    total_tokens = sum(m.tokens for m in successes)

    return BenchmarkResult(
        scheduler=scheduler,
        concurrency=concurrency,
        n_requests=len(metrics),
        n_success=len(successes),
        throughput_rps=round(len(successes) / max(elapsed_sec, 1e-6), 3),
        tokens_per_sec=round(total_tokens / max(elapsed_sec, 1e-6), 1),
        e2e_p50_ms=round(percentile(e2es, 0.50), 2),
        e2e_p95_ms=round(percentile(e2es, 0.95), 2),
        e2e_p99_ms=round(percentile(e2es, 0.99), 2),
        queue_wait_p50_ms=round(percentile(waits, 0.50), 2),
        queue_wait_p99_ms=round(percentile(waits, 0.99), 2),
        feature_p50_ms=round(percentile(features, 0.50), 3),
        predict_p50_ms=round(percentile(predicts, 0.50), 3),
        schedule_p50_ms=round(percentile(schedules, 0.50), 3),
        error_rate=round((len(metrics) - len(successes)) / max(len(metrics), 1) * 100, 2),
        elapsed_sec=round(elapsed_sec, 2),
    )


def results_to_dict(results: List[BenchmarkResult]) -> list[dict]:
    return [asdict(r) for r in results]
