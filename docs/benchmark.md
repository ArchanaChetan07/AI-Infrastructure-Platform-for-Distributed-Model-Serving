# Benchmark Guide

## Run Scheduler Comparison

```bash
make benchmark-scheduler
# or
PYTHONPATH=python:. python python/benchmark/compare.py
```

## Schedulers Compared

| Scheduler | Description |
|-----------|-------------|
| **FCFS** | First-come-first-served baseline |
| **Oracle SJF** | Perfect knowledge of output length (upper bound) |
| **Predicted SJF** | ML-predicted output length (production path) |

## Concurrency Levels

Default: 1, 2, 4, 8, 16, 32 (configurable in `configs/training.yaml`)

## Output Artifacts

```
python/benchmark/results/
  comparison_*.json
  comparison_*.csv
  comparison_*.md
  latency_throughput_*.png
  queue_wait_*.png
```

## Metrics Collected

- End-to-end latency (p50, p95, p99)
- Queue wait time
- Feature extraction latency
- Prediction latency
- Throughput (RPS, tokens/sec)
- Scheduler overhead

## Evaluation Report

```bash
python python/evaluation/generate_report.py
```

Produces `docs/reports/evaluation_report.{md,html,pdf}`.
