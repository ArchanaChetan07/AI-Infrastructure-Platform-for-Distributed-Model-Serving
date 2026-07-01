#!/usr/bin/env python3
"""Generate production benchmark report (FCFS vs Oracle SJF vs Predicted SJF)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

import asyncio

from benchmark.compare import main as compare_main


def run(concurrency: int, n_requests: int, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    # compare.py writes markdown/json/csv to python/benchmark/results/
    asyncio.run(compare_main())
    results_dir = ROOT / "python" / "benchmark" / "results"
    latest = sorted(results_dir.glob("comparison_*.json"))[-1]
    data = json.loads(latest.read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = output_dir / f"production_bench_{stamp}.json"
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    md = output_dir / f"production_bench_{stamp}.md"
    md.write_text(_to_markdown(data), encoding="utf-8")
    return {"json": str(out), "markdown": str(md)}


def _to_markdown(data: dict) -> str:
    lines = ["# Scheduler Benchmark Report", ""]
    for algo, metrics in data.items():
        lines.append(f"## {algo}")
        for k, v in metrics.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "reports")
    args = parser.parse_args()
    paths = run(args.concurrency, args.requests, args.output)
    print(json.dumps(paths, indent=2))
