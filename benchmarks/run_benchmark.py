#!/usr/bin/env python3
"""
Benchmark — SmolLM3 Native vLLM Port vs Transformers Backend (Project 2A)
==========================================================================
Measures throughput (tokens/sec) for the native port vs the Transformers
fallback at batch sizes 1, 8, 32.

Usage:
  # Benchmark native port:
  python benchmarks/run_benchmark.py --base-url http://localhost:8000

  # With results saved:
  python benchmarks/run_benchmark.py --base-url http://localhost:8000 --output-dir results/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import httpx

PROMPTS = [
    "Explain what a transformer neural network is.",
    "Write a Python function to compute Fibonacci numbers.",
    "What are the main advantages of using vLLM for LLM serving?",
    "Describe the attention mechanism in machine learning.",
    "How does grouped query attention reduce memory usage?",
]


@dataclass
class BenchResult:
    concurrency: int
    n_requests: int
    n_success: int
    total_tokens: int
    elapsed_sec: float
    throughput_rps: float
    tokens_per_sec: float
    e2e_p50_ms: float
    e2e_p99_ms: float
    error_rate: float


async def run_request(
    client: httpx.AsyncClient, model: str, prompt: str, max_tokens: int
) -> tuple[bool, float, int]:
    t0 = time.monotonic()
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": False,
                "temperature": 0.0,
            },
            timeout=120.0,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        if resp.status_code == 200:
            data = resp.json()
            tokens = data.get("usage", {}).get("completion_tokens", max_tokens)
            return True, elapsed_ms, tokens
        return False, elapsed_ms, 0
    except Exception as e:
        elapsed_ms = (time.monotonic() - t0) * 1000
        print(f"  Request error: {e}")
        return False, elapsed_ms, 0


async def run_level(
    base_url: str, model: str, concurrency: int,
    n_requests: int, max_tokens: int
) -> BenchResult:
    sem = asyncio.Semaphore(concurrency)
    results = []

    async with httpx.AsyncClient(base_url=base_url) as client:
        async def bounded(i: int):
            async with sem:
                prompt = PROMPTS[i % len(PROMPTS)]
                return await run_request(client, model, prompt, max_tokens)

        t_start = time.monotonic()
        results = await asyncio.gather(*[bounded(i) for i in range(n_requests)])
        elapsed = time.monotonic() - t_start

    successes = [r for r in results if r[0]]
    e2es = sorted(r[1] for r in results)
    total_tokens = sum(r[2] for r in successes)

    def pct(data, p):
        return data[max(0, int(len(data) * p) - 1)] if data else 0.0

    return BenchResult(
        concurrency=concurrency,
        n_requests=n_requests,
        n_success=len(successes),
        total_tokens=total_tokens,
        elapsed_sec=round(elapsed, 2),
        throughput_rps=round(len(successes) / elapsed, 2),
        tokens_per_sec=round(total_tokens / elapsed, 1),
        e2e_p50_ms=round(pct(e2es, 0.50), 1),
        e2e_p99_ms=round(pct(e2es, 0.99), 1),
        error_rate=round((n_requests - len(successes)) / n_requests * 100, 1),
    )


def render_table(results: List[BenchResult]) -> str:
    lines = [
        "| Concurrency | Requests | Success | RPS | Tok/s | E2E p50 | E2E p99 | Err |",
        "|-------------|----------|---------|-----|------------|--------------|--------------|--------|",
    ]
    for r in results:
        lines.append(
            f"| {r.concurrency} | {r.n_requests} | {r.n_success} "
            f"| {r.throughput_rps} | {r.tokens_per_sec} "
            f"| {r.e2e_p50_ms} | {r.e2e_p99_ms} | {r.error_rate}% |"
        )
    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM3-3B")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--output-dir", default="benchmarks/results")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\nvLLM SmolLM3 Port Benchmark")
    print(f"Endpoint: {args.base_url}  Model: {args.model}")
    print("=" * 60)

    all_results = []
    for c in args.concurrency:
        print(f"Running concurrency={c}, n={args.requests}...")
        r = await run_level(args.base_url, args.model, c, args.requests, args.max_tokens)
        all_results.append(r)
        print(f"  RPS={r.throughput_rps}  Tok/s={r.tokens_per_sec}  "
              f"p99={r.e2e_p99_ms}ms  errors={r.error_rate}%")

    table = render_table(all_results)
    print(f"\n{table}")

    # Save
    json_path = Path(args.output_dir) / f"smollm3_bench_{ts}.json"
    md_path = Path(args.output_dir) / f"smollm3_bench_{ts}.md"

    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)

    md = f"# SmolLM3 Native Port Benchmark\n\n**{datetime.now().isoformat()}**\n\n{table}\n"
    with open(md_path, "w") as f:
        f.write(md)

    print(f"\nSaved: {json_path}\n       {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
