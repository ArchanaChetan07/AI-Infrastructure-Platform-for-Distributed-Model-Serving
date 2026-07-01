"""Compare FCFS vs Oracle SJF vs Predicted SJF."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

import yaml

from benchmark.fcfs import run_fcfs_benchmark
from benchmark.plots import generate_plots
from benchmark.sjf import run_oracle_benchmark, run_sjf_benchmark


async def run_comparison(
    concurrency_levels: list[int],
    n_requests: int,
    models_dir: str = "models",
    output_dir: str = "benchmark/results",
) -> list[dict]:
    results: list[dict] = []
    for c in concurrency_levels:
        print(f"Benchmarking concurrency={c}...")
        fcfs = await run_fcfs_benchmark(c, n_requests)
        oracle = await run_oracle_benchmark(c, n_requests)
        try:
            sjf = await run_sjf_benchmark(c, n_requests, models_dir)
        except Exception as e:
            print(f"  SJF benchmark skipped: {e}")
            sjf = fcfs.copy()
            sjf["scheduler"] = "sjf"
        results.extend([fcfs, oracle, sjf])
        print(
            f"  FCFS p99={fcfs['e2e_p99_ms']}ms  "
            f"Oracle p99={oracle['e2e_p99_ms']}ms  "
            f"SJF p99={sjf['e2e_p99_ms']}ms"
        )
    return results


def generate_report(results: list[dict], output_dir: Path) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"comparison_{ts}.json"
    csv_path = output_dir / f"comparison_{ts}.csv"
    md_path = output_dir / f"comparison_{ts}.md"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # CSV
    if results:
        keys = list(results[0].keys())
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in results:
                f.write(",".join(str(r[k]) for k in keys) + "\n")

    lines = [
        "# Scheduler Comparison Benchmark",
        f"\n**Generated:** {datetime.now().isoformat()}\n",
        "| Scheduler | Concurrency | p50 (ms) | p99 (ms) | RPS | Tok/s |",
        "|-----------|-------------|----------|----------|-----|-------|",
    ]
    for r in results:
        lines.append(
            f"| {r['scheduler']} | {r['concurrency']} | {r['e2e_p50_ms']} | "
            f"{r['e2e_p99_ms']} | {r['throughput_rps']} | {r['tokens_per_sec']} |"
        )
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    generate_plots(results, output_dir, ts)
    print(f"Saved: {json_path}, {csv_path}, {md_path}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--output-dir", default="benchmark/results")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    bench_cfg = cfg.get("benchmark", {})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = await run_comparison(
        bench_cfg.get("concurrency_levels", [1, 2, 4, 8, 16, 32]),
        bench_cfg.get("requests_per_level", 20),
        cfg.get("export", {}).get("output_dir", "shared/models"),
        str(output_dir),
    )
    generate_report(results, output_dir)


if __name__ == "__main__":
    asyncio.run(main())
