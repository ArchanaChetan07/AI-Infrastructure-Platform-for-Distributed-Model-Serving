"""Generate benchmark plots."""

from __future__ import annotations

from pathlib import Path
from typing import List


def generate_plots(results: List[dict], output_dir: Path, timestamp: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots")
        return

    schedulers = sorted(set(r["scheduler"] for r in results))

    # Latency plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for sched in schedulers:
        subset = [r for r in results if r["scheduler"] == sched]
        subset.sort(key=lambda x: x["concurrency"])
        axes[0].plot(
            [r["concurrency"] for r in subset],
            [r["e2e_p50_ms"] for r in subset],
            marker="o",
            label=f"{sched} p50",
        )
        axes[0].plot(
            [r["concurrency"] for r in subset],
            [r["e2e_p99_ms"] for r in subset],
            marker="s",
            linestyle="--",
            label=f"{sched} p99",
        )

    axes[0].set_xlabel("Concurrency")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("End-to-End Latency")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for sched in schedulers:
        subset = [r for r in results if r["scheduler"] == sched]
        subset.sort(key=lambda x: x["concurrency"])
        axes[1].plot(
            [r["concurrency"] for r in subset],
            [r["throughput_rps"] for r in subset],
            marker="o",
            label=sched,
        )

    axes[1].set_xlabel("Concurrency")
    axes[1].set_ylabel("Requests/sec")
    axes[1].set_title("Throughput")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f"latency_throughput_{timestamp}.png", dpi=150)
    plt.close(fig)

    # Queue wait comparison
    fig2, ax = plt.subplots(figsize=(10, 5))
    for sched in schedulers:
        subset = [r for r in results if r["scheduler"] == sched]
        subset.sort(key=lambda x: x["concurrency"])
        ax.plot(
            [r["concurrency"] for r in subset],
            [r["queue_wait_p99_ms"] for r in subset],
            marker="o",
            label=sched,
        )
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Queue Wait p99 (ms)")
    ax.set_title("Queue Wait Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig2.savefig(output_dir / f"queue_wait_{timestamp}.png", dpi=150)
    plt.close(fig2)

    print(f"Plots saved to {output_dir}")
