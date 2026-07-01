#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=python:.
python python/benchmark/compare.py "$@"
python python/evaluation/generate_report.py
echo "Benchmark complete."
