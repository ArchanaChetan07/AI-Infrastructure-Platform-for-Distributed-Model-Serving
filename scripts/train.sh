#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=python:.
python python/training/prepare_dataset.py "$@"
python python/training/train.py "$@"
python python/training/evaluate.py "$@"
echo "Training complete."
