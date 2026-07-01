#!/usr/bin/env bash
set -euo pipefail
echo "==> Ruff"
ruff check predictor/ scheduler/ benchmark/ training/ tests/
echo "==> Black"
black --check predictor/ scheduler/ benchmark/ training/ tests/
echo "==> isort"
isort --check-only predictor/ scheduler/ benchmark/ training/ tests/
echo "==> mypy"
mypy predictor/ scheduler/ --ignore-missing-imports
echo "==> pytest"
pytest tests/ -m "unit or integration" -v --cov=predictor --cov=scheduler --cov-report=term-missing
echo "==> Docker build"
docker build -t vllm-sjf-scheduler:latest -f docker/Dockerfile.scheduler .
echo "CI passed."
