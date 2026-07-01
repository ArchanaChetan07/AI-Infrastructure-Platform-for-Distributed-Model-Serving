# Contributing

Thank you for your interest in contributing to the AI Infrastructure Platform for Distributed Model Serving.

## Development setup

```bash
git clone https://github.com/ArchanaChetan07/AI-Infrastructure-Platform-for-Distributed-Model-Serving.git
cd AI-Infrastructure-Platform-for-Distributed-Model-Serving
pip install -r requirements-dev.txt
```

On Windows, use `scripts/setup_venv.ps1` if a clean virtual environment is required.

Set `PYTHONPATH` to the repository root and `python/`:

```powershell
$env:PYTHONPATH = "$PWD;$PWD\python"
```

## Running tests

```bash
pytest tests/ -v --cov=python --cov-report=term-missing
cmake -S cpp -B cpp/build && cmake --build cpp/build --config Release
ctest --test-dir cpp/build
```

## Code style

- Python: `ruff check`, `black`, `isort`
- Go: `go fmt`, `go vet`
- C++: follow existing CMake layout and GoogleTest patterns

## Pull requests

1. Fork the repository and create a feature branch from `main`.
2. Add or update tests for behavior changes.
3. Ensure CI passes (lint, unit tests, Docker builds).
4. Open a PR with a clear description and test plan.

## Security

Do not commit secrets, API keys, or Hugging Face tokens. Report vulnerabilities via GitHub Security Advisories or open a private issue with the maintainers.
