# Changelog

All notable changes to this project are documented in this file.

## [1.0.0] - 2026-06-30

### Added

- ML-predicted Shortest-Job-First scheduler with priority aging
- FastAPI gateway with OpenAI-compatible chat API, streaming, and Prometheus metrics
- Output-length predictor (PyTorch, ONNX, TorchScript export)
- C++20 scheduler runtime with GoogleTest suite
- Go API gateway control plane
- Docker images for scheduler, gateway, and ML pipeline
- Kubernetes manifests and Prometheus monitoring config
- Native SmolLM3 vLLM model port (`vllm_port/`)
- Comprehensive test suite (92 tests) including stress and certification tests
- CI/CD pipeline (Python, Go, C++, Docker, benchmarks)
- Production certification report and GPU setup documentation

### Fixed

- Gateway connect/request timeouts and backend error handling (502/504)
- Windows multiprocessing pickling in `parallel_extract`
- Go module sum and package naming for Docker builds
- Python 3.11 asyncio compatibility in priority queue

[1.0.0]: https://github.com/ArchanaChetan07/AI-Infrastructure-Platform-for-Distributed-Model-Serving/releases/tag/v1.0.0
