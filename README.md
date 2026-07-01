# AI Infrastructure Platform for Distributed Model Serving

Production-grade ML scheduling and serving infrastructure for [vLLM](https://github.com/vllm-project/vllm). Predicts output length before decoding and schedules inference requests using Shortest-Job-First (SJF) with aging to reduce tail latency under load.

[![CI](https://github.com/ArchanaChetan07/AI-Infrastructure-Platform-for-Distributed-Model-Serving/actions/workflows/ci.yml/badge.svg)](https://github.com/ArchanaChetan07/AI-Infrastructure-Platform-for-Distributed-Model-Serving/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Go 1.22+](https://img.shields.io/badge/go-1.22+-00ADD8?logo=go&logoColor=white)](https://go.dev/)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Why this platform?

Default vLLM scheduling is first-come-first-served (FCFS). Under contention, a single long-running request can block the queue and inflate **p99 latency** for all clients.

This platform replaces naive FCFS with **ML-predicted SJF**:

```
Client → API Gateway → Scheduler → Feature Extraction → ML Predictor
      → Priority Queue → vLLM → GPU → Prometheus / OpenTelemetry
```

| Layer | Stack | Role |
|-------|-------|------|
| **Gateway** | Python (FastAPI) + Go | OpenAI-compatible API, health probes, metrics |
| **Scheduler** | Python + C++20 | FCFS, Oracle SJF, Predicted SJF with aging |
| **Predictor** | PyTorch / ONNX / TorchScript | Output-length estimation from prompt features |
| **Inference** | vLLM | PagedAttention, continuous batching, GPU execution |

---

## Features

- **Multiple scheduling policies** — FCFS, Oracle SJF, Predicted SJF with priority aging
- **Sub-millisecond overhead targets** — feature extraction, prediction, and queue ops tuned for production
- **Observability** — Prometheus metrics, structured logging, `/health`, `/ready`, `/live`
- **Resilience** — connect/request timeouts, graceful backend failure handling, cancellation
- **Multi-language runtime** — Python ML pipeline, C++ scheduler core, Go control-plane gateway
- **Docker & Kubernetes** — production images, compose stacks, K8s manifests
- **Native SmolLM3 vLLM port** — `HuggingFaceTB/SmolLM3-3B` integration in `vllm_port/`

---

## Quick start

### Prerequisites

- Python 3.10+
- Go 1.22+ (optional; Docker image available)
- CMake 3.20+ and a C++20 compiler
- Docker with NVIDIA Container Toolkit (for GPU inference)
- Hugging Face token (`HF_TOKEN`) for gated models — see [docs/GPU_SETUP.md](docs/GPU_SETUP.md)

### Install

```bash
git clone https://github.com/ArchanaChetan07/AI-Infrastructure-Platform-for-Distributed-Model-Serving.git
cd AI-Infrastructure-Platform-for-Distributed-Model-Serving
pip install -r requirements-dev.txt
```

```powershell
# Windows
.\scripts\setup_venv.ps1
$env:PYTHONPATH = "$PWD;$PWD\python"
```

### Train the predictor

```bash
make train
```

Artifacts are exported to `shared/models/` (PyTorch, ONNX, TorchScript, metadata).

### Build C++ scheduler

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build --config Release
ctest --test-dir cpp/build
```

### Start the gateway

```bash
make gateway
# http://localhost:8080 → proxies vLLM at http://localhost:8002
```

### Run benchmarks

```bash
make benchmark-scheduler
```

Compares FCFS, Oracle SJF, and Predicted SJF at concurrency 1–128.

### Full stack (Docker)

```bash
docker compose -f docker/docker-compose.yml up -d
```

---

## Repository layout

```
├── python/           # ML pipeline, scheduler, FastAPI gateway
├── cpp/              # High-performance C++ scheduler runtime
├── go/               # Go API gateway (alternative control plane)
├── vllm_port/        # Native SmolLM3 vLLM model integration
├── shared/models/    # Exported predictor artifacts
├── configs/          # Runtime and training configuration
├── docker/           # Multi-stage production images
├── kubernetes/       # Kubernetes manifests
├── monitoring/       # Prometheus & Grafana
├── tests/            # Python, integration, stress, and certification tests
├── scripts/          # Setup, benchmark report generation
└── docs/             # Architecture, deployment, GPU setup, certification
```

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Architecture](docs/architecture.md) | System design and data flow |
| [Deployment](docs/deployment.md) | Production deployment guide |
| [GPU Setup](docs/GPU_SETUP.md) | CUDA, vLLM, and `HF_TOKEN` configuration |
| [API Reference](docs/api.md) | Gateway and scheduler endpoints |
| [Benchmark Guide](docs/benchmark.md) | Performance evaluation methodology |
| [Production Certification](docs/PRODUCTION_CERTIFICATION.md) | Certification checklist and results |

---

## Performance targets

| Component | Target |
|-----------|--------|
| Feature extraction | &lt; 1 ms |
| Prediction | &lt; 0.2 ms |
| Scheduling overhead | &lt; 0.5 ms |
| Overall overhead | &lt; 1% of end-to-end latency |
| Concurrent requests | 1000+ |

---

## Testing

```bash
pytest tests/ -v --cov=python --cov-report=term-missing
ruff check python tests vllm_port
```

GPU integration tests require `HF_TOKEN` and a CUDA-capable GPU; they skip gracefully otherwise.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security reports: [SECURITY.md](SECURITY.md).

---

## License

[MIT](LICENSE) — Copyright (c) 2026 Archana Chetan
