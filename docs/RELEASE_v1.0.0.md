# v1.0.0 — Production Release

Production-certified ML scheduling and serving platform for vLLM inference.

## Highlights

- **Predicted SJF scheduler** with priority aging — reduces tail latency under load
- **FastAPI gateway** — OpenAI-compatible API, streaming, Prometheus metrics, health probes
- **Multi-language runtime** — Python ML pipeline, C++ scheduler core, Go control plane
- **92 automated tests** — unit, integration, stress, and certification suites
- **Docker & Kubernetes** — production images and deployment manifests
- **91.58% test coverage on core packages (scheduler, predictor)**

## Quick start

```bash
pip install -r requirements-dev.txt
make train && make gateway
pytest tests/ -v
```

See [README](https://github.com/ArchanaChetan07/AI-Infrastructure-Platform-for-Distributed-Model-Serving/blob/main/README.md) and [GPU Setup](https://github.com/ArchanaChetan07/AI-Infrastructure-Platform-for-Distributed-Model-Serving/blob/main/docs/GPU_SETUP.md).

## Certification

Full 23-section report: [docs/PRODUCTION_CERTIFICATION_REPORT.md](https://github.com/ArchanaChetan07/AI-Infrastructure-Platform-for-Distributed-Model-Serving/blob/main/docs/PRODUCTION_CERTIFICATION_REPORT.md)

**Production readiness: 9.0 / 10 (internal audit)**

## External requirements

- `HF_TOKEN` for GPU/SmolLM3 integration tests
- CUDA GPU for full vLLM end-to-end validation
