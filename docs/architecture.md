# vLLM Intelligent Scheduler — Architecture

## System Design

Three-tier architecture optimized for latency-sensitive inference scheduling.

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────┐
│   Client    │────▶│  Go API Gateway  │────▶│ C++ Runtime SJF │────▶│ vLLM │
└─────────────┘     └──────────────────┘     └─────────────────┘     └──────┘
                            │                         │
                            ▼                         ▼
                     Prometheus               Python ML Pipeline
                     OpenTelemetry            (train / export)
```

## Go Control Plane

- REST API compatible with OpenAI `/v1/chat/completions`
- SJF priority queue with configurable aging
- Prometheus metrics at `/metrics`
- Health and readiness probes for Kubernetes
- Hot-reload configuration via YAML

## C++ Runtime

- Lock-free priority queue with SJF ordering
- SIMD-friendly feature extraction (<1 ms)
- ONNX Runtime inference path (<0.2 ms target)
- Thread-safe request lifecycle (submit, acquire, complete, cancel)

## Python ML Pipeline

- 40-dimensional prompt feature vector
- 2-layer MLP with Huber loss
- Synthetic and HuggingFace dataset support
- Export to PyTorch, ONNX, TorchScript
- Benchmark orchestration and evaluation reports

## Scheduling Algorithm

1. Extract features from incoming prompt
2. Predict output token count via MLP
3. Enqueue with priority = predicted length (shorter = higher priority)
4. Apply aging to prevent starvation
5. Dispatch to vLLM worker pool
6. Collect queue and latency metrics

## Fallback

On predictor failure or overload, the system falls back to FCFS to maintain availability.
