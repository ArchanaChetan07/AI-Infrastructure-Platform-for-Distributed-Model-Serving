# Deployment Guide

## Local Development

```bash
# Terminal 1 — vLLM backend
docker run --gpus all -p 8002:8000 vllm-smollm3:latest

# Terminal 2 — Go gateway
cd go && go run ./cmd/gateway

# Terminal 3 — verify
curl http://localhost:8080/health
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"HuggingFaceTB/SmolLM3-3B","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

## Docker Compose

```bash
docker compose -f docker/docker-compose.yml up -d
```

Services:
- `vllm` — inference backend (port 8002)
- `gateway` — intelligent scheduler (port 8080)
- `prometheus` — metrics (port 9091)

## Kubernetes

```bash
kubectl create namespace vllm-scheduler
kubectl apply -f kubernetes/
```

Includes Deployment, Service, Ingress, HPA, and ConfigMap.

## Configuration

| File | Purpose |
|------|---------|
| `configs/gateway.yaml` | Go gateway and vLLM connection |
| `configs/training.yaml` | ML training hyperparameters |
| `configs/scheduler.yaml` | Legacy Python scheduler settings |

Environment overrides:
- `CONFIG_PATH` — gateway config file
- `VLLM_URL` — vLLM backend URL

## Model Artifacts

Place trained models in `shared/models/`:
- `output_length_mlp.pt`
- `output_length.onnx`
- `output_length.ts`
- `metadata.json`

Run `make train` to generate these automatically.
