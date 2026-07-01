# API Reference

## Go Gateway

Base URL: `http://localhost:8080`

### Health

```
GET /health
GET /ready
```

### Metrics

```
GET /metrics
```

Prometheus format. Key metrics:
- `gateway_requests_total{status}`
- `gateway_queue_depth`

### Scheduler Stats

```
GET /scheduler/stats
```

### Chat Completions

```
POST /v1/chat/completions
```

OpenAI-compatible. Requests are scheduled via SJF before forwarding to vLLM.

**Request:**
```json
{
  "model": "HuggingFaceTB/SmolLM3-3B",
  "messages": [{"role": "user", "content": "Explain attention."}],
  "max_tokens": 128,
  "stream": false
}
```

**Streaming:** Set `"stream": true` for SSE responses.

## gRPC (proto definitions)

See `go/proto/scheduler.proto` and `go/proto/predictor.proto`.

Services:
- `SchedulerService` — submit, acquire, complete, cancel, stats
- `PredictorService` — predict, predict batch

## C++ Runtime

Binary: `cpp/build/schedulerd`

Library: `libscheduler_runtime.a`

## Python ML API

```python
from predictor import OutputLengthPredictor, FeatureExtractor

predictor = OutputLengthPredictor.from_checkpoint("shared/models/output_length_mlp.pt")
result = predictor.predict("Write a Python function to sort a list.")
print(result.predicted_tokens)
```
