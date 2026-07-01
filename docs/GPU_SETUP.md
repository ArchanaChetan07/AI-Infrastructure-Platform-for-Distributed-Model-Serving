# GPU & HuggingFace Integration Setup

This guide enables the **7 skipped GPU/vLLM tests** in `tests/test_smollm3.py`.

## Prerequisites

- NVIDIA GPU with CUDA (verify: `nvidia-smi`)
- Python 3.10+ with PyTorch CUDA build
- HuggingFace account and access token
- Optional: vLLM installed (`pip install vllm`)

## 1. Set HF_TOKEN

```bash
# Linux/macOS
export HF_TOKEN="hf_xxxxxxxxxxxxxxxx"

# Windows PowerShell
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxx"
```

Create a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with **read** access.

## 2. Verify GPU

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. Run GPU integration tests

```bash
export PYTHONPATH=.:python
export HF_TOKEN=hf_...
pytest tests/test_smollm3.py -m "accuracy or integration or benchmark" -v
```

Tests are **skipped** (not failed) when `HF_TOKEN` is unset — CI remains green.

## 4. Run full vLLM Docker stack

```bash
docker build -t vllm-smollm3:latest -f docker/Dockerfile .
docker run --gpus all -e HF_TOKEN=$HF_TOKEN -p 8002:8000 vllm-smollm3:latest
```

## 5. Scheduler + gateway against vLLM

```bash
# Terminal 1: vLLM on :8002
# Terminal 2:
export PYTHONPATH=.:python
uvicorn scheduler.gateway:app --host 0.0.0.0 --port 8080
curl http://localhost:8080/ready
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"HuggingFaceTB/SmolLM3-3B","messages":[{"role":"user","content":"Hello"}]}'
```

## CI configuration

Add repository secret `HF_TOKEN` in GitHub → Settings → Secrets. The optional `gpu` job in `.github/workflows/ci.yml` runs when the secret is present.
