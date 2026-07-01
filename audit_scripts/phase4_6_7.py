#!/usr/bin/env python3
"""Phases 4, 6, 7 smoke / correctness scripts."""
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT.parent / "audit_logs"
AUDIT.mkdir(parents=True, exist_ok=True)
VENV_PY = ROOT / ".venv_audit" / "Scripts" / "python.exe"
PY = str(VENV_PY if VENV_PY.exists() else sys.executable)
ENV = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + str(ROOT / "python")}


def log(name, text):
    p = AUDIT / name
    with open(p, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text[:500])


def phase4_smollm3():
    log("phase4_smollm3.log", f"=== Phase 4a SmolLM3 forward ===\n{datetime.now().isoformat()}")
    script = r'''
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from vllm_port.smollm3 import SmolLM3ForCausalLM

cfg = {
    "architectures": ["SmolLM3ForCausalLM"],
    "hidden_size": 256, "intermediate_size": 512, "num_hidden_layers": 4,
    "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 32,
    "vocab_size": 256, "max_position_embeddings": 512, "rope_theta": 136000.0,
    "rms_norm_eps": 1e-5, "tie_word_embeddings": True, "nope_layer_interval": 2,
    "hidden_act": "silu",
}
model = SmolLM3ForCausalLM(cfg)
model.eval()
ids = torch.randint(0, 256, (2, 16))
with torch.no_grad():
    out = model(ids)
logits = out.logits if hasattr(out, "logits") else out[0]
print("shape", tuple(logits.shape))
print("dtype", logits.dtype)
print("finite", bool(torch.isfinite(logits).all()))
'''
    r = subprocess.run([PY, "-c", script], cwd=ROOT, env=ENV, capture_output=True, text=True)
    log("phase4_smollm3.log", f"exit={r.returncode}\n{r.stdout}\n{r.stderr}")


def phase4_predictor():
    log("phase4_predictor.log", f"=== Phase 4b predictor train+onnx ===\n{datetime.now().isoformat()}")
    script = r'''
import os, sys, tempfile
import numpy as np
import torch
sys.path.insert(0, "python")
from predictor.model import OutputLengthMLP
from predictor.trainer import Trainer, TrainConfig
from predictor.dataset import SyntheticDataset
from predictor.export import export_onnx
import onnxruntime as ort

torch.manual_seed(0)
ds = SyntheticDataset(n_samples=64, seed=0)
model = OutputLengthMLP(input_dim=ds.feature_dim, hidden_dims=[32, 16])
cfg = TrainConfig(epochs=1, batch_size=16, lr=1e-3, device="cpu")
trainer = Trainer(model, cfg)
trainer.fit(ds)
tmpdir = tempfile.mkdtemp()
pt = os.path.join(tmpdir, "m.pt")
torch.save(model.state_dict(), pt)
onnx_path = os.path.join(tmpdir, "m.onnx")
export_onnx(model, onnx_path, input_dim=ds.feature_dim)
x = torch.randn(4, ds.feature_dim)
with torch.no_grad():
    pt_out = model(x).numpy()
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
onnx_out = sess.run(None, {"features": x.numpy()})[0]
abs_err = float(np.max(np.abs(pt_out - onnx_out)))
rel_err = float(np.max(np.abs(pt_out - onnx_out) / (np.abs(pt_out) + 1e-8)))
print("max_abs_err", abs_err)
print("max_rel_err", rel_err)
print("onnx_path", onnx_path)
'''
    r = subprocess.run([PY, "-c", script], cwd=ROOT, env=ENV, capture_output=True, text=True)
    log("phase4_predictor.log", f"exit={r.returncode}\n{r.stdout}\n{r.stderr}")


def phase4_gateway():
    log("phase4_gateway.log", f"=== Phase 4c gateway smoke ===\n{datetime.now().isoformat()}")
    script = r'''
import asyncio, json, time
import httpx
from httpx import ASGITransport
import sys
sys.path.insert(0, "python")
from scheduler.gateway import create_app

async def mock_handler(request):
    if request.url.path == "/health":
        return httpx.Response(200, json={"status": "ok"})
    if request.url.path == "/v1/chat/completions":
        body = {
            "id": "stub", "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        return httpx.Response(200, json=body)
    return httpx.Response(404)

async def main():
    app = create_app()
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for path in ["/health", "/metrics", "/scheduler/stats"]:
            t0 = time.perf_counter()
            r = await client.get(path)
            print(path, r.status_code, round((time.perf_counter()-t0)*1000, 2), "ms")
            print("body", r.text[:300])
        t0 = time.perf_counter()
        r = await client.post("/v1/chat/completions", json={"model":"m","messages":[{"role":"user","content":"x"}]})
        print("/v1/chat/completions", r.status_code, round((time.perf_counter()-t0)*1000, 2), "ms")
        print("body", r.text[:500])
asyncio.run(main())
'''
    r = subprocess.run([PY, "-c", script], cwd=ROOT, env=ENV, capture_output=True, text=True)
    log("phase4_gateway.log", f"exit={r.returncode}\n{r.stdout}\n{r.stderr}")


def phase7_nope_rope():
    log("phase7_numerical.log", f"=== Phase 7 NoPE/RoPE + tied embeddings ===\n{datetime.now().isoformat()}")
    script = r'''
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from vllm_port.smollm3 import SmolLM3ForCausalLM, SmolLM3Model

cfg = {
    "hidden_size": 256, "intermediate_size": 512, "num_hidden_layers": 8,
    "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 32,
    "vocab_size": 256, "max_position_embeddings": 512, "rope_theta": 136000.0,
    "rms_norm_eps": 1e-5, "tie_word_embeddings": True, "nope_layer_interval": 2,
    "hidden_act": "silu",
}
model = SmolLM3ForCausalLM(cfg)
for i, layer in enumerate(model.model.layers):
    use_rope = getattr(layer.self_attn, "use_rope", None)
    print(f"layer {i}: use_rope={use_rope}")
# tied
tied = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
print("tie_word_embeddings=True data_ptr_match", tied)
cfg2 = dict(cfg)
cfg2["tie_word_embeddings"] = False
model2 = SmolLM3ForCausalLM(cfg2)
untied = model2.lm_head.weight.data_ptr() != model2.model.embed_tokens.weight.data_ptr()
print("tie_word_embeddings=False data_ptr_differ", untied)
'''
    r = subprocess.run([PY, "-c", script], cwd=ROOT, env=ENV, capture_output=True, text=True)
    log("phase7_numerical.log", f"exit={r.returncode}\n{r.stdout}\n{r.stderr}")


def phase6_concurrency():
    log("phase6_concurrency.log", f"=== Phase 6 concurrency ===\n{datetime.now().isoformat()}")
    script = r'''
import asyncio, time
import httpx
from httpx import ASGITransport
import sys
sys.path.insert(0, "python")
from scheduler.gateway import create_app

call_count = 0
class FlakyTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request):
        nonlocal call_count
        call_count += 1
        if request.url.path == "/health":
            return httpx.Response(200, json={"status":"ok"})
        if request.url.path == "/v1/chat/completions":
            await asyncio.sleep(0.05)
            return httpx.Response(200, json={
                "choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
            })
        return httpx.Response(404)

async def main():
    app = create_app()
    # Patch vllm client by monkeypatching gateway after startup - use direct ASGI only
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=30.0) as c:
        async def one(i):
            try:
                r = await c.post("/v1/chat/completions", json={"model":"m","messages":[{"role":"user","content":str(i)}]})
                return r.status_code
            except Exception as e:
                return str(e)
        t0 = time.perf_counter()
        results = await asyncio.gather(*[one(i) for i in range(50)])
        print("50 concurrent statuses sample", results[:10], "errors", sum(1 for x in results if x != 200))
        print("elapsed", round(time.perf_counter()-t0, 3))
        r = await c.get("/scheduler/stats")
        print("stats", r.text[:400])
asyncio.run(main())
'''
    r = subprocess.run([PY, "-c", script], cwd=ROOT, env=ENV, capture_output=True, text=True)
    log("phase6_concurrency.log", f"exit={r.returncode}\n{r.stdout}\n{r.stderr}")


if __name__ == "__main__":
    phase4_smollm3()
    phase4_predictor()
    phase4_gateway()
    phase7_nope_rope()
    phase6_concurrency()
