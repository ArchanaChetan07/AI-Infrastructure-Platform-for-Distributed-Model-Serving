"""Phase 4/6/7 audit smoke tests — run as script files."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

AUDIT = ROOT.parent / "audit_logs"


def write_log(name, content):
    p = AUDIT / name
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)


def test_smollm3_forward():
    import torch
    from vllm_port.smollm3 import SmolLM3ForCausalLM

    cfg = {
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
    lines = [
        f"shape={tuple(logits.shape)}",
        f"dtype={logits.dtype}",
        f"finite={bool(torch.isfinite(logits).all())}",
    ]
    write_log("phase4_smollm3.log", "\n".join(lines))
    print("\n".join(lines))


def test_predictor_onnx():
    import os
    import tempfile
    import numpy as np
    import torch
    from predictor.model import OutputLengthMLP
    from predictor.trainer import Trainer, TrainConfig
    from predictor.dataset import SyntheticDataset
    from predictor.export import export_onnx
    import onnxruntime as ort

    ds = SyntheticDataset(n_samples=64, seed=0)
    model = OutputLengthMLP(input_dim=ds.feature_dim, hidden_dims=[32, 16])
    cfg = TrainConfig(epochs=1, batch_size=16, lr=1e-3, device="cpu")
    Trainer(model, cfg).fit(ds)
    tmpdir = tempfile.mkdtemp()
    onnx_path = os.path.join(tmpdir, "m.onnx")
    export_onnx(model, onnx_path, input_dim=ds.feature_dim)
    x = torch.randn(4, ds.feature_dim)
    with torch.no_grad():
        pt_out = model(x).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"features": x.numpy()})[0]
    abs_err = float(np.max(np.abs(pt_out - onnx_out)))
    rel_err = float(np.max(np.abs(pt_out - onnx_out) / (np.abs(pt_out) + 1e-8)))
    lines = [f"max_abs_err={abs_err}", f"max_rel_err={rel_err}", f"onnx_path={onnx_path}"]
    write_log("phase4_predictor.log", "\n".join(lines))
    print("\n".join(lines))


def test_gateway():
    import asyncio
    import time
    import httpx
    from httpx import ASGITransport
    from scheduler.gateway import create_app

    async def main():
        app = create_app()
        transport = ASGITransport(app=app)
        lines = []
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=60.0) as client:
            for path in ["/health", "/metrics", "/scheduler/stats"]:
                t0 = time.perf_counter()
                r = await client.get(path)
                lines.append(f"{path} status={r.status_code} latency_ms={round((time.perf_counter()-t0)*1000,2)}")
                lines.append(f"body={r.text[:400]}")
            t0 = time.perf_counter()
            r = await client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hello"}]},
            )
            lines.append(f"/v1/chat/completions status={r.status_code} latency_ms={round((time.perf_counter()-t0)*1000,2)}")
            lines.append(f"body={r.text[:500]}")
        write_log("phase4_gateway.log", "\n".join(lines))
        print("\n".join(lines))

    asyncio.run(main())


def test_nope_rope():
    from vllm_port.smollm3 import SmolLM3ForCausalLM

    cfg = {
        "hidden_size": 256, "intermediate_size": 512, "num_hidden_layers": 8,
        "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 32,
        "vocab_size": 256, "max_position_embeddings": 512, "rope_theta": 136000.0,
        "rms_norm_eps": 1e-5, "tie_word_embeddings": True, "nope_layer_interval": 2,
        "hidden_act": "silu",
    }
    model = SmolLM3ForCausalLM(cfg)
    lines = []
    for i, layer in enumerate(model.model.layers):
        use_rope = getattr(layer.self_attn, "use_rope", None)
        lines.append(f"layer {i}: use_rope={use_rope}")
    tied = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    lines.append(f"tie_word_embeddings=True data_ptr_match={tied}")
    cfg2 = dict(cfg)
    cfg2["tie_word_embeddings"] = False
    model2 = SmolLM3ForCausalLM(cfg2)
    untied = model2.lm_head.weight.data_ptr() != model2.model.embed_tokens.weight.data_ptr()
    lines.append(f"tie_word_embeddings=False data_ptr_differ={untied}")
    write_log("phase7_numerical.log", "\n".join(lines))
    print("\n".join(lines))


def test_concurrency():
    import asyncio
    import time
    import httpx
    from httpx import ASGITransport
    from scheduler.gateway import create_app

    async def main():
        app = create_app()
        transport = ASGITransport(app=app)
        lines = []
        async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=60.0) as c:
            async def one(i):
                try:
                    r = await c.post(
                        "/v1/chat/completions",
                        json={"model": "m", "messages": [{"role": "user", "content": str(i)}]},
                    )
                    return r.status_code
                except Exception as e:
                    return str(e)

            t0 = time.perf_counter()
            results = await asyncio.gather(*[one(i) for i in range(50)])
            lines.append(f"50_concurrent_errors={sum(1 for x in results if x != 200)}")
            lines.append(f"sample_statuses={results[:10]}")
            lines.append(f"elapsed_s={round(time.perf_counter()-t0, 3)}")
            r = await c.get("/scheduler/stats")
            lines.append(f"stats={r.text[:400]}")
        write_log("phase6_concurrency.log", "\n".join(lines))
        print("\n".join(lines))

    asyncio.run(main())


if __name__ == "__main__":
    import traceback

    for fn in [test_smollm3_forward, test_predictor_onnx, test_gateway, test_nope_rope, test_concurrency]:
        name = fn.__name__
        try:
            print(f"=== {name} ===")
            fn()
            print(f"{name} OK")
        except Exception:
            err = traceback.format_exc()
            write_log(f"{name}.err.log", err)
            print(err)
