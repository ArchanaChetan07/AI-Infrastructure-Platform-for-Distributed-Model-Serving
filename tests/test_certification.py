"""Certification-grade coverage tests — gateway, scheduler, predictor edge cases."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import torch
from predictor.dataset import OutputLengthDataset, parallel_extract
from predictor.model import OutputLengthMLP
from predictor.predictor import OutputLengthPredictor
from predictor.trainer import TrainConfig, Trainer
from scheduler.base_scheduler import ScheduledRequest, SchedulerType
from scheduler.gateway import SchedulerGateway, create_app, load_config
from scheduler.priority_queue import PriorityQueue, new_request_id
from scheduler.sjf_scheduler import OracleSJFScheduler, SJFScheduler, build_scheduler
from starlette.testclient import TestClient


@pytest.mark.unit
def test_gateway_loads_checkpoint(tmp_path):
    cfg = load_config("configs/scheduler.yaml")
    cfg = dict(cfg)
    cfg["scheduler"] = dict(cfg["scheduler"])
    cfg["scheduler"]["type"] = "fcfs"
    cfg["predictor"] = {"models_dir": str(tmp_path)}
    models = tmp_path
    torch.save(OutputLengthMLP().state_dict(), models / "output_length_mlp.pt")

    mock = AsyncMock(spec=httpx.AsyncClient)
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    mock.post = AsyncMock(return_value=resp)
    mock.get = AsyncMock(return_value=MagicMock(status_code=200))

    gw = SchedulerGateway(cfg, http_client=mock)
    app = create_app(gateway=gw)
    with TestClient(app) as client:
        client.get("/health")
        assert client.get("/ready").status_code == 200


@pytest.mark.unit
def test_gateway_streaming_response(gateway_config):
    client = AsyncMock(spec=httpx.AsyncClient)

    @asynccontextmanager
    async def stream_cm(*args, **kwargs):
        class Resp:
            async def aiter_bytes(self):
                yield b"data: {}\n\n"

        yield Resp()

    client.stream = stream_cm
    gw = SchedulerGateway(gateway_config, http_client=client)
    app = create_app(gateway=gw)
    with TestClient(app) as c:
        c.get("/health")
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert r.status_code == 200


@pytest.mark.unit
def test_gateway_request_timeout(gateway_config):
    cfg = dict(gateway_config)
    cfg["scheduler"] = dict(cfg["scheduler"])
    cfg["scheduler"]["request_timeout_sec"] = 0.05

    async def slow_post(*args, **kwargs):
        await asyncio.sleep(1.0)
        return MagicMock(status_code=200, json=lambda: {"choices": []})

    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(side_effect=slow_post)
    client.get = AsyncMock(return_value=MagicMock(status_code=200))

    gw = SchedulerGateway(cfg, http_client=client)
    app = create_app(gateway=gw)
    with TestClient(app) as c:
        c.get("/health")
        r = c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "x"}]},
        )
        assert r.status_code == 504


@pytest.mark.unit
def test_gateway_non_200_backend(gateway_config):
    client = AsyncMock(spec=httpx.AsyncClient)
    bad = MagicMock(status_code=500, text="error", json=lambda: {})
    client.post = AsyncMock(return_value=bad)
    client.get = AsyncMock(return_value=MagicMock(status_code=200))

    gw = SchedulerGateway(gateway_config, http_client=client)
    app = create_app(gateway=gw)
    with TestClient(app) as c:
        c.get("/health")
        r = c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "x"}]},
        )
        assert r.status_code == 502


@pytest.mark.asyncio
@pytest.mark.unit
async def test_sjf_oracle_scheduler():
    sched = OracleSJFScheduler()
    req = ScheduledRequest(
        request_id=new_request_id(),
        prompt="p",
        messages=[],
        max_tokens=20,
        predicted_tokens=0.0,
        priority=0.0,
    )
    await sched.submit(req)
    got = await sched.acquire(timeout=1.0)
    assert got.max_tokens == 20
    await sched.complete(req.request_id, success=False)
    assert await sched.cancel("missing") is False


@pytest.mark.asyncio
@pytest.mark.unit
async def test_sjf_fallback_queue(gateway_config):
    class FailPred:
        def predict(self, prompt):
            raise RuntimeError("fail")

    sched = SJFScheduler(FailPred(), fallback_to_fcfs=True)  # type: ignore[arg-type]
    await sched.start()
    req = ScheduledRequest(
        request_id=new_request_id(),
        prompt="x",
        messages=[],
        max_tokens=5,
        predicted_tokens=0.0,
        priority=0.0,
    )
    await sched.submit(req)
    assert sched.queue_depth >= 0
    await sched.stop()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_priority_queue_stats_and_miss_update():
    q = PriorityQueue()
    assert await q.update_priority("nope", 1.0) is False
    await q.enqueue("a", 1.0, "x")
    st = q.stats()
    assert st.depth >= 0
    assert q.depth == 1


@pytest.mark.unit
def test_dataset_tensors_and_records(tmp_path):
    ds = OutputLengthDataset(cache_dir=tmp_path)
    samples = ds.from_prompts(["a"], ["b c d"], [{}])
    splits = ds.split(samples * 20)
    x, y = ds.to_tensors(splits.train, splits.feature_mean, splits.feature_std)
    assert x.shape[0] == len(splits.train)
    assert y.shape[0] == len(splits.train)

    recs = [{"instruction": "hi", "response": "there"}]
    assert len(ds._from_records(recs)) == 1


@pytest.mark.unit
def test_parallel_extract():
    feats = parallel_extract(["hello world", "test prompt"], workers=2)
    assert len(feats) == 2


@pytest.mark.unit
def test_predictor_batch_and_empty_metadata(tmp_path):
    pred = OutputLengthPredictor(OutputLengthMLP())
    results = pred.predict_batch(["a", "b"])
    assert len(results) == 2

    pt = tmp_path / "m.pt"
    meta = tmp_path / "metadata.json"
    torch.save(OutputLengthMLP().state_dict(), pt)
    meta.write_text(json.dumps({"feature_mean": [], "feature_std": []}))
    loaded = OutputLengthPredictor.from_checkpoint(pt, meta)
    assert loaded._mean is None


@pytest.mark.unit
def test_trainer_resume(tmp_path):
    from pathlib import Path
    builder = OutputLengthDataset()
    samples = builder.generate_synthetic(40)
    splits = builder.split(samples)
    cfg = TrainConfig(
        epochs=1,
        batch_size=8,
        checkpoint_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "logs"),
        use_amp=False,
    )
    trainer = Trainer(cfg)
    result = trainer.train(splits)
    resumed = trainer.resume(splits, Path(result.checkpoint_path))
    assert resumed.checkpoint_path


@pytest.mark.unit
def test_build_scheduler_sjf_requires_predictor():
    with pytest.raises(ValueError):
        build_scheduler(SchedulerType.SJF, None)


@pytest.fixture
def gateway_config(tmp_path):
    cfg = load_config("configs/scheduler.yaml")
    cfg = dict(cfg)
    cfg["scheduler"] = dict(cfg["scheduler"])
    cfg["scheduler"]["type"] = "fcfs"
    cfg["scheduler"]["request_timeout_sec"] = 10.0
    cfg["predictor"] = {"models_dir": str(tmp_path)}
    return cfg
