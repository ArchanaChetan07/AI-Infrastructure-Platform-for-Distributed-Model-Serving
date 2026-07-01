"""Extended scheduler and gateway coverage tests."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from predictor.model import OutputLengthMLP
from predictor.predictor import OutputLengthPredictor
from scheduler.aging import AgingConfig
from scheduler.base_scheduler import ScheduledRequest, SchedulerType
from scheduler.gateway import SchedulerGateway, create_app, load_config
from scheduler.priority_queue import PriorityQueue, new_request_id
from scheduler.sjf_scheduler import SJFScheduler, build_scheduler
from starlette.testclient import TestClient


@pytest.mark.asyncio
@pytest.mark.unit
async def test_priority_queue_timeout_and_cancel():
    q = PriorityQueue()
    with pytest.raises(asyncio.TimeoutError):
        await q.dequeue(timeout=0.01)
    await q.enqueue("a", 1.0, "payload")
    assert await q.cancel("a")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_priority_queue_update_priority():
    q = PriorityQueue()
    await q.enqueue("x", 10.0, "p")
    assert await q.update_priority("x", 5.0)
    item = await q.dequeue(timeout=1.0)
    assert item.priority == 5.0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_sjf_prediction_failure_fallback():
    class BadPredictor:
        def predict(self, prompt):
            raise RuntimeError("boom")

    sched = SJFScheduler(BadPredictor(), fallback_to_fcfs=True)  # type: ignore[arg-type]
    await sched.start()
    req = ScheduledRequest(
        request_id=new_request_id(),
        prompt="hi",
        messages=[],
        max_tokens=10,
        predicted_tokens=0.0,
        priority=0.0,
    )
    await sched.submit(req)
    acquired = await sched.acquire(timeout=2.0)
    assert acquired is not None
    await sched.stop()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_sjf_on_reload():
    predictor = OutputLengthPredictor(OutputLengthMLP())
    sched = SJFScheduler(predictor, aging_config=AgingConfig())
    await sched.start()
    await sched.on_reload({"aging": {"policy": "exponential", "factor": 0.01, "max_boost": 100.0}})
    await sched.stop()


@pytest.mark.unit
def test_build_scheduler_types():
    pred = OutputLengthPredictor(OutputLengthMLP())
    assert build_scheduler(SchedulerType.FCFS, pred).__class__.__name__ == "FCFSScheduler"
    oracle = build_scheduler(SchedulerType.ORACLE_SJF, pred)
    assert oracle.__class__.__name__ == "OracleSJFScheduler"
    with pytest.raises(ValueError):
        build_scheduler(SchedulerType.SJF, None)


@pytest.mark.unit
def test_gateway_ready_and_proxy(gateway_config, mock_http_client):
    mock_http_client.get = AsyncMock(return_value=MagicMock(status_code=200))
    mock_http_client.request = AsyncMock(
        return_value=MagicMock(status_code=200, content=b"{}", headers={})
    )
    gw = SchedulerGateway(gateway_config, http_client=mock_http_client)
    app = create_app(gateway=gw)
    with TestClient(app) as client:
        client.get("/health")
        ready = client.get("/ready")
        assert ready.status_code == 200
        body = ready.json()
        assert body["status"] in ("ready", "starting")
        live = client.get("/live")
        assert live.json()["status"] == "alive"
        proxy = client.get("/v1/models")
        assert proxy.status_code == 200


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fcfs_cancel():
    from scheduler.fcfs_scheduler import FCFSScheduler

    sched = FCFSScheduler()
    req = ScheduledRequest(
        request_id=new_request_id(),
        prompt="x",
        messages=[],
        max_tokens=5,
        predicted_tokens=1.0,
        priority=1.0,
    )
    await sched.submit(req)
    assert await sched.cancel(req.request_id)


@pytest.fixture
def gateway_config(tmp_path):
    cfg = load_config("configs/scheduler.yaml")
    cfg = dict(cfg)
    cfg["scheduler"] = dict(cfg["scheduler"])
    cfg["scheduler"]["type"] = "fcfs"
    cfg["scheduler"]["request_timeout_sec"] = 10.0
    cfg["predictor"] = {"models_dir": str(tmp_path)}
    return cfg


@pytest.fixture
def mock_http_client():
    client = AsyncMock(spec=httpx.AsyncClient)
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    client.post = AsyncMock(return_value=resp)
    return client
