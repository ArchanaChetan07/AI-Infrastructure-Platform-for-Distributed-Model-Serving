"""Concurrency and stress tests (mocked vLLM backend)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from scheduler.gateway import SchedulerGateway, create_app, load_config
from starlette.testclient import TestClient


def _client_fixture(gateway_config, mock_http_client):
    gw = SchedulerGateway(gateway_config, http_client=mock_http_client)
    app = create_app(gateway=gw)
    with TestClient(app) as test_client:
        test_client.get("/health")
        yield test_client


@pytest.fixture
def gateway_config(tmp_path):
    cfg = load_config("configs/scheduler.yaml")
    cfg = dict(cfg)
    cfg["scheduler"] = dict(cfg["scheduler"])
    cfg["scheduler"]["type"] = "fcfs"
    cfg["scheduler"]["max_workers"] = 16
    cfg["scheduler"]["request_timeout_sec"] = 30.0
    cfg["predictor"] = {"models_dir": str(tmp_path)}
    return cfg


@pytest.fixture
def mock_http_client():
    async def slow_post(*args, **kwargs):
        await asyncio.sleep(0.001)
        resp = MagicMock(status_code=200)
        resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        return resp

    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(side_effect=slow_post)
    client.get = AsyncMock(return_value=MagicMock(status_code=200))
    return client


@pytest.mark.unit
@pytest.mark.parametrize("n", [1, 10, 50, 100])
def test_sequential_load(gateway_config, mock_http_client, n):
    gw = SchedulerGateway(gateway_config, http_client=mock_http_client)
    app = create_app(gateway=gw)
    with TestClient(app) as client:
        client.get("/health")
        codes = []
        for i in range(n):
            r = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": str(i)}]},
            )
            codes.append(r.status_code)
        assert all(c == 200 for c in codes)
        stats = client.get("/scheduler/stats")
        assert stats.status_code == 200
