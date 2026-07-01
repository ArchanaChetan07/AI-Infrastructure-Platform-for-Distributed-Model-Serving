"""Gateway integration tests with mocked vLLM backend."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from scheduler.gateway import SchedulerGateway, create_app, load_config
from starlette.testclient import TestClient


def _mock_vllm_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.text = ""
    resp.json.return_value = {
        "id": "stub",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    return resp


@pytest.fixture
def gateway_config(tmp_path):
    cfg = load_config("configs/scheduler.yaml")
    cfg = dict(cfg)
    cfg["scheduler"] = dict(cfg["scheduler"])
    cfg["scheduler"]["type"] = "fcfs"
    cfg["scheduler"]["vllm_connect_timeout_sec"] = 1.0
    cfg["scheduler"]["request_timeout_sec"] = 10.0
    cfg["predictor"] = {"models_dir": str(tmp_path)}
    return cfg


@pytest.fixture
def mock_http_client():
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(return_value=_mock_vllm_response())
    return client


@pytest.fixture
def client(gateway_config, mock_http_client):
    gw = SchedulerGateway(gateway_config, http_client=mock_http_client)
    app = create_app(gateway=gw)
    with TestClient(app) as test_client:
        test_client.get("/health")
        yield test_client, mock_http_client


@pytest.mark.unit
def test_health_metrics_stats(client):
    test_client, _ = client
    metrics = test_client.get("/metrics")
    assert metrics.status_code == 200
    assert "scheduler_requests_total" in metrics.text

    stats = test_client.get("/scheduler/stats")
    assert stats.status_code == 200
    assert "queue_depth" in stats.json()


@pytest.mark.unit
def test_chat_completions_mocked_backend(client):
    test_client, mock_http_client = client
    resp = test_client.post(
        "/v1/chat/completions",
        json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "hello"
    mock_http_client.post.assert_called()


@pytest.mark.unit
def test_chat_completions_backend_error(gateway_config):
    backend = AsyncMock(spec=httpx.AsyncClient)
    backend.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    gw = SchedulerGateway(gateway_config, http_client=backend)
    app = create_app(gateway=gw)
    with TestClient(app) as test_client:
        test_client.get("/health")
        resp = test_client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 502


@pytest.mark.unit
def test_concurrent_chat_requests(client):
    test_client, _ = client
    for i in range(20):
        resp = test_client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": str(i)}]},
        )
        assert resp.status_code == 200
