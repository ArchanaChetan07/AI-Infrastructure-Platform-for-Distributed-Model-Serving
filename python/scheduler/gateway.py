"""vLLM scheduler gateway — production integration layer."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from predictor.predictor import OutputLengthPredictor
from prometheus_client import Counter, Gauge, Histogram, generate_latest

from scheduler.base_scheduler import ScheduledRequest, SchedulerType
from scheduler.priority_queue import new_request_id
from scheduler.sjf_scheduler import SJFScheduler, build_scheduler

logger = logging.getLogger(__name__)

REQUESTS_TOTAL = Counter("scheduler_requests_total", "Total requests", ["scheduler", "status"])
QUEUE_DEPTH = Gauge("scheduler_queue_depth", "Current queue depth", ["scheduler"])
REQUESTS_TIMEOUT = Counter("scheduler_timeouts_total", "Timed out requests", ["scheduler"])
REQUESTS_DROPPED = Counter("scheduler_dropped_total", "Dropped/cancelled requests", ["scheduler"])
PREDICTED_TOKENS = Histogram("scheduler_predicted_tokens", "Predicted output tokens")
FEATURE_LATENCY = Histogram("scheduler_feature_latency_ms", "Feature extraction latency ms")
PREDICT_LATENCY = Histogram("scheduler_predict_latency_ms", "Prediction latency ms")
SCHEDULE_LATENCY = Histogram("scheduler_overhead_ms", "Scheduling overhead ms")
E2E_LATENCY = Histogram("scheduler_e2e_latency_ms", "End-to-end latency ms")


class SchedulerGateway:
    """Async gateway that schedules requests before forwarding to vLLM."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.config = config
        self.vllm_url = config.get("vllm", {}).get("base_url", "http://localhost:8000")
        self.model = config.get("vllm", {}).get("model", "HuggingFaceTB/SmolLM3-3B")
        sched_cfg = config.get("scheduler", {})
        self.scheduler_type = SchedulerType(sched_cfg.get("type", "sjf"))
        self.max_workers = sched_cfg.get("max_workers", 32)
        self.request_timeout = float(sched_cfg.get("request_timeout_sec", 300.0))
        self.connect_timeout = float(sched_cfg.get("vllm_connect_timeout_sec", 5.0))
        self._predictor: Optional[OutputLengthPredictor] = None
        self._scheduler: Any = None
        self._client: Optional[httpx.AsyncClient] = http_client
        self._owns_client = http_client is None
        self._workers: list[asyncio.Task[None]] = []
        self._running = False

    async def startup(self) -> None:
        if self._running:
            return
        models_dir = Path(self.config.get("predictor", {}).get("models_dir", "models"))
        pt_path = models_dir / "output_length_mlp.pt"
        if pt_path.exists():
            self._predictor = OutputLengthPredictor.from_checkpoint(pt_path)
        else:
            from predictor.model import OutputLengthMLP

            self._predictor = OutputLengthPredictor(OutputLengthMLP())

        self._scheduler = build_scheduler(
            self.scheduler_type,
            self._predictor,
            self.config.get("scheduler", {}),
        )
        if isinstance(self._scheduler, SJFScheduler):
            await self._scheduler.start()

        if self._client is None:
            timeout = httpx.Timeout(
                connect=self.connect_timeout,
                read=self.request_timeout,
                write=30.0,
                pool=5.0,
            )
            self._client = httpx.AsyncClient(base_url=self.vllm_url, timeout=timeout)
            self._owns_client = True
        self._running = True
        for _ in range(self.max_workers):
            self._workers.append(asyncio.create_task(self._dispatch_loop()))
        logger.info("Gateway started: scheduler=%s vllm=%s", self.scheduler_type, self.vllm_url)

    async def shutdown(self) -> None:
        self._running = False
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        if isinstance(self._scheduler, SJFScheduler):
            await self._scheduler.stop()
        if self._client and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def submit_chat(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages", [])
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        req = ScheduledRequest(
            request_id=new_request_id(),
            prompt=prompt,
            messages=messages,
            max_tokens=int(body.get("max_tokens", 128)),
            predicted_tokens=0.0,
            priority=0.0,
            metadata={
                "body": body,
                "future": asyncio.get_running_loop().create_future(),
            },
        )
        await self._scheduler.submit(req)
        QUEUE_DEPTH.labels(scheduler=self.scheduler_type.value).set(self._scheduler.queue_depth)
        future: asyncio.Future[dict[str, Any]] = req.metadata["future"]
        if req.predicted_tokens > 0:
            PREDICTED_TOKENS.observe(req.predicted_tokens)
        try:
            return await asyncio.wait_for(future, timeout=self.request_timeout)
        except asyncio.TimeoutError:
            await self._scheduler.cancel(req.request_id)
            REQUESTS_TIMEOUT.labels(scheduler=self.scheduler_type.value).inc()
            REQUESTS_DROPPED.labels(scheduler=self.scheduler_type.value).inc()
            if not future.done():
                future.cancel()
            raise

    async def submit_chat_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        messages = body.get("messages", [])
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        req = ScheduledRequest(
            request_id=new_request_id(),
            prompt=prompt,
            messages=messages,
            max_tokens=int(body.get("max_tokens", 128)),
            predicted_tokens=0.0,
            priority=0.0,
            metadata={"body": body, "stream_queue": asyncio.Queue()},
        )
        await self._scheduler.submit(req)
        stream_queue: asyncio.Queue[Optional[bytes]] = req.metadata["stream_queue"]
        while True:
            chunk = await stream_queue.get()
            if chunk is None:
                break
            yield chunk

    async def _dispatch_loop(self) -> None:
        while self._running:
            try:
                req = await self._scheduler.acquire(timeout=1.0)
            except asyncio.TimeoutError:
                continue
            asyncio.create_task(self._forward_request(req))

    async def _forward_request(self, req: ScheduledRequest) -> None:
        t0 = time.monotonic()
        body = dict(req.metadata.get("body", {}))
        body["model"] = body.get("model", self.model)
        stream = body.get("stream", False)
        assert self._client is not None

        try:
            if stream:
                queue: asyncio.Queue[Optional[bytes]] = req.metadata["stream_queue"]
                async with self._client.stream(
                    "POST", "/v1/chat/completions", json={**body, "stream": True}
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        await queue.put(chunk)
                await queue.put(None)
            else:
                resp = await self._client.post("/v1/chat/completions", json=body)
                result = resp.json() if resp.status_code == 200 else {"error": resp.text}
                future: asyncio.Future = req.metadata["future"]
                if not future.done():
                    future.set_result(result)
            await self._scheduler.complete(req.request_id, success=True)
            REQUESTS_TOTAL.labels(scheduler=self.scheduler_type.value, status="success").inc()
        except Exception as e:
            logger.exception("Forward failed for %s: %s", req.request_id, e)
            await self._scheduler.complete(req.request_id, success=False)
            REQUESTS_TOTAL.labels(scheduler=self.scheduler_type.value, status="error").inc()
            if "future" in req.metadata:
                fut: asyncio.Future = req.metadata["future"]
                if not fut.done():
                    fut.set_result({"error": str(e)})
        finally:
            E2E_LATENCY.observe((time.monotonic() - t0) * 1000.0)
            QUEUE_DEPTH.labels(scheduler=self.scheduler_type.value).set(self._scheduler.queue_depth)

    def stats(self) -> dict[str, Any]:
        return {
            "scheduler": self._scheduler.stats().__dict__ if self._scheduler else {},
            "queue_depth": self._scheduler.queue_depth if self._scheduler else 0,
        }


def load_config(path: str = "configs/scheduler.yaml") -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def create_app(
    config_path: str = "configs/scheduler.yaml",
    *,
    gateway: Optional[SchedulerGateway] = None,
) -> FastAPI:
    config = load_config(config_path)
    gateway = gateway or SchedulerGateway(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await gateway.startup()
        app.state.gateway = gateway
        yield
        await gateway.shutdown()

    app = FastAPI(title="vLLM SJF Scheduler Gateway", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request) -> dict[str, Any]:
        gw: SchedulerGateway = request.app.state.gateway
        backend_ok = False
        if gw._client is not None:
            try:
                resp = await gw._client.get("/health", timeout=gw.connect_timeout)
                backend_ok = resp.status_code < 500
            except Exception:
                backend_ok = False
        status = "ready" if gw._running and gw._scheduler is not None else "starting"
        return {
            "status": status,
            "scheduler": gw.scheduler_type.value,
            "queue_depth": gw._scheduler.queue_depth if gw._scheduler else 0,
            "vllm_backend": "ok" if backend_ok else "unavailable",
        }

    @app.get("/live")
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type="text/plain")

    @app.get("/scheduler/stats")
    async def scheduler_stats(request: Request) -> dict[str, Any]:
        gw: SchedulerGateway = request.app.state.gateway
        return gw.stats()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        body = await request.json()
        gw: SchedulerGateway = request.app.state.gateway
        if body.get("stream"):
            return StreamingResponse(
                gw.submit_chat_stream(body),
                media_type="text/event-stream",
            )
        try:
            result = await gw.submit_chat(body)
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Request timed out") from exc
        if "error" in result:
            raise HTTPException(status_code=502, detail=result["error"])
        return JSONResponse(result)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def proxy(request: Request, path: str) -> Response:
        """Proxy other vLLM endpoints directly."""
        gw: SchedulerGateway = request.app.state.gateway
        assert gw._client is not None
        url = f"/{path}"
        body = await request.body()
        resp = await gw._client.request(
            request.method, url, content=body, headers=dict(request.headers)
        )
        return Response(content=resp.content, status_code=resp.status_code)

    return app


app = create_app()
