from __future__ import annotations

import json

import fakeredis.aioredis
import httpx
import pytest
from fastapi.testclient import TestClient

from conductor.backends import OpenAIBackend, split_think
from conductor.breaker import CircuitBreaker, State
from conductor.config import BreakerConfig, Endpoint, MonitorConfig
from conductor.monitor import MemoryStore, Monitor, RedisStore
from conductor.server import create_app
from conductor.telemetry import Metrics

from .conftest import build

# ---------------------------------------------------------------- breaker


def test_breaker_lifecycle():
    t = [0.0]
    b = CircuitBreaker("x", BreakerConfig(failure_threshold=2, window_s=10, cooldown_s=5), clock=lambda: t[0])
    b.record_failure()
    assert b.available()
    b.record_failure()
    assert b.state is State.OPEN and not b.available()
    t[0] = 6
    assert b.acquire() and b.state is State.HALF_OPEN
    assert not b.acquire()  # one probe at a time
    b.record_failure()
    assert b.state is State.OPEN
    t[0] = 12
    assert b.acquire()
    b.record_success()
    assert b.state is State.CLOSED


def test_breaker_release_frees_probe():
    t = [0.0]
    b = CircuitBreaker("x", BreakerConfig(failure_threshold=1, cooldown_s=1), clock=lambda: t[0])
    b.record_failure()
    t[0] = 2
    assert b.acquire()
    b.release()
    assert b.acquire()


def test_breaker_failures_expire():
    t = [0.0]
    b = CircuitBreaker("x", BreakerConfig(failure_threshold=2, window_s=10), clock=lambda: t[0])
    b.record_failure()
    t[0] = 11
    b.record_failure()
    assert b.state is State.CLOSED


# ---------------------------------------------------------------- monitor


def _cfg(**kw) -> MonitorConfig:
    return MonitorConfig(**({"balance_min_samples": 10, "collapse_min_samples": 10, "uniform_hold_s": 100} | kw))


async def test_balanced_traffic_raises_nothing():
    m = Monitor(_cfg(), Metrics(), clock=lambda: 1000.0)
    for c in ("code", "reasoning", "vision", "long_ctx") * 10:
        await m.record([c])
    reports = await m.check()
    assert not reports["balance"].violations and not reports["collapse"].violations
    assert not m.uniform_mode()


async def test_collapse_triggers_uniform_mode_and_expires():
    t = [1000.0]
    cfg = _cfg()
    m = Monitor(cfg, Metrics(), store=MemoryStore(cfg, clock=lambda: t[0]), clock=lambda: t[0])
    for _ in range(90):
        await m.record(["code"])
    for c in ("reasoning", "vision", "long_ctx") * 4:
        await m.record([c])
    reports = await m.check()
    assert any(v.startswith("code=") for v in reports["collapse"].violations)
    assert m.uniform_mode()
    assert m.metrics.uniform_mode._value.get() == 1
    t[0] += cfg.collapse_window_s + 200  # traffic aged out, hold expired
    await m.check()
    assert not m.uniform_mode()


async def test_load_balance_alert_dedupes():
    m = Monitor(_cfg(collapse_min_samples=10**9), Metrics(), clock=lambda: 1000.0)
    for c in ["code"] * 70 + ["reasoning"] * 30:
        await m.record([c])
    await m.check()
    await m.check()
    assert m.metrics.alerts.labels("load_balance", "code")._value.get() == 1
    assert m.metrics.alerts.labels("load_balance", "vision")._value.get() == 1


async def test_redis_store_shares_state():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    cfg = _cfg()
    a = Monitor(cfg, Metrics(), store=RedisStore(cfg, client), clock=lambda: 5000.0)
    b = Monitor(cfg, Metrics(), store=RedisStore(cfg, client), clock=lambda: 5000.0)
    for _ in range(20):
        await a.record(["vision"])
    reports = await b.check()  # replica b sees replica a's traffic
    assert reports["collapse"].shares["vision"] == 1.0
    assert b.uniform_mode()
    await a.check()
    assert a.uniform_mode()


# ---------------------------------------------------------------- OpenAI-compatible backend


def test_split_think():
    assert split_think("<think>abc</think> answer", "") == ("answer", "abc")
    assert split_think("abc</think>answer", "") == ("answer", "abc")
    assert split_think("plain", "r") == ("plain", "r")


async def test_openai_backend_complete_and_stream():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.update(body)
        if body["stream"]:
            chunks = [
                {"choices": [{"delta": {"reasoning_content": "think "}}]},
                {"choices": [{"delta": {"content": "Hel"}}]},
                {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"completion_tokens": 2}},
            ]
            sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi", "reasoning_content": "r"},
                                                       "finish_reason": "stop"}], "usage": {"completion_tokens": 1}})

    ep = Endpoint(base_url="http://x/v1", model="m", extra_body={"top_k": 20})
    client = httpx.AsyncClient(base_url="http://x/v1", transport=httpx.MockTransport(handler))
    be = OpenAIBackend(ep, client=client)
    c = await be.complete([{"role": "user", "content": "q"}], max_tokens=5, temperature=0.0)
    assert (c.content, c.reasoning) == ("hi", "r")
    assert seen["max_tokens"] == 5 and seen["temperature"] == 0.0 and seen["top_k"] == 20
    deltas = [d async for d in be.stream([{"role": "user", "content": "q"}])]
    assert "".join(d.content for d in deltas) == "Hello"
    assert "".join(d.reasoning for d in deltas) == "think "
    assert seen["stream_options"] == {"include_usage": True}


async def test_openai_backend_http_error():
    from conductor.backends import BackendError

    client = httpx.AsyncClient(base_url="http://x/v1", transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")))
    be = OpenAIBackend(Endpoint(base_url="http://x/v1", model="m"), client=client)
    with pytest.raises(BackendError):
        await be.complete([{"role": "user", "content": "q"}])


# ---------------------------------------------------------------- HTTP server


@pytest.fixture
def client(settings):
    orch, _ = build(settings)
    with TestClient(create_app(settings, orch)) as c:
        yield c


def test_chat_completion_with_debug(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "prove this math equation then write python code"}], "debug": True})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["x_conductor"]["path"] == "multi" and body["x_conductor"]["synthesised"] is True
    assert set(body["x_conductor"]["specialists"]) == set(body["x_conductor"]["selected"])


def test_chat_stream_sse(client):
    with client.stream("POST", "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "fix this python bug"}], "stream": True}) as r:
        lines = [ln for ln in r.iter_lines() if ln.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(ln[5:]) for ln in lines[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text.startswith("[code]")
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_route_endpoint_and_pool(client):
    r = client.post("/v1/route", json={"messages": [{"role": "user", "content": "summarize this document"}]})
    assert r.json()["selected"][0] == "long_ctx"
    pool = client.get("/v1/pool").json()
    assert pool["tiers"][0]["experts"] == 128 and len(pool["tiers"]) == 6


def test_metrics_and_health(client):
    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}]})
    m = client.get("/metrics").text
    assert 'conductor_requests_total{path="fast",stream="false"} 1.0' in m
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json()["ready"] is True
    assert "uniform_mode" in client.get("/v1/monitor").json()


def test_no_user_message_is_400(client):
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "system", "content": "x"}]})
    assert r.status_code == 400


def test_api_key_auth(settings, monkeypatch):
    monkeypatch.setenv("CONDUCTOR_API_KEYS", "k1,k2")
    orch, _ = build(settings)
    with TestClient(create_app(settings, orch)) as c:
        payload = {"messages": [{"role": "user", "content": "hello"}]}
        assert c.post("/v1/chat/completions", json=payload).status_code == 401
        assert c.post("/v1/chat/completions", json=payload, headers={"Authorization": "Bearer k2"}).status_code == 200
        assert c.get("/healthz").status_code == 200
