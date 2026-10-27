from __future__ import annotations

import json

import pytest

from conductor.orchestrator import ServiceUnavailable
from conductor.schemas import ChatRequest

from .conftest import build


def chat(text, images=0, **kw) -> ChatRequest:
    content = [{"type": "text", "text": text}] + [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}} for _ in range(images)]
    return ChatRequest(messages=[{"role": "user", "content": content if images else text}], **kw)


async def test_trivial_returns_fast_path(settings):
    orch, b = build(settings)
    r = await orch.run(chat("hello"))
    assert r.plan.path == "fast" and r.content.startswith("[fast]")
    assert not b["code"].calls and not b["reasoning"].calls


async def test_single_specialist_skips_synthesis(settings):
    orch, b = build(settings)
    r = await orch.run(chat("fix this python function bug in my script"))
    assert r.plan.selected == ["code"] and not r.synthesised
    assert r.content.startswith("[code]")
    assert len(b["conductor"].calls) == 1  # routing only


async def test_multi_synthesis_is_probability_weighted(settings):
    orch, b = build(settings)
    r = await orch.run(chat("prove this math equation then write python code for the function"))
    assert set(r.plan.selected) == {"reasoning", "code"} and r.synthesised
    synth_prompt = b["conductor"].calls[-1]["messages"][-1]["content"]
    assert "routing probability" in synth_prompt and "Fast draft" in synth_prompt
    first, second = synth_prompt.index("Reasoning Specialist"), synth_prompt.index("Code Specialist")
    hi = max(r.plan.merge_weights, key=r.plan.merge_weights.get)
    assert (first < second) == (hi == "reasoning")


async def test_fast_path_fires_on_every_request(settings):
    orch, b = build(settings)
    await orch.run(chat("prove this math equation then write python code"))
    assert len(b["fast"].calls) == 1


async def test_subqueries_and_reasoning_hidden_by_default(settings):
    orch, b = build(settings)
    r = await orch.run(chat("prove this math logic puzzle"))
    assert "[Focus for this part of the task]" in b["reasoning"].calls[0]["messages"][-1]["content"]
    assert r.reasoning == ""
    r2 = await orch.run(chat("prove this math logic puzzle", show_reasoning=True))
    assert "mock reasoning trace" in r2.reasoning


async def test_images_only_reach_image_capable_specialists(settings):
    orch, b = build(settings)
    await orch.run(chat("what does this python code screenshot do", images=1))
    vision_msg = b["vision"].calls[0]["messages"][-1]["content"]
    code_msg = b["code"].calls[0]["messages"][-1]["content"]
    assert isinstance(vision_msg, list) and any(p["type"] == "image_url" for p in vision_msg)
    assert isinstance(code_msg, str) and "1 image(s) attached" in code_msg
    routing_msg = b["conductor"].calls[0]["messages"][-1]["content"]
    assert "base64" not in routing_msg  # attachments are opaque to L0


async def test_failed_specialist_degrades_to_single(settings):
    orch, _ = build(settings, code="fail=1")
    r = await orch.run(chat("prove this math equation then write python code for the function"))
    assert r.plan.path == "single" and r.content.startswith("[reasoning]")
    assert not r.outputs["code"].ok


async def test_all_specialists_fail_returns_fast(settings):
    orch, _ = build(settings, code="fail=1", reasoning="fail=1")
    r = await orch.run(chat("prove this math equation then write python code"))
    assert r.plan.path == "fast" and r.content.startswith("[fast]")


async def test_synthesis_failure_returns_highest_weighted_output(settings):
    orch, b = build(settings)
    real = b["conductor"].complete
    calls = {"n": 0}

    async def flaky(messages, **params):
        calls["n"] += 1
        if calls["n"] > 1:
            from conductor.backends import BackendError
            raise BackendError("synth down")
        return await real(messages, **params)

    b["conductor"].complete = flaky
    r = await orch.run(chat("prove this math equation then write python code for the function"))
    best = max(r.plan.merge_weights, key=r.plan.merge_weights.get)
    assert not r.synthesised and r.content.startswith(f"[{best}]")


async def test_malformed_routing_retries_once_then_fast(settings):
    orch, b = build(settings, conductor="malformed=1")
    r = await orch.run(chat("prove this math equation"))
    assert r.plan.path == "fast" and "router fallback: malformed_json" in r.plan.reasons
    assert len(b["conductor"].calls) == 2


async def test_router_backend_down_falls_back(settings):
    orch, _ = build(settings, conductor="fail=1")
    r = await orch.run(chat("prove this"))
    assert r.plan.path == "fast" and r.content.startswith("[fast]")


async def test_fast_down_on_trivial_is_503(settings):
    orch, _ = build(settings, fast="fail=1")
    with pytest.raises(ServiceUnavailable):
        await orch.run(chat("hello"))


async def test_breaker_opens_and_reroutes(settings):
    orch, b = build(settings, code="fail=1")
    threshold = settings.specialist("code").breaker.failure_threshold
    for _ in range(threshold):
        await orch.run(chat("fix this python function bug"))
    assert orch.breakers["code"].state.value == "open"
    before = len(b["code"].calls)
    r = await orch.run(chat("fix this python function bug"))
    assert len(b["code"].calls) == before
    assert "code" not in r.plan.selected


async def test_monitor_records_invocations(settings):
    orch, _ = build(settings)
    await orch.run(chat("fix this python function bug"))
    await orch.run(chat("prove this math equation then write python code"))
    counts = await orch.monitor.store.counts(3600, orch.monitor._clock())
    assert counts["code"] == 2 and counts["reasoning"] == 1


async def collect(orch, req):
    plan, text, reasoning, done = None, "", "", None
    async for ev in orch.stream(req):
        if ev.kind == "plan":
            plan = ev.plan
        elif ev.kind == "delta":
            text += ev.delta.content
            reasoning += ev.delta.reasoning
        elif ev.kind == "done":
            done = ev.info
    return plan, text, reasoning, done


async def test_stream_fast(settings):
    orch, _ = build(settings)
    plan, text, _, done = await collect(orch, chat("hello", stream=True))
    assert plan.path == "fast" and text.startswith("[fast]") and done["path"] == "fast"


async def test_stream_single_hides_reasoning(settings):
    orch, _ = build(settings)
    plan, text, reasoning, done = await collect(orch, chat("prove this math logic", stream=True))
    assert plan.selected == ["reasoning"] and text.startswith("[reasoning]") and reasoning == ""
    _, _, reasoning, _ = await collect(orch, chat("prove this math logic", stream=True, show_reasoning=True))
    assert "mock reasoning trace" in reasoning


async def test_stream_single_falls_back_before_first_token(settings):
    orch, _ = build(settings, reasoning="fail=1")
    plan, text, _, done = await collect(orch, chat("prove this math logic", stream=True))
    assert text.startswith("[fast]") and done["path"] == "fast"
    assert any("unavailable before first token" in r for r in plan.reasons)


async def test_stream_multi_synthesises(settings):
    orch, _ = build(settings)
    plan, text, _, done = await collect(orch, chat("prove this math equation then write python code", stream=True))
    assert plan.path == "multi" and text.startswith("[conductor]") and done["synthesised"] is True


async def test_route_json_schema_is_sent(settings):
    orch, b = build(settings)
    await orch.route(chat("prove this"))
    rf = b["conductor"].calls[0]["response_format"]
    assert rf["type"] == "json_schema"
    assert set(rf["json_schema"]["schema"]["properties"]["weights"]["required"]) == {"code", "reasoning", "vision", "long_ctx"}
    json.dumps(rf)  # serialisable
