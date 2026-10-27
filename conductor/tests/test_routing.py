from __future__ import annotations

import pytest
import yaml

from conductor.config import Settings
from conductor.router import normalise, plan, routing_input
from conductor.schemas import ChatRequest, RoutingDecision

from .conftest import POOL


def req(text: str = "q", images: int = 0) -> ChatRequest:
    content = [{"type": "text", "text": text}] + [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}} for _ in range(images)]
    return ChatRequest(messages=[{"role": "user", "content": content if images else text}])


def decision(**w: float) -> RoutingDecision:
    base = {"code": 0.0, "reasoning": 0.0, "vision": 0.0, "long_ctx": 0.0} | w
    return RoutingDecision(weights=base)


def run(settings, d, r=None, available=lambda c: True, uniform=False):
    return plan(d, r or req(), settings, available=available, uniform_mode=uniform)


def test_pool_matches_spec(settings):
    assert settings.conductor.experts == 128 and settings.conductor.experts_per_token == 8
    reasoning = settings.specialist("reasoning")
    assert (reasoning.experts, reasoning.experts_per_token, reasoning.shared_experts) == (512, 10, 1)
    assert settings.fast_path.architecture == "dense"
    assert sum(s.architecture == "moe" for s in settings.specialists) == 4
    assert settings.routing.capacity_factor == 2


def test_config_rejects_missing_cluster():
    raw = yaml.safe_load(POOL.read_text())
    raw["specialists"] = raw["specialists"][:3]
    with pytest.raises(ValueError):
        Settings.model_validate(raw)


def test_normalise_handles_garbage():
    assert normalise({"code": -1, "reasoning": 0}) == {c: 0.25 for c in ("code", "reasoning", "vision", "long_ctx")}
    n = normalise({"code": 2, "reasoning": 2})
    assert n["code"] == pytest.approx(0.5) and n["vision"] == 0


def test_soft_top_k_with_merge_weights(settings):
    p = run(settings, decision(code=0.05, reasoning=0.72, vision=0.18, long_ctx=0.05))
    assert p.selected == ["reasoning", "vision"] and p.path == "multi"
    assert p.merge_weights["reasoning"] == pytest.approx(0.72 / 0.90)
    assert sum(p.merge_weights.values()) == pytest.approx(1.0)


def test_single_when_one_cluster_dominates(settings):
    p = run(settings, decision(code=0.95, reasoning=0.05))
    assert p.selected == ["code"] and p.path == "single"


def test_capacity_factor_bounds_fanout(settings):
    p = run(settings, decision(code=0.34, reasoning=0.33, long_ctx=0.33))
    assert len(p.selected) == 2
    assert any("capacity: dropped" in r for r in p.reasons)


def test_trivial_goes_fast(settings):
    d = decision(code=0.5, reasoning=0.5)
    d.trivial = True
    assert run(settings, d).path == "fast"


def test_trivial_ignored_when_images_attached(settings):
    d = decision(code=1.0)
    d.trivial = True
    p = run(settings, d, req("look", images=1))
    assert "vision" in p.selected


def test_images_force_vision_and_floor_weight(settings):
    p = run(settings, decision(code=0.99, vision=0.01), req("what does this code screenshot do", images=1))
    assert p.selected == ["vision", "code"]
    assert p.merge_weights["vision"] >= settings.routing.min_weight / (settings.routing.min_weight + 0.99) - 1e-9


def test_long_context_forced(settings):
    long_text = "x" * int(settings.routing.long_ctx_force_tokens * settings.routing.chars_per_token + 10)
    p = run(settings, decision(reasoning=1.0), req(long_text))
    assert p.selected[0] == "long_ctx" and "reasoning" in p.selected


def test_open_breaker_passes_slot_to_next_cluster(settings):
    p = run(settings, decision(code=0.6, reasoning=0.3, vision=0.1), available=lambda c: c != "code")
    assert p.selected == ["reasoning", "vision"]
    assert any("breaker open" in r for r in p.reasons)


def test_all_breakers_open_goes_fast(settings):
    p = run(settings, decision(code=1.0), available=lambda c: False)
    assert p.path == "fast" and p.selected == []


def test_uniform_mode_flattens_distribution(settings):
    p = run(settings, decision(code=1.0), uniform=True)
    assert p.probs["code"] == pytest.approx(0.5 * 1.0 + 0.5 * 0.25)
    assert p.probs["vision"] == pytest.approx(0.125)
    assert len(p.selected) == 2  # flattened mass pushes a second cluster over min_weight


def test_routing_input_is_bounded(settings):
    text = "a" * 100_000
    out = routing_input(req(text, images=2), settings)
    assert "images_attached: 2" in out
    assert len(out) < settings.conductor.routing_input_chars + 1000
    assert "elided for routing" in out
