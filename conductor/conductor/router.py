"""L0: the Conductor's routing pass and the planner that turns its softmax into a plan."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable

from pydantic import ValidationError

from .backends import Backend, BackendError
from .config import ROUTABLE_CLUSTERS, Settings
from .schemas import ChatRequest, RoutePlan, RoutingDecision, routing_json_schema
from .telemetry import Metrics

log = logging.getLogger("conductor.router")


class RoutingFailed(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _clip_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n[... {len(text) - limit} chars elided for routing ...]\n{text[-half:]}"


def routing_input(req: ChatRequest, settings: Settings) -> str:
    """The Conductor never sees attachments or full long contexts, only a bounded view plus metadata."""
    cfg = settings.conductor
    est_tokens = int(req.total_chars() / settings.routing.chars_per_token)
    history = [m for m in req.messages[:-1] if m.role in ("user", "assistant")][-4:]
    hist = "\n".join(f"{m.role}: {_clip_middle(m.text(), 400)}" for m in history)
    query = _clip_middle(req.last_user().text(), cfg.routing_input_chars)
    return (
        f"images_attached: {req.image_count()}\n"
        f"estimated_context_tokens: {est_tokens}\n"
        f"recent_history:\n{hist or '(none)'}\n"
        f"query: {query}"
    )


def normalise(weights: dict[str, float]) -> dict[str, float]:
    w = {c: max(0.0, float(weights.get(c, 0.0) or 0.0)) for c in ROUTABLE_CLUSTERS}
    total = sum(w.values())
    if total <= 0:
        return {c: 1.0 / len(ROUTABLE_CLUSTERS) for c in ROUTABLE_CLUSTERS}
    return {c: v / total for c, v in w.items()}


class Router:
    def __init__(self, settings: Settings, backend: Backend, metrics: Metrics) -> None:
        self.settings = settings
        self.backend = backend
        self.metrics = metrics

    async def decide(self, req: ChatRequest) -> RoutingDecision:
        """Call the Conductor with guided decoding. Retry once on malformed output."""
        cfg = self.settings.conductor
        messages = [
            {"role": "system", "content": cfg.routing_prompt},
            {"role": "user", "content": routing_input(req, self.settings)},
        ]
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "routing_decision", "schema": routing_json_schema(), "strict": True},
        }
        last_error = "unknown"
        for attempt in range(cfg.route_retries + 1):
            if attempt:
                self.metrics.router_retries.inc()
            try:
                out = await self.backend.complete(
                    messages, max_tokens=cfg.routing_max_tokens, temperature=0.0, response_format=response_format
                )
            except BackendError as e:
                raise RoutingFailed("backend_error") from e
            try:
                decision = RoutingDecision.model_validate(json.loads(out.content))
                if not any(c in decision.weights for c in ROUTABLE_CLUSTERS):
                    raise ValueError("no routable weights")
                return decision
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                last_error = f"{type(e).__name__}: {e}"
                log.warning("malformed routing output (attempt %d): %s", attempt + 1, last_error)
        raise RoutingFailed("malformed_json")


def plan(
    decision: RoutingDecision,
    req: ChatRequest,
    settings: Settings,
    *,
    available: Callable[[str], bool],
    uniform_mode: bool,
    metrics: Metrics | None = None,
) -> RoutePlan:
    """Soft routing -> constraints -> capacity factor -> breaker-aware top-k."""
    rcfg = settings.routing
    probs = normalise(decision.weights)
    reasons: list[str] = []

    if uniform_mode and rcfg.collapse_uniform_mix > 0:
        u = 1.0 / len(ROUTABLE_CLUSTERS)
        a = rcfg.collapse_uniform_mix
        probs = {c: (1 - a) * p + a * u for c, p in probs.items()}
        reasons.append(f"collapse fallback: mixed {a:.0%} uniform")

    if decision.trivial and not req.image_count():
        return RoutePlan(decision=decision, probs=probs, selected=[], merge_weights={}, path="fast",
                         reasons=reasons + ["conductor marked trivial"], uniform_mode=uniform_mode)

    forced: list[str] = []
    if rcfg.images_force_vision and req.image_count():
        forced.append("vision")
        reasons.append("images attached: vision forced")
    if req.total_chars() / rcfg.chars_per_token >= rcfg.long_ctx_force_tokens:
        forced.append("long_ctx")
        reasons.append("context exceeds threshold: long_ctx forced")

    ranked = sorted(ROUTABLE_CLUSTERS, key=lambda c: probs[c], reverse=True)
    candidates = forced + [c for c in ranked if c not in forced and probs[c] >= rcfg.min_weight]
    if not candidates:
        candidates = ranked[:1]

    selected: list[str] = []
    for c in candidates:
        if len(selected) >= rcfg.capacity_factor:
            if metrics is not None and probs[c] >= rcfg.min_weight:
                metrics.capacity_drops.labels(c).inc()
            reasons.append(f"capacity: dropped {c} (p={probs[c]:.2f})")
            continue
        if not available(c):
            reasons.append(f"breaker open: skipped {c}")
            continue
        selected.append(c)

    # Breakers may have emptied the plan; fill from the remaining ranking.
    if not selected:
        for c in ranked:
            if available(c):
                selected.append(c)
                reasons.append(f"all preferred clusters unavailable: rerouted to {c}")
                break

    if not selected:
        return RoutePlan(decision=decision, probs=probs, selected=[], merge_weights={}, path="fast",
                         reasons=reasons + ["no specialist available"], uniform_mode=uniform_mode)

    # A forced cluster keeps at least min_weight of merge mass even if L0 under-weighted it.
    eff = {c: max(probs[c], rcfg.min_weight) if c in forced else probs[c] for c in selected}
    mass = sum(eff.values()) or 1.0
    merge = {c: eff[c] / mass for c in selected}
    return RoutePlan(decision=decision, probs=probs, selected=selected, merge_weights=merge,
                     path="single" if len(selected) == 1 else "multi", reasons=reasons, uniform_mode=uniform_mode)


def fallback_plan(reason: str, uniform_mode: bool) -> RoutePlan:
    u = 1.0 / len(ROUTABLE_CLUSTERS)
    return RoutePlan(decision=None, probs={c: u for c in ROUTABLE_CLUSTERS}, selected=[], merge_weights={}, path="fast",
                     reasons=[f"router fallback: {reason}"], uniform_mode=uniform_mode)


def timed() -> Callable[[], float]:
    t0 = time.perf_counter()
    return lambda: (time.perf_counter() - t0) * 1000
