"""The request lifecycle: fast-path shadow, L0 routing, L1 fan-out, weighted synthesis.

    t=0      fast path starts (streaming into a buffer)    L0 Conductor routes
    ~50ms    routing decision -> plan (soft top-k, capacity, breakers)
    ~200ms   selected L1 specialists fire in parallel (each runs its own L2 token gating)
    ~1.2s+   k=1: specialist output returned directly
             k>1: Conductor re-invoked as synthesiser, outputs weighted by L0 probability
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .backends import Backend, BackendError, Completion, Delta, make_backend
from .breaker import CircuitBreaker, State
from .config import Settings
from .monitor import Monitor
from .router import Router, RoutingFailed, fallback_plan, plan
from .schemas import ChatRequest, ContentPart, RoutePlan
from .telemetry import Metrics, Tracer

log = logging.getLogger("conductor")


class ServiceUnavailable(RuntimeError):
    pass


@dataclass
class SpecialistResult:
    cluster: str
    ok: bool
    completion: Completion | None = None
    error: str | None = None
    latency_ms: float = 0.0


@dataclass
class Result:
    content: str
    reasoning: str
    plan: RoutePlan
    outputs: dict[str, SpecialistResult] = field(default_factory=dict)
    synthesised: bool = False
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass
class StreamEvent:
    kind: str  # "plan" | "delta" | "done"
    plan: RoutePlan | None = None
    delta: Delta | None = None
    info: dict[str, Any] = field(default_factory=dict)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n[... truncated {len(text) - limit} chars ...]"


class FastShadow:
    """Streams the dense fast path into a buffer so it can be replayed live or awaited whole."""

    def __init__(self, orch: "Orchestrator", messages: list[dict[str, Any]], max_tokens: int | None) -> None:
        self._orch = orch
        self._deltas: list[Delta] = []
        self._changed = asyncio.Event()
        self.done = False
        self.error: str | None = None
        self.t0 = time.perf_counter()
        self.ttft_ms: float | None = None
        self.task = asyncio.create_task(self._run(messages, max_tokens), name="fast-path")

    async def _run(self, messages: list[dict[str, Any]], max_tokens: int | None) -> None:
        orch = self._orch
        breaker = orch.breakers["fast"]
        try:
            if not breaker.acquire():
                raise BackendError("fast path breaker open")
            with orch.tracer.span("l1.fast"):
                async for d in orch.backends["fast"].stream(messages, max_tokens=max_tokens):
                    if self.ttft_ms is None and (d.content or d.reasoning):
                        self.ttft_ms = (time.perf_counter() - self.t0) * 1000
                        orch.metrics.fast_ttft.observe(self.ttft_ms / 1000)
                    self._deltas.append(d)
                    self._changed.set()
            breaker.record_success()
            orch.metrics.invocations.labels("fast", "ok").inc()
        except asyncio.CancelledError:
            orch.metrics.invocations.labels("fast", "cancelled").inc()
            breaker.release()
            raise
        except Exception as e:  # noqa: BLE001 - shadow path must never raise into the request
            self.error = str(e)
            breaker.record_failure()
            orch.metrics.invocations.labels("fast", "error").inc()
            log.warning("fast path failed: %s", e)
        finally:
            self.done = True
            self._changed.set()

    def text(self) -> str:
        return "".join(d.content for d in self._deltas)

    async def result(self) -> str | None:
        try:
            await asyncio.shield(self.task)
        except asyncio.CancelledError:
            return None
        return None if self.error else self.text()

    def ready_text(self) -> str | None:
        return self.text() if self.done and not self.error else None

    async def replay(self) -> AsyncIterator[Delta]:
        i = 0
        while True:
            while i < len(self._deltas):
                yield self._deltas[i]
                i += 1
            if self.done:
                return
            self._changed.clear()
            if i < len(self._deltas) or self.done:
                continue
            await self._changed.wait()

    def cancel(self) -> None:
        if not self.task.done():
            self.task.cancel()


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        *,
        backends: dict[str, Backend] | None = None,
        metrics: Metrics | None = None,
        monitor: Monitor | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.settings = settings
        self.metrics = metrics or Metrics()
        self.tracer = tracer or Tracer(settings.telemetry)
        self.monitor = monitor or Monitor(settings.monitor, self.metrics)
        if backends is None:
            backends = {"conductor": make_backend(settings.conductor.endpoint)}
            for s in [*settings.specialists, settings.fast_path]:
                backends[s.cluster] = make_backend(s.endpoint)
        self.backends = backends
        self.breakers = {
            s.cluster: CircuitBreaker(s.cluster, s.breaker) for s in [*settings.specialists, settings.fast_path]
        }
        self.router = Router(settings, backends["conductor"], self.metrics)

    async def start(self) -> None:
        self.monitor.start()

    async def aclose(self) -> None:
        await self.monitor.stop()
        for b in self.backends.values():
            await b.aclose()

    def _sync_breaker_gauges(self) -> None:
        for c, b in self.breakers.items():
            self.metrics.breaker_state.labels(c).set(1 if b.state is State.OPEN else 0)

    # ------------------------------------------------------------ message construction

    def _specialist_messages(self, cluster: str, req: ChatRequest, subquery: str | None) -> list[dict[str, Any]]:
        spec = self.settings.specialist(cluster)
        msgs: list[dict[str, Any]] = [{"role": "system", "content": spec.system_prompt}]
        user_msgs = [m for m in req.messages if m.role != "system"]
        sys_extra = [m.text() for m in req.messages if m.role == "system"]
        if sys_extra:
            msgs[0]["content"] += "\n\n" + "\n".join(sys_extra)
        last_user_idx = max(i for i, m in enumerate(user_msgs) if m.role == "user")
        for i, m in enumerate(user_msgs):
            msg = m.to_openai(keep_images=spec.accepts_images)
            if i == last_user_idx and subquery:
                focus = f"\n\n[Focus for this part of the task]: {subquery}"
                if isinstance(msg["content"], list):
                    msg["content"] = [*msg["content"], ContentPart(type="text", text=focus).model_dump(exclude_none=True)]
                else:
                    msg["content"] = f"{msg['content']}{focus}"
            msgs.append(msg)
        return msgs

    def _synthesis_messages(self, req: ChatRequest, rp: RoutePlan, ok: dict[str, SpecialistResult],
                            fast_text: str | None) -> list[dict[str, Any]]:
        cfg = self.settings
        mass = sum(rp.merge_weights[c] for c in ok) or 1.0
        parts = [f"User request:\n{_clip(req.last_user().text(), cfg.conductor.routing_input_chars)}"]
        if req.image_count():
            parts.append(f"({req.image_count()} image(s) were attached; the vision specialist has read them.)")
        for c in sorted(ok, key=lambda c: rp.merge_weights[c], reverse=True):
            spec = cfg.specialist(c)
            w = rp.merge_weights[c] / mass
            parts.append(f"### {spec.name} output (routing probability {w:.2f})\n"
                         f"{_clip(ok[c].completion.content, cfg.synthesis.max_chars_per_output)}")
        if fast_text and cfg.synthesis.attach_fast_path:
            parts.append(f"### Fast draft (reference only, lowest trust)\n{_clip(fast_text, 4000)}")
        parts.append("Write the final answer to the user now.")
        return [{"role": "system", "content": cfg.conductor.synthesis_prompt}, {"role": "user", "content": "\n\n".join(parts)}]

    # ------------------------------------------------------------ stages

    async def route(self, req: ChatRequest) -> RoutePlan:
        t0 = time.perf_counter()
        uniform = self.monitor.uniform_mode()
        with self.tracer.span("l0.route") as span:
            try:
                decision = await self.router.decide(req)
                rp = plan(decision, req, self.settings, available=lambda c: self.breakers[c].available(),
                          uniform_mode=uniform, metrics=self.metrics)
            except RoutingFailed as e:
                self.metrics.router_fallbacks.labels(e.reason).inc()
                rp = fallback_plan(e.reason, uniform)
            rp.route_ms = (time.perf_counter() - t0) * 1000
            span.set_attribute("conductor.path", rp.path)
            span.set_attribute("conductor.selected", ",".join(rp.selected))
        self.metrics.stage_latency.labels("route").observe(rp.route_ms / 1000)
        for c, p in rp.probs.items():
            self.metrics.route_prob.labels(c).observe(p)
        return rp

    async def _call(self, cluster: str, messages: list[dict[str, Any]], max_tokens: int | None) -> SpecialistResult:
        breaker = self.breakers[cluster]
        spec = self.settings.specialist(cluster)
        if not breaker.acquire():
            self.metrics.invocations.labels(cluster, "breaker_open").inc()
            return SpecialistResult(cluster, False, error="breaker open")
        t0 = time.perf_counter()
        with self.tracer.span(f"l1.{cluster}", **{"gen_ai.request.model": spec.endpoint.model}):
            try:
                comp = await asyncio.wait_for(self.backends[cluster].complete(messages, max_tokens=max_tokens),
                                              timeout=spec.endpoint.timeout_s)
            except asyncio.CancelledError:
                breaker.release()
                raise
            except (BackendError, asyncio.TimeoutError) as e:
                breaker.record_failure()
                self.metrics.invocations.labels(cluster, "error").inc()
                log.warning("specialist %s failed: %r", cluster, e)
                return SpecialistResult(cluster, False, error=repr(e), latency_ms=(time.perf_counter() - t0) * 1000)
        breaker.record_success()
        latency = (time.perf_counter() - t0) * 1000
        self.metrics.invocations.labels(cluster, "ok").inc()
        self.metrics.specialist_latency.labels(cluster).observe(latency / 1000)
        self.metrics.tokens.labels(cluster).inc(comp.usage.get("completion_tokens", 0))
        return SpecialistResult(cluster, True, completion=comp, latency_ms=latency)

    async def _dispatch(self, req: ChatRequest, rp: RoutePlan) -> dict[str, SpecialistResult]:
        subq = rp.decision.subqueries if rp.decision else {}
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *(self._call(c, self._specialist_messages(c, req, subq.get(c)), req.max_tokens) for c in rp.selected)
        )
        self.metrics.stage_latency.labels("dispatch").observe(time.perf_counter() - t0)
        await self.monitor.record(rp.selected)
        self._sync_breaker_gauges()
        return {r.cluster: r for r in results}

    def _reasoning(self, req: ChatRequest, ok: dict[str, SpecialistResult]) -> str:
        if not req.show_reasoning:
            return ""
        return "\n\n".join(f"[{self.settings.specialist(c).name}]\n{r.completion.reasoning}"
                           for c, r in ok.items() if r.completion and r.completion.reasoning)

    # ------------------------------------------------------------ non-streaming

    async def run(self, req: ChatRequest) -> Result:
        t0 = time.perf_counter()
        fast = FastShadow(self, self._specialist_messages("fast", req, None), req.max_tokens)
        try:
            with self.tracer.span("conductor.request", **{"conductor.session": req.session_id}):
                result = await self._run(req, fast)
        finally:
            fast.cancel()
        result.latency_ms = (time.perf_counter() - t0) * 1000
        self.metrics.requests.labels(result.plan.path, "false").inc()
        self.metrics.request_latency.labels(result.plan.path).observe(result.latency_ms / 1000)
        return result

    async def _fast_result(self, rp: RoutePlan, fast: FastShadow, note: str | None = None) -> Result:
        text = await fast.result()
        if text is None:
            raise ServiceUnavailable(f"fast path unavailable ({fast.error or 'cancelled'}) and no specialist answered")
        if note:
            rp.reasons.append(note)
        return Result(content=text, reasoning="", plan=rp)

    async def _run(self, req: ChatRequest, fast: FastShadow) -> Result:
        rp = await self.route(req)
        if rp.path == "fast":
            return await self._fast_result(rp, fast)

        outputs = await self._dispatch(req, rp)
        ok = {c: r for c, r in outputs.items() if r.ok}
        if not ok:
            rp.path = "fast"
            return await self._fast_result(rp, fast, "all specialists failed: fast-path answer returned")

        if len(ok) == 1:
            (c, r), = ok.items()
            if len(rp.selected) > 1:
                rp.reasons.append(f"only {c} succeeded: synthesis skipped")
            rp.path = "single"
            return Result(content=r.completion.content, reasoning=self._reasoning(req, ok), plan=rp, outputs=outputs,
                          usage=r.completion.usage)

        t1 = time.perf_counter()
        messages = self._synthesis_messages(req, rp, ok, fast.ready_text())
        with self.tracer.span("l0.synthesise"):
            try:
                comp = await self.backends["conductor"].complete(
                    messages, max_tokens=req.max_tokens or self.settings.synthesis.max_tokens,
                    temperature=self.settings.synthesis.temperature)
                content, synthesised = comp.content, True
            except BackendError as e:
                best = max(ok, key=lambda c: rp.merge_weights[c])
                rp.reasons.append(f"synthesis failed ({e}); returned highest-weighted output: {best}")
                content, synthesised = ok[best].completion.content, False
        self.metrics.stage_latency.labels("synthesis").observe(time.perf_counter() - t1)
        return Result(content=content, reasoning=self._reasoning(req, ok), plan=rp, outputs=outputs, synthesised=synthesised)

    # ------------------------------------------------------------ streaming

    async def stream(self, req: ChatRequest) -> AsyncIterator[StreamEvent]:
        fast = FastShadow(self, self._specialist_messages("fast", req, None), req.max_tokens)
        try:
            async for ev in self._stream(req, fast):
                yield ev
        finally:
            fast.cancel()

    async def _stream(self, req: ChatRequest, fast: FastShadow) -> AsyncIterator[StreamEvent]:
        rp = await self.route(req)

        if rp.path == "fast":
            yield StreamEvent("plan", plan=rp)
            async for ev in self._replay_fast(rp, fast):
                yield ev
            return

        if rp.path == "single":
            c = rp.selected[0]
            yield StreamEvent("plan", plan=rp)
            async for ev in self._stream_single(req, rp, c, fast):
                yield ev
            return

        outputs = await self._dispatch(req, rp)
        ok = {c: r for c, r in outputs.items() if r.ok}
        if not ok:
            rp.path = "fast"
            rp.reasons.append("all specialists failed: fast-path answer returned")
            yield StreamEvent("plan", plan=rp)
            async for ev in self._replay_fast(rp, fast):
                yield ev
            return

        yield StreamEvent("plan", plan=rp)
        if req.show_reasoning and (reasoning := self._reasoning(req, ok)):
            yield StreamEvent("delta", delta=Delta(reasoning=reasoning))

        if len(ok) == 1:
            (c, r), = ok.items()
            rp.reasons.append(f"only {c} succeeded: synthesis skipped")
            yield StreamEvent("delta", delta=Delta(content=r.completion.content))
            yield StreamEvent("done", info={"path": "single", "finish_reason": r.completion.finish_reason or "stop"})
            self.metrics.requests.labels("single", "true").inc()
            return

        messages = self._synthesis_messages(req, rp, ok, fast.ready_text())
        emitted = False
        try:
            with self.tracer.span("l0.synthesise"):
                async for d in self.backends["conductor"].stream(
                        messages, max_tokens=req.max_tokens or self.settings.synthesis.max_tokens,
                        temperature=self.settings.synthesis.temperature):
                    if d.reasoning and not req.show_reasoning:
                        d = Delta(content=d.content, finish_reason=d.finish_reason)
                    if d.content or d.reasoning:
                        emitted = True
                    yield StreamEvent("delta", delta=d)
        except BackendError as e:
            if emitted:
                yield StreamEvent("done", info={"path": "multi", "finish_reason": "error", "error": str(e)})
                return
            best = max(ok, key=lambda c: rp.merge_weights[c])
            yield StreamEvent("delta", delta=Delta(content=ok[best].completion.content))
            yield StreamEvent("done", info={"path": "multi", "finish_reason": "stop", "synthesised": False})
            self.metrics.requests.labels("multi", "true").inc()
            return
        yield StreamEvent("done", info={"path": "multi", "finish_reason": "stop", "synthesised": True})
        self.metrics.requests.labels("multi", "true").inc()

    async def _replay_fast(self, rp: RoutePlan, fast: FastShadow) -> AsyncIterator[StreamEvent]:
        async for d in fast.replay():
            yield StreamEvent("delta", delta=Delta(content=d.content, finish_reason=d.finish_reason))
        if fast.error:
            raise ServiceUnavailable(f"fast path failed: {fast.error}")
        yield StreamEvent("done", info={"path": "fast", "finish_reason": "stop"})
        self.metrics.requests.labels("fast", "true").inc()

    async def _stream_single(self, req: ChatRequest, rp: RoutePlan, c: str, fast: FastShadow) -> AsyncIterator[StreamEvent]:
        breaker = self.breakers[c]
        subq = rp.decision.subqueries.get(c) if rp.decision else None
        emitted = False
        await self.monitor.record([c])
        if breaker.acquire():
            t0 = time.perf_counter()
            try:
                with self.tracer.span(f"l1.{c}"):
                    async for d in self.backends[c].stream(self._specialist_messages(c, req, subq), max_tokens=req.max_tokens):
                        if d.reasoning and not req.show_reasoning:
                            d = Delta(content=d.content, finish_reason=d.finish_reason)
                        if not (d.content or d.reasoning or d.finish_reason):
                            continue
                        if d.content:
                            emitted = True
                        yield StreamEvent("delta", delta=d)
                breaker.record_success()
                self.metrics.invocations.labels(c, "ok").inc()
                self.metrics.specialist_latency.labels(c).observe(time.perf_counter() - t0)
                yield StreamEvent("done", info={"path": "single", "finish_reason": "stop"})
                self.metrics.requests.labels("single", "true").inc()
                return
            except (asyncio.CancelledError, GeneratorExit):
                breaker.release()
                raise
            except BackendError as e:
                breaker.record_failure()
                self.metrics.invocations.labels(c, "error").inc()
                log.warning("streaming specialist %s failed: %s", c, e)
                if emitted:
                    yield StreamEvent("done", info={"path": "single", "finish_reason": "error", "error": str(e)})
                    return
            finally:
                self._sync_breaker_gauges()
        else:
            self.metrics.invocations.labels(c, "breaker_open").inc()
        rp.reasons.append(f"{c} unavailable before first token: fast-path answer streamed")
        async for ev in self._replay_fast(rp, fast):
            yield ev
