"""OpenAI-compatible HTTP surface for the Conductor."""

from __future__ import annotations

import hmac
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .config import Settings, load_settings
from .orchestrator import Orchestrator, Result, ServiceUnavailable
from .schemas import ChatRequest, RoutePlan

log = logging.getLogger("conductor.server")


def _plan_meta(rp: RoutePlan, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = {
        "path": rp.path,
        "probs": {c: round(p, 4) for c, p in rp.probs.items()},
        "selected": rp.selected,
        "merge_weights": {c: round(w, 4) for c, w in rp.merge_weights.items()},
        "intent": rp.decision.intent if rp.decision else None,
        "reasons": rp.reasons,
        "uniform_mode": rp.uniform_mode,
        "route_ms": round(rp.route_ms, 1),
    }
    if extra:
        meta.update(extra)
    return meta


def _completion_body(req: ChatRequest, res: Result, rid: str) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": res.content}
    if res.reasoning:
        message["reasoning_content"] = res.reasoning
    body: dict[str, Any] = {
        "id": rid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": res.usage,
    }
    if req.debug:
        body["x_conductor"] = _plan_meta(res.plan, {
            "synthesised": res.synthesised,
            "latency_ms": round(res.latency_ms, 1),
            "specialists": {c: {"ok": r.ok, "latency_ms": round(r.latency_ms, 1), "error": r.error}
                            for c, r in res.outputs.items()},
        })
    return body


def _chunk(rid: str, model: str, delta: dict[str, Any], finish: str | None = None, **extra: Any) -> str:
    body = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"


def create_app(settings: Settings | None = None, orchestrator: Orchestrator | None = None) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        orch = orchestrator or Orchestrator(settings)
        app.state.orch = orch
        await orch.start()
        try:
            yield
        finally:
            await orch.aclose()

    app = FastAPI(title="Conductor", version=settings.version, lifespan=lifespan)
    keys_env = settings.server.api_keys_env
    api_keys = [k.strip() for k in os.environ.get(keys_env, "").split(",") if k.strip()] if keys_env else []

    async def auth(request: Request) -> None:
        if not api_keys:
            return
        header = request.headers.get("authorization", "")
        token = header.removeprefix("Bearer ").strip()
        if not any(hmac.compare_digest(token, k) for k in api_keys):
            raise HTTPException(status_code=401, detail="invalid API key")

    def orch_of(request: Request) -> Orchestrator:
        return request.app.state.orch

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        size = request.headers.get("content-length")
        if size and int(size) > settings.server.max_request_bytes:
            return JSONResponse({"error": {"message": "request too large"}}, status_code=413)
        return await call_next(request)

    @app.post("/v1/chat/completions", dependencies=[Depends(auth)])
    async def chat(req: ChatRequest, request: Request):
        orch = orch_of(request)
        try:
            req.last_user()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        rid = f"chatcmpl-{uuid.uuid4().hex}"

        if not req.stream:
            try:
                res = await orch.run(req)
            except ServiceUnavailable as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
            return _completion_body(req, res, rid)

        async def sse():
            yield _chunk(rid, req.model, {"role": "assistant", "content": ""})
            try:
                async for ev in orch.stream(req):
                    if ev.kind == "plan" and req.debug:
                        yield _chunk(rid, req.model, {}, x_conductor=_plan_meta(ev.plan))
                    elif ev.kind == "delta" and ev.delta:
                        delta: dict[str, Any] = {}
                        if ev.delta.content:
                            delta["content"] = ev.delta.content
                        if ev.delta.reasoning:
                            delta["reasoning_content"] = ev.delta.reasoning
                        if delta:
                            yield _chunk(rid, req.model, delta)
                    elif ev.kind == "done":
                        yield _chunk(rid, req.model, {}, ev.info.get("finish_reason", "stop"))
            except ServiceUnavailable as e:
                yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'service_unavailable'}})}\n\n"
            except Exception:
                log.exception("stream failed")
                yield f"data: {json.dumps({'error': {'message': 'internal error', 'type': 'server_error'}})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/v1/route", dependencies=[Depends(auth)])
    async def route_only(req: ChatRequest, request: Request):
        """Run L0 only and return the plan. No specialists fire. For router evaluation."""
        return _plan_meta(await orch_of(request).route(req))

    @app.get("/v1/models", dependencies=[Depends(auth)])
    async def models():
        return {"object": "list", "data": [{"id": "conductor", "object": "model", "owned_by": "self-hosted"}]}

    @app.get("/v1/pool", dependencies=[Depends(auth)])
    async def pool(request: Request):
        orch = orch_of(request)
        tiers = [{"tier": "L0", "name": settings.conductor.name, "model": settings.conductor.endpoint.model,
                  "experts": settings.conductor.experts, "experts_per_token": settings.conductor.experts_per_token}]
        for s in [*settings.specialists, settings.fast_path]:
            tiers.append({"tier": "L1", "cluster": s.cluster, "name": s.name, "model": s.endpoint.model,
                          "architecture": s.architecture, "experts": s.experts, "experts_per_token": s.experts_per_token,
                          "breaker": orch.breakers[s.cluster].state.value})
        return {"version": settings.version, "routing": settings.routing.model_dump(), "tiers": tiers}

    @app.get("/v1/monitor", dependencies=[Depends(auth)])
    async def monitor(request: Request):
        reports = await orch_of(request).monitor.check()
        return {k: {"window_s": r.window_s, "total": r.total, "shares": r.shares, "violations": r.violations}
                for k, r in reports.items()} | {"uniform_mode": orch_of(request).monitor.uniform_mode()}

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request):
        orch = orch_of(request)
        status = {name: await b.health() for name, b in orch.backends.items()}
        ready = status.get("conductor", False) and status.get("fast", False)
        return JSONResponse({"ready": ready, "backends": status}, status_code=200 if ready else 503)

    @app.get("/metrics")
    async def metrics(request: Request):
        return Response(generate_latest(orch_of(request).metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    uvicorn.run(create_app(), host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8080")),
                timeout_keep_alive=75)


if __name__ == "__main__":
    main()
