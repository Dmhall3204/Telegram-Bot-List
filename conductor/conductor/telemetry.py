"""Prometheus metrics and OpenTelemetry tracing.

Tracing uses the OTel API, which is a no-op until an SDK is configured; the SDK
and OTLP exporter are optional dependencies enabled by ``telemetry.otlp_endpoint``.
"""

from __future__ import annotations

import logging

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from .config import TelemetryConfig

log = logging.getLogger("conductor")

try:
    from opentelemetry import trace as _otel_trace
except ImportError:  # pragma: no cover - API is a hard dependency, kept defensive
    _otel_trace = None

_LATENCY_BUCKETS = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2, 4, 8, 16, 32, 64, 128)


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.requests = Counter("conductor_requests_total", "Requests by execution path", ["path", "stream"], registry=r)
        self.request_latency = Histogram("conductor_request_seconds", "End-to-end latency (non-streaming)", ["path"],
                                         buckets=_LATENCY_BUCKETS, registry=r)
        self.stage_latency = Histogram("conductor_stage_seconds", "Stage latency", ["stage"], buckets=_LATENCY_BUCKETS, registry=r)
        self.route_prob = Histogram("conductor_route_probability", "L0 probability per cluster", ["cluster"],
                                    buckets=(0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0), registry=r)
        self.router_fallbacks = Counter("conductor_router_fallbacks_total", "L0 failures that fell back to fast path",
                                        ["reason"], registry=r)
        self.router_retries = Counter("conductor_router_retries_total", "L0 malformed-output retries", registry=r)
        self.invocations = Counter("conductor_specialist_invocations_total", "Specialist calls", ["cluster", "status"], registry=r)
        self.specialist_latency = Histogram("conductor_specialist_seconds", "Specialist latency", ["cluster"],
                                            buckets=_LATENCY_BUCKETS, registry=r)
        self.capacity_drops = Counter("conductor_capacity_drops_total", "Clusters above min_weight dropped by capacity factor",
                                      ["cluster"], registry=r)
        self.breaker_state = Gauge("conductor_breaker_open", "1 if the specialist breaker is open", ["cluster"], registry=r)
        self.load_share = Gauge("conductor_load_share", "Share of L1 invocations per cluster", ["cluster", "window"], registry=r)
        self.uniform_mode = Gauge("conductor_uniform_mode", "1 while routing-collapse fallback is active", registry=r)
        self.alerts = Counter("conductor_alerts_total", "Monitor alerts raised", ["kind", "cluster"], registry=r)
        self.fast_ttft = Histogram("conductor_fast_path_ttft_seconds", "Fast-path time to first token", buckets=_LATENCY_BUCKETS,
                                   registry=r)
        self.tokens = Counter("conductor_completion_tokens_total", "Completion tokens by cluster", ["cluster"], registry=r)


class _NoopSpan:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def set_attribute(self, *a, **k):
        return None


class Tracer:
    def __init__(self, cfg: TelemetryConfig) -> None:
        self._tracer = None
        if _otel_trace is None:
            return
        if cfg.otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                from opentelemetry.sdk.resources import Resource
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                provider = TracerProvider(resource=Resource.create({"service.name": cfg.service_name}))
                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=cfg.otlp_endpoint)))
                _otel_trace.set_tracer_provider(provider)
            except ImportError:
                log.warning("otlp_endpoint set but opentelemetry-sdk / exporter not installed; tracing disabled")
        self._tracer = _otel_trace.get_tracer("conductor")

    def span(self, name: str, **attrs):
        if self._tracer is None:
            return _NoopSpan()
        return self._tracer.start_as_current_span(name, attributes={k: v for k, v in attrs.items() if v is not None})
