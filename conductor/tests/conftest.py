from __future__ import annotations

from pathlib import Path

import pytest

from conductor.backends import MockBackend
from conductor.config import load_settings
from conductor.orchestrator import Orchestrator
from conductor.telemetry import Metrics

POOL = Path(__file__).parent.parent / "config" / "pool.yaml"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("CONDUCTOR_URL", "CODE_URL", "REASONING_URL", "VISION_URL", "LONG_CTX_URL", "FAST_URL",
                "MONITOR_STORE", "CONDUCTOR_API_KEYS"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def settings():
    return load_settings(POOL)


def build(settings, **flags: str) -> tuple[Orchestrator, dict[str, MockBackend]]:
    """Orchestrator over mock backends. flags: cluster -> query string, e.g. code="fail=1"."""
    backends: dict[str, MockBackend] = {}
    endpoints = {"conductor": settings.conductor.endpoint}
    endpoints |= {s.cluster: s.endpoint for s in [*settings.specialists, settings.fast_path]}
    for name, ep in endpoints.items():
        q = flags.get(name)
        url = f"mock://{name}" + (f"?{q}" if q else "")
        backends[name] = MockBackend(ep.model_copy(update={"base_url": url}), latency_ms=1)
    return Orchestrator(settings, backends=backends, metrics=Metrics()), backends
