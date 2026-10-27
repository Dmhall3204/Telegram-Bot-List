"""Typed configuration for the Conductor pool.

Everything that varies between deployments (endpoints, checkpoints, thresholds)
lives in a YAML file validated by these models. Checkpoints rotate; roles do not.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

ROUTABLE_CLUSTERS = ("code", "reasoning", "vision", "long_ctx")

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


class Endpoint(BaseModel):
    """An OpenAI-compatible chat endpoint (vLLM / SGLang) or ``mock://<name>``."""

    base_url: str
    model: str
    api_key_env: str | None = None
    timeout_s: float = 120.0
    connect_timeout_s: float = 5.0
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.8
    extra_body: dict = Field(default_factory=dict)

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None

    @property
    def is_mock(self) -> bool:
        return self.base_url.startswith("mock://")


class BreakerConfig(BaseModel):
    failure_threshold: int = 5
    window_s: float = 60.0
    cooldown_s: float = 30.0
    half_open_max_calls: int = 1


class SpecialistConfig(BaseModel):
    name: str
    cluster: str
    role: str
    architecture: Literal["moe", "dense"]
    total_params_b: float
    active_params_b: float
    experts: int | None = None
    experts_per_token: int | None = None
    shared_experts: int = 0
    native_context: int
    accepts_images: bool = False
    returns_reasoning: bool = False
    system_prompt: str
    endpoint: Endpoint
    breaker: BreakerConfig = Field(default_factory=BreakerConfig)

    @model_validator(mode="after")
    def _moe_has_gate(self) -> "SpecialistConfig":
        if self.architecture == "moe" and not (self.experts and self.experts_per_token):
            raise ValueError(f"{self.name}: MoE specialist needs experts and experts_per_token")
        return self


class ConductorConfig(BaseModel):
    name: str = "Conductor"
    total_params_b: float
    active_params_b: float
    experts: int
    experts_per_token: int
    native_context: int
    endpoint: Endpoint
    routing_prompt: str
    synthesis_prompt: str
    routing_max_tokens: int = 512
    routing_input_chars: int = 12000
    route_retries: int = 1


class RoutingConfig(BaseModel):
    capacity_factor: int = 2
    min_weight: float = 0.10
    images_force_vision: bool = True
    long_ctx_force_tokens: int = 24000
    chars_per_token: float = 3.5
    collapse_uniform_mix: float = 0.5

    @field_validator("capacity_factor")
    @classmethod
    def _cap(cls, v: int) -> int:
        if not 1 <= v <= len(ROUTABLE_CLUSTERS):
            raise ValueError("capacity_factor must be between 1 and the number of routable clusters")
        return v

    @field_validator("collapse_uniform_mix")
    @classmethod
    def _mix(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("collapse_uniform_mix must be within [0, 1]")
        return v


class MonitorConfig(BaseModel):
    store: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://localhost:6379/0"
    key_prefix: str = "conductor"
    bucket_s: int = 60
    balance_window_s: int = 86400
    balance_low: float = 0.05
    balance_high: float = 0.60
    balance_min_samples: int = 500
    collapse_window_s: int = 3600
    collapse_high: float = 0.80
    collapse_low: float = 0.02
    collapse_min_samples: int = 100
    uniform_hold_s: int = 900
    check_interval_s: float = 30.0
    alert_webhook_url: str | None = None


class SynthesisConfig(BaseModel):
    max_tokens: int = 4096
    temperature: float = 0.4
    max_chars_per_output: int = 24000
    attach_fast_path: bool = True


class TelemetryConfig(BaseModel):
    service_name: str = "conductor"
    otlp_endpoint: str | None = None


class ServerConfig(BaseModel):
    api_keys_env: str | None = "CONDUCTOR_API_KEYS"
    max_request_bytes: int = 32 * 1024 * 1024


class Settings(BaseModel):
    version: str = "0.3"
    conductor: ConductorConfig
    specialists: list[SpecialistConfig]
    fast_path: SpecialistConfig
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @model_validator(mode="after")
    def _one_specialist_per_cluster(self) -> "Settings":
        clusters = [s.cluster for s in self.specialists]
        if sorted(clusters) != sorted(ROUTABLE_CLUSTERS):
            raise ValueError(f"specialists must cover exactly {ROUTABLE_CLUSTERS}, got {clusters}")
        if self.fast_path.cluster != "fast":
            raise ValueError("fast_path.cluster must be 'fast'")
        if self.fast_path.architecture != "dense":
            raise ValueError("fast_path must be dense (no internal gating) to keep TTFT low")
        return self

    def specialist(self, cluster: str) -> SpecialistConfig:
        if cluster == "fast":
            return self.fast_path
        for s in self.specialists:
            if s.cluster == cluster:
                return s
        raise KeyError(cluster)


def _expand_env(text: str) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` so one YAML serves every environment."""

    def repl(m: re.Match[str]) -> str:
        return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")

    return _ENV_PATTERN.sub(repl, text)


def load_settings(path: str | os.PathLike[str] | None = None) -> Settings:
    path = Path(path or os.environ.get("CONDUCTOR_CONFIG", Path(__file__).parent.parent / "config" / "pool.yaml"))
    raw = yaml.safe_load(_expand_env(path.read_text()))
    return Settings.model_validate(raw)
