"""Load-balance surveillance and routing-collapse detection.

Counts L1 invocations per cluster in time buckets. Shares are computed over all
routable invocations in a window, so they sum to 1 (uniform = 0.25 for four
clusters).

- Load balance (24h window): any cluster < balance_low or > balance_high -> alert.
  System analogue of the auxiliary load-balancing loss.
- Collapse (1h window): one cluster > collapse_high or any < collapse_low ->
  alert and enable uniform-mix routing for uniform_hold_s. Analogue of
  dead-expert detection.

The Redis store shares counters and the uniform flag across replicas.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from .config import ROUTABLE_CLUSTERS, MonitorConfig
from .telemetry import Metrics

log = logging.getLogger("conductor.monitor")


class Store(Protocol):
    async def incr(self, clusters: list[str], now: float) -> None: ...
    async def counts(self, window_s: int, now: float) -> dict[str, int]: ...
    async def set_uniform(self, hold_s: int) -> None: ...
    async def uniform_active(self) -> bool: ...
    async def mark_alert(self, key: str, ttl_s: int) -> bool: ...


class MemoryStore:
    def __init__(self, cfg: MonitorConfig, clock=time.time) -> None:
        self.cfg = cfg
        self._clock = clock
        self._buckets: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._uniform_until = 0.0
        self._alerts: dict[str, float] = {}

    def _bucket(self, now: float) -> int:
        return int(now // self.cfg.bucket_s)

    async def incr(self, clusters: list[str], now: float) -> None:
        b = self._buckets[self._bucket(now)]
        for c in clusters:
            b[c] += 1
        horizon = self._bucket(now - max(self.cfg.balance_window_s, self.cfg.collapse_window_s)) - 1
        for k in [k for k in self._buckets if k < horizon]:
            del self._buckets[k]

    async def counts(self, window_s: int, now: float) -> dict[str, int]:
        lo = self._bucket(now - window_s)
        out = {c: 0 for c in ROUTABLE_CLUSTERS}
        for k, b in self._buckets.items():
            if k > lo:
                for c, n in b.items():
                    out[c] = out.get(c, 0) + n
        return out

    async def set_uniform(self, hold_s: int) -> None:
        self._uniform_until = self._clock() + hold_s

    async def uniform_active(self) -> bool:
        return self._clock() < self._uniform_until

    async def mark_alert(self, key: str, ttl_s: int) -> bool:
        now = self._clock()
        if self._alerts.get(key, 0) > now:
            return False
        self._alerts[key] = now + ttl_s
        return True


class RedisStore:
    def __init__(self, cfg: MonitorConfig, client=None) -> None:
        import redis.asyncio as aioredis

        self.cfg = cfg
        self.r = client or aioredis.from_url(cfg.redis_url, decode_responses=True)
        self.p = cfg.key_prefix

    def _key(self, bucket: int) -> str:
        return f"{self.p}:lb:{bucket}"

    async def incr(self, clusters: list[str], now: float) -> None:
        key = self._key(int(now // self.cfg.bucket_s))
        ttl = max(self.cfg.balance_window_s, self.cfg.collapse_window_s) + 2 * self.cfg.bucket_s
        pipe = self.r.pipeline()
        for c in clusters:
            pipe.hincrby(key, c, 1)
        pipe.expire(key, ttl)
        await pipe.execute()

    async def counts(self, window_s: int, now: float) -> dict[str, int]:
        hi = int(now // self.cfg.bucket_s)
        lo = int((now - window_s) // self.cfg.bucket_s)
        pipe = self.r.pipeline()
        for b in range(lo + 1, hi + 1):
            pipe.hgetall(self._key(b))
        out = {c: 0 for c in ROUTABLE_CLUSTERS}
        for h in await pipe.execute():
            for c, n in (h or {}).items():
                out[c] = out.get(c, 0) + int(n)
        return out

    async def set_uniform(self, hold_s: int) -> None:
        await self.r.set(f"{self.p}:uniform", "1", ex=hold_s)

    async def uniform_active(self) -> bool:
        return bool(await self.r.exists(f"{self.p}:uniform"))

    async def mark_alert(self, key: str, ttl_s: int) -> bool:
        return bool(await self.r.set(f"{self.p}:alert:{key}", "1", ex=ttl_s, nx=True))


@dataclass
class Report:
    window_s: int
    total: int
    shares: dict[str, float]
    violations: list[str] = field(default_factory=list)


class Monitor:
    def __init__(self, cfg: MonitorConfig, metrics: Metrics, store: Store | None = None, clock=time.time) -> None:
        self.cfg = cfg
        self.metrics = metrics
        self._clock = clock
        if store is None:
            store = RedisStore(cfg) if cfg.store == "redis" else MemoryStore(cfg, clock)
        self.store = store
        self._uniform_cached = False
        self._task: asyncio.Task | None = None
        self._http: httpx.AsyncClient | None = None

    async def record(self, clusters: list[str]) -> None:
        if clusters:
            try:
                await self.store.incr(clusters, self._clock())
            except Exception:  # monitoring must never fail a request
                log.exception("monitor record failed")

    def uniform_mode(self) -> bool:
        return self._uniform_cached

    async def _report(self, window_s: int) -> Report:
        counts = await self.store.counts(window_s, self._clock())
        total = sum(counts.values())
        shares = {c: (counts.get(c, 0) / total if total else 0.0) for c in ROUTABLE_CLUSTERS}
        return Report(window_s=window_s, total=total, shares=shares)

    async def check(self) -> dict[str, Report]:
        cfg = self.cfg
        balance = await self._report(cfg.balance_window_s)
        collapse = await self._report(cfg.collapse_window_s)
        for c in ROUTABLE_CLUSTERS:
            self.metrics.load_share.labels(c, "balance").set(balance.shares[c])
            self.metrics.load_share.labels(c, "collapse").set(collapse.shares[c])

        if balance.total >= cfg.balance_min_samples:
            for c, s in balance.shares.items():
                if s < cfg.balance_low or s > cfg.balance_high:
                    balance.violations.append(f"{c}={s:.3f}")
                    await self._alert("load_balance", c, f"cluster {c} share {s:.1%} outside "
                                      f"[{cfg.balance_low:.0%}, {cfg.balance_high:.0%}] over {cfg.balance_window_s}s; review router prompt")

        if collapse.total >= cfg.collapse_min_samples:
            for c, s in collapse.shares.items():
                if s > cfg.collapse_high or s < cfg.collapse_low:
                    collapse.violations.append(f"{c}={s:.3f}")
            if collapse.violations:
                await self.store.set_uniform(cfg.uniform_hold_s)
                await self._alert("routing_collapse", ",".join(collapse.violations),
                                  f"routing collapse {collapse.violations}; uniform-mix routing for {cfg.uniform_hold_s}s, "
                                  "manual router prompt review required")

        self._uniform_cached = await self.store.uniform_active()
        self.metrics.uniform_mode.set(1 if self._uniform_cached else 0)
        return {"balance": balance, "collapse": collapse}

    async def _alert(self, kind: str, cluster: str, message: str) -> None:
        # One alert per kind+cluster per hour across all replicas.
        if not await self.store.mark_alert(f"{kind}:{cluster}", 3600):
            return
        self.metrics.alerts.labels(kind, cluster).inc()
        log.warning("ALERT %s: %s", kind, message)
        if self.cfg.alert_webhook_url:
            self._http = self._http or httpx.AsyncClient(timeout=5.0)
            try:
                await self._http.post(self.cfg.alert_webhook_url, json={"kind": kind, "cluster": cluster, "text": message})
            except httpx.HTTPError:
                log.exception("alert webhook failed")

    async def _loop(self) -> None:
        while True:
            try:
                await self.check()
            except Exception:
                log.exception("monitor check failed")
            await asyncio.sleep(self.cfg.check_interval_s)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="conductor-monitor")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._http:
            await self._http.aclose()
