"""Per-specialist circuit breaker.

An open breaker removes a cluster from selection, so its capacity slot passes to
the next-ranked cluster: the system-level analogue of token re-routing when an
expert is over capacity.
"""

from __future__ import annotations

import time
from collections import deque
from enum import Enum

from .config import BreakerConfig


class State(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, name: str, cfg: BreakerConfig, clock=time.monotonic) -> None:
        self.name = name
        self.cfg = cfg
        self._clock = clock
        self._failures: deque[float] = deque()
        self._opened_at = 0.0
        self._half_open_inflight = 0
        self.state = State.CLOSED

    def _trim(self, now: float) -> None:
        while self._failures and now - self._failures[0] > self.cfg.window_s:
            self._failures.popleft()

    def available(self) -> bool:
        """True if a call may be attempted now. Does not reserve a half-open slot."""
        now = self._clock()
        if self.state is State.OPEN and now - self._opened_at >= self.cfg.cooldown_s:
            self.state = State.HALF_OPEN
            self._half_open_inflight = 0
        if self.state is State.HALF_OPEN:
            return self._half_open_inflight < self.cfg.half_open_max_calls
        return self.state is State.CLOSED

    def acquire(self) -> bool:
        if not self.available():
            return False
        if self.state is State.HALF_OPEN:
            self._half_open_inflight += 1
        return True

    def release(self) -> None:
        """Return a half-open probe slot without a verdict (call was cancelled)."""
        if self.state is State.HALF_OPEN:
            self._half_open_inflight = max(0, self._half_open_inflight - 1)

    def record_success(self) -> None:
        if self.state is State.HALF_OPEN:
            self._half_open_inflight = max(0, self._half_open_inflight - 1)
        self.state = State.CLOSED
        self._failures.clear()

    def record_failure(self) -> None:
        now = self._clock()
        if self.state is State.HALF_OPEN:
            self._trip(now)
            return
        self._failures.append(now)
        self._trim(now)
        if len(self._failures) >= self.cfg.failure_threshold:
            self._trip(now)

    def _trip(self, now: float) -> None:
        self.state = State.OPEN
        self._opened_at = now
        self._half_open_inflight = 0
        self._failures.clear()
