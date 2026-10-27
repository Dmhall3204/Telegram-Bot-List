"""Model backends: an OpenAI-compatible HTTP client and a deterministic mock.

vLLM and SGLang both expose ``/v1/chat/completions`` with ``response_format``
JSON-schema guided decoding and a separate ``reasoning_content`` field when a
reasoning parser is enabled, so one client covers every specialist.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .config import Endpoint

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


class BackendError(RuntimeError):
    pass


@dataclass
class Completion:
    content: str
    reasoning: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    ttft_ms: float = 0.0
    latency_ms: float = 0.0
    finish_reason: str | None = None


@dataclass
class Delta:
    content: str = ""
    reasoning: str = ""
    finish_reason: str | None = None


class Backend(Protocol):
    endpoint: Endpoint

    async def complete(self, messages: list[dict[str, Any]], **params: Any) -> Completion: ...

    def stream(self, messages: list[dict[str, Any]], **params: Any) -> AsyncIterator[Delta]: ...

    async def health(self) -> bool: ...

    async def aclose(self) -> None: ...


def split_think(content: str, reasoning: str) -> tuple[str, str]:
    """Move inline ``<think>`` blocks into the reasoning channel (parser-less servers)."""
    found = _THINK_RE.findall(content)
    if found:
        reasoning = (reasoning + "\n" + "\n".join(found)).strip()
        content = _THINK_RE.sub("", content)
    elif "</think>" in content:  # template opened the tag in the prompt
        head, _, content = content.partition("</think>")
        reasoning = (reasoning + "\n" + head).strip()
    return content.strip(), reasoning


class OpenAIBackend:
    def __init__(self, endpoint: Endpoint, client: httpx.AsyncClient | None = None) -> None:
        self.endpoint = endpoint
        headers = {"Content-Type": "application/json"}
        if endpoint.api_key:
            headers["Authorization"] = f"Bearer {endpoint.api_key}"
        self._client = client or httpx.AsyncClient(
            base_url=endpoint.base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(endpoint.timeout_s, connect=endpoint.connect_timeout_s),
            limits=httpx.Limits(max_connections=512, max_keepalive_connections=128),
        )

    def _body(self, messages: list[dict[str, Any]], stream: bool, params: dict[str, Any]) -> dict[str, Any]:
        ep = self.endpoint
        temperature = params.pop("temperature", None)
        max_tokens = params.pop("max_tokens", None)
        body: dict[str, Any] = {
            "model": ep.model,
            "messages": messages,
            "max_tokens": max_tokens or ep.max_tokens,
            "temperature": ep.temperature if temperature is None else temperature,
            "top_p": ep.top_p,
            "stream": stream,
        }
        body.update(ep.extra_body)
        body.update({k: v for k, v in params.items() if v is not None})
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    async def complete(self, messages: list[dict[str, Any]], **params: Any) -> Completion:
        t0 = time.perf_counter()
        try:
            r = await self._client.post("/chat/completions", json=self._body(messages, False, dict(params)))
        except httpx.HTTPError as e:
            raise BackendError(f"{self.endpoint.model}: transport error: {e!r}") from e
        if r.status_code >= 400:
            raise BackendError(f"{self.endpoint.model}: HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        try:
            choice = data["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError) as e:
            raise BackendError(f"{self.endpoint.model}: malformed response: {data!r:.500}") from e
        content, reasoning = split_think(msg.get("content") or "", msg.get("reasoning_content") or msg.get("reasoning") or "")
        latency = (time.perf_counter() - t0) * 1000
        return Completion(
            content=content,
            reasoning=reasoning,
            usage=data.get("usage") or {},
            ttft_ms=latency,
            latency_ms=latency,
            finish_reason=choice.get("finish_reason"),
        )

    async def stream(self, messages: list[dict[str, Any]], **params: Any) -> AsyncIterator[Delta]:
        body = self._body(messages, True, dict(params))
        try:
            async with self._client.stream("POST", "/chat/completions", json=body) as r:
                if r.status_code >= 400:
                    text = (await r.aread()).decode(errors="replace")
                    raise BackendError(f"{self.endpoint.model}: HTTP {r.status_code}: {text[:500]}")
                in_think = False
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    if not chunk.get("choices"):
                        continue
                    choice = chunk["choices"][0]
                    d = choice.get("delta") or {}
                    content = d.get("content") or ""
                    reasoning = d.get("reasoning_content") or d.get("reasoning") or ""
                    # Parser-less servers inline <think>; route it to the reasoning channel.
                    if "<think>" in content:
                        in_think = True
                        content = content.replace("<think>", "")
                    if in_think:
                        if "</think>" in content:
                            r_part, _, content = content.partition("</think>")
                            reasoning += r_part
                            in_think = False
                        else:
                            reasoning += content
                            content = ""
                    if content or reasoning or choice.get("finish_reason"):
                        yield Delta(content=content, reasoning=reasoning, finish_reason=choice.get("finish_reason"))
        except httpx.HTTPError as e:
            raise BackendError(f"{self.endpoint.model}: transport error: {e!r}") from e

    async def health(self) -> bool:
        try:
            r = await self._client.get("/models", timeout=3.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------- mock

_KEYWORDS = {
    "code": ("code", "function", "bug", "python", "javascript", "sql", "compile", "stack trace", "refactor", "api", "regex", "script"),
    "reasoning": ("prove", "proof", "math", "solve", "logic", "probability", "plan", "why", "derive", "equation", "puzzle", "optimal"),
    "vision": ("image", "screenshot", "chart", "photo", "diagram", "ocr", "picture"),
    "long_ctx": ("document", "transcript", "summarize", "summarise", "codebase", "report", "pdf", "chapter", "contract"),
}
_TRIVIAL = ("hi", "hello", "hey", "thanks", "thank you", "good morning", "what time")


class MockBackend:
    """Deterministic stand-in for a model server. Used for tests and GPU-less runs.

    The mock Conductor routes on keywords and image presence; mock specialists
    echo a tagged answer. ``mock://name?fail=1`` always fails; ``latency_ms``
    simulates model time.
    """

    def __init__(self, endpoint: Endpoint, latency_ms: float = 5.0) -> None:
        self.endpoint = endpoint
        self.name = endpoint.base_url.removeprefix("mock://").split("?")[0]
        self.fail = "fail=1" in endpoint.base_url
        self.malformed = "malformed=1" in endpoint.base_url
        self.latency_ms = latency_ms
        self.calls: list[dict[str, Any]] = []

    def _route(self, messages: list[dict[str, Any]]) -> str:
        text = str(messages[-1]["content"]).lower()
        images = "images_attached: 0" not in text
        query = text.split("query:", 1)[-1].strip()
        scores = {c: 0.05 + sum(query.count(k) for k in kws) for c, kws in _KEYWORDS.items()}
        if images:
            scores["vision"] += 3
        total = sum(scores.values())
        weights = {c: round(v / total, 4) for c, v in scores.items()}
        trivial = (not images) and (query.strip(" ?!.") in _TRIVIAL or (len(query) < 25 and max(scores.values()) < 1))
        top = sorted(weights, key=weights.get, reverse=True)[:2]
        return json.dumps(
            {
                "intent": "mock-classified",
                "trivial": trivial,
                "weights": weights,
                "subqueries": {c: f"[{c}] {query[:200]}" for c in top},
                "synthesise": weights[top[1]] >= 0.10,
            }
        )

    def _answer(self, messages: list[dict[str, Any]], params: dict[str, Any]) -> tuple[str, str]:
        if self.malformed and "response_format" in params:
            return "not json at all", ""
        if "response_format" in params:
            return self._route(messages), ""
        last = messages[-1]["content"]
        text = last if isinstance(last, str) else " ".join(p.get("text", "") for p in last if p.get("type") == "text")
        reasoning = f"mock reasoning trace for: {text[:80]}" if self.name == "reasoning" else ""
        return f"[{self.name}] answer to: {text[:400]}", reasoning

    async def complete(self, messages: list[dict[str, Any]], **params: Any) -> Completion:
        self.calls.append({"messages": messages, **params})
        await asyncio.sleep(self.latency_ms / 1000)
        if self.fail:
            raise BackendError(f"mock://{self.name} configured to fail")
        content, reasoning = self._answer(messages, params)
        return Completion(content=content, reasoning=reasoning, usage={"completion_tokens": len(content.split())},
                          ttft_ms=self.latency_ms, latency_ms=self.latency_ms, finish_reason="stop")

    async def stream(self, messages: list[dict[str, Any]], **params: Any) -> AsyncIterator[Delta]:
        self.calls.append({"messages": messages, "stream": True, **params})
        await asyncio.sleep(self.latency_ms / 1000)
        if self.fail:
            raise BackendError(f"mock://{self.name} configured to fail")
        content, reasoning = self._answer(messages, params)
        if reasoning:
            yield Delta(reasoning=reasoning)
        for word in re.findall(r"\S+\s*", content):
            yield Delta(content=word)
        yield Delta(finish_reason="stop")

    async def health(self) -> bool:
        return not self.fail

    async def aclose(self) -> None:
        return None


def make_backend(endpoint: Endpoint) -> Backend:
    return MockBackend(endpoint) if endpoint.is_mock else OpenAIBackend(endpoint)
