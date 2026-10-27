"""Wire formats: the OpenAI-compatible request surface and the L0 routing contract."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .config import ROUTABLE_CLUSTERS

# ---------------------------------------------------------------- request surface


class ImageURL(BaseModel):
    url: str
    detail: str | None = None


class ContentPart(BaseModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageURL | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentPart] | None = None
    name: str | None = None

    def text(self) -> str:
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        return "\n".join(p.text or "" for p in self.content if p.type == "text")

    def images(self) -> list[ContentPart]:
        if isinstance(self.content, list):
            return [p for p in self.content if p.type == "image_url"]
        return []

    def to_openai(self, *, keep_images: bool) -> dict[str, Any]:
        if isinstance(self.content, list):
            if keep_images:
                content: Any = [p.model_dump(exclude_none=True) for p in self.content]
            else:
                content = self.text()
                n = len(self.images())
                if n:
                    content = f"{content}\n[{n} image(s) attached; described separately by the vision specialist]".strip()
        else:
            content = self.content or ""
        msg: dict[str, Any] = {"role": self.role, "content": content}
        if self.name:
            msg["name"] = self.name
        return msg


class ChatRequest(BaseModel):
    model: str = "conductor"
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None
    user: str | None = None
    # Conductor extensions
    session_id: str | None = None
    show_reasoning: bool = False
    debug: bool = False

    def last_user(self) -> ChatMessage:
        for m in reversed(self.messages):
            if m.role == "user":
                return m
        raise ValueError("request contains no user message")

    def image_count(self) -> int:
        return sum(len(m.images()) for m in self.messages)

    def total_chars(self) -> int:
        return sum(len(m.text()) for m in self.messages)


# ---------------------------------------------------------------- L0 routing contract


class RoutingDecision(BaseModel):
    """What the Conductor emits at L0. Weights form a softmax over routable clusters."""

    intent: str = ""
    trivial: bool = False
    weights: dict[str, float]
    subqueries: dict[str, str] = Field(default_factory=dict)
    synthesise: bool = True


def routing_json_schema() -> dict[str, Any]:
    """JSON schema used for guided decoding (xgrammar / outlines via response_format)."""
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "maxLength": 200},
            "trivial": {"type": "boolean"},
            "weights": {
                "type": "object",
                "properties": {c: {"type": "number", "minimum": 0, "maximum": 1} for c in ROUTABLE_CLUSTERS},
                "required": list(ROUTABLE_CLUSTERS),
                "additionalProperties": False,
            },
            "subqueries": {
                "type": "object",
                "properties": {c: {"type": "string", "maxLength": 2000} for c in ROUTABLE_CLUSTERS},
                "additionalProperties": False,
            },
            "synthesise": {"type": "boolean"},
        },
        "required": ["intent", "trivial", "weights", "subqueries", "synthesise"],
        "additionalProperties": False,
    }


class RoutePlan(BaseModel):
    """The executed plan after normalisation, constraints, capacity and breaker checks."""

    decision: RoutingDecision | None
    probs: dict[str, float]
    selected: list[str]
    merge_weights: dict[str, float]
    path: Literal["fast", "single", "multi"]
    reasons: list[str] = Field(default_factory=list)
    uniform_mode: bool = False
    route_ms: float = 0.0
