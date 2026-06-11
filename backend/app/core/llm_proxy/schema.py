"""Unified chat request schema (OpenAI-flavored) shared by all adapters."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Provider = Literal["openai", "openai-codex", "openrouter", "anthropic", "custom"]


class ToolSpec(BaseModel):
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ChatRequest(BaseModel):
    provider: Provider
    model: str
    messages: list[ChatMessage]
    tools: list[ToolSpec] = Field(default_factory=list)
    # Keys ride along per request from the browser and are never persisted
    # or logged server-side (spec: Part B).
    api_key: str | None = None
    base_url: str | None = None  # "custom" provider only
    max_tokens: int = Field(default=4096, ge=1, le=200_000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
