"""LLMChat node -- call an external chat model from a graph workflow."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

from ...core.llm_proxy import anthropic, codex, openai_like
from ...core.llm_proxy.schema import ChatMessage, ChatRequest, Provider
from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)

ProviderAdapter = Callable[[ChatRequest, httpx.AsyncClient], AsyncIterator[dict[str, Any]]]

PROVIDER_OPTIONS = ["ChatGPT API", "Codex", "Claude API", "Ollama"]

_PROVIDER_ALIASES: dict[str, Provider] = {
    "chatgpt api": "openai",
    "openai": "openai",
    "openai api": "openai",
    "codex": "openai-codex",
    "openai-codex": "openai-codex",
    "claude api": "anthropic",
    "claude": "anthropic",
    "anthropic": "anthropic",
    "anthropic api": "anthropic",
    "ollama": "custom",
}

_ADAPTERS: dict[Provider, ProviderAdapter] = {
    "openai": openai_like.stream_chat,
    "custom": openai_like.stream_chat,
    "anthropic": anthropic.stream_chat,
    "openai-codex": codex.stream_chat,
}


class LLMChatNode(BaseNode):
    NODE_NAME = "LLMChat"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Send text, an image tensor, or JSON-like array data to a chat LLM and "
        "emit the assistant response as STRING. Providers: ChatGPT API, Codex, "
        "Claude API, and local Ollama."
    )
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="text",
                data_type=DataType.STRING,
                description="Optional text prompt or context from an upstream node.",
                optional=True,
            ),
            PortDefinition(
                name="image",
                data_type=DataType.IMAGE,
                description=(
                    "Optional image tensor. OpenAI-compatible/Ollama and Claude "
                    "receive it as multimodal input."
                ),
                optional=True,
            ),
            PortDefinition(
                name="array",
                data_type=DataType.LIST,
                description="Optional array/object-like context serialized as JSON for the prompt.",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="text",
                data_type=DataType.STRING,
                description="Assistant response text.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="provider",
                param_type=ParamType.SELECT,
                default="ChatGPT API",
                options=PROVIDER_OPTIONS,
                description=(
                    "LLM backend to call. Ollama uses its OpenAI-compatible /v1 endpoint."
                ),
            ),
            ParamDefinition(
                name="model",
                param_type=ParamType.STRING,
                default="gpt-5.2",
                description="Model id, for example gpt-5.2, claude-sonnet-4-6, or llama3.2.",
            ),
            ParamDefinition(
                name="prompt",
                param_type=ParamType.STRING,
                default="Summarize the provided input.",
                description="User prompt. Connected text/array/image inputs are appended to this prompt.",
            ),
            ParamDefinition(
                name="system_prompt",
                param_type=ParamType.STRING,
                default="You are a helpful assistant.",
                description="System instruction sent before the user prompt.",
            ),
            ParamDefinition(
                name="openai_api_key",
                param_type=ParamType.STRING,
                default="",
                description=(
                    "OpenAI API key. Prefer OPENAI_API_KEY or CODEFYUI_OPENAI_API_KEY "
                    "so saved graphs do not contain secrets."
                ),
                visible_when={"provider": "ChatGPT API"},
            ),
            ParamDefinition(
                name="anthropic_api_key",
                param_type=ParamType.STRING,
                default="",
                description=(
                    "Anthropic API key. Prefer ANTHROPIC_API_KEY or "
                    "CODEFYUI_ANTHROPIC_API_KEY so saved graphs do not contain secrets."
                ),
                visible_when={"provider": "Claude API"},
            ),
            ParamDefinition(
                name="ollama_base_url",
                param_type=ParamType.STRING,
                default="http://127.0.0.1:11434/v1",
                description="Ollama OpenAI-compatible endpoint base URL.",
                visible_when={"provider": "Ollama"},
            ),
            ParamDefinition(
                name="max_tokens",
                param_type=ParamType.INT,
                default=1024,
                description="Maximum output tokens.",
                min_value=1,
                max_value=200_000,
            ),
            ParamDefinition(
                name="temperature",
                param_type=ParamType.FLOAT,
                default=0.7,
                description="Sampling temperature.",
                min_value=0.0,
                max_value=2.0,
            ),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        provider = _normalize_provider(params.get("provider", "ChatGPT API"))
        req = _build_request(provider, inputs, params)
        text, usage = asyncio.run(_collect_chat(req, _ADAPTERS[provider], progress_callback))
        result: dict[str, Any] = {"text": text}
        if usage:
            result["__usage__"] = usage
        return result


def _normalize_provider(raw: Any) -> Provider:
    key = str(raw or "ChatGPT API").strip().lower()
    provider = _PROVIDER_ALIASES.get(key)
    if provider is None:
        supported = ", ".join(PROVIDER_OPTIONS)
        raise ValueError(f"Unsupported LLM provider {raw!r}. Choose one of: {supported}")
    return provider


def _build_request(provider: Provider, inputs: dict[str, Any], params: dict[str, Any]) -> ChatRequest:
    model = str(params.get("model") or "").strip()
    if not model:
        raise ValueError("LLMChat requires a model id")

    api_key = _api_key_for(provider, params)
    base_url = _base_url_for(provider, params)

    if provider == "openai" and not api_key:
        raise ValueError("ChatGPT API provider requires openai_api_key or OPENAI_API_KEY")
    if provider == "anthropic" and not api_key:
        raise ValueError("Claude API provider requires anthropic_api_key or ANTHROPIC_API_KEY")

    system_prompt = str(params.get("system_prompt") or "").strip()
    user_content = _build_user_content(provider, inputs, params)
    max_tokens = int(params.get("max_tokens", 1024))
    temperature = float(params.get("temperature", 0.7))

    messages: list[ChatMessage] = []
    if system_prompt:
        messages.append(ChatMessage(role="system", content=system_prompt))
    messages.append(ChatMessage(role="user", content=user_content))

    return ChatRequest(
        provider=provider,
        model=model,
        messages=messages,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _api_key_for(provider: Provider, params: dict[str, Any]) -> str | None:
    if provider == "openai":
        return _first_non_empty(
            params.get("openai_api_key"),
            params.get("api_key"),
            os.environ.get("CODEFYUI_OPENAI_API_KEY"),
            os.environ.get("OPENAI_API_KEY"),
        )
    if provider == "anthropic":
        return _first_non_empty(
            params.get("anthropic_api_key"),
            params.get("api_key"),
            os.environ.get("CODEFYUI_ANTHROPIC_API_KEY"),
            os.environ.get("ANTHROPIC_API_KEY"),
        )
    return None


def _base_url_for(provider: Provider, params: dict[str, Any]) -> str | None:
    if provider != "custom":
        return None
    return str(params.get("ollama_base_url") or "http://127.0.0.1:11434/v1").strip()


def _first_non_empty(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _build_user_content(
    provider: Provider,
    inputs: dict[str, Any],
    params: dict[str, Any],
) -> str | list[dict[str, Any]]:
    prompt_parts: list[str] = []
    prompt = str(params.get("prompt") or "").strip()
    if prompt:
        prompt_parts.append(prompt)

    if inputs.get("text") is not None:
        prompt_parts.append(str(inputs["text"]))

    if inputs.get("array") is not None:
        prompt_parts.append("Array input:\n" + _jsonish(inputs["array"]))

    image_value = inputs.get("image")
    image_url = _image_to_data_url(image_value) if image_value is not None else None
    text = "\n\n".join(part for part in prompt_parts if part).strip()

    if image_url is None:
        if not text:
            raise ValueError("LLMChat needs a prompt, text input, image input, or array input")
        return text

    if provider == "openai-codex":
        codex_note = (
            "Image input was provided, but the Codex provider currently accepts "
            "text only in CodefyUI."
        )
        return "\n\n".join(part for part in (text, codex_note) if part)

    return [
        {"type": "text", "text": text or "Describe the image."},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]


def _jsonish(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=_json_default)
    except (TypeError, ValueError):
        return repr(value)


def _json_default(value: Any) -> Any:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        return {
            "type": type(value).__name__,
            "shape": [int(x) for x in shape],
            "dtype": str(dtype) if dtype is not None else "",
        }
    return repr(value)


def _image_to_data_url(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("data:image/"):
            return text
        if text:
            return f"data:image/png;base64,{text}"

    if isinstance(value, dict):
        url = value.get("data_url") or value.get("url")
        if isinstance(url, str) and url.startswith("data:image/"):
            return url
        data = value.get("base64") or value.get("data")
        if isinstance(data, str) and data:
            media_type = str(value.get("media_type") or "image/png")
            return f"data:{media_type};base64,{data}"

    image = _coerce_pil_image(value)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _coerce_pil_image(value: Any) -> Any:
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")

    try:
        import torch

        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu()
            if arr.ndim == 4:
                arr = arr[0]
            if arr.ndim == 3 and int(arr.shape[0]) in (1, 3, 4):
                arr = arr.permute(1, 2, 0)
            if arr.is_floating_point():
                arr = arr.clamp(0, 1).mul(255).to(torch.uint8)
            else:
                arr = arr.clamp(0, 255).to(torch.uint8)
            np_arr = arr.numpy()
            if np_arr.ndim == 3 and np_arr.shape[2] == 1:
                np_arr = np_arr[:, :, 0]
            return Image.fromarray(np_arr).convert("RGB")
    except ImportError:
        pass

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            arr = value
            if arr.ndim == 4:
                arr = arr[0]
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                arr = arr.transpose(1, 2, 0)
            if arr.dtype.kind == "f":
                arr = (arr.clip(0, 1) * 255).astype("uint8")
            else:
                arr = arr.clip(0, 255).astype("uint8")
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr[:, :, 0]
            return Image.fromarray(arr).convert("RGB")
    except ImportError:
        pass

    raise TypeError(f"Unsupported image input type for LLMChat: {type(value).__name__}")


async def _collect_chat(
    req: ChatRequest,
    adapter: ProviderAdapter,
    progress_callback: Any | None = None,
) -> tuple[str, dict[str, int]]:
    chunks: list[str] = []
    async with httpx.AsyncClient() as client:
        async for event in adapter(req, client):
            etype = event.get("type")
            if etype == "text_delta":
                delta = str(event.get("text", ""))
                chunks.append(delta)
                if progress_callback is not None:
                    progress_callback({"text": "".join(chunks)})
            elif etype == "error":
                raise RuntimeError(str(event.get("message", "LLM provider error")))
            elif etype == "done":
                message = event.get("message") or {}
                content = str(message.get("content") or "".join(chunks))
                raw_usage = event.get("usage")
                usage = raw_usage if isinstance(raw_usage, dict) else {}
                return content, {
                    str(k): int(v)
                    for k, v in usage.items()
                    if isinstance(v, (int, float))
                }
    return "".join(chunks), {}