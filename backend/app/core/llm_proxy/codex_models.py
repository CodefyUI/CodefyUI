"""Account-aware model discovery for the ChatGPT Codex backend.

The Codex backend exposes richer metadata than the public OpenAI ``/v1/models``
endpoint.  Treat that response as an untrusted, best-effort catalog: retain only
picker fields this proxy understands, cache by ChatGPT account, and keep a
built-in catalog so settings remain usable while logged out or offline.
"""

from __future__ import annotations

import hashlib
import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from time import monotonic
from typing import Any

import httpx

from . import codex_auth

logger = logging.getLogger(__name__)

MODELS_URL = "https://chatgpt.com/backend-api/codex/models"
CLIENT_VERSION = "0.144.0"
CACHE_TTL_S = 5 * 60
FETCH_TIMEOUT_S = 5.0

_MAX_ID_LENGTH = 256
_MAX_DISPLAY_NAME_LENGTH = 120
_MAX_DESCRIPTION_LENGTH = 500
_MAX_EFFORT_LENGTH = 64
_MAX_MODELS = 500

_MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._:/-]*$")
_EFFORT_RE = re.compile(r"^[a-z][a-z0-9_-]*$")

_EFFORT_DESCRIPTIONS = {
    "low": "Fast responses with lighter reasoning",
    "medium": "Balances speed and reasoning depth for everyday tasks",
    "high": "Greater reasoning depth for complex problems",
    "xhigh": "Extra high reasoning depth for complex problems",
    "max": "Maximum reasoning depth for the hardest problems",
}

# The three GPT-5.6 variants mirror the compatible portion of the Codex client
# catalog.  ``ultra`` is deliberately absent: it is a Codex orchestration mode
# backed by automatic delegation, not a portable single-request effort value.
_FALLBACK_SPECS: tuple[
    tuple[str, str, str, str | None, tuple[str, ...]], ...
] = (
    (
        "gpt-5.6-sol",
        "GPT-5.6 Sol",
        "Latest frontier agentic coding model.",
        "low",
        ("low", "medium", "high", "xhigh", "max"),
    ),
    (
        "gpt-5.6-terra",
        "GPT-5.6 Terra",
        "Balanced agentic coding model for everyday work.",
        "medium",
        ("low", "medium", "high", "xhigh", "max"),
    ),
    (
        "gpt-5.6-luna",
        "GPT-5.6 Luna",
        "Fast and affordable agentic coding model.",
        "medium",
        ("low", "medium", "high", "xhigh", "max"),
    ),
    (
        "gpt-5.5",
        "GPT-5.5",
        "Previous frontier model for complex coding and research.",
        "medium",
        ("low", "medium", "high", "xhigh"),
    ),
    (
        "gpt-5.4",
        "GPT-5.4",
        "Legacy general-purpose coding model.",
        "medium",
        ("low", "medium", "high", "xhigh"),
    ),
    (
        "gpt-5.4-mini",
        "GPT-5.4 Mini",
        "Legacy small and cost-efficient coding model.",
        "medium",
        ("low", "medium", "high", "xhigh"),
    ),
    (
        "gpt-5.3-codex-spark",
        "GPT-5.3 Codex Spark",
        "Legacy low-latency Codex research-preview model.",
        None,
        (),
    ),
)


@dataclass
class _CacheEntry:
    models: list[dict[str, Any]]
    etag: str | None
    fetched_at: float


_CACHE: dict[str, _CacheEntry] = {}
_CACHE_GENERATION = 0


def clear_cache() -> None:
    """Discard all sanitized model metadata (used on logout and in tests)."""
    global _CACHE_GENERATION
    _CACHE.clear()
    # Invalidate in-flight fetches so a response that started before logout or
    # an explicit clear cannot repopulate the cache after this point.
    _CACHE_GENERATION += 1


def _clean_text(value: Any, max_length: int) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > max_length:
        return None
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        return None
    return value


def _clean_effort(value: Any) -> str | None:
    effort = _clean_text(value, _MAX_EFFORT_LENGTH)
    if effort is None or not _EFFORT_RE.fullmatch(effort) or effort == "ultra":
        return None
    return effort


def _clean_model_id(value: Any) -> str | None:
    model_id = _clean_text(value, _MAX_ID_LENGTH)
    if model_id is None or not _MODEL_ID_RE.fullmatch(model_id):
        return None
    return model_id


def _sanitize_model(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    visibility = raw.get("visibility")
    if visibility not in (None, "list") or raw.get("supported_in_api") is False:
        return None

    model_id = _clean_model_id(raw.get("slug")) or _clean_model_id(raw.get("id"))
    if model_id is None:
        return None
    display_name = (
        _clean_text(raw.get("display_name"), _MAX_DISPLAY_NAME_LENGTH)
        or model_id
    )
    description = _clean_text(raw.get("description"), _MAX_DESCRIPTION_LENGTH)

    raw_levels = raw.get(
        "supported_reasoning_levels",
        raw.get("supported_reasoning_efforts", []),
    )
    levels: list[dict[str, str]] = []
    seen_efforts: set[str] = set()
    if isinstance(raw_levels, list):
        for raw_level in raw_levels:
            if isinstance(raw_level, dict):
                effort = _clean_effort(
                    raw_level.get(
                        "effort",
                        raw_level.get("reasoning_effort", raw_level.get("level")),
                    )
                )
                level_description = _clean_text(
                    raw_level.get("description"), _MAX_DESCRIPTION_LENGTH
                )
            else:
                effort = _clean_effort(raw_level)
                level_description = None
            if effort is None or effort in seen_efforts:
                continue
            seen_efforts.add(effort)
            levels.append({
                "effort": effort,
                "description": level_description or effort,
            })

    default_effort = _clean_effort(
        raw.get("default_reasoning_level", raw.get("default_reasoning_effort"))
    )
    if default_effort not in seen_efforts:
        default_effort = (
            "medium" if "medium" in seen_efforts
            else levels[0]["effort"] if levels
            else None
        )

    return {
        # ``id`` preserves the existing /api/llm/models client contract.
        "id": model_id,
        "display_name": display_name,
        "description": description,
        "default_reasoning_effort": default_effort,
        "supported_reasoning_efforts": levels,
    }


def _sanitize_catalog(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        return []
    raw_models = payload["models"]
    # The upstream catalog assigns lower numbers to preferred models.  Do not
    # expose priority itself, but preserve its picker order where valid.
    raw_models = sorted(
        raw_models,
        key=lambda model: (
            model.get("priority", 1_000_000)
            if isinstance(model, dict) and isinstance(model.get("priority"), int)
            else 1_000_000
        ),
    )

    models: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_model in raw_models:
        model = _sanitize_model(raw_model)
        if model is None or model["id"] in seen_ids:
            continue
        seen_ids.add(model["id"])
        models.append(model)
        if len(models) >= _MAX_MODELS:
            break
    return models


def fallback_models() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for model_id, display_name, description, default_effort, efforts in _FALLBACK_SPECS:
        models.append({
            "id": model_id,
            "display_name": display_name,
            "description": description,
            "default_reasoning_effort": default_effort,
            "supported_reasoning_efforts": [
                {
                    "effort": effort,
                    "description": _EFFORT_DESCRIPTIONS.get(effort, effort),
                }
                for effort in efforts
            ],
        })
    return models


def _cache_key(account_id: str, access_token: str) -> str:
    if account_id:
        return f"account:{account_id}"
    # Older/incomplete sessions may lack account_id.  Hashing the access token
    # still prevents one account from receiving another account's catalog.
    digest = hashlib.sha256(access_token.encode()).hexdigest()
    return f"token:{digest}"


def _stored_cache_key() -> str | None:
    tokens = codex_auth.load_tokens()
    if not tokens:
        return None
    access_token = tokens.get("access_token")
    account_id = tokens.get("account_id", "")
    if not isinstance(access_token, str) or not access_token:
        return None
    return _cache_key(account_id if isinstance(account_id, str) else "", access_token)


def _rotate_legacy_cache_key(
    previous_key: str | None, current_key: str, *, lineage_matches: bool
) -> _CacheEntry | None:
    """Carry a legacy token-keyed cache across an access-token refresh.

    Modern sessions use the stable ChatGPT account id, but older saved sessions
    can lack it and therefore fall back to an access-token hash.  Refreshing
    such a session may rotate only the token or may also recover a stable
    account id.  Move (and do not copy) the old entry to either resulting key
    so subsequent failures can still use stale data without leaving the same
    catalog reachable under an obsolete token key.
    """
    if (
        not lineage_matches
        or previous_key is None
        or not previous_key.startswith("token:")
        or previous_key == current_key
    ):
        return _CACHE.get(current_key)

    previous_entry = _CACHE.pop(previous_key, None)
    if previous_entry is not None:
        # A cache already associated with the rotated token is more recent and
        # must not be replaced by the legacy entry.
        _CACHE.setdefault(current_key, previous_entry)
    return _CACHE.get(current_key)


def _response(
    models: list[dict[str, Any]], *, source: str, stale: bool = False
) -> dict[str, Any]:
    return {"models": deepcopy(models), "source": source, "stale": stale}


def _context_is_current(auth_generation: int, cache_generation: int) -> bool:
    return (
        codex_auth.session_generation() == auth_generation
        and _CACHE_GENERATION == cache_generation
    )


def _stale_or_fallback(
    cache_key: str | None, *, auth_generation: int, cache_generation: int
) -> dict[str, Any]:
    if not _context_is_current(auth_generation, cache_generation):
        return _response(fallback_models(), source="fallback")
    if cache_key is not None and (entry := _CACHE.get(cache_key)) is not None:
        return _response(entry.models, source="stale", stale=True)
    return _response(fallback_models(), source="fallback")


async def list_models(client: httpx.AsyncClient) -> dict[str, Any]:
    """Return a sanitized account catalog with cache/stale/fallback behavior."""
    cache_generation = _CACHE_GENERATION
    initial_auth_generation = codex_auth.session_generation()
    stored_cache_key = _stored_cache_key()
    try:
        access_token, account_id = await codex_auth.get_valid_access(client)
    except codex_auth.CodexNotLoggedIn:
        return _response(fallback_models(), source="fallback")
    except httpx.HTTPError as exc:
        logger.warning("Codex model auth refresh failed: %s", type(exc).__name__)
        return _stale_or_fallback(
            stored_cache_key,
            auth_generation=initial_auth_generation,
            cache_generation=cache_generation,
        )

    auth_generation = codex_auth.session_generation()
    if _CACHE_GENERATION != cache_generation:
        return _response(fallback_models(), source="fallback")
    cache_key = _cache_key(account_id, access_token)
    entry = _rotate_legacy_cache_key(
        stored_cache_key,
        cache_key,
        lineage_matches=initial_auth_generation == auth_generation,
    )
    now = monotonic()
    if entry is not None and now - entry.fetched_at < CACHE_TTL_S:
        if _context_is_current(auth_generation, cache_generation):
            return _response(entry.models, source="cache")
        return _response(fallback_models(), source="fallback")

    async def request_catalog(token: str, account: str) -> httpx.Response:
        headers = {
            "authorization": f"Bearer {token}",
            "chatgpt-account-id": account,
            "originator": codex_auth.ORIGINATOR,
            "accept": "application/json",
        }
        if entry is not None and entry.etag:
            headers["if-none-match"] = entry.etag
        return await client.get(
            MODELS_URL,
            params={"client_version": CLIENT_VERSION},
            headers=headers,
            timeout=FETCH_TIMEOUT_S,
        )

    try:
        response = await request_catalog(access_token, account_id)
        if response.status_code == 401:
            previous_cache_key = cache_key
            previous_auth_generation = auth_generation
            access_token, account_id = await codex_auth.get_valid_access(
                client, force_refresh=True
            )
            auth_generation = codex_auth.session_generation()
            if _CACHE_GENERATION != cache_generation:
                return _response(fallback_models(), source="fallback")
            cache_key = _cache_key(account_id, access_token)
            entry = _rotate_legacy_cache_key(
                previous_cache_key,
                cache_key,
                lineage_matches=previous_auth_generation == auth_generation,
            )
            response = await request_catalog(access_token, account_id)

        if response.status_code == 304 and entry is not None:
            if not _context_is_current(auth_generation, cache_generation):
                return _response(fallback_models(), source="fallback")
            entry.fetched_at = monotonic()
            return _response(entry.models, source="cache")
        response.raise_for_status()
        models = _sanitize_catalog(response.json())
        if not models:
            raise ValueError("Codex model catalog contained no visible valid models")
        if not _context_is_current(auth_generation, cache_generation):
            return _response(fallback_models(), source="fallback")
        _CACHE[cache_key] = _CacheEntry(
            models=deepcopy(models),
            etag=response.headers.get("etag"),
            fetched_at=monotonic(),
        )
        return _response(models, source="live")
    except (codex_auth.CodexNotLoggedIn, httpx.HTTPError, ValueError) as exc:
        logger.warning("Codex model catalog refresh failed: %s", type(exc).__name__)
        return _stale_or_fallback(
            cache_key,
            auth_generation=auth_generation,
            cache_generation=cache_generation,
        )
