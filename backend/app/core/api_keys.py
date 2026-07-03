"""API-key minting/verification + the three Stage-2 auth dependencies.

Three-tier rule (spec Section 7):

- ``require_api_key``            — invoke only; NON-RAISING result object
  (the handler assigns run_id FIRST and envelopes the 401 itself, with
  ``WWW-Authenticate: Bearer``).
- ``require_api_key_or_session`` — published reads; RAISING, plain
  ``{"detail": ...}`` 401. Exists because Stage 3's editor UI reads runs
  with the session token — an API key must never need to live in the
  browser.
- ``require_session_token``      — management surface; RAISING; reuses the
  ``auth_guard`` compare and 403 shape (app/main.py).

Keys survive restarts by construction (DB, not process memory) — the
deliberate contrast with the rotating session token (app/core/auth.py).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request

from .auth import TOKEN_HEADER, constant_time_equals, session_token
from .db import Database, utc_now_iso

KEY_PREFIX = "cdui_"
PREFIX_DISPLAY_CHARS = 12

_MISSING_KEY_MESSAGE = "missing or malformed Authorization: Bearer header"
_NOT_AN_API_KEY_MESSAGE = (
    "this endpoint takes an API key (cdui_...), not the editor session token"
)
_UNKNOWN_KEY_MESSAGE = "unknown or revoked API key"


def mint_token() -> str:
    """A fresh API key: ``cdui_`` + 256-bit urlsafe secret (Decision C2).

    256-bit CSPRNG secrets make slow hashes pointless — sha256 storage is
    the deliberate choice; bcrypt exists for low-entropy passwords.
    """
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """sha256 hex of the FULL token string (prefix included)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def get_db(request: Request) -> Database:
    """App-state DB with the getattr-or-503 access rule
    (routes_execution_outputs._get_store precedent) — never a raw
    AttributeError when the lifespan didn't run (ASGITransport tests)."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="database unavailable")
    return db


@dataclass
class ApiKeyResult:
    """Outcome of the NON-RAISING invoke key check.

    ``key_row`` is the api_keys row (as a dict) when the key verified,
    else None with ``failure`` naming why — the invoke handler envelopes
    the 401 itself so the response still carries a ``run_id``.
    """

    key_row: dict[str, Any] | None
    failure: str | None = None

    @property
    def ok(self) -> bool:
        return self.key_row is not None


async def require_api_key(request: Request) -> ApiKeyResult:
    """NON-RAISING key check for the invoke route (spec Section 7).

    Steps: parse ``Authorization: Bearer`` and require the ``cdui_``
    prefix; sha256 the credential; SELECT by hash where not revoked;
    ``hmac.compare_digest`` the fetched hash (belt-and-braces, the
    auth.constant_time_equals precedent); UPDATE ``last_used_at`` on the
    same serialized connection.
    """
    header = request.headers.get("Authorization", "")
    scheme, _, credential = header.partition(" ")
    credential = credential.strip()
    if scheme.lower() != "bearer" or not credential:
        return ApiKeyResult(None, _MISSING_KEY_MESSAGE)
    if not credential.startswith(KEY_PREFIX):
        # Self-diagnosing (spec Section 6.3): the expected failure mode is
        # someone pasting the editor session token where a key belongs.
        return ApiKeyResult(None, _NOT_AN_API_KEY_MESSAGE)

    db = get_db(request)
    computed = hash_token(credential)

    def _lookup(conn: sqlite3.Connection) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT id, name, prefix, token_hash, created_at, last_used_at, "
            "revoked_at FROM api_keys "
            "WHERE token_hash = ? AND revoked_at IS NULL",
            (computed,),
        ).fetchone()
        if row is None:
            return None
        if not hmac.compare_digest(row["token_hash"], computed):
            return None
        conn.execute(
            "UPDATE api_keys SET last_used_at = ? WHERE id = ?",
            (utc_now_iso(), row["id"]),
        )
        return dict(row)

    row = await db.run(_lookup)
    if row is None:
        return ApiKeyResult(None, _UNKNOWN_KEY_MESSAGE)
    return ApiKeyResult(row)


async def require_session_token(request: Request) -> None:
    """Management-surface auth: the exact ``auth_guard`` compare + 403
    shape, applied at route level because the middleware exempts the
    /api/apps and /api/keys prefixes (and never covered GETs anyway)."""
    provided = request.headers.get(TOKEN_HEADER)
    if not constant_time_equals(provided, session_token()):
        raise HTTPException(
            status_code=403,
            detail=f"Missing or invalid {TOKEN_HEADER} header",
        )


async def require_api_key_or_session(request: Request) -> None:
    """Published reads: EITHER a valid API key or the editor session token.

    Raising, plain ``{"detail": ...}`` 401 (the contract-endpoint style).
    """
    provided = request.headers.get(TOKEN_HEADER)
    if constant_time_equals(provided, session_token()):
        return
    result = await require_api_key(request)
    if result.ok:
        return
    raise HTTPException(
        status_code=401,
        detail=(
            "provide a valid API key (Authorization: Bearer cdui_...) "
            f"or the editor session token ({TOKEN_HEADER} header)"
        ),
    )
