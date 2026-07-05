"""API-key management endpoints (Stage 2): mint, list, soft-revoke.

Session-token-only (``require_session_token`` on every route) — key
management is the operator's surface. The token plaintext appears in the
POST response ONCE and is never stored or logged; lists show ``prefix``
(first 12 chars) only. Nothing is ever deleted from ``api_keys`` —
revoke sets ``revoked_at`` so ``runs.api_key_id`` stays meaningful.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..core.api_keys import (
    PREFIX_DISPLAY_CHARS,
    get_db,
    hash_token,
    mint_token,
    require_session_token,
)
from ..core.db import utc_now_iso

router = APIRouter(prefix="/api/keys", tags=["api-keys"])

_KEY_COLUMNS = "id, name, prefix, created_at, last_used_at, revoked_at"


def _key_error(
    status_code: int, code: str, message: str,
    details: list[Any] | None = None,
) -> HTTPException:
    """Management-surface error: plain ``{"detail": ...}`` transport with a
    stable ``code`` inside — same shape as routes_apps._manage_error."""
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message, "details": details},
    )


class CreateKeyRequest(BaseModel):
    name: str = Field(min_length=1)


@router.post("", dependencies=[Depends(require_session_token)])
async def create_key(body: CreateKeyRequest, request: Request):
    """Mint a key. The token is in THIS response only — copy it now."""
    db = get_db(request)
    token = mint_token()
    token_hash = hash_token(token)
    prefix = token[:PREFIX_DISPLAY_CHARS]
    now = utc_now_iso()

    def _insert(conn: sqlite3.Connection) -> int:
        cur = conn.execute(
            "INSERT INTO api_keys (name, prefix, token_hash, created_at) "
            "VALUES (?, ?, ?, ?)",
            (body.name, prefix, token_hash, now),
        )
        return cur.lastrowid

    key_id = await db.run(_insert)
    return {"id": key_id, "name": body.name, "prefix": prefix, "token": token}


@router.get("", dependencies=[Depends(require_session_token)])
async def list_keys(request: Request):
    """All keys newest-first, secrets never included; revoked rows REMAIN."""
    db = get_db(request)

    def _select(conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = conn.execute(
            f"SELECT {_KEY_COLUMNS} FROM api_keys ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    return await db.run(_select)


@router.post("/{key_id}/revoke", dependencies=[Depends(require_session_token)])
async def revoke_key(key_id: int, request: Request):
    """Soft revoke: set ``revoked_at`` once; the row stays listed.

    Deliberately POST /revoke, not DELETE — nothing is ever removed from
    ``api_keys``. Revoking an already-revoked key keeps the original
    ``revoked_at``.
    """
    db = get_db(request)
    now = utc_now_iso()

    def _revoke(conn: sqlite3.Connection) -> dict[str, Any] | None:
        conn.execute(
            "UPDATE api_keys SET revoked_at = ? "
            "WHERE id = ? AND revoked_at IS NULL",
            (now, key_id),
        )
        row = conn.execute(
            f"SELECT {_KEY_COLUMNS} FROM api_keys WHERE id = ?", (key_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    row = await db.run(_revoke)
    if row is None:
        raise _key_error(404, "key_not_found", f"API key {key_id} not found")
    return row
