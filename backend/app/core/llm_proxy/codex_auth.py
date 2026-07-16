"""ChatGPT-account OAuth for the "openai-codex" provider.

Reimplements the open-source Codex CLI login flow (PKCE against
auth.openai.com with a one-shot localhost callback listener) so users can
spend ChatGPT subscription quota instead of API credits. Reuses the CLI's
public client id -- the documented-risk approach accepted in the design
spec.

Facts verified against github.com/openai/codex (2026-06):
- account id lives in the id_token claim
  ["https://api.openai.com/auth"]["chatgpt_account_id"] and is persisted at
  login because post-refresh access tokens omit the claim;
- access tokens carry expiry in the JWT "exp" claim; the CLI refreshes
  5 minutes early and unconditionally after 8 days.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from platformdirs import user_data_dir

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTH_BASE = "https://auth.openai.com"
SCOPE = "openid profile email offline_access"
ORIGINATOR = "codex_cli_rs"
CALLBACK_PORTS: tuple[int, ...] = (1455, 1457)
LOGIN_TIMEOUT_S = 300
_REFRESH_EARLY_S = 300          # refresh when exp is < 5 minutes away
_REFRESH_MAX_AGE = timedelta(days=8)


class CodexNotLoggedIn(Exception):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _llm_dir() -> Path:
    override = os.environ.get("CODEFYUI_USER_DATA_DIR")
    base = Path(override) if override else Path(user_data_dir("codefyui", appauthor=False))
    return base / "llm"


def auth_file() -> Path:
    return _llm_dir() / "codex_auth.json"


# -- PKCE / JWT helpers -------------------------------------------------------

def generate_pkce() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(os.urandom(64)).rstrip(b"=").decode()
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def decode_jwt_claims(token: str) -> dict[str, Any]:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except (IndexError, ValueError, json.JSONDecodeError):
        return {}


def account_id_from_id_token(id_token: str) -> str | None:
    auth = decode_jwt_claims(id_token).get("https://api.openai.com/auth")
    if isinstance(auth, dict):
        value = auth.get("chatgpt_account_id")
        return value if isinstance(value, str) else None
    return None


def email_from_id_token(id_token: str) -> str | None:
    value = decode_jwt_claims(id_token).get("email")
    return value if isinstance(value, str) else None


# -- token storage ------------------------------------------------------------

# Process-local lineage for the stored ChatGPT session.  Route handlers and
# OAuth callbacks run on one event loop, so synchronous increments plus a
# compare-before-write are sufficient to invalidate work that was suspended at
# an HTTP await.  Refreshes within the same session deliberately do not bump it.
_SESSION_GENERATION = 0


def session_generation() -> int:
    """Return the current in-process auth-session lineage."""
    return _SESSION_GENERATION


def _advance_session_generation() -> None:
    global _SESSION_GENERATION
    _SESSION_GENERATION += 1


def load_tokens() -> dict[str, Any] | None:
    p = auth_file()
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) and data.get("access_token") else None


def _write_tokens(tokens: dict[str, Any]) -> None:
    p = auth_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(tokens, indent=2), encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass  # best-effort on Windows


def save_tokens(tokens: dict[str, Any]) -> None:
    """Replace the stored session and invalidate suspended prior-session work."""
    _write_tokens(tokens)
    _advance_session_generation()


def clear_tokens() -> None:
    try:
        auth_file().unlink(missing_ok=True)
    except OSError:
        pass
    finally:
        # Bump even when no file exists: logout is an explicit invalidation of
        # any refresh or catalog operation already suspended at an await.
        _advance_session_generation()


# -- refresh ------------------------------------------------------------------

def needs_refresh(tokens: dict[str, Any]) -> bool:
    exp = decode_jwt_claims(tokens.get("access_token", "")).get("exp")
    now = datetime.now(timezone.utc)
    if isinstance(exp, (int, float)):
        if datetime.fromtimestamp(exp, tz=timezone.utc) - now < timedelta(seconds=_REFRESH_EARLY_S):
            return True
    try:
        last = datetime.fromisoformat(tokens.get("last_refresh", ""))
    except ValueError:
        return True
    return now - last > _REFRESH_MAX_AGE


async def _refresh(client: httpx.AsyncClient, tokens: dict[str, Any]) -> dict[str, Any]:
    resp = await client.post(f"{AUTH_BASE}/oauth/token", json={
        "grant_type": "refresh_token",
        "client_id": CLIENT_ID,
        "refresh_token": tokens["refresh_token"],
    })
    resp.raise_for_status()
    fresh = resp.json()
    updated = dict(tokens)
    updated["access_token"] = fresh.get("access_token", tokens["access_token"])
    if fresh.get("refresh_token"):
        updated["refresh_token"] = fresh["refresh_token"]
    if fresh.get("id_token"):
        updated["id_token"] = fresh["id_token"]
    # Post-refresh JWTs omit the account-id claim -- keep the stored value.
    updated["last_refresh"] = now_iso()
    return updated


async def get_valid_access(
    client: httpx.AsyncClient, *, force_refresh: bool = False
) -> tuple[str, str]:
    tokens = load_tokens()
    if not tokens:
        raise CodexNotLoggedIn("Not signed in to ChatGPT")
    if force_refresh or needs_refresh(tokens):
        generation = session_generation()
        refresh_source = dict(tokens)
        refreshed = await _refresh(client, refresh_source)
        current = load_tokens()
        if (
            generation != session_generation()
            or current is None
            or current.get("access_token") != refresh_source.get("access_token")
            or current.get("refresh_token") != refresh_source.get("refresh_token")
        ):
            raise CodexNotLoggedIn("ChatGPT session changed during token refresh")
        # Same event loop and no await between the CAS above and this write.
        # Keep the generation stable: this is credential rotation within the
        # same authenticated session, not a login/session replacement.
        _write_tokens(refreshed)
        tokens = refreshed
    return tokens["access_token"], tokens.get("account_id", "")


# -- interactive login flow ----------------------------------------------------

class _LoginFlow:
    def __init__(self) -> None:
        self.state = secrets.token_urlsafe(32)
        self.verifier, self.challenge = generate_pkce()
        self.port: int = 0
        self.server: asyncio.AbstractServer | None = None
        self.exchange_client: httpx.AsyncClient | None = None
        self.settled = asyncio.Event()
        self.timeout_task: asyncio.Task | None = None


# The flow singleton is mutated only from the server's single event loop
# (route handlers + the callback connection handler); no threads.
_flow: _LoginFlow | None = None


def build_authorize_url(*, port: int, state: str, challenge: str) -> str:
    return f"{AUTH_BASE}/oauth/authorize?" + urlencode({
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": f"http://localhost:{port}/auth/callback",
        "scope": SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "originator": ORIGINATOR,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
    })


async def _exchange_code(client: httpx.AsyncClient, *, code: str, verifier: str,
                         port: int) -> dict[str, Any]:
    resp = await client.post(
        f"{AUTH_BASE}/oauth/token",
        content=urlencode({
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": f"http://localhost:{port}/auth/callback",
            "client_id": CLIENT_ID,
            "code_verifier": verifier,
        }),
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    resp.raise_for_status()
    data = resp.json()
    id_token = data.get("id_token", "")
    return {
        "id_token": id_token,
        "access_token": data["access_token"],
        "refresh_token": data.get("refresh_token", ""),
        "account_id": account_id_from_id_token(id_token) or "",
        "last_refresh": now_iso(),
    }


def _http_response(status: int, body: str) -> bytes:
    reason = {200: "OK", 400: "Bad Request", 404: "Not Found"}.get(status, "OK")
    payload = body.encode()
    return (
        f"HTTP/1.1 {status} {reason}\r\n"
        f"content-type: text/html; charset=utf-8\r\n"
        f"content-length: {len(payload)}\r\n"
        "connection: close\r\n\r\n"
    ).encode() + payload


async def _handle_conn(reader: asyncio.StreamReader,
                       writer: asyncio.StreamWriter) -> None:
    global _flow
    flow = _flow
    try:
        request_line = await asyncio.wait_for(reader.readline(), timeout=10)
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=10)
            if line in (b"\r\n", b"\n", b""):
                break
        parts = request_line.decode("latin-1").split()
        path = parts[1] if len(parts) >= 2 else "/"
        parsed = urlparse(path)
        if flow is None or parsed.path != "/auth/callback":
            writer.write(_http_response(404, "Not found"))
            return
        q = parse_qs(parsed.query)
        if q.get("state", [""])[0] != flow.state:
            writer.write(_http_response(400, "State mismatch - retry sign-in from CodefyUI."))
            return
        code = q.get("code", [""])[0]
        if not code:
            writer.write(_http_response(400, "Missing authorization code."))
            return
        exchange_generation = session_generation()
        client = flow.exchange_client or httpx.AsyncClient()
        try:
            tokens = await _exchange_code(client, code=code,
                                          verifier=flow.verifier, port=flow.port)
        finally:
            if flow.exchange_client is None:
                await client.aclose()
        # Logout, cancellation, or a newer login flow may have happened while
        # the OAuth token exchange was awaiting the network.  The captured
        # callback must not resurrect or replace that newer session.
        if _flow is not flow or session_generation() != exchange_generation:
            writer.write(_http_response(400, "Sign-in was cancelled or superseded."))
            return
        save_tokens(tokens)
        writer.write(_http_response(
            200,
            "<html><body style='font-family:sans-serif'>"
            "Signed in to ChatGPT. You can close this tab and return to CodefyUI."
            "</body></html>",
        ))
        _settle()
    except Exception:
        try:
            writer.write(_http_response(400, "Sign-in failed - retry from CodefyUI."))
        except Exception:
            pass
    finally:
        try:
            await writer.drain()
            writer.close()
        except Exception:
            pass


def _settle() -> None:
    global _flow
    flow = _flow
    if flow is None:
        return
    _flow = None
    flow.settled.set()
    if flow.server is not None:
        try:
            flow.server.close()
        except Exception:
            pass  # loop already closed (e.g. sync fixture teardown on Windows)
    if flow.timeout_task is not None:
        try:
            flow.timeout_task.cancel()
        except Exception:
            pass
    if flow.exchange_client is not None:
        try:
            asyncio.get_running_loop()
            asyncio.ensure_future(flow.exchange_client.aclose())
        except RuntimeError:
            pass  # no running loop -> client transport is GC-closed


def cancel_login() -> None:
    _settle()


def active_login_port() -> int | None:
    return _flow.port if _flow is not None else None


async def wait_login_settled(timeout: float = 10.0) -> None:
    flow = _flow
    if flow is not None:
        # Flow object survives _settle() locally; waiters use its event.
        await asyncio.wait_for(flow.settled.wait(), timeout)


async def start_login(*, ports: tuple[int, ...] = CALLBACK_PORTS,
                      exchange_client: httpx.AsyncClient | None = None) -> str:
    global _flow
    cancel_login()
    flow = _LoginFlow()
    flow.exchange_client = exchange_client
    last_err: OSError | None = None
    for port in ports:
        try:
            flow.server = await asyncio.start_server(_handle_conn, "127.0.0.1", port)
            flow.port = flow.server.sockets[0].getsockname()[1]
            break
        except OSError as exc:
            last_err = exc
    if flow.server is None:
        raise RuntimeError(f"Could not bind a callback port {ports}: {last_err}")

    async def _timeout() -> None:
        await asyncio.sleep(LOGIN_TIMEOUT_S)
        cancel_login()

    flow.timeout_task = asyncio.ensure_future(_timeout())
    _flow = flow
    return build_authorize_url(port=flow.port, state=flow.state,
                               challenge=flow.challenge)


def status() -> dict[str, Any]:
    if _flow is not None:
        return {"status": "pending"}
    tokens = load_tokens()
    if not tokens:
        return {"status": "logged_out"}
    return {"status": "logged_in",
            "email": email_from_id_token(tokens.get("id_token", ""))}


def logout() -> None:
    cancel_login()
    clear_tokens()
