"""Per-process session token + middleware enforcing it on mutating requests.

Threat model:

* The backend has no real user accounts — it's a desktop tool. "Authentication"
  here means **proving the request came from a process on this machine that
  could read a 0600-protected file**, which is enough to defeat:
  - Browser CSRF (a malicious page can issue cross-origin requests but cannot
    read our local token file).
  - DNS rebinding (the rebinding flow gives the attacker a 127.0.0.1 socket
    but the ``Host`` header still says ``attacker.com``; we reject those).
* This does NOT defend against an attacker who already runs code on the host —
  any same-user process can read the token file. That risk is unavoidable
  without OS-level isolation (Docker / sandboxing).

Token lifecycle:

1. On import (process startup) we generate a fresh URL-safe token.
2. We write it to ``<USER_DATA>/codefyui/session.token`` with mode 0600 so
   CLI tools (``cdui plugin install`` calling ``/api/plugins/reload``) can
   read it back.
3. The browser frontend fetches the token once via ``GET /api/auth/bootstrap``,
   which the middleware allows through (it's a read-only GET). All subsequent
   mutating requests must echo the token back in ``X-CodefyUI-Token``.

We keep this module dependency-free of FastAPI so tests can poke at the token
generation/file logic without spinning up an app.
"""

from __future__ import annotations

import hmac
import logging
import os
import secrets
import sys
from pathlib import Path
from typing import Iterable

from platformdirs import user_data_dir

logger = logging.getLogger(__name__)

TOKEN_HEADER = "X-CodefyUI-Token"
TOKEN_QUERY_PARAM = "token"  # WebSocket can't set custom headers from browser

# We generate the token at import time. The server process keeps it in memory;
# every restart rotates the token, which is the desired behaviour (no stale
# tokens in random files after a crash).
_SESSION_TOKEN = secrets.token_urlsafe(32)


def session_token() -> str:
    """Return the current process's session token."""
    return _SESSION_TOKEN


def _token_dir() -> Path:
    """Where the session token file lives.

    Honors ``CODEFYUI_USER_DATA_DIR`` for dev mode so a repo-local
    ``.codefyui_dev/`` install can keep its own session token without
    clobbering a global ``cdui start`` running in parallel. Mirrors the
    same override in ``plugin_loader.plugins_user_root``.
    """
    override = os.environ.get("CODEFYUI_USER_DATA_DIR")
    if override:
        return Path(override)
    return Path(user_data_dir("codefyui", appauthor=False))


def token_file_path() -> Path:
    """Where we persist the token on disk for CLI tool consumption."""
    return _token_dir() / "session.token"


def write_token_file() -> Path:
    """Write the session token to a 0600-protected file. Returns the path."""
    p = token_file_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    # Use os.open with O_CREAT|O_WRONLY|O_TRUNC and explicit mode so we don't
    # leak the token through a wider umask window. On Windows the mode arg is
    # advisory (NTFS ACLs ignore it), but the file ends up in the per-user
    # AppData dir which is already private to the OS account.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    mode = 0o600
    fd = os.open(str(p), flags, mode)
    try:
        os.write(fd, _SESSION_TOKEN.encode("ascii"))
    finally:
        os.close(fd)
    # On Windows, os.open ignores the mode arg — best-effort chmod for
    # documentation purposes (doesn't change the actual NTFS ACL).
    if sys.platform != "win32":
        os.chmod(p, 0o600)
    return p


def constant_time_equals(provided: str | None, expected: str) -> bool:
    """``hmac.compare_digest`` with a friendly ``None`` handler."""
    if not provided:
        return False
    return hmac.compare_digest(provided, expected)


# ── Host header whitelist ──────────────────────────────────────────────

# Anything in this set is accepted as a same-host request. The list expands
# at process start once we know the listening port — `init_allowed_hosts`
# below populates it from settings.HOST / settings.PORT plus the dev origins.

_ALLOWED_HOSTS: set[str] = set()


def init_allowed_hosts(host: str, port: int, extra: Iterable[str] = ()) -> None:
    """Populate the Host header whitelist used by ``host_is_allowed``.

    ``host`` is the address uvicorn binds to (typically ``127.0.0.1``); we
    automatically include the obvious aliases so a user typing ``localhost``
    in the address bar still works, but we do **not** accept arbitrary hosts
    (that's exactly the DNS-rebinding hole we're closing).
    """
    _ALLOWED_HOSTS.clear()
    bases: set[str] = {"127.0.0.1", "localhost", "[::1]", "::1"}
    if host and host not in ("0.0.0.0", "::"):
        bases.add(host)
    for base in bases:
        _ALLOWED_HOSTS.add(f"{base}:{port}")
        # Some clients omit the port when it's the protocol default; we don't
        # serve on 80/443 by default but accept the bare host for forgiveness.
        _ALLOWED_HOSTS.add(base)
    for entry in extra:
        _ALLOWED_HOSTS.add(entry)


def host_is_allowed(host_header: str) -> bool:
    """Return True iff *host_header* matches one of the whitelisted hosts.

    Comparison is case-insensitive (per RFC 7230 §5.4 hosts are case-insensitive)
    and tolerant of trailing whitespace. Empty / missing headers fail closed.
    """
    if not host_header:
        return False
    return host_header.strip().lower() in {h.lower() for h in _ALLOWED_HOSTS}


def allowed_hosts() -> frozenset[str]:
    """Snapshot of the current whitelist (mostly for tests / error messages)."""
    return frozenset(_ALLOWED_HOSTS)


def local_interface_ips() -> list[str]:
    """Best-effort enumeration of this machine's IPv4 addresses.

    Used when binding 0.0.0.0: ``init_allowed_hosts`` deliberately skips
    the wildcard itself, so each concrete interface IP gets whitelisted as
    ``{ip}:{port}`` instead. This does NOT weaken the DNS-rebinding
    defense — a rebound browser still sends the attacker's hostname in
    ``Host``, which stays unlisted.
    """
    import socket

    ips: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    # UDP-connect trick: finds the outbound-default interface without
    # sending a packet (helps when the hostname resolves to 127.0.1.1).
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("192.0.2.1", 80))  # TEST-NET-1, never routed
            ips.add(s.getsockname()[0])
        finally:
            s.close()
    except OSError:
        pass
    ips.discard("127.0.0.1")
    return sorted(ips)
