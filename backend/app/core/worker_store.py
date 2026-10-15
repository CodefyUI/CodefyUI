"""Worker enrollment and credentials (#780): the ``workers`` and
``worker_enrollments`` tables (migration 009).

The flow, modeled on ``kubeadm join``:

1. The Workers panel asks for an **enrollment token** (``POST
   /api/workers/enrollments``): ``[a-z0-9]{6}.[a-z0-9]{16}``, the kubeadm
   bootstrap-token shape. The six characters before the dot are its public
   id (listed, shown in the panel); the whole token is the secret. It
   expires (24 h by default) and has a number of uses (1 by default).
2. ``cdui agent start --control URL --token T`` sends it to ``POST
   /api/workers/enroll`` with its name and facts, and gets back a
   **worker token**: 32 random bytes, URL-safe, which it keeps in a 0600
   file and presents on every ``/ws/worker`` dial from then on.

Neither secret is stored: each table keeps the sha256 hex of its secret
(:func:`hash_secret`), as ``api_keys`` does. A 256-bit (worker) or 82-bit
(enrollment, also short-lived and rate-limited) random secret makes a slow
hash pointless.

A name is taken once enrolled: an ordinary token cannot enroll a name
that already has a row (:class:`NameTaken`, 409), which would hand that
worker's identity to whoever holds the token. To give an enrolled worker a
new credential (a reinstalled machine, a lost credential file), the panel
creates a token **bound** to it (``worker`` column): that token can enroll
that name only, and re-keys its row, so the old worker token stops working.
Revoking a worker deletes its row.

Every method is one ``Database.run`` call (the Decision A2 seam). This
module knows nothing about the registry or the socket; ``routes_workers``
and ``ws_worker`` put the two together.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import sqlite3
import string
from dataclasses import dataclass
from typing import Any

from .db import Database, transaction, utc_iso_after, utc_now_iso

logger = logging.getLogger(__name__)

#: ``abcdef.0123456789abcdef``: kubeadm's bootstrap-token shape.
ENROLLMENT_TOKEN_PATTERN = re.compile(r"([a-z0-9]{6})\.([a-z0-9]{16})")
_ALPHABET = string.ascii_lowercase + string.digits

DEFAULT_TTL_S = 24 * 3600
#: The longest an enrollment token may live: a week.
MAX_TTL_S = 7 * 24 * 3600
MIN_TTL_S = 60
DEFAULT_USES = 1
MAX_USES = 100

STATE_PENDING = "pending"
STATE_EXPIRED = "expired"
STATE_USED = "used"


def hash_secret(secret: str) -> str:
    """sha256 hex of *secret*: the only form a credential is stored in."""
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def is_enrollment_token(token: str | None) -> bool:
    """Whether *token* has the enrollment-token shape (never a session or
    worker token: those are ``token_urlsafe`` and carry no dot)."""
    return bool(token) and ENROLLMENT_TOKEN_PATTERN.fullmatch(
        token.strip()) is not None


def mint_enrollment_token() -> tuple[str, str]:
    """``(public id, whole token)``."""
    public = "".join(secrets.choice(_ALPHABET) for _ in range(6))
    secret = "".join(secrets.choice(_ALPHABET) for _ in range(16))
    return public, f"{public}.{secret}"


def mint_worker_token() -> str:
    return secrets.token_urlsafe(32)


class EnrollmentRefused(Exception):
    """An enrollment token that cannot enroll; the message says why.
    ``status`` is the HTTP status the route answers with."""

    status = 401


class WrongWorker(EnrollmentRefused):
    """A token bound to one worker, used for another name."""

    status = 403


class NameTaken(EnrollmentRefused):
    """The name is enrolled (or held by a live agent) and the token is not
    bound to it."""

    status = 409


@dataclass(frozen=True)
class KnownWorker:
    """One ``workers`` row, without its credential."""

    name: str
    platform: dict[str, str]
    mounts: dict[str, str]
    devices: list[dict[str, Any]]
    created_at: str
    last_seen_at: str | None


def _loads(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        value = json.loads(raw)
    except ValueError:
        return default
    return value if isinstance(value, type(default)) else default


def _enrollment_state(row: sqlite3.Row, now: str) -> str:
    if row["uses_left"] <= 0:
        return STATE_USED
    if row["expires_at"] <= now:
        return STATE_EXPIRED
    return STATE_PENDING


def _describe_enrollment(row: sqlite3.Row, now: str) -> dict[str, Any]:
    return {"id": row["id"], "label": row["label"], "worker": row["worker"],
            "created_at": row["created_at"], "expires_at": row["expires_at"],
            "uses_left": row["uses_left"],
            "state": _enrollment_state(row, now)}


def name_taken_message(name: str) -> str:
    return (f"worker {name} is already enrolled; remove it or create a "
            "re-key token for it")


class WorkerStore:
    """The two enrollment tables over one :class:`Database`."""

    def __init__(self, db: Database) -> None:
        self.db = db

    # ── enrollment tokens ─────────────────────────────────────────────────

    async def create_enrollment(self, *, ttl_s: int = DEFAULT_TTL_S,
                                uses: int = DEFAULT_USES,
                                label: str | None = None,
                                worker: str | None = None) -> dict[str, Any]:
        """A new enrollment token. The returned ``token`` is the only copy.
        *worker* binds it: it can then enroll (re-key) that name only."""
        def _create(conn: sqlite3.Connection) -> dict[str, Any]:
            while True:
                public, token = mint_enrollment_token()
                now = utc_now_iso()
                try:
                    conn.execute(
                        "INSERT INTO worker_enrollments (id, token_hash, "
                        "expires_at, uses_left, created_at, label, worker) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (public, hash_secret(token), utc_iso_after(ttl_s),
                         uses, now, label, worker))
                except sqlite3.IntegrityError:
                    continue  # the public half collided; draw again
                row = conn.execute(
                    "SELECT * FROM worker_enrollments WHERE id = ?",
                    (public,)).fetchone()
                return {**_describe_enrollment(row, now), "token": token}

        return await self.db.run(_create)

    async def list_enrollments(self) -> list[dict[str, Any]]:
        def _list(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            now = utc_now_iso()
            rows = conn.execute(
                "SELECT * FROM worker_enrollments ORDER BY created_at DESC"
            ).fetchall()
            return [_describe_enrollment(row, now) for row in rows]

        return await self.db.run(_list)

    async def delete_enrollment(self, enrollment_id: str) -> bool:
        def _delete(conn: sqlite3.Connection) -> bool:
            return conn.execute(
                "DELETE FROM worker_enrollments WHERE id = ?",
                (enrollment_id,)).rowcount > 0

        return await self.db.run(_delete)

    async def enroll(self, token: str, *, name: str,
                     platform: dict[str, str], mounts: dict[str, str],
                     devices: list[dict[str, Any]],
                     held_elsewhere: bool = False) -> tuple[str, bool]:
        """Spend one use of *token* on *name*. Returns ``(worker token,
        re-keyed)``; ``re-keyed`` is True when *name* was enrolled before
        (its old worker token no longer works).

        Raises :class:`EnrollmentRefused` for an unknown, expired or
        used-up token, :class:`WrongWorker` for a token bound to another
        name, and :class:`NameTaken` when *name* is enrolled already, or
        *held_elsewhere* (a live same-host agent uses it), and the token is
        not bound to it. A refused enrollment spends nothing. All in one
        transaction, so two agents racing for the last use cannot both get
        it.
        """
        token_hash = hash_secret(token.strip())
        worker_token = mint_worker_token()

        def _enroll(conn: sqlite3.Connection) -> bool:
            with transaction(conn):
                now = utc_now_iso()
                row = conn.execute(
                    "SELECT * FROM worker_enrollments WHERE token_hash = ?",
                    (token_hash,)).fetchone()
                if row is None:
                    raise EnrollmentRefused(
                        "the enrollment token is not valid (never issued, "
                        "or deleted); generate a new one in the Workers "
                        "panel")
                state = _enrollment_state(row, now)
                if state == STATE_USED:
                    raise EnrollmentRefused(
                        f"enrollment token {row['id']} has already been "
                        "used; generate a new one in the Workers panel")
                if state == STATE_EXPIRED:
                    raise EnrollmentRefused(
                        f"enrollment token {row['id']} expired at "
                        f"{row['expires_at']}; generate a new one in the "
                        "Workers panel")
                bound = row["worker"]
                if bound is not None and bound != name:
                    raise WrongWorker(
                        f"enrollment token {row['id']} re-keys worker "
                        f"{bound} only; start the agent with --name {bound}")
                existing = conn.execute(
                    "SELECT 1 FROM workers WHERE id = ?", (name,)).fetchone()
                if (existing is not None or held_elsewhere) and bound != name:
                    raise NameTaken(name_taken_message(name))
                conn.execute(
                    "UPDATE worker_enrollments SET uses_left = uses_left - 1 "
                    "WHERE id = ?", (row["id"],))
                facts = (json.dumps(platform), json.dumps(mounts),
                         json.dumps(devices))
                if existing is None:
                    conn.execute(
                        "INSERT INTO workers (id, name, token_hash, platform, "
                        "mounts_json, devices_json, created_at, last_seen_at, "
                        "enabled) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)",
                        (name, name, hash_secret(worker_token), *facts, now,
                         now))
                else:
                    conn.execute(
                        "UPDATE workers SET token_hash = ?, platform = ?, "
                        "mounts_json = ?, devices_json = ?, last_seen_at = ?, "
                        "enabled = 1 WHERE id = ?",
                        (hash_secret(worker_token), *facts, now, name))
                logger.info("worker %s enrolled with enrollment token %s%s",
                            name, row["id"],
                            " (re-keyed: its old worker token no longer "
                            "works)" if existing is not None else "")
                return existing is not None

        rekeyed = await self.db.run(_enroll)
        return worker_token, rekeyed

    # ── workers ───────────────────────────────────────────────────────────

    async def authenticate(self, worker_token: str | None) -> str | None:
        """The name of the enabled worker *worker_token* belongs to, or None."""
        if not worker_token:
            return None
        token_hash = hash_secret(worker_token)

        def _lookup(conn: sqlite3.Connection) -> str | None:
            row = conn.execute(
                "SELECT name FROM workers WHERE token_hash = ? AND enabled = 1",
                (token_hash,)).fetchone()
            return None if row is None else row["name"]

        return await self.db.run(_lookup)

    async def touch(self, name: str, *,
                    platform: dict[str, str] | None = None,
                    mounts: dict[str, str] | None = None,
                    devices: list[dict[str, Any]] | None = None,
                    machine_id: str | None = None) -> None:
        """Record that *name* was seen now, with the facts it reported."""
        def _touch(conn: sqlite3.Connection) -> None:
            sets, values = ["last_seen_at = ?"], [utc_now_iso()]
            for column, value in (("platform", platform),
                                  ("mounts_json", mounts),
                                  ("devices_json", devices)):
                if value is not None:
                    sets.append(f"{column} = ?")
                    values.append(json.dumps(value))
            if machine_id is not None:
                sets.append("machine_id = ?")
                values.append(machine_id)
            conn.execute(f"UPDATE workers SET {', '.join(sets)} WHERE id = ?",
                         (*values, name))

        await self.db.run(_touch)

    async def delete_worker(self, name: str) -> bool:
        """Revoke *name*'s credential by deleting its row."""
        def _delete(conn: sqlite3.Connection) -> bool:
            return conn.execute("DELETE FROM workers WHERE id = ?",
                                (name,)).rowcount > 0

        return await self.db.run(_delete)

    async def list_workers(self) -> list[KnownWorker]:
        def _list(conn: sqlite3.Connection) -> list[KnownWorker]:
            rows = conn.execute(
                "SELECT * FROM workers WHERE enabled = 1 ORDER BY name"
            ).fetchall()
            return [KnownWorker(
                name=row["name"], platform=_loads(row["platform"], {}),
                mounts=_loads(row["mounts_json"], {}),
                devices=_loads(row["devices_json"], []),
                created_at=row["created_at"],
                last_seen_at=row["last_seen_at"]) for row in rows]

        return await self.db.run(_list)
