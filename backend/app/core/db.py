"""SQLite storage for published apps, API keys, and run records (Stage 2).

Decision A2 contract, verbatim:

- ``sqlite3.connect(path, check_same_thread=False, isolation_level=None)``
  — both kwargs MANDATORY. The default isolation level silently
  de-atomizes DDL migrations and breaks ``BEGIN IMMEDIATE`` (proven by
  test during spec review). All transaction control is explicit.
- ONE shared ``asyncio.Lock`` lives on the ``Database`` object (never
  created per-call). Every DB operation is one sync fn executed via ONE
  ``asyncio.to_thread`` call under that lock — ``Database.run`` is the
  seam, so swapping in a pool or aiosqlite later is a one-module change.
- Kept FastAPI-free (``api_contract`` pure-module precedent); the
  dependency plumbing lives in ``app.core.api_keys``.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, TypeVar

from .migrations import MIGRATIONS, iter_statements

logger = logging.getLogger(__name__)

T = TypeVar("T")

_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp, microsecond precision, trailing ``Z``.

    One fixed format so TEXT comparison (retention's ``created_at < ?``,
    the runs ``before=`` cursor) is lexicographically correct.
    """
    return datetime.now(timezone.utc).strftime(_TIMESTAMP_FORMAT)


class Database:
    """Serialized async facade over one sqlite3 connection."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._last_prune_monotonic: float | None = None

    def connect(self) -> None:
        """Open the connection, apply pragmas, run pending migrations.

        Sync on purpose: called once from the lifespan (via
        ``asyncio.to_thread``) or directly from a test fixture.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self.path), check_same_thread=False, isolation_level=None,
        )
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            self._apply_migrations(conn)
        except BaseException:
            conn.close()
            raise
        self._conn = conn

    @staticmethod
    def _apply_migrations(conn: sqlite3.Connection) -> None:
        """Run ``MIGRATIONS[user_version:]``, bumping ``user_version``.

        Each migration and its version bump commit atomically —
        ``BEGIN IMMEDIATE`` works only because ``isolation_level=None``.
        """
        current = conn.execute("PRAGMA user_version").fetchone()[0]
        for index in range(current, len(MIGRATIONS)):
            conn.execute("BEGIN IMMEDIATE")
            try:
                for statement in iter_statements(MIGRATIONS[index]):
                    conn.execute(statement)
                # PRAGMA user_version cannot take bound params — interpolate
                # the int (it is our own loop index, not user input).
                conn.execute(f"PRAGMA user_version = {index + 1}")
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    def close(self) -> None:
        """Release the handle (Windows needs this to free the WAL sidecars)."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    async def run(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Execute ``fn(conn)`` in a worker thread, serialized by the lock.

        THE seam: every route-level DB operation goes through here (one
        sync fn, one ``to_thread``, one lock — Decision A2). Multi-statement
        transactions live INSIDE ``fn`` with explicit BEGIN/COMMIT/ROLLBACK.
        """
        async with self._lock:
            conn = self._conn
            if conn is None:
                raise RuntimeError("Database is not connected")
            return await asyncio.to_thread(fn, conn)

    async def prune_runs(self, retention_days: int, *, force: bool = False) -> int:
        """Delete runs older than ``retention_days`` days; 0 disables.

        ``force=True`` (lifespan startup) always prunes; otherwise the call
        is the piggyback-on-writes path, rate-limited to once per hour via
        a monotonic timestamp on this object. Every prune logs loudly.
        """
        if retention_days <= 0:
            return 0
        now = time.monotonic()
        if (
            not force
            and self._last_prune_monotonic is not None
            and now - self._last_prune_monotonic < 3600.0
        ):
            return 0
        self._last_prune_monotonic = now
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=retention_days)
        ).strftime(_TIMESTAMP_FORMAT)

        def _prune(conn: sqlite3.Connection) -> int:
            return conn.execute(
                "DELETE FROM runs WHERE created_at < ?", (cutoff,)
            ).rowcount

        pruned = await self.run(_prune)
        if pruned:
            logger.warning(
                "pruned %d runs older than %dd; "
                "set CODEFYUI_RUNS_RETENTION_DAYS=0 to keep",
                pruned, retention_days,
            )
        return pruned
