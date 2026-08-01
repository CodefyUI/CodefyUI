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
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterator, TypeVar

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


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """``BEGIN IMMEDIATE`` … ``COMMIT``, rolling back on any exception.

    The inline form of this (``BEGIN IMMEDIATE`` / try / ``COMMIT`` /
    except ``BaseException`` / ``ROLLBACK`` / re-raise) is already the house
    idiom in ``routes_apps.py`` and ``_apply_migrations``; this is the same
    thing factored out for the modules that need it in several places.
    Those two existing call sites are deliberately left inline — rewriting
    shipped transaction code is not worth the risk in an unrelated change,
    so the two forms coexist until something else has to touch them.

    IMMEDIATE, not the default DEFERRED: the write lock is taken up front,
    so a read-then-write inside the block (``SELECT MAX(...) + 1`` then
    ``INSERT``) cannot interleave with another writer. DEFERRED would
    upgrade the lock only at the INSERT, leaving the read stale.

    Only usable because ``Database.connect`` passes ``isolation_level=None``
    — sqlite3's default would emit its own implicit BEGIN and de-atomize
    this.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            # A failed COMMIT can already have rolled the transaction back;
            # never let that mask the original error on its way out.
            pass
        raise


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

        Cancelling the caller does NOT abandon the worker
        --------------------------------------------------
        ``asyncio.to_thread`` cancellation cancels the *awaiter*, never the
        thread — a running sqlite3 statement cannot be interrupted. A bare
        ``await asyncio.to_thread(...)`` would therefore unwind the
        ``async with`` and RELEASE THE LOCK while ``fn`` was still executing
        on the shared connection, so the next caller would land in the
        middle of someone else's transaction: its writes silently swallowed
        by that transaction's ROLLBACK, swept into its COMMIT, or refused
        with "cannot start a transaction within a transaction".

        So the cancellation is absorbed (``shield``) and re-raised only once
        the thread has finished and the transaction is settled. The loop
        handles repeated cancellation — during shutdown a task can be
        cancelled more than once, and every one of those must be held until
        the connection is quiet. ``shield`` ALONE is not enough: the outer
        await still raises immediately and the lock still goes early.

        The cost is bounded: cancelling a DB call now takes as long as the
        statement in flight, which ``PRAGMA busy_timeout=5000`` caps.
        """
        async with self._lock:
            conn = self._conn
            if conn is None:
                raise RuntimeError("Database is not connected")
            worker = asyncio.ensure_future(asyncio.to_thread(fn, conn))
            cancellation: BaseException | None = None
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError as exc:
                    if not worker.done():
                        cancellation = exc   # ours, not the worker's
                except BaseException:
                    pass                     # worker.result() re-raises below
            if cancellation is not None:
                raise cancellation
            return worker.result()

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
