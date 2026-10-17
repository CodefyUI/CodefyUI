"""Per-run spool of the frames the agent sends about a run (#783).

A run's subprocess keeps running when the connection to the control plane
drops (a laptop control plane sleeps, a VPN changes networks). So that its
events are not lost, every per-run frame is appended to the run's spool
BEFORE it is sent, and stays there until the control plane acknowledges it:

    <runs_dir>/<run_id>/spool.jsonl      one encoded frame per line

- ``ack{run_id, seq}`` (cumulative) moves the acknowledged offset past
  every frame up to ``seq``. The file is compacted (the acknowledged prefix
  cut off) only once that prefix is :data:`COMPACT_BYTES` or half the file,
  so an ack costs no I/O most of the time.
- After a reconnect, :meth:`Spool.read_after` hands the frames after a seq
  back in bounded batches, in the order they were produced, and the
  connection replays them after ``run_alive``. The control plane drops the
  ones it had already seen (by ``seq``).
- Once the frame acknowledged is the run's ``run_finished`` the spool is
  complete: the file is deleted.

A spool file left by a previous agent process is stale: that agent's run
subprocesses ended with it (they are its children), so nothing will ever
finish those runs. :class:`SpoolStore` deletes such files when it starts;
the control plane ends the runs ``interrupted`` (the reconnected agent
does not list them).

Every method that touches the file blocks: the connection calls them in a
thread (``asyncio.to_thread``). A lock per spool keeps an ack's compaction
from moving the file under a concurrent append or read. Not fsynced: the
spool protects against a network drop, not against the agent itself
dying, which ends the runs anyway.
"""

from __future__ import annotations

import bisect
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

#: The spool's file name inside a run's directory. A run directory that
#: holds one is never pruned by retention (``runner.prune_run_dirs``).
SPOOL_FILE = "spool.jsonl"
#: The acknowledged prefix is cut off the file once it is this large, or
#: half the file, whichever comes first.
COMPACT_BYTES = 4 * 1024 * 1024


class Spool:
    """One run's frames that the control plane has not acknowledged yet."""

    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self.path = run_dir / SPOOL_FILE
        #: Highest seq the control plane acknowledged.
        self.acked = 0
        #: Highest seq appended.
        self.last_seq = 0
        #: Highest seq sent, in order, on the current connection. Owned by
        #: the connection (event loop); reset to :attr:`acked` on reconnect.
        self.sent = 0
        #: The ``run_finished`` frame's seq, once it is spooled.
        self.final_seq: int | None = None
        #: Seq and end offset of every line in the file, in file order;
        #: ``_first`` is the index of the first unacknowledged one.
        self._seqs: list[int] = []
        self._ends: list[int] = []
        self._first = 0
        self._size = 0
        self._lock = threading.Lock()
        run_dir.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "wb")  # noqa: SIM115 -- held open

    @property
    def pending(self) -> int:
        """Frames not acknowledged yet."""
        return len(self._seqs) - self._first

    @property
    def complete(self) -> bool:
        """True once the run's ``run_finished`` was acknowledged."""
        return self.final_seq is not None and self.acked >= self.final_seq

    def append(self, seq: int, line: str, final: bool = False) -> None:
        """Spool one encoded frame. *final*: it is the ``run_finished``."""
        data = line.encode("utf-8") + b"\n"
        with self._lock:
            self._handle.write(data)
            self._handle.flush()
            self._size += len(data)
            self._seqs.append(seq)
            self._ends.append(self._size)
            self.last_seq = max(self.last_seq, seq)
            if final:
                self.final_seq = seq

    def ack(self, seq: int) -> bool:
        """Every frame up to *seq* is recorded. True when complete."""
        with self._lock:
            if seq > self.acked:
                self.acked = seq
                self._first = bisect.bisect_right(self._seqs, seq,
                                                  lo=self._first)
                self._maybe_compact()
            return self.complete

    def _acked_bytes(self) -> int:
        return self._ends[self._first - 1] if self._first else 0

    def _maybe_compact(self) -> None:
        cut = self._acked_bytes()
        if cut == 0 or (cut < COMPACT_BYTES and cut * 2 < self._size):
            return
        self._handle.close()
        if self._first == len(self._seqs):
            self._handle = open(self.path, "wb")  # noqa: SIM115
        else:
            temporary = self.path.with_suffix(".tmp")
            with open(self.path, "rb") as source, \
                    open(temporary, "wb") as target:
                source.seek(cut)
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
            os.replace(temporary, self.path)
            self._handle = open(self.path, "ab")  # noqa: SIM115
        self._seqs = self._seqs[self._first:]
        self._ends = [end - cut for end in self._ends[self._first:]]
        self._first = 0
        self._size -= cut

    def read_after(self, seq: int, limit: int) -> list[tuple[int, str]]:
        """Up to *limit* ``(seq, line)`` after *seq* and after the
        acknowledged ones, in order. Streams: one batch in memory."""
        with self._lock:
            self._handle.flush()
            start_index = bisect.bisect_right(self._seqs, seq,
                                              lo=self._first)
            stop_index = min(start_index + limit, len(self._seqs))
            if start_index >= stop_index:
                return []
            start = self._ends[start_index - 1] if start_index else 0
            with open(self.path, "rb") as source:
                source.seek(start)
                data = source.read(self._ends[stop_index - 1] - start)
            out: list[tuple[int, str]] = []
            previous = 0
            for index in range(start_index, stop_index):
                end = self._ends[index] - start
                out.append((self._seqs[index], data[previous:end]
                            .rstrip(b"\n").decode("utf-8")))
                previous = end
            return out

    def close(self, *, delete: bool) -> None:
        with self._lock:
            try:
                self._handle.close()
            except OSError:
                pass
            if delete:
                self.path.unlink(missing_ok=True)


class SpoolStore:
    """Every run's spool on this agent, by run id. Thread-safe."""

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir
        self._spools: dict[str, Spool] = {}
        self._lock = threading.Lock()
        discard_stale_spools(runs_dir)

    def spool(self, run_id: str) -> Spool:
        """*run_id*'s spool, created on its first frame. Blocking."""
        with self._lock:
            spool = self._spools.get(run_id)
            if spool is None:
                spool = Spool(self.runs_dir / run_id, run_id)
                self._spools[run_id] = spool
            return spool

    def spools(self) -> list[Spool]:
        with self._lock:
            return list(self._spools.values())

    def ack(self, run_id: str, seq: int) -> bool:
        """Apply an ``ack``. True when it completed (and deleted) the
        spool. Blocking."""
        with self._lock:
            spool = self._spools.get(run_id)
        if spool is None:
            logger.debug("ack for run %s, which has no spool", run_id)
            return False
        if not spool.ack(seq):
            return False
        spool.close(delete=True)
        with self._lock:
            self._spools.pop(run_id, None)
        return True


def discard_stale_spools(runs_dir: Path) -> int:
    """Delete the spool files a previous agent process left. Returns how many."""
    removed = 0
    try:
        candidates = list(runs_dir.iterdir())
    except OSError:
        return 0
    for path in candidates:
        try:
            (path / SPOOL_FILE).unlink()
        except OSError:
            continue
        removed += 1
    if removed:
        logger.warning("discarded %d spool(s) left by a previous agent "
                       "process; their runs ended with it", removed)
    return removed
