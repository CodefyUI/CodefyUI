"""Per-run spool of the frames the agent sends about a run (#783).

A run's subprocess keeps running when the connection to the control plane
drops (a laptop control plane sleeps, a VPN changes networks). So that its
events are not lost, every per-run frame is appended to the run's spool
BEFORE it is sent, and stays there until the control plane acknowledges it:

    <runs_dir>/<run_id>/spool.jsonl      one encoded frame per line

- ``ack{run_id, seq}`` (cumulative) drops every frame up to ``seq``: the
  file is cut down to the frames after it. Acks arrive every few dozen
  frames or once a second, so what is left is short while connected.
- After a reconnect, :meth:`Spool.unacked` yields every frame not yet
  acknowledged, in the order they were produced, and the connection
  replays them after ``run_alive``. The control plane drops the ones it
  had already seen (by ``seq``).
- Once the frame acknowledged is the run's ``run_finished`` the spool is
  complete: the file is deleted.

A spool file left by a previous agent process is stale: that agent's run
subprocesses ended with it (they are its children), so nothing will ever
finish those runs. :class:`SpoolStore` deletes such files when it starts;
the control plane ends the runs ``interrupted`` (the reconnected agent
does not list them).

Not fsynced: the spool protects against a network drop, not against the
agent itself dying, which ends the runs anyway. Event-loop only.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path


logger = logging.getLogger(__name__)

#: The spool's file name inside a run's directory. A run directory that
#: holds one is never pruned by retention (``runner.prune_run_dirs``).
SPOOL_FILE = "spool.jsonl"


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
        #: Highest seq sent on the current connection.
        self.sent = 0
        #: The ``run_finished`` frame's seq, once it is spooled.
        self.final_seq: int | None = None
        #: ``(seq, end offset)`` of every line in the file, in file order.
        self._lines: list[tuple[int, int]] = []
        self._size = 0
        run_dir.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "wb")  # noqa: SIM115 -- held open

    @property
    def pending(self) -> int:
        """Frames not acknowledged yet."""
        return len(self._lines)

    @property
    def complete(self) -> bool:
        """True once the run's ``run_finished`` was acknowledged."""
        return self.final_seq is not None and self.acked >= self.final_seq

    def append(self, seq: int, line: str, *, final: bool = False) -> None:
        """Spool one encoded frame. *final*: it is the ``run_finished``."""
        data = line.encode("utf-8") + b"\n"
        self._handle.write(data)
        self._handle.flush()
        self._size += len(data)
        self._lines.append((seq, self._size))
        self.last_seq = max(self.last_seq, seq)
        if final:
            self.final_seq = seq

    def ack(self, seq: int) -> bool:
        """Drop every frame up to *seq*. True when the spool is complete."""
        if seq <= self.acked:
            return self.complete
        self.acked = seq
        self.sent = max(self.sent, seq)
        cut_index = 0
        while cut_index < len(self._lines) \
                and self._lines[cut_index][0] <= seq:
            cut_index += 1
        if cut_index:
            self._truncate(cut_index)
        return self.complete

    def _truncate(self, count: int) -> None:
        """Remove the first *count* lines from the file."""
        cut = self._lines[count - 1][1]
        remaining = self._lines[count:]
        self._handle.close()
        if not remaining:
            self._handle = open(self.path, "wb")  # noqa: SIM115
        else:
            with open(self.path, "rb") as source:
                source.seek(cut)
                rest = source.read()
            temporary = self.path.with_suffix(".tmp")
            with open(temporary, "wb") as target:
                target.write(rest)
            os.replace(temporary, self.path)
            self._handle = open(self.path, "ab")  # noqa: SIM115
        self._lines = [(s, end - cut) for s, end in remaining]
        self._size -= cut

    def unacked(self) -> list[tuple[int, str]]:
        """``(seq, line)`` for every frame not acknowledged, in order.

        Read in one go: an ``ack`` arriving while the caller replays them
        rewrites the file underneath.
        """
        self._handle.flush()
        lines = list(self._lines)
        with open(self.path, "rb") as source:
            data = source.read(lines[-1][1] if lines else 0)
        out: list[tuple[int, str]] = []
        start = 0
        for seq, end in lines:
            out.append((seq, data[start:end].rstrip(b"\n").decode("utf-8")))
            start = end
        return out

    def close(self, *, delete: bool) -> None:
        try:
            self._handle.close()
        except OSError:
            pass
        if delete:
            self.path.unlink(missing_ok=True)


class SpoolStore:
    """Every run's spool on this agent, by run id."""

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir
        self._spools: dict[str, Spool] = {}
        discard_stale_spools(runs_dir)

    def get(self, run_id: str) -> Spool | None:
        return self._spools.get(run_id)

    def spool(self, run_id: str) -> Spool:
        """*run_id*'s spool, created on its first frame."""
        spool = self._spools.get(run_id)
        if spool is None:
            spool = Spool(self.runs_dir / run_id, run_id)
            self._spools[run_id] = spool
        return spool

    def spools(self) -> list[Spool]:
        return list(self._spools.values())

    def ack(self, run_id: str, seq: int) -> bool:
        """Apply an ``ack``. True when it completed (and deleted) the spool."""
        spool = self._spools.get(run_id)
        if spool is None:
            logger.debug("ack for run %s, which has no spool", run_id)
            return False
        if not spool.ack(seq):
            return False
        spool.close(delete=True)
        del self._spools[run_id]
        return True

    def close(self) -> None:
        """Close every file, keeping them (the agent is stopping)."""
        for spool in self._spools.values():
            spool.close(delete=False)
        self._spools.clear()


def discard_stale_spools(runs_dir: Path) -> int:
    """Delete the spool files a previous agent process left. Returns how many."""
    removed = 0
    try:
        candidates = list(runs_dir.iterdir())
    except OSError:
        return 0
    for path in candidates:
        spool = path / SPOOL_FILE
        try:
            spool.unlink()
        except OSError:
            continue
        removed += 1
    if removed:
        logger.warning("discarded %d spool(s) left by a previous agent "
                       "process; their runs ended with it", removed)
    return removed
