"""``cdui agent start --one-shot``: take one run, report it, exit (#790).

GitHub's ephemeral self-hosted runner, for a SLURM job or an ``ssh``
command: the batch system hands out the GPU, the agent connects to the
control plane, takes exactly one run, reports it and exits with a code
the job script can read. ``--run-id ID`` takes that run only.

The agent says ``hello.one_shot`` (with ``one_shot_run_id`` and its
``instance_id``); the control plane keeps it out of ``target: any``
placement and places at most one run on it (``core.workers``). On this
side :class:`OneShot` follows that run:

- the first ``assign`` it accepts is THE run (``taken``); any other
  assignment is refused (``run_finished`` ``failed``, which never happens
  with a control plane that keeps the rule);
- its ``run_finished`` gives the outcome (``finished``);
- the control plane's ``ack`` of that ``run_finished`` says the outcome is
  recorded there (``delivered``): only then does the agent exit, so a
  connection that drops mid-run is ridden out the usual way (spool,
  reconnect, replay, #783). If the outcome cannot be delivered within
  the window the control plane keeps a disconnected run
  (``welcome.resume_grace_s``), the agent gives up and exits anyway; the
  spool stays on disk and a later agent with the same user data dir
  replays it (a late result there).

Exit codes (:meth:`OneShot.exit_code`): :data:`EXIT_SUCCEEDED` (0) the run
succeeded and the control plane recorded it; :data:`EXIT_RUN_FAILED` (1) it
failed, was cancelled or interrupted, or its outcome could not be
delivered; 2 the control plane refused the agent (``reject``, a pinned
certificate that does not match, or a bad flag), as for every agent;
:data:`EXIT_NO_RUN` (3) the agent was stopped (SIGTERM, Ctrl-C, a batch
system's time limit) before a run arrived.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable

from ..core import worker_protocol as wp

logger = logging.getLogger(__name__)

EXIT_SUCCEEDED = 0
EXIT_RUN_FAILED = 1
EXIT_NO_RUN = 3

#: The shortest wait for the control plane to record the outcome.
MIN_DELIVER_WINDOW_S = 60.0


@dataclass
class OneShot:
    """The one run of a ``--one-shot`` agent."""

    #: ``--run-id``: the only run this agent takes, or None.
    run_id: str | None = None
    #: The run it took, once assigned.
    taken: str | None = None
    #: That run's ``run_finished.status``.
    status: str | None = None
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    delivered: asyncio.Event = field(default_factory=asyncio.Event)

    def refusal(self, run_id: str) -> str | None:
        """Why this agent does not take *run_id*, or None (it takes it)."""
        if self.taken is not None and self.taken != run_id:
            return (f"this one-shot agent already took run {self.taken}")
        if self.run_id is not None and self.run_id != run_id:
            return f"this one-shot agent takes only run {self.run_id}"
        return None

    def take(self, run_id: str) -> None:
        if self.taken is None:
            self.taken = run_id
            logger.info("one-shot: took run %s", run_id)

    def sent(self, frame: object) -> None:
        """Called with every frame the agent sends: notes the run's end."""
        if isinstance(frame, wp.RunFinished) and frame.run_id == self.taken \
                and self.status is None:
            self.status = frame.status
            self.finished.set()

    def acked(self, run_id: str) -> None:
        """The control plane recorded *run_id*'s ``run_finished``."""
        if run_id == self.taken and self.finished.is_set():
            self.delivered.set()

    async def wait_done(self, stop: asyncio.Event,
                        deliver_window_s: Callable[[], float]) -> None:
        """Set *stop* once the run's outcome is delivered, or once it could
        not be within *deliver_window_s()* seconds of the run's end."""
        await self.finished.wait()
        window = max(MIN_DELIVER_WINDOW_S, float(deliver_window_s()))
        try:
            await asyncio.wait_for(self.delivered.wait(), timeout=window)
        except asyncio.TimeoutError:
            logger.error(
                "one-shot: run %s ended %s, but the control plane did not "
                "record it within %.0f s; exiting. Its frames stay spooled "
                "and a later agent with this user data dir replays them",
                self.taken, self.status, window)
        stop.set()

    def exit_code(self, code: int) -> int:
        """The agent's exit code, given the connection loop's *code*."""
        if code != 0:
            return code
        if self.taken is None:
            logger.warning("one-shot: stopped before a run arrived (exit %d)",
                           EXIT_NO_RUN)
            return EXIT_NO_RUN
        ok = self.status == wp.FINISHED_SUCCEEDED and self.delivered.is_set()
        result = EXIT_SUCCEEDED if ok else EXIT_RUN_FAILED
        logger.info("one-shot: run %s %s%s; exit %d", self.taken,
                    self.status or "did not finish",
                    "" if self.delivered.is_set() or self.status is None
                    else " (not recorded by the control plane)", result)
        return result
