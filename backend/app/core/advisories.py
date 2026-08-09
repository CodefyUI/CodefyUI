"""One non-fatal sentence, delivered to every surface a user might watch.

#252 built this delivery for ``TrainingLoop``'s schedule-length notes
(#205/#244) and #149 needs exactly the same shape from
``CheckpointLoader``, one package over. The *decision* -- is there anything
to say, and what -- stays with the node that knows; only the delivery lives
here, because getting it right means knowing three unobvious things about
this codebase and there is no reason for a second node author to rediscover
them:

* ``logger.warning`` reaches the server console and the rotating log file,
  and is the ONLY channel a ``run_graph.py`` invocation or an exported
  Python script has at all.
* ``context.log_warning`` reaches the run's durable event log, which the
  Runs panel reads back as a ``warning``-toned line **while the run is
  still going**. This is the one that arrives on time. It is silently
  discarded on a run with no durable consumer, like every other signal.
* the returned string is what the caller puts in its result under
  ``__log__``, which is the only result key the canvas Log tab renders
  (``core.output_entries``); dunder keys are filtered out of recorded
  outputs and port summaries, so it adds a log line and nothing else. It
  appears only when the node finishes -- late, but the canvas has no
  earlier channel (``useGraphExecution`` registers no ``run_warning``
  handler and ``LogEntry.type`` has no warning severity).

Because the Log tab has no severity of its own, the ``prefix`` in the text
is the only thing distinguishing an advisory from a ``Print`` node's
output. Every caller passes one.

Nothing here raises: an advisory must never be able to fail the run it is
about.
"""

from __future__ import annotations

import logging
from typing import Any

_fallback_logger = logging.getLogger(__name__)


def emit_advisory(
    note: str,
    *,
    kind: str,
    prefix: str,
    context: Any = None,
    logger: logging.Logger | None = None,
) -> str:
    """Send *note* to all three surfaces and return the ``__log__`` line.

    *kind* is the stable token a client may branch on (it becomes
    ``WarningSignal.kind``); *note* is the human sentence. Both are set by
    the caller, because a warning nobody can act on is worse than none.

    *logger* is the CALLER's logger, so the server-log line is attributed
    to the node that had something to say rather than to this module.
    """
    log = logger or _fallback_logger
    log.warning("%s", note)

    log_warning = getattr(context, "log_warning", None)
    if callable(log_warning):
        try:
            log_warning(kind, note)
        except Exception:  # noqa: BLE001 - see the module docstring
            log.debug("could not record the %s warning", kind, exc_info=True)
    return prefix + note


def join_notes(*notes: str | None) -> str | None:
    """Fold several advisories into one ``__log__`` value, or ``None``.

    A node can have more than one thing to say in a single run (a resumed
    ``TrainingLoop`` can both fail to reposition its schedule and be handed
    a schedule of the wrong length). ``__log__`` is a single string, so they
    are stacked with a blank line between them rather than one silently
    winning.
    """
    present = [n for n in notes if n]
    if not present:
        return None
    return "\n\n".join(present)
