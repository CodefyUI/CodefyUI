"""How a pack install fails, in terms the UI can act on.

An install can fail in six shapes, and they lead to six different next
steps for the person watching. Collapsing them into one ``RuntimeError``
would put that decision back on the frontend, which would then have to
match on message text:

* :class:`PackInstallError` -- it did not work, here is what pip said. The
  optional ``hint`` carries the last lines of output, which is what a
  learner has to paste into an issue and what a teacher reads first.
* :class:`PackCancelled` -- the user pressed Stop. NOT an install error: it
  is the system doing as it was told, so it deliberately does not inherit
  from :class:`PackInstallError` and no ``except PackInstallError`` will
  report it as a failure.
* :class:`PackNeedsRestart` -- the resolver cannot do this while the server
  is running (something already imported would have to be replaced).
  ``command`` is the exact thing to type in a terminal instead.
* :class:`PackInsufficientDisk` -- there is not enough room. ``needed`` and
  ``free`` are bytes, so the UI can say how much short it is rather than
  "an error occurred".
* :class:`PendingExists` -- a restart-mode install is ALREADY waiting to
  run, claimed by a server process that is still alive. Nothing failed yet
  and nothing was overwritten; the next step is to wait for that one, not
  to retry this one.
* :class:`RestartRefused` -- this server COULD restart itself to finish the
  install, and will not do it right now. ``reason`` says what is in the way
  and ``command`` is the way round it. Nothing has been changed, and the
  same request usually succeeds a minute later.

Stdlib only, and imported by ``packs/__init__.py``: node code catches these
without dragging the installer's machinery into a graph run.
"""

from __future__ import annotations


class PackInstallError(RuntimeError):
    """Installing a pack failed. ``hint`` is the operator-facing detail."""

    def __init__(self, message: str, *, hint: str | None = None):
        super().__init__(message)
        self.hint = hint


class PackCancelled(Exception):
    """The install was cancelled. Not a failure -- see the module docstring."""


class PackNeedsRestart(PackInstallError):
    """This install cannot finish inside the running server.

    ``command`` is what to run instead, spelled out in full: a message that
    says "restart required" without saying what to type leaves the user to
    guess at a CLI they have never run.
    """

    def __init__(self, message: str, *, command: str, hint: str | None = None):
        super().__init__(message, hint=hint)
        self.command = command


class PackInsufficientDisk(PackInstallError):
    """There is not enough free space. ``needed`` and ``free`` are BYTES.

    Raised BEFORE anything is downloaded: finding out at 90% of a 470 MB
    model that the disk was always too small wastes the download and leaves
    a half-written cache behind.
    """

    def __init__(self, message: str, *, needed: int, free: int,
                 hint: str | None = None):
        super().__init__(message, hint=hint)
        self.needed = needed
        self.free = free


class PendingExists(PackInstallError):
    """Another restart-mode install is already pending, and still live.

    The pending file is a CLAIM on one interpreter: the helper it names will
    reinstall packages into ``backend/.venv`` as soon as the server holding
    it exits. Two claims at once means two ``uv`` runs over one
    site-packages, so the second one is refused rather than allowed to
    overwrite the first.

    A subclass of :class:`PackInstallError` so the routes' existing mapping
    reports it without a new branch -- and so a caller that only knows
    "install errors" cannot miss it. What makes it different is that nothing
    has been changed or lost: the right next step is to wait for the restart
    already under way (``restart.clear_stale_pending`` removes the claim of
    a server that died holding it).
    """


class RestartRefused(PackInstallError):
    """A restart this server could have done, and will not do NOW.

    Not :class:`~app.core.packs.service.RestartUnavailable`, and the
    difference is the user's next move. "Unavailable" is a property of how
    the server was started and will not change while it runs, so the only
    way forward is the terminal. This is a passing condition -- a graph
    running, another restart already claimed -- so the answer is "not yet",
    and the same button works a minute later.

    ``reason`` is a short phrase naming what is in the way, for a UI that
    wants to say more than "409"; ``command`` is the same full line every
    other refusal carries, because a user who does not want to wait is
    entitled to do it from a terminal instead.

    A subclass of :class:`PackInstallError` for the same reason
    :class:`PendingExists` is (which is one of the conditions this reports):
    a caller that only knows "install errors" must not miss it. The routes
    map it BEFORE the generic 500 branch -- nothing failed here.
    """

    def __init__(self, message: str, *, reason: str, command: str,
                 hint: str | None = None):
        super().__init__(message, hint=hint)
        self.reason = reason
        self.command = command
