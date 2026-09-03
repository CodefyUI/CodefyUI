"""What can go wrong between a typed source and an installed plugin.

The CLI answers each of these with two lines of bilingual prose and an exit
code, because a terminal has nowhere else to put the detail. A GUI cannot do
that: "install failed" with a pip log folded into the sentence is a dead end
in a dialog, and re-deriving what happened by matching on message text is how
a translated string turns into a broken button.

So the REASON is carried in attributes and the PROSE stays with whoever is
talking to the user. ``UnknownCatalogName`` knows the spec, the ids this
install really has and where its catalog file lives; the CLI turns that into
the same four-line bilingual message it has always printed, and a route can
turn the identical exception into a list the frontend renders. Neither one
has to parse the other's sentence.

The base classes are chosen so that no existing ``except`` clause has to
change. ``ManifestError`` and ``SourceError`` are ``ValueError`` subclasses
because ``validate_manifest`` and ``parse_source`` have always raised
``ValueError`` and their callers still catch it; ``PluginInstallError`` is a
``RuntimeError`` for the same reason on the install side.

Stdlib and ``app.core.jobs``, and imported by ``plugins/__init__.py``: a
caller that only wants to catch a plugin failure never has to drag the
installer, the AST gate or the network in to do it. The one import that is
not the standard library is the job runner's own ``JobBusy``, which is
itself domain-free and stdlib-only -- and a refusal to START an install is a
refusal about a JOB, so a client that catches one installer's busy answer
must be able to catch the other's with the same clause.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..jobs import JobBusy


class PluginInstallError(RuntimeError):
    """Installing a plugin failed. ``hint`` is the operator-facing detail.

    The hint is the last lines of whatever went wrong -- pip's output, a
    tarball error -- which is what a learner pastes into an issue and what a
    teacher reads first. It is deliberately separate from the message so a
    UI can show one line and keep the other behind a disclosure.
    """

    def __init__(self, message: str, *, hint: str | None = None):
        super().__init__(message)
        self.hint = hint


class PluginCancelled(Exception):
    """The install was cancelled. NOT a failure.

    It is the system doing as it was told, so it deliberately does not
    inherit from :class:`PluginInstallError`: no ``except PluginInstallError``
    may report a cancel as something that went wrong.
    """


class PluginNeedsRestart(PluginInstallError):
    """This install cannot finish inside the running server.

    ``command`` is what to run instead, spelled out in full: a message that
    says "restart required" without saying what to type leaves the user to
    guess at a CLI they have never run.
    """

    def __init__(self, message: str, *, command: str, hint: str | None = None):
        super().__init__(message, hint=hint)
        self.command = command


class ConsentRequired(PluginInstallError):
    """The install stopped at a decision only the user can make.

    Two shapes of consent, one exception, because the caller's next move is
    the same for both: show what is being asked for and ask. The CLI asks at
    a prompt (``capability_gate``, ``--trust-author``); a GUI has to ask in a
    dialog, and it can only build that dialog from a list -- which is why
    ``missing_capabilities`` and ``allowed_modules`` are tuples of names
    rather than a sentence naming them.

    An install error rather than a sibling of :class:`PluginCancelled`: the
    install did not complete, and a caller that only knows "install errors"
    must not silently treat an ungranted capability as success.
    """

    def __init__(
        self,
        message: str,
        *,
        missing_capabilities: Iterable[str] = (),
        allowed_modules: Iterable[str] = (),
        hint: str | None = None,
    ):
        super().__init__(message, hint=hint)
        self.missing_capabilities = tuple(missing_capabilities)
        self.allowed_modules = tuple(allowed_modules)


class TrustAuthorRequired(ConsentRequired):
    """The user has not agreed to this plugin's ``allowed_modules``.

    A subclass rather than a second class, because every caller that already
    catches :class:`ConsentRequired` -- the CLI's ``_install_github``, the
    flow -- is asking the same question and must keep getting the same
    answer. What it adds is a NAME for the half of consent that is not about
    capabilities: the two are refused with different words and answered with
    different controls (a list of checkboxes against a single "I trust this
    author"), and a caller that has to tell them apart by asking which tuple
    happens to be populated is one bad ``if`` away from showing the wrong
    dialog.
    """


class InspectBusy(PluginInstallError):
    """Something else is being inspected right now. Try again in a moment.

    Reading a source means a network round trip or a manifest off the disk,
    and one at a time is the whole design: a panel that fires an inspection
    per keystroke would otherwise open a socket per keystroke. NOT queued --
    a caller that waited would get its answer long after the person had
    typed something else -- so this is refused immediately and the caller
    asks again.
    """


class InspectionExpired(KeyError):
    """No inspection with this id: it was used, evicted, or timed out.

    A ``KeyError`` because that is what a lookup that finds nothing raises,
    and every one of those three endings is the same fact to the caller: the
    consent screen it is holding describes a source this server no longer
    remembers reading, so the answer is to inspect it again rather than to
    install from a manifest nobody can now produce.
    """

    def __init__(self, inspection_id: str):
        self.inspection_id = inspection_id
        super().__init__(inspection_id)


class PluginBusy(JobBusy):
    """An install is already running. ``job_id`` is the one to follow.

    The Plugin Center's wording for :class:`~app.core.jobs.JobBusy`, and the
    twin of ``packs.service.PackBusy``: the two installers share one
    interpreter, so each of them refuses while the OTHER is running as well
    as while its own job is. ``reason`` is what says which of the two this
    is -- ``None`` for our own job (the id already implies it) and
    ``"pack_install_running"`` when the job in the way belongs to the
    Package Center, because "follow the install you started" and "wait for
    somebody else's" are different offers for a panel to make.
    """

    def __init__(self, job_id: str, *, reason: str | None = None,
                 message: str | None = None):
        super().__init__(
            job_id,
            message or f"a plugin install is already running (job {job_id})",
            reason=reason)


class AlreadyInstalled(PluginInstallError):
    """This plugin is here already, and nobody asked to replace it.

    Its own class because it is the one install failure that is not a
    failure of the plugin: nothing is wrong with the source, the answer is
    "you already have it", and the caller's next move is an offer -- a
    ``--force`` in the terminal, a Reinstall button in the panel -- rather
    than a message about what went wrong. A route turns this into 409 where
    every other install error is a 500, which is a distinction no ``except
    PluginInstallError`` can make by reading the sentence.

    ``plugin_id`` travels with it because the offer has to name the plugin,
    and a caller holding an id it parsed back out of the message is a caller
    that breaks the day the message is translated.
    """

    def __init__(self, message: str, *, plugin_id: str, hint: str | None = None):
        super().__init__(message, hint=hint)
        self.plugin_id = plugin_id


class ReservedPluginId(PluginInstallError):
    """The manifest declares an id something in this build already owns.

    A route under ``/api/plugins/``, a pack that ships in this release, or
    another repository's catalog row -- :func:`~.catalog.reserved_id_holder`
    decides which, and ``taken_by`` is the noun phrase it answered with. The
    id is what the lockfile, the catalog card and ``/api/plugins/{id}`` all
    key on, so installing a second thing under it would not be a conflict the
    user could see; it would be one pack quietly standing where another was.

    Its own class for the reason :class:`AlreadyInstalled` has one: the two
    things a caller has to say -- WHICH id clashed and WHAT holds it -- are
    the two things a caller cannot recover from the sentence. The route used
    to match a regular expression against the message to find the id, which
    made the wording of an English sentence part of the wire contract; the
    CLI prints that sentence, so the sentence and the attributes now travel
    together and neither is derived from the other.
    """

    def __init__(self, message: str, *, plugin_id: str, taken_by: str,
                 hint: str | None = None):
        super().__init__(message, hint=hint)
        self.plugin_id = plugin_id
        self.taken_by = taken_by


class SourceError(ValueError):
    """What the user typed cannot be turned into something to install.

    A ``ValueError`` on purpose: ``parse_source`` has raised one since it
    existed and every caller -- ``cmd_install``, ``cmd_info``,
    ``scripts/project.py`` -- still catches ``ValueError`` around it. Making
    the structured version a subclass means those keep working untouched.
    """


class UnknownCatalogName(SourceError):
    """A bare word this install's catalog has never heard of.

    Its own class because the generic "expected a catalog name, owner/repo or
    a URL" is a dead end here: the user *did* type a catalog name, and what
    they cannot see is that their copy of ``registry.json`` does not have it
    (usually an install old enough to predate the pack). Answering that needs
    the ids this install really has and the file they were read from, so both
    travel with the exception instead of being re-derived by each caller.
    """

    def __init__(self, spec: str, known: Iterable[str], catalog_path: Path):
        self.spec = spec
        self.known = tuple(known)
        self.catalog_path = catalog_path
        super().__init__(
            f"no plugin pack named {spec!r} in the catalog at {catalog_path}"
        )


class UnparseableSource(SourceError):
    """Not a catalog name, not ``owner/repo``, not a GitHub URL.

    Carries only the spec: there is nothing else true to say about it, and
    the suggestion of what to type instead depends on the catalog the caller
    is showing, not on this failure.
    """

    def __init__(self, spec: str):
        self.spec = spec
        super().__init__(f"could not parse plugin source {spec!r}")


class GitHubError(RuntimeError):
    """Resolving or downloading from GitHub failed.

    ``status`` is the HTTP status when there was one and ``None`` when the
    request never got that far (DNS, TLS, a timeout). The difference is the
    user's next step -- a 404 is a typo in the repo name, a missing status is
    the network -- and it is not recoverable from the message text.
    """

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


class ManifestError(ValueError):
    """A ``cdui.plugin.toml`` this build will not install.

    A ``ValueError`` for the same reason :class:`SourceError` is: every
    ``except ValueError`` around ``validate_manifest`` -- in the CLI and in
    the tests -- predates this class and keeps working.
    """
