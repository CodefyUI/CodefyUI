"""One plugin install, start to finish. The server job and the CLI both call this.

Deliberately SYNCHRONOUS and free of any transport, the way
``packs/flows.py`` is. ``cdui plugin install`` and the Plugin Center differ
only in where the events go -- a terminal in one case, a long poll in the
other -- so the ORDER of an install, what counts as a failure and what a
failure is called are written down exactly once, here, and both get the same
answers. The console and the panel showing different steps for the same
install was the shape this replaces.

The order is the design, and it is not the order the CLI grew into:

1. **resolve** decides, before a byte moves, whether this install may happen
   at all. The sha is the one the user was shown -- never re-resolved, or a
   branch that moved between the preview and the install would substitute a
   commit nobody looked at -- and the reserved-id and already-installed
   questions are asked HERE rather than after the download, which is where
   the CLI asks them and why it can spend a minute fetching a repository it
   was always going to refuse.
2. **download** and **extract**, into a temporary directory that the
   ``with`` block takes away whatever happens.
3. the **identity check**, which is the reason inspecting is worth anything:
   the manifest that comes out of the tarball is compared against the one
   the user agreed to, and a plugin that grew a capability -- or changed its
   id -- between the two is refused. Consent to a manifest that is not the
   manifest being installed is not consent.
4. **verify**: the AST gate over the whole extracted tree.
5. **deps**, BEFORE anything is staged. A pip run that conflicts, or that
   the user stops halfway, then leaves nothing on disk to roll back -- where
   the CLI installs dependencies after staging and has to undo the staging
   when they fail.
6. **stage**: copy to ``.staging/<id>-<sha7>``, move any existing install to
   ``<id>.old-<timestamp>``, rename into place, and put the backup back if
   the rename fails.
7. **lock**: the lockfile entry, written last and atomically, so a lockfile
   that names a plugin is a lockfile whose plugin is on the disk.

A built-in pack takes the same road with most of it missing: its files
already shipped in this release, so there is nothing to download, nothing to
unpack and nobody outside this repository to be asked about. It says so once
(``built-in pack: gate skipped``) rather than emitting empty steps, because
a step that renders as done without having run is worse than no step.

``reload`` is NOT a step here. Re-discovering the registry touches a dict the
event loop reads, and this function runs on a worker thread; the caller runs
it after this returns (see ``JobRunner.start``'s ``after_work``).
"""

from __future__ import annotations

import http.client
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.core import plugin_loader

from . import catalog as catalog_module
from . import consent, deps, gate
from . import github as github_client
from .errors import (
    AlreadyInstalled,
    ConsentRequired,
    GitHubError,
    PluginCancelled,
    PluginInstallError,
)
from .inspect import Inspection
from .manifest import (
    manifest_allowed_modules,
    manifest_capabilities,
    manifest_python_deps,
    read_manifest,
    validate_manifest,
)

#: Where a download is assembled before it replaces anything. A sibling of
#: the installed plugins rather than a temp directory, so the rename that
#: puts it in place is a rename within one filesystem -- across devices it
#: would be a copy, and a copy is not atomic.
STAGING_DIRNAME = ".staging"

#: Minimum seconds between two ``progress`` events for the tarball. The
#: download client throttles its own callback to the same figure; this one
#: exists because the callback is a contract with whoever supplies the
#: client, and a client that reports every chunk must not become one event
#: per 64 KB on a long poll.
PROGRESS_MIN_INTERVAL_S = 0.25

#: The clock the progress throttle reads. A module attribute rather than a
#: direct call to ``time.monotonic`` so a test can hand this flow a clock it
#: moves itself: throttling is a rule with a deadline inside it, and the
#: only honest way to test one is to control the time it is measured against.
monotonic = time.monotonic


@dataclass(frozen=True)
class InstallPlan:
    """Everything one install needs, decided before the install starts.

    The point of the split is that every decision a PERSON is entitled to
    make -- which capabilities are granted, whether the author is trusted,
    whether an existing copy may be replaced -- is made while there is still
    somebody to ask, and is written down here. What follows is machinery: it
    reads the plan, and the only thing it may do with a manifest that does
    not match the plan is refuse.

    ``sha`` is the commit the user was shown. ``manifest`` is what they were
    shown of it. ``granted_capabilities`` is the answer they gave, not the
    request; ``trust_author`` says the same thing about ``allowed_modules``.
    """

    kind: Literal["builtin", "github"]
    plugin_id: str
    catalog_id: str | None
    owner: str | None
    repo: str | None
    ref: str
    sha: str | None
    manifest: dict[str, Any]
    granted_capabilities: tuple[str, ...]
    trust_author: bool
    force: bool
    mode: Literal["install", "update"]
    prior: dict[str, Any] | None


@dataclass(frozen=True)
class InstallOutcome:
    """What one install actually did, for the caller to report.

    ``plugin_dir`` is the directory this install WROTE, which is why a
    built-in pack has none: it activates the copy that shipped with the
    release and puts nothing anywhere.
    """

    plugin_id: str
    sha: str | None
    deps_installed: tuple[str, ...]
    tombstone_cleared: bool
    replaced: bool
    plugin_dir: Path | None


def plan_from_inspection(
    inspection: Inspection,
    *,
    accept_capabilities: Iterable[str] = (),
    trust_author: bool = False,
    force: bool = False,
) -> InstallPlan:
    """Turn what the user was shown, plus what they answered, into a plan.

    This is where consent is ENFORCED -- before a job exists, on the caller's
    thread, where the refusal is still an answer to a request rather than a
    failed install somebody has to go and read events about. A capability
    nobody granted raises :class:`~.errors.ConsentRequired` carrying the
    list, and so does an untrusted ``allowed_modules``; the caller shows the
    list and asks again.

    *accept_capabilities* is a list of capability ids and never a boolean.
    "Yes to everything" is the caller enumerating what it is saying yes to,
    which is the only form that still means something when the manifest asks
    for one more thing than it did when the dialog was drawn.

    A built-in pack skips both questions (``consent_required`` is False on
    its inspection): it arrived through a pull request in this repository, so
    there is no third party here for the user to be asked about. Its
    capabilities are still RECORDED -- ``cdui plugin list`` answers "which of
    my plugins reaches the network" for every pack, wherever it came from.

    Never touches the network: it plans from the :class:`~.inspect.Inspection`
    it was handed, whose sha and manifest are already pinned to one commit.
    """
    prior = inspection.installed
    prior_capabilities = prior["capabilities"] if prior is not None else None
    prior_modules = prior["trusted_modules"] if prior is not None else None

    if inspection.consent_required:
        decision = consent.decide_capabilities(
            inspection.capabilities,
            accepted=accept_capabilities,
            prior=prior_capabilities,
        )
        if decision.missing:
            raise ConsentRequired(
                f"{inspection.plugin_id} asks for capabilities this install "
                f"has not granted: {', '.join(decision.missing)}.",
                missing_capabilities=decision.missing,
            )
        # Raises with the whole module list when the author is not trusted
        # for it -- including an update that asks for more than the modules
        # the user trusted last time.
        consent.check_trust(
            inspection.allowed_modules,
            trust_author=trust_author,
            prior_trusted=prior_modules,
        )
        granted: tuple[str, ...] = decision.granted
    else:
        granted = inspection.capabilities

    return InstallPlan(
        kind=inspection.kind,
        plugin_id=inspection.plugin_id,
        catalog_id=inspection.catalog_id,
        owner=inspection.owner,
        repo=inspection.repo,
        ref=inspection.ref or "",
        sha=inspection.sha,
        manifest=inspection.manifest,
        granted_capabilities=granted,
        # The plan records the ANSWER -- "this install may carry
        # allowed_modules" -- not the flag that was passed. ``check_trust``
        # lets an update through on the modules the user already trusted
        # without the flag being set a second time, and a plan that recorded
        # the flag would then have the identity check below refuse the very
        # tarball this call just approved. A manifest that asks for no
        # modules trusts nobody, which is what makes the check downstream
        # refuse a DOWNLOADED manifest that grew a module list.
        trust_author=bool(inspection.allowed_modules),
        force=force,
        mode=inspection.mode,
        prior=prior,
    )


def install_plugin_live(
    plan: InstallPlan,
    *,
    emit: Callable[[dict], None],
    cancel_check: Callable[[], bool],
    github: object | None = None,
) -> InstallOutcome:
    """Install *plan*, reporting every step to *emit*.

    Synchronous and blocking: the caller owns the thread. Raises
    :class:`~.errors.PluginCancelled` when *cancel_check* goes true,
    :class:`~.errors.AlreadyInstalled` when the plugin is here and nobody
    asked to replace it, :class:`~.errors.PluginNeedsRestart` when its
    dependencies cannot be installed into a running server, and
    :class:`~.errors.PluginInstallError` for everything else that fails.

    *github* is the client the GitHub half calls -- any object exposing
    ``download_tarball`` and ``extract_tarball``; ``None`` means this
    package's own :mod:`~.github`. It is a hook because ``scripts/plugins.py``
    re-exports the client under its own module attributes precisely so a
    test can replace them, and a flow that reached through the core module
    would leave those patches pointing at a name nothing calls -- a suite
    that passes while talking to the real network. A client that supplies
    only some of the names falls back to the core one per name, which is
    what lets the CLI hand over its own module unchanged.
    """
    if plan.kind == "builtin":
        return _install_builtin(plan, emit=emit, cancel_check=cancel_check)
    if plan.kind == "github":
        client = github_client if github is None else github
        return _install_from_github(
            plan, emit=emit, cancel_check=cancel_check, client=client
        )
    raise ValueError(f"install kind {plan.kind!r} has no live install")


# ── the two roads ──────────────────────────────────────────────────────────

def _install_builtin(
    plan: InstallPlan,
    *,
    emit: Callable[[dict], None],
    cancel_check: Callable[[], bool],
) -> InstallOutcome:
    """Activate a pack that shipped with this release."""
    plugin_dir = plugin_loader.plugins_builtin_root() / plan.plugin_id

    with _step(emit, "resolve", f"Reading {plan.plugin_id}"):
        manifest = _manifest_at(plugin_dir, source=str(plugin_dir))
        found_id = _manifest_id(manifest)
        if found_id != plan.plugin_id:
            raise PluginInstallError(
                f"The pack in {plugin_dir.name} installs as {found_id!r}.",
                hint=f"This install was planned for {plan.plugin_id!r}.",
            )
        if plan.prior is not None and not plan.force:
            raise _already_installed(plan)

    # No download, so no third-party code, so nothing for the AST gate to
    # look at that a pull request in this repository has not already been
    # through. Said out loud once: a "verify" step that renders as done
    # without having scanned anything is a claim this install cannot make.
    emit({"type": "log", "line": "built-in pack: gate skipped"})

    allowed_modules = tuple(manifest_allowed_modules(manifest))
    specs = tuple(deps.dep_specs(manifest_python_deps(manifest)))
    _stop_if_cancelled(plan, cancel_check)
    if specs:
        deps.install_deps_step(specs, emit=emit, cancel_check=cancel_check)
        _stop_if_cancelled(plan, cancel_check)

    with _step(emit, "lock", f"Recording {plan.plugin_id}"):
        cleared = _write_lockfile_entry(
            plan, _lockfile_record(plan, manifest, allowed_modules)
        )

    return InstallOutcome(
        plugin_id=plan.plugin_id,
        sha=None,
        deps_installed=specs,
        tombstone_cleared=cleared,
        replaced=plan.prior is not None,
        plugin_dir=None,
    )


def _install_from_github(
    plan: InstallPlan,
    *,
    emit: Callable[[dict], None],
    cancel_check: Callable[[], bool],
    client: object,
) -> InstallOutcome:
    """Fetch one commit of a repository and put it on the disk."""
    download_tarball = _client_call(client, "download_tarball")
    extract_tarball = _client_call(client, "extract_tarball")
    user_root = plugin_loader.plugins_user_root()
    plugin_dir = user_root / plan.plugin_id
    origin = f"{plan.owner}/{plan.repo}"

    with _step(emit, "resolve", f"Resolving {origin}"):
        sha = plan.sha
        if not sha:
            # Never re-resolved here: a plan with no sha is one built by
            # something that skipped the inspection, and guessing a commit
            # is the substitution the pinned sha exists to prevent.
            raise PluginInstallError(
                f"The plan for {plan.plugin_id!r} names no commit.",
                hint="An install plan is built from an inspection, which "
                     "pins the commit it read the manifest at.",
            )
        taken_by = catalog_module.reserved_id_holder(
            plan.plugin_id, owner=plan.owner, repo=plan.repo
        )
        if taken_by is not None:
            raise PluginInstallError(
                f"Plugin id {plan.plugin_id!r} is reserved by this build.",
                hint=(
                    f"{origin} declares an id that names {taken_by}; it "
                    f"cannot be installed under that id."
                ),
            )
        installed = plan.prior is not None or plugin_dir.exists()
        if installed and not plan.force and plan.mode != "update":
            raise _already_installed(plan)

    short_sha = sha[:7]
    staging = user_root / STAGING_DIRNAME / f"{plan.plugin_id}-{short_sha}"
    _stop_if_cancelled(plan, cancel_check)

    with tempfile.TemporaryDirectory(prefix="codefyui-plugin-") as workdir:
        tarball = Path(workdir) / "src.tar.gz"
        with _step(emit, "download", f"Downloading {origin}@{short_sha}"):
            try:
                download_tarball(
                    plan.owner,
                    plan.repo,
                    sha,
                    tarball,
                    cancel_check=cancel_check,
                    progress=_tarball_progress(emit),
                )
            except http.client.HTTPException as exc:
                # The client translates the OSError family and leaves this
                # one, which is not one. Unreachable from a URL built out of
                # a 40-hex sha, and one line cheaper than finding out.
                raise GitHubError(str(exc), status=None) from exc
        _stop_if_cancelled(plan, cancel_check)

        with _step(emit, "extract", f"Unpacking {plan.plugin_id}"):
            unpacked = Path(workdir) / "extracted"
            unpacked.mkdir()
            root = extract_tarball(tarball, unpacked)
            manifest = _manifest_at(root, source=f"{origin}@{short_sha}")
            allowed_modules = tuple(manifest_allowed_modules(manifest))
            _refuse_a_changed_manifest(plan, manifest, allowed_modules, short_sha)
        _stop_if_cancelled(plan, cancel_check)

        with _step(emit, "verify", f"Scanning {plan.plugin_id} for unsafe code"):
            try:
                gate.validate_plugin_dir(
                    root, list(allowed_modules), plan.granted_capabilities
                )
            except gate.PluginValidationError as exc:
                # The gate's own message names the file and says why, which
                # is the whole of what an operator needs and more than a
                # dialog's first line has room for -- so it becomes the hint.
                raise PluginInstallError(
                    f"{plan.plugin_id} was refused by the security scan.",
                    hint=str(exc),
                ) from exc
        _stop_if_cancelled(plan, cancel_check)

        specs = tuple(deps.dep_specs(manifest_python_deps(manifest)))
        if specs:
            deps.install_deps_step(specs, emit=emit, cancel_check=cancel_check)
            _stop_if_cancelled(plan, cancel_check)

        with _step(emit, "stage", f"Installing {plan.plugin_id}"):
            backup = _stage(
                root, staging, plugin_dir, plan=plan, cancel_check=cancel_check
            )

    # No cancel between here and the lockfile write, deliberately. The files
    # are already in place, so stopping now would not undo an install -- it
    # would leave one on the disk that no lockfile mentions, which is the one
    # state nothing in this system ever looks at again. Past the rename, the
    # only way out is to finish.
    with _step(emit, "lock", f"Recording {plan.plugin_id}"):
        cleared = _write_lockfile_entry(
            plan, _lockfile_record(plan, manifest, allowed_modules)
        )

    # Only now. The backup is the only copy of what was here before, so it
    # outlives every step that could still fail -- including the lockfile
    # write, whose failure leaves a directory beside the install that a
    # person can rename back.
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)

    return InstallOutcome(
        plugin_id=plan.plugin_id,
        sha=sha,
        deps_installed=specs,
        tombstone_cleared=cleared,
        replaced=backup is not None,
        plugin_dir=plugin_dir,
    )


# ── the steps' shared parts ────────────────────────────────────────────────

@contextmanager
def _step(
    emit: Callable[[dict], None], step: str, label: str
) -> Iterator[None]:
    """Emit ``step_started`` now and ``step_done`` only if the body finished.

    A step that raised is not a step that is done, and the client draws its
    list from these two events alone: closing the step from a ``finally``
    would tick a checkbox for work that failed.
    """
    emit({"type": "step_started", "step": step, "label": label})
    yield
    emit({"type": "step_done", "step": step})


def _stop_if_cancelled(
    plan: InstallPlan, cancel_check: Callable[[], bool]
) -> None:
    """Raise if the user has asked for this install to stop."""
    if cancel_check():
        raise PluginCancelled(f"install of {plan.plugin_id} cancelled")


def _tarball_progress(
    emit: Callable[[dict], None]
) -> Callable[[int, int | None], None]:
    """Adapt the download client's ``(done, total)`` callback to events.

    The client reports two positional numbers because it knows nothing about
    an event vocabulary; the frames a client reads carry the same four keys
    whoever produced them, so the shape is put on here.

    The first frame and the last one are forced past the throttle. A bar that
    stops at 97% because the download finished inside a throttle window is
    worse than no bar -- and the last frame is only knowable when there was a
    ``Content-Length``, which is also the only case in which a percentage
    exists to be left stuck.
    """
    last_report = 0.0
    reported = False

    def report(bytes_done: int, bytes_total: int | None) -> None:
        nonlocal last_report, reported
        stamp = monotonic()
        finished = bytes_total is not None and bytes_done >= bytes_total
        if reported and not finished and stamp - last_report < PROGRESS_MIN_INTERVAL_S:
            return
        last_report = stamp
        reported = True
        percent = None
        if bytes_total:
            percent = round(min(100.0, 100.0 * bytes_done / bytes_total), 1)
        emit({"type": "progress", "item": "tarball", "bytes_done": bytes_done,
              "bytes_total": bytes_total, "percent": percent})

    return report


def _client_call(client: object, name: str) -> Callable[..., Any]:
    """*name* off the GitHub client this install was handed.

    Missing names fall back to the core client one at a time, because the
    caller that supplies one at all is ``scripts/plugins.py``, which
    re-exports two of the three functions and reaches the rest through the
    module. Requiring the whole surface would make the hook unusable by the
    one caller it exists for.
    """
    found = getattr(client, name, None)
    return getattr(github_client, name) if found is None else found


def _manifest_at(plugin_root: Path, *, source: str) -> dict[str, Any]:
    """Read and validate the manifest in *plugin_root*.

    Both failures become :class:`~.errors.PluginInstallError` because this
    runs inside a job: a ``FileNotFoundError`` or a ``ManifestError`` out of
    here reaches a runner that has no mapping for either and reports a
    perfectly ordinary "this is not a plugin" as a crash.
    """
    try:
        manifest = read_manifest(plugin_root)
    except FileNotFoundError as exc:
        raise PluginInstallError(
            f"{source} has no {plugin_loader.MANIFEST_FILENAME}.",
            hint="A plugin is a directory with a manifest at its root.",
        ) from exc
    try:
        validate_manifest(manifest)
    except ValueError as exc:
        raise PluginInstallError(
            f"The manifest in {source} is not one this build installs.",
            hint=str(exc),
        ) from exc
    return manifest


def _manifest_id(manifest: dict[str, Any]) -> str:
    """The ``[plugin] id`` of *manifest*, or ``""``."""
    plugin_meta = manifest.get("plugin")
    plugin_meta = plugin_meta if isinstance(plugin_meta, dict) else {}
    found = plugin_meta.get("id")
    return found if isinstance(found, str) else ""


def _already_installed(plan: InstallPlan) -> AlreadyInstalled:
    return AlreadyInstalled(
        f"Plugin {plan.plugin_id!r} is already installed.",
        plugin_id=plan.plugin_id,
        hint="Install it again with force to replace the copy that is there.",
    )


def _refuse_a_changed_manifest(
    plan: InstallPlan,
    manifest: dict[str, Any],
    allowed_modules: Sequence[str],
    short_sha: str,
) -> None:
    """Refuse a tarball whose manifest is not the one that was consented to.

    The inspection read the manifest from ``raw.githubusercontent.com`` at a
    pinned sha and the tarball comes from ``codeload`` at the same sha, so in
    the ordinary case these two are the same file arriving twice. This is
    what makes the preview MEAN something: without it, the dialog describes
    one manifest and the installer obeys another, and every capability the
    user unticked is a capability the tarball can simply declare.

    Three ways they can differ, all fatal: a different id (the lockfile key,
    the card and the ``/api/plugins/{id}`` URL would all be somebody else's),
    a capability nobody granted, and an ``allowed_modules`` list that is not
    the one the user read.

    That last one is a comparison and not a flag. ``trust_author`` says the
    user agreed to a LIST -- the list the inspection showed them -- so a
    tarball that ships ``["subprocess", "os"]`` where the preview said
    ``["subprocess"]`` has taken an answer about one module list as
    permission for another, and every extra name is another door the AST
    gate is told to leave open. The untrusted case stays underneath it as
    the coarser refusal: a module list nobody was asked about at all.
    """
    reasons: list[str] = []
    found_id = _manifest_id(manifest)
    if found_id != plan.plugin_id:
        reasons.append(f"it installs as {found_id!r}, not {plan.plugin_id!r}")
    grew = tuple(
        capability
        for capability in manifest_capabilities(manifest)
        if capability not in plan.granted_capabilities
    )
    if grew:
        reasons.append(f"it asks for {', '.join(grew)}, which was not granted")
    consented = set(manifest_allowed_modules(plan.manifest))
    grew_modules = tuple(
        module for module in allowed_modules if module not in consented
    )
    if grew_modules:
        reasons.append(
            f"it asks to import {', '.join(grew_modules)}, which was not "
            f"consented to"
        )
    elif allowed_modules and not plan.trust_author:
        reasons.append(
            f"it asks to import {', '.join(allowed_modules)}, which needs the "
            f"author trusted"
        )
    if not reasons:
        return
    raise PluginInstallError(
        f"manifest at {short_sha} differs from the one you consented to",
        hint="; ".join(reasons) + ".",
    )


def _stage(
    source: Path,
    staging: Path,
    plugin_dir: Path,
    *,
    plan: InstallPlan,
    cancel_check: Callable[[], bool],
) -> Path | None:
    """Put *source* at *plugin_dir*; returns the backup it moved aside.

    Copy first, replace second. The copy is the long part and it touches
    nothing anybody is using, so it is also the last moment at which stopping
    costs nothing -- which is why the cancel is checked between the two
    halves rather than only between steps.

    A rename that fails puts the backup straight back, because the window
    between the two renames is the only moment in an install when the user
    has no plugin at all -- and when putting it back fails too, the failure
    names the directory it was left in, which is by then the only copy of
    what was there. Whatever goes wrong -- including halfway through
    the copy -- the staging copy goes with it: a ``.staging`` directory
    nothing points at is invisible, permanent, and the same size as the
    plugin.

    Both renames are translated. On Windows a file the editor, an antivirus
    or a running server still has open makes either one a raw
    ``PermissionError``, and an install that ends in one has told the user
    nothing about which plugin it was or what to close.
    """
    staging.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        shutil.rmtree(staging)

    backup: Path | None = None
    try:
        shutil.copytree(source, staging)
        _stop_if_cancelled(plan, cancel_check)
        if plugin_dir.exists():
            backup = plugin_dir.with_name(
                f"{plan.plugin_id}.old-{int(time.time())}"
            )
            try:
                plugin_dir.rename(backup)
            except OSError as exc:
                # Nothing has moved yet, so there is nothing to put back --
                # only the staging copy to throw away, which the handler
                # below does.
                backup = None
                raise PluginInstallError(
                    f"Could not move the previous {plan.plugin_id} aside.",
                    hint=str(exc),
                ) from exc
        try:
            staging.rename(plugin_dir)
        except OSError as exc:
            detail = str(exc)
            if backup is not None:
                try:
                    backup.rename(plugin_dir)
                except OSError as restore_exc:
                    # The usual Windows cause -- something holds the
                    # destination open -- breaks BOTH renames, so the restore
                    # is the failure most likely to happen here rather than
                    # the one that never does. Unguarded, its raw OSError
                    # replaced the translated one on the way out and nobody
                    # was told that the previous install is still on the disk
                    # under another name, which is the only fact that makes
                    # this recoverable by hand.
                    detail = (
                        f"{exc}; the previous install could not be put back "
                        f"({restore_exc}) and is at {backup.name}"
                    )
            raise PluginInstallError(
                f"Could not put {plan.plugin_id} in place.",
                hint=detail,
            ) from exc
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return backup


def _lockfile_record(
    plan: InstallPlan,
    manifest: dict[str, Any],
    allowed_modules: Sequence[str],
) -> dict[str, Any]:
    """The lockfile entry for a finished install.

    The keys and their order are the ones ``cdui plugin install`` has always
    written -- this is the same file, read by the same loader, and a second
    spelling of an entry is a lockfile that two halves of one program
    disagree about.

    ``capabilities`` is what was GRANTED and ``manifest`` is the ``[plugin]``
    table of the manifest that was actually installed. Both matter for the
    next update: the granted list is what a re-grant is measured against, and
    a manifest read back off the disk would be whatever the plugin says about
    itself now.
    """
    plugin_meta = manifest.get("plugin")
    record: dict[str, Any] = {
        "source_kind": "builtin" if plan.kind == "builtin" else "github_url",
        "source": (
            plan.plugin_id
            if plan.kind == "builtin"
            else f"{plan.owner}/{plan.repo}" + (f"@{plan.ref}" if plan.ref else "")
        ),
    }
    if plan.kind == "github":
        record["url"] = f"https://github.com/{plan.owner}/{plan.repo}"
        record["ref"] = plan.ref
        record["sha"] = plan.sha
    record["installed_at"] = plugin_loader.now_iso()
    record["manifest"] = plugin_meta if isinstance(plugin_meta, dict) else {}
    record["trusted_modules"] = list(allowed_modules)
    record["capabilities"] = list(plan.granted_capabilities)
    record["enabled"] = True
    if plan.catalog_id is not None:
        # Only when there really was a catalog row. Writing the key
        # unconditionally would have every free-text install claim a catalog
        # identity it does not have, and "installed from the catalog" is
        # exactly the claim a reader wants to trust.
        record["catalog_id"] = plan.catalog_id
    return record


def _write_lockfile_entry(plan: InstallPlan, record: dict[str, Any]) -> bool:
    """Record the install; returns whether a tombstone went with it.

    The lockfile is read here rather than carried in the plan, as late as the
    write allows: it is one file that the CLI and the server both edit, and
    every second between the read and the write is a second in which the
    other one's edit can be lost. ``save_lockfile`` is atomic, so the file
    is never half-written -- only, at worst, one edit behind.

    Installing is the undo for having uninstalled, so the tombstone goes with
    it (#175): otherwise ``cdui plugin sync`` would keep skipping a pack that
    is now installed. Only built-in packs are ever tombstoned, so this is a
    no-op for a repository plugin -- asked unconditionally because "which
    kinds get tombstoned" is the uninstall's rule to change, not this one's.
    """
    lockfile = plugin_loader.load_lockfile()
    cleared = plugin_loader.clear_removed(lockfile, plan.plugin_id)
    lockfile.setdefault("plugins", {})[plan.plugin_id] = record
    plugin_loader.save_lockfile(lockfile)
    return cleared
