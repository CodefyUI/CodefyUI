"""Turning a plugin off, on, or out -- the three writes, without a caller.

Enable, disable and uninstall are three lockfile edits with rules attached:
a disable of something already disabled must not rewrite the file, an
uninstall must never delete a directory it did not download, and a built-in
pack has to leave a tombstone behind so ``cdui plugin sync`` does not put
back what the user just threw away (#175). Those rules lived twice -- once in
``scripts/plugins.py``, printing bilingually as it went, and once in
``routes_plugins.py``, raising ``HTTPException`` as it went -- and the second
copy was written by reading the first. A rule that exists twice is a rule
that will disagree with itself, and here the disagreement would be about
which files get deleted.

So the writes happen here and produce a VALUE. What the caller does with it
is the caller's business: the CLI prints its zh/en pair, the route returns
JSON, and neither has to know what the other says. Nothing in this module
prints, raises for control flow, or reloads the registry -- the reload is
the caller's next line (``rediscover_now()`` in the server, an HTTP POST to
a possibly-absent server in the CLI), and doing it here would make an
uninstall in the CLI process pointlessly re-import every pack that is left.

:class:`UninstallOutcome` is deliberately fuller than what either caller
needs today: an uninstall leaves the plugin's Python dependencies behind
(uninstalling packages from inside the process that imported them is how you
get a half-loaded interpreter serving requests -- see ``routes_packs.py``),
and the only honest thing to do about that is to say so and hand over the
command -- for the ones nothing else still needs, since a line the user is
told to run must not remove what CodefyUI or another plugin depends on
(#414). The value carries the answer so that neither caller has to re-derive
it from a manifest that is, by then, sometimes deleted.
"""

from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core import plugin_loader

from .catalog import builtin_catalog_packs, load_catalog
from .deps import (
    _shell_quote,
    is_safe_dep_name,
    manual_uninstall_command,
    orphaned_deps,
)
from .errors import SourceError
from .lockfile_lock import locked_lockfile
from .manifest import PLUGIN_ID_RE, manifest_python_deps
from .sources import parse_github_url, parse_source

logger = logging.getLogger(__name__)


def set_enabled(
    plugin_id: str,
    enabled: bool,
    *,
    lock_timeout: float | None = None,
) -> bool | None:
    """Flip a plugin's ``enabled`` flag. Returns what happened.

    ``None`` means there is no such plugin in the lockfile, ``False`` that it
    was already in that state and nothing was written, ``True`` that the flag
    was flipped and saved. Three answers rather than two because the callers
    say three different things: the CLI prints "already disabled (no-op)", the
    route 404s, and only the last one is a change worth reloading for.

    The file is deliberately NOT rewritten for a no-op. A lockfile rewrite is
    what a backup tool, a file watcher and a project diff all see, and "the
    user pressed disable twice" is not a change any of them should be told
    about. That is also why the lock is held around a read that may write
    nothing: whether there is anything to write is decided from the document
    this call read, and deciding it outside the lock would answer from one
    document and write over another (#412).

    *lock_timeout* is how long to wait for the writer's lock before giving
    up; ``None`` is the lock module's own default. The CLI can afford that; a
    request handler passes something short, because this is called on the
    event loop.

    :raises LockfileBusy: another writer holds the lockfile. Nothing was
        read and nothing was written.
    """
    with locked_lockfile(timeout=lock_timeout) as lockfile:
        entry = lockfile.get("plugins", {}).get(plugin_id)
        if not entry:
            return None
        if plugin_loader.is_enabled(entry) == enabled:
            return False
        entry["enabled"] = enabled
        lockfile.save()
    return True


@dataclass(frozen=True)
class UninstallOutcome:
    """What an uninstall did, and what it deliberately left behind."""

    plugin_id: str
    #: The lockfile entry is gone, i.e. the plugin is uninstalled. ``False``
    #: means the uninstall was ABANDONED and the plugin is still installed --
    #: see ``error`` for why. There is no half state: nothing is popped
    #: unless the files this install downloaded are actually gone.
    removed: bool
    #: A ``removed`` record was written, so ``cdui plugin sync`` will not
    #: re-add this pack. Only built-in packs get one (#175).
    tombstoned: bool
    #: ``True`` the downloaded directory is gone, ``False`` it could not be
    #: deleted (and then ``removed`` is ``False`` too), ``None`` there was
    #: never a copy of ours to delete (a built-in pack is repo code; a linked
    #: one is the author's own working tree).
    files_removed: bool | None
    #: ``[python_deps]`` names this plugin asked for that nothing else still
    #: needs: they are not CodefyUI itself, no other installed distribution
    #: requires them, no other installed plugin declares them, and no
    #: Package Center pack installs them (``deps.orphaned_deps``). The rest
    #: are left out because the command below would take them from whatever
    #: needs them (#414). Empty, too, when that could not be worked out: it
    #: is advice, and never the reason an uninstall fails. See the module
    #: docstring for why nothing uninstalls these either.
    python_deps_left: tuple[str, ...]
    #: The command that WOULD remove those packages, to run by hand with the
    #: server stopped, quoted as the install command is; ``None`` when there
    #: are none.
    uninstall_command: str | None
    #: The command that puts this plugin back, built by
    #: :func:`reinstall_command` from what the lockfile recorded about where
    #: it came from; ``None`` when no command can (a linked directory that is
    #: gone, a built-in pack this build's catalog no longer lists). Shown
    #: whenever it is not ``None``, by the CLI and the panel alike, whether
    #: or not any Python packages were left behind (#506).
    reinstall_hint: str | None
    #: The directory this uninstall deleted, or tried to; ``None`` when there
    #: was never a copy of ours to delete. Reported rather than left for the
    #: caller to rebuild from the id and the user root: that rebuild is this
    #: module's own rule about where a pack's files live, and a caller
    #: repeating it would name the wrong path the day the rule changes --
    #: while what a failed uninstall has to tell the user is exactly WHICH
    #: directory is still there.
    directory: Path | None = None
    #: Why the uninstall was abandoned, in one line, or ``None`` when it was
    #: not. Text rather than the exception: the caller says it in its own
    #: language and its own envelope, and neither wants a traceback.
    error: str | None = None


def uninstall_plugin(
    plugin_id: str,
    *,
    builtin_ids: Collection[str] | None = None,
    catalog: dict[str, Any] | None = None,
    lock_timeout: float | None = None,
) -> UninstallOutcome | None:
    """Remove a plugin from this install. ``None`` when it was not installed.

    The order is load-bearing: the manifest is read while the files are still
    there, the files go next, and only then is the lockfile written. That is
    the reason the delete comes first -- if it fails, the entry stays, the
    plugin stays installed, and the caller is told why (``removed=False``
    with an ``error``). The alternative, popping the entry anyway, would
    leave a directory no lockfile mentions, which nothing in this system
    would ever look at again, let alone clean up.

    Nothing here touches ``sys.modules``. The CLI runs in its own process and
    never imported the plugin, so there would be nothing to drop; and the
    in-process caller -- the DELETE route -- purges the plugin's modules
    itself, immediately before ``rediscover_now()``, which is where the
    namespace finder is re-created for everything still installed. Purging
    from in here would put the two half a call apart with no reload between
    them, and a namespace dropped without being rebuilt is a plugin nothing
    can import until the next full re-discovery.

    *builtin_ids* overrides which ids count as built-in packs for the
    tombstone rule. It exists for ``scripts/plugins.py``, whose tests fake
    the catalog by patching the CLI's own root: without it this would read
    past the patch and answer from the real ``registry.json``. Same reason
    :func:`~app.core.plugins.catalog.catalog_path` takes a root. *catalog*
    overrides the parsed ``registry.json`` the reinstall command is checked
    against, for the same reason.

    The writer's lock is held across the WHOLE of that order, ``rmtree`` and
    all -- which is the widest read-to-write gap in the plugin system and the
    one #412 was filed about. An install finishing anywhere inside the delete
    used to be erased by the save at the end of it, silently, leaving a pack
    on disk that no lockfile mentions. The delete cannot move after the write
    (see above), so the lock is what makes the two ends agree; *lock_timeout*
    is how long to wait for it, short from a request handler.

    :raises LockfileBusy: another writer holds the lockfile. Nothing was
        deleted and nothing was written.
    """
    with locked_lockfile(timeout=lock_timeout) as lockfile:
        entry = lockfile.get("plugins", {}).get(plugin_id)
        if not entry:
            return None

        deps = _python_deps_left_behind(plugin_id, lockfile)
        known_builtins = (
            builtin_catalog_packs(catalog) if builtin_ids is None else builtin_ids
        )
        # Read off the entry while it still exists, which is the only time the
        # lockfile knows where this plugin came from (#506).
        reinstall = reinstall_command(
            plugin_id, entry,
            catalog=load_catalog() if catalog is None else catalog,
            builtin_ids=known_builtins,
        )

        files_removed: bool | None = None
        directory: Path | None = None
        if entry.get("source_kind") == "github_url":
            files_removed, failure, directory = _remove_downloaded_files(plugin_id)
            if not files_removed:
                return _outcome(
                    plugin_id, removed=False, tombstoned=False,
                    files_removed=False, deps=deps, error=failure,
                    directory=directory, reinstall=reinstall,
                )

        lockfile["plugins"].pop(plugin_id, None)

        # Remember the decision instead of merely forgetting the pack (#175).
        # Popping the entry made "never installed" and "removed on purpose" the
        # same state, so `cdui plugin sync` would have to either re-install what
        # the user just threw away or nag about it forever. Only built-in packs
        # are tombstoned: they are the only ones sync can put back uninvited, and
        # a tombstone nothing reads is dead data the user would still have to
        # explain.
        tombstoned = (
            entry.get("source_kind") == "builtin" or plugin_id in known_builtins
        )
        if tombstoned:
            plugin_loader.mark_removed(
                lockfile, plugin_id, source_kind=entry.get("source_kind")
            )
        lockfile.save()

    return _outcome(
        plugin_id, removed=True, tombstoned=tombstoned,
        files_removed=files_removed, deps=deps, error=None,
        directory=directory, reinstall=reinstall,
    )


def _outcome(
    plugin_id: str,
    *,
    removed: bool,
    tombstoned: bool,
    files_removed: bool | None,
    deps: tuple[str, ...],
    error: str | None,
    reinstall: str | None,
    directory: Path | None = None,
) -> UninstallOutcome:
    """One :class:`UninstallOutcome`, so the two exits agree on the fields
    that describe the PLUGIN rather than the attempt.

    The dependency facts are reported either way: they are true of the plugin
    whether or not this call removed it, and a caller showing "and these
    packages stay installed" must not have to ask which exit it came from.
    """
    return UninstallOutcome(
        plugin_id=plugin_id,
        removed=removed,
        tombstoned=tombstoned,
        files_removed=files_removed,
        python_deps_left=deps,
        uninstall_command=manual_uninstall_command(deps) if deps else None,
        reinstall_hint=reinstall,
        directory=directory,
        error=error,
    )


def reinstall_command(
    plugin_id: str,
    entry: dict[str, Any],
    *,
    catalog: dict[str, Any],
    builtin_ids: Collection[str],
) -> str | None:
    """The command that installs *entry* again, or ``None`` when none can.

    ``cdui plugin install <id>`` only works for an id the catalog lists, so a
    plugin id is never offered on its own say-so (#506). In order:

    * A built-in pack, or a plugin installed from a catalog row, gets its
      catalog name -- provided the catalog still lists it and the name still
      resolves there. That is the command that brings a tombstoned pack back.
    * A plugin installed from a GitHub repository gets that repository at the
      ref it was installed from (``owner/repo@ref``; no ``@`` for the default
      branch), read from the recorded ``url`` and checked by
      :func:`~.sources.parse_source` to resolve to that same repository.
    * A linked directory gets ``cdui plugin link <path>`` while the directory
      and its manifest are still there; uninstalling never deletes it.

    Anything else -- a source this cannot vouch for -- gets ``None``, and the
    caller says nothing rather than print a command that would fail or
    install something else.
    """
    plugins = catalog.get("plugins", {})
    kind = entry.get("source_kind")

    catalog_name = entry.get("catalog_id")
    if kind == "builtin" or plugin_id in builtin_ids:
        catalog_name = plugin_id
    if (
        isinstance(catalog_name, str)
        and PLUGIN_ID_RE.fullmatch(catalog_name)
        and isinstance(plugins.get(catalog_name), dict)
        and _resolves_as(catalog_name, catalog, ("catalog", catalog_name, "", ""))
    ):
        return f"cdui plugin install {catalog_name}"

    if kind == "github_url":
        url = entry.get("url")
        ref = entry.get("ref") or ""
        repo = parse_github_url(url) if isinstance(url, str) else None
        if repo is None or not isinstance(ref, str):
            return None
        owner, name = repo
        spec = f"{owner}/{name}" + (f"@{ref}" if ref else "")
        if not _resolves_as(spec, catalog, ("github", owner, name, ref)):
            return None
        # Every character a valid spec is built from is safe bare in every
        # shell bar a non-ASCII letter, which gets the shared quoting rule.
        bare = _BARE_SPEC.fullmatch(spec)
        return f"cdui plugin install {spec if bare else _shell_quote(spec)}"

    if kind == "local":
        path = entry.get("path")
        if (
            isinstance(path, str)
            and Path(path).is_absolute()
            and not _UNSAFE_PATH_CHARS.intersection(path)
            and (Path(path) / plugin_loader.MANIFEST_FILENAME).is_file()
        ):
            return f"cdui plugin link {_shell_quote(path)}"
    return None


#: An ``owner/repo@ref`` that needs no quoting anywhere.
_BARE_SPEC = re.compile(r"[A-Za-z0-9._/@-]+")

#: Characters a double-quoted argument does not neutralise in every shell a
#: user may paste into: a linked path holding one gets no command at all.
_UNSAFE_PATH_CHARS = frozenset('"`$!%\n\r')


def _resolves_as(
    spec: str, catalog: dict[str, Any], expected: tuple[str, str, str, str],
) -> bool:
    """Does ``cdui plugin install <spec>`` resolve to *expected*?"""
    try:
        return tuple(parse_source(spec, catalog=catalog)) == expected
    except SourceError:
        return False


def _python_deps_left_behind(
    plugin_id: str, lockfile: dict[str, Any],
) -> tuple[str, ...]:
    """The declared packages this uninstall leaves that nothing else needs.

    Read where the manifest is, and for the same reason: before the files
    go. The one fact only the lockfile holds -- which OTHER plugins are
    installed -- is gathered here; the rest of "does anything still need
    it" is :func:`~.deps.orphaned_deps`'s. A plugin that declared nothing
    costs nothing more: the distribution walk behind that call runs inside
    this uninstall's lock, and only when there is a name to protect.

    Never raises. This is advice about what the uninstall leaves, worked
    out inside the uninstall's lock, and an exception here would abandon an
    uninstall that has nothing wrong with it -- which is what a malformed
    ``CODEFYUI_*`` variable did, through an ``app.config`` import on this
    path (#414). So anything unexpected costs the advice and nothing else:
    no package is named, and the reason is logged -- one line, no
    traceback, because in the CLI that line lands on the user's terminal.
    ``_reload_target`` in ``scripts/plugins.py`` strikes the same bargain
    around its own ``app.config`` import.
    """
    try:
        declared = _declared_python_deps(plugin_id, lockfile)
        if not declared:
            return ()
        return tuple(sorted(orphaned_deps(
            declared,
            declared_elsewhere=_python_deps_of_other_plugins(
                plugin_id, lockfile),
        )))
    except Exception as exc:
        # ``%r``, not ``%s``: the ``str`` of a KeyError is the bare key, and a
        # line that ends in a quoted word names no failure at all.
        logger.warning(
            "plugin uninstall: could not tell which of %s's Python packages "
            "nothing else needs, so none are named: %r",
            plugin_id,
            exc,
        )
        return ()


def _declared_python_deps(plugin_id: str, lockfile: dict[str, Any]) -> list[str]:
    """The ``[python_deps]`` names in the installed plugin's manifest.

    Read off the disk rather than out of the lockfile because the lockfile
    records the ``[plugin]`` table only -- and read BEFORE the files go,
    which is the whole reason this is a separate step.

    Vetted by :func:`~.deps.is_safe_dep_name`, the rule an INSTALL applies to
    the same table. A manifest is untrusted text and these names end up in
    ``uninstall_command``, which is a line the user is invited to paste into
    a shell: ``evil @ git+https://attacker.example/evil`` as a key would
    otherwise travel from a manifest, through a REST response, into somebody's
    terminal. A name that could never have been installed cannot have left a
    package behind either, so nothing true is lost by dropping it.
    """
    plugin_dir = installed_dir(plugin_id, lockfile)
    if plugin_dir is None:
        return []
    manifest = plugin_loader.read_manifest_safe(plugin_dir)
    return [name for name in manifest_python_deps(manifest)
            if is_safe_dep_name(name)]


def _python_deps_of_other_plugins(
    plugin_id: str, lockfile: dict[str, Any],
) -> list[str]:
    """Every ``[python_deps]`` name another installed plugin declares.

    Disabled plugins count: switching one off keeps its files, and switching
    it back on has to find its packages where it left them. Not vetted --
    these names are only compared, never printed -- and a plugin whose
    manifest cannot be read declares nothing this can see. Walked through
    ``iter_plugin_dirs``, the rule discovery itself uses, like
    :func:`installed_dir`.
    """
    names: list[str] = []
    for other_id, plugin_dir in plugin_loader.iter_plugin_dirs(
        plugin_loader.plugins_builtin_root(),
        plugin_loader.plugins_user_root(),
        lockfile,
        include_disabled=True,
    ):
        if other_id != plugin_id:
            names.extend(manifest_python_deps(
                plugin_loader.read_manifest_safe(plugin_dir)))
    return names


def installed_dir(plugin_id: str, lockfile: dict[str, Any]) -> Path | None:
    """Where an installed plugin's files are, or ``None`` when they are gone.

    Goes through ``iter_plugin_dirs`` rather than rebuilding its
    source-kind-to-directory rule (built-in root, user root, or the absolute
    path a linked plugin recorded), which is the rule discovery itself uses.
    A plugin whose directory or manifest has disappeared is not yielded, so
    ``None`` here is exactly the Plugin Center's ``missing_files``.
    """
    for pid, plugin_dir in plugin_loader.iter_plugin_dirs(
        plugin_loader.plugins_builtin_root(),
        plugin_loader.plugins_user_root(),
        lockfile,
        include_disabled=True,
    ):
        if pid == plugin_id:
            return plugin_dir
    return None


def _remove_downloaded_files(plugin_id: str) -> tuple[bool, str | None, Path]:
    """Delete ``<user root>/<plugin_id>/``. Answers ``(gone, why not, where)``.

    The directory is returned rather than left to be rebuilt by a caller,
    because building it is this function's own rule (see below) and a caller
    repeating it is a second copy that can name a different path.

    The containment check is the point: this is the one place in the plugin
    system that runs ``rmtree`` on a path built from an id, so the resolved
    directory has to sit DIRECTLY in the resolved user root or nothing is
    deleted. Resolving both sides is what makes a symlinked pack directory --
    or a symlinked user root, which is what a temp directory is on macOS --
    answer the question about the real target.

    A failure is reported rather than raised, and the caller stops on it: a
    pack whose files are still there is a pack that will load again on the
    next start, so calling it uninstalled would be a lie the lockfile then
    tells forever. An ABSENT directory is success -- there is nothing of ours
    left, which is all "removed" ever meant.

    Only the containment refusal is also LOGGED. Every failure here is
    returned, and the caller says it -- logging the ordinary one (a file
    Windows is holding open) would print the same sentence twice in the same
    terminal. A path that resolves outside the user root is not an ordinary
    failure and not a user mistake, so the record of it should exist
    somewhere other than one line the user may have already scrolled past.
    """
    user_root = plugin_loader.plugins_user_root()
    target = user_root / plugin_id
    try:
        resolved = target.resolve()
        root = user_root.resolve()
    except OSError as exc:  # pragma: no cover - resolve() rarely fails
        return False, str(exc), target
    if resolved.parent != root:
        logger.warning(
            "plugin uninstall: refusing to delete %s -- it is not directly "
            "inside %s",
            resolved,
            root,
        )
        return False, f"{resolved} is not directly inside {root}", target
    if not resolved.exists():
        return True, None, target
    try:
        shutil.rmtree(resolved)
    except OSError as exc:
        return False, str(exc), target
    return True, None, target
