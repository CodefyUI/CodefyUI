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
command. The value carries the answer so that neither caller has to re-derive
it from a manifest that is, by then, sometimes deleted.
"""

from __future__ import annotations

import logging
import shutil
import sys
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core import plugin_loader

from .catalog import builtin_catalog_packs
from .manifest import manifest_python_deps

logger = logging.getLogger(__name__)


def set_enabled(plugin_id: str, enabled: bool) -> bool | None:
    """Flip a plugin's ``enabled`` flag. Returns what happened.

    ``None`` means there is no such plugin in the lockfile, ``False`` that it
    was already in that state and nothing was written, ``True`` that the flag
    was flipped and saved. Three answers rather than two because the callers
    say three different things: the CLI prints "already disabled (no-op)", the
    route 404s, and only the last one is a change worth reloading for.

    The file is deliberately NOT rewritten for a no-op. A lockfile rewrite is
    what a backup tool, a file watcher and a project diff all see, and "the
    user pressed disable twice" is not a change any of them should be told
    about.
    """
    lockfile = plugin_loader.load_lockfile()
    entry = lockfile.get("plugins", {}).get(plugin_id)
    if not entry:
        return None
    if plugin_loader.is_enabled(entry) == enabled:
        return False
    entry["enabled"] = enabled
    plugin_loader.save_lockfile(lockfile)
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
    #: ``[python_deps]`` names this plugin asked for, which are still
    #: installed in the interpreter. See the module docstring for why nothing
    #: uninstalls them.
    python_deps_left: tuple[str, ...]
    #: The command that WOULD remove those packages, to run by hand with the
    #: server stopped; ``None`` when the plugin declared none.
    uninstall_command: str | None
    #: How to get this plugin back.
    reinstall_hint: str
    #: Why the uninstall was abandoned, in one line, or ``None`` when it was
    #: not. Text rather than the exception: the caller says it in its own
    #: language and its own envelope, and neither wants a traceback.
    error: str | None = None


def uninstall_plugin(
    plugin_id: str,
    *,
    builtin_ids: Collection[str] | None = None,
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
    :func:`~app.core.plugins.catalog.catalog_path` takes a root.
    """
    lockfile = plugin_loader.load_lockfile()
    entry = lockfile.get("plugins", {}).get(plugin_id)
    if not entry:
        return None

    deps = tuple(sorted(_declared_python_deps(plugin_id, lockfile)))

    files_removed: bool | None = None
    if entry.get("source_kind") == "github_url":
        files_removed, failure = _remove_downloaded_files(plugin_id)
        if not files_removed:
            return _outcome(
                plugin_id, removed=False, tombstoned=False,
                files_removed=False, deps=deps, error=failure,
            )

    lockfile["plugins"].pop(plugin_id, None)

    # Remember the decision instead of merely forgetting the pack (#175).
    # Popping the entry made "never installed" and "removed on purpose" the
    # same state, so `cdui plugin sync` would have to either re-install what
    # the user just threw away or nag about it forever. Only built-in packs
    # are tombstoned: they are the only ones sync can put back uninvited, and
    # a tombstone nothing reads is dead data the user would still have to
    # explain.
    known_builtins = builtin_catalog_packs() if builtin_ids is None else builtin_ids
    tombstoned = (
        entry.get("source_kind") == "builtin" or plugin_id in known_builtins
    )
    if tombstoned:
        plugin_loader.mark_removed(
            lockfile, plugin_id, source_kind=entry.get("source_kind")
        )
    plugin_loader.save_lockfile(lockfile)

    return _outcome(
        plugin_id, removed=True, tombstoned=tombstoned,
        files_removed=files_removed, deps=deps, error=None,
    )


def _outcome(
    plugin_id: str,
    *,
    removed: bool,
    tombstoned: bool,
    files_removed: bool | None,
    deps: tuple[str, ...],
    error: str | None,
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
        uninstall_command=(
            f"uv pip uninstall --python {sys.executable} {' '.join(deps)}"
            if deps
            else None
        ),
        reinstall_hint=f"cdui plugin install {plugin_id}",
        error=error,
    )


def _declared_python_deps(plugin_id: str, lockfile: dict[str, Any]) -> list[str]:
    """The ``[python_deps]`` names in the installed plugin's manifest.

    Read off the disk rather than out of the lockfile because the lockfile
    records the ``[plugin]`` table only -- and read BEFORE the files go,
    which is the whole reason this is a separate step.
    """
    plugin_dir = installed_dir(plugin_id, lockfile)
    if plugin_dir is None:
        return []
    manifest = plugin_loader.read_manifest_safe(plugin_dir)
    return [name for name in manifest_python_deps(manifest) if isinstance(name, str)]


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


def _remove_downloaded_files(plugin_id: str) -> tuple[bool, str | None]:
    """Delete ``<user root>/<plugin_id>/``. Answers ``(gone, why not)``.

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
        return False, str(exc)
    if resolved.parent != root:
        logger.warning(
            "plugin uninstall: refusing to delete %s -- it is not directly "
            "inside %s",
            resolved,
            root,
        )
        return False, f"{resolved} is not directly inside {root}"
    if not resolved.exists():
        return True, None
    try:
        shutil.rmtree(resolved)
    except OSError as exc:
        return False, str(exc)
    return True, None
