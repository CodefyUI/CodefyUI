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
    #: The lockfile entry is gone. Always true in a returned outcome -- the
    #: field exists so a caller can say it rather than assume it.
    removed: bool
    #: A ``removed`` record was written, so ``cdui plugin sync`` will not
    #: re-add this pack. Only built-in packs get one (#175).
    tombstoned: bool
    #: ``True`` the downloaded directory is gone, ``False`` it could not be
    #: deleted and is still there, ``None`` there was never a copy of ours to
    #: delete (a built-in pack is repo code; a linked one is the author's own
    #: working tree).
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


def uninstall_plugin(
    plugin_id: str,
    *,
    builtin_ids: Collection[str] | None = None,
) -> UninstallOutcome | None:
    """Remove a plugin from this install. ``None`` when it was not installed.

    The order is load-bearing. The manifest is read while the files are still
    there, the modules are dropped from ``sys.modules`` before anything moves
    (a re-install of the same id would otherwise reload the OLD path a cached
    module remembers), the files go, and only then is the lockfile written --
    so an interrupted uninstall leaves an entry pointing at a directory that
    is gone, which every reader already handles, rather than a directory no
    entry mentions, which nothing would ever clean up.

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
    plugin_loader.purge_plugin_modules(plugin_id)

    files_removed: bool | None = None
    if entry.get("source_kind") == "github_url":
        files_removed = _remove_downloaded_files(plugin_id)

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

    return UninstallOutcome(
        plugin_id=plugin_id,
        removed=True,
        tombstoned=tombstoned,
        files_removed=files_removed,
        python_deps_left=deps,
        uninstall_command=(
            f"uv pip uninstall --python {sys.executable} {' '.join(deps)}"
            if deps
            else None
        ),
        reinstall_hint=f"cdui plugin install {plugin_id}",
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


def _remove_downloaded_files(plugin_id: str) -> bool:
    """Delete ``<user root>/<plugin_id>/``; report whether it is gone.

    The containment check is the point: this is the one place in the plugin
    system that runs ``rmtree`` on a path built from an id, so the resolved
    directory has to sit DIRECTLY in the resolved user root or nothing is
    deleted. Resolving both sides is what makes a symlinked pack directory --
    or a symlinked user root, which is what a temp directory is on macOS --
    answer the question about the real target.

    A failure is reported, not raised: the entry is removed either way, so a
    pack whose files Windows is holding open stops claiming to be installed
    instead of becoming impossible to uninstall.
    """
    user_root = plugin_loader.plugins_user_root()
    target = user_root / plugin_id
    try:
        resolved = target.resolve()
        root = user_root.resolve()
    except OSError as exc:  # pragma: no cover - resolve() rarely fails
        logger.warning("plugin uninstall: cannot resolve %s: %s", target, exc)
        return False
    if resolved.parent != root:
        logger.warning(
            "plugin uninstall: refusing to delete %s -- it is not directly "
            "inside %s",
            resolved,
            root,
        )
        return False
    if not resolved.exists():
        return True
    try:
        shutil.rmtree(resolved)
    except OSError as exc:
        logger.warning("plugin uninstall: failed to delete %s: %s", resolved, exc)
        return False
    return True
