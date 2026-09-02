"""``plugins/registry.json``: the packs this build says it ships.

An index, not a package manager. A ``builtin`` entry names a directory that
came with the release and is activated in place; a ``github`` entry names a
repository the installer may fetch. Nothing else can be installed by NAME, so
this file is the whole of what ``cdui plugin install <word>`` can reach.

Two readers with two different needs, which is why there are two shapes:

* :func:`load_catalog` hands back the parsed JSON exactly as it is on disk.
  Every caller that predates this module reads it that way -- and a test that
  fakes a catalog fakes a plain dict -- so the raw shape stays supported and
  keeps its old habit of answering an empty catalog rather than raising when
  the file is missing or corrupt. A CLI that cannot install anything is bad;
  a CLI that traces back on ``cdui plugin list`` is worse.
* :func:`validate_catalog` turns the same dict into :class:`CatalogEntry`
  records, DROPPING anything malformed with a log line. That is what a GUI
  needs: a card per pack, with fields it can render without asking whether
  ``tags`` is a list this time. Dropping rather than raising for the same
  reason as above -- one typo in one entry must not empty the Plugin Center.

Reading the file is the only thing here that touches the disk, and nothing
touches the network.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core import plugin_loader

from .manifest import PLUGIN_ID_RE

logger = logging.getLogger(__name__)

#: Ids a plugin may never claim: each one is a fixed path segment under
#: ``/api/plugins/``, so a pack called ``install`` would sit where the install
#: route already lives and the router -- not the plugin -- would decide which
#: one wins. Reserved by NAME here rather than by trying to detect the clash
#: later, because the clash is only visible from the routing table.
RESERVED_PLUGIN_IDS = frozenset({
    "catalog", "inspect", "install", "jobs", "generation", "reload",
})

#: ``owner/repo``, in the characters GitHub allows in either half.
_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")

#: The two things a catalog entry can be. Anything else is a newer catalog
#: read by an older build, and an entry this build cannot install is worse
#: than no entry: it would render as a card whose button does nothing.
_KINDS = frozenset({"builtin", "github"})


def catalog_path(builtin_root: Path | None = None) -> Path:
    """Where ``registry.json`` is read from.

    *builtin_root* overrides the install's own plugin directory. It exists
    for one caller: ``scripts/plugins.py`` re-exports this under its old name
    and its tests redirect the built-in root by patching the CLI's copy of
    ``plugins_builtin_root``, so the CLI passes its own answer in rather than
    letting this module reach past the patch to the real repository.
    """
    root = plugin_loader.plugins_builtin_root() if builtin_root is None else builtin_root
    return root / "registry.json"


def load_catalog(builtin_root: Path | None = None) -> dict[str, Any]:
    """The parsed ``registry.json``, or an empty catalog.

    Every failure -- no file, unreadable file, invalid JSON -- is the same
    answer, deliberately: this is read on the way to ``plugin list``,
    ``search`` and every ``install``, and a catalog nobody can read means
    "there are no packs to install by name", which those commands can say.
    Raising here would take down commands that have nothing to do with the
    catalog.
    """
    p = catalog_path(builtin_root)
    if not p.exists():
        return {"schema": 1, "plugins": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"schema": 1, "plugins": {}}


def builtin_catalog_packs(
    catalog: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """The ``kind = "builtin"`` half of the catalog -- packs that ship in-repo.

    *catalog* overrides the parsed registry, for the same reason
    :func:`catalog_path` takes a root: the CLI's tests patch the CLI's
    ``load_catalog``, and this must answer from the catalog the caller has,
    not from the one on disk.
    """
    data = load_catalog() if catalog is None else catalog
    return {
        pack_id: entry
        for pack_id, entry in data.get("plugins", {}).items()
        if isinstance(entry, dict) and entry.get("kind") == "builtin"
    }


def available_builtin_packs(
    *,
    read_catalog: Callable[[], dict[str, Any]] | None = None,
    read_lockfile: Callable[[], dict[str, Any]] | None = None,
) -> list[tuple[str, str]]:
    """Built-in packs shipped on disk that this install has made no decision about.

    A release can add a pack (``stats`` did), and its files land on disk with
    the update -- but the server only loads what the lockfile records, and
    nothing re-syncs it. So the pack is fully installable and completely
    invisible: the nodes never appear, and no message anywhere says why.

    "No decision" is the operative phrase (#175): a pack the user uninstalled
    is subtracted too. Before uninstall left a tombstone, the entry was simply
    popped, so a removed pack was indistinguishable from one this install had
    never seen -- which is why the notices used to nag about a pack the user
    had already thrown away, once per start, forever.

    Returns ``(id, display name)`` pairs, sorted, so callers can name them.

    The two readers are injectable because this is the one function the CLI
    re-exports whose inputs its tests fake by patching ``plugins.load_catalog``
    / ``plugins.load_lockfile``. They are resolved INSIDE the guard below, so
    a reader that raises is swallowed like every other failure here.
    """
    try:
        catalog_reader = load_catalog if read_catalog is None else read_catalog
        lockfile_reader = (
            plugin_loader.load_lockfile if read_lockfile is None else read_lockfile
        )
        catalog = builtin_catalog_packs(catalog_reader())
        lockfile = lockfile_reader()
        installed = lockfile.get("plugins", {})
        tombstoned = plugin_loader.removed_ids(lockfile)
    except Exception:  # never let discoverability break a caller
        return []
    out: list[tuple[str, str]] = []
    for pack_id, entry in catalog.items():
        if pack_id in installed or pack_id in tombstoned:
            continue
        out.append((pack_id, str(entry.get("name") or pack_id)))
    return sorted(out)


@dataclass(frozen=True)
class CatalogEntry:
    """One catalog row, in the shape something can render or install from.

    Flat and fully populated on purpose: the caller never has to ask whether
    a key is present, what type it is this time, or which keys go with which
    ``kind``. Absent optional values are the empty string or an empty tuple
    rather than ``None`` so a template can interpolate them directly; ``path``
    and ``repo`` are the exception, because ``None`` there means "this kind of
    entry does not have one" and an empty string would read as a real path.
    """

    id: str
    name: str
    description: str
    kind: str                        # "builtin" | "github"
    path: str | None = None          # builtin: repo-relative pack directory
    repo: str | None = None          # github: "owner/repo"
    ref: str = ""                    # github: tag/branch; "" = default branch
    homepage: str = ""
    tags: tuple[str, ...] = ()
    chapters: tuple[str, ...] = ()
    official: bool = False


def _string_tuple(value: Any) -> tuple[str, ...]:
    """The string members of *value* when it is a list, else empty."""
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _text(value: Any) -> str:
    """*value* when it is a string, else the empty string."""
    return value if isinstance(value, str) else ""


def validate_catalog(data: Any) -> dict[str, CatalogEntry]:
    """The well-formed entries of a parsed ``registry.json``, keyed by id.

    Drops rather than raises, one ``logger.warning`` per casualty. The
    catalog is data this build ships, so a malformed entry is a bug in a PR
    that a log line will get fixed -- while raising would mean one bad row
    empties the Plugin Center, hides the four good packs beside it, and takes
    ``cdui plugin list`` down with it.

    What gets dropped is what something downstream would otherwise trip over:
    an id that is not a legal plugin id (it becomes a directory name and a
    URL segment), a ``kind`` this build cannot install, a ``builtin`` with no
    directory to activate, and a ``github`` entry whose ``repo`` is not
    ``owner/repo``. A ``github`` entry WITH a ``path`` is dropped too -- the
    two kinds are installed by completely different code, and an entry
    claiming both is a question this module must not answer by guessing.
    """
    plugins = data.get("plugins") if isinstance(data, dict) else None
    if not isinstance(plugins, dict):
        logger.warning(
            "plugin catalog has no readable 'plugins' table; treating it as empty"
        )
        return {}

    entries: dict[str, CatalogEntry] = {}
    for plugin_id, entry in plugins.items():
        if not isinstance(plugin_id, str) or not PLUGIN_ID_RE.match(plugin_id):
            logger.warning(
                "plugin catalog: dropping %r -- not a valid plugin id", plugin_id
            )
            continue
        if not isinstance(entry, dict):
            logger.warning(
                "plugin catalog: dropping %r -- entry is not a table", plugin_id
            )
            continue
        kind = entry.get("kind")
        if kind not in _KINDS:
            logger.warning(
                "plugin catalog: dropping %r -- unknown kind %r", plugin_id, kind
            )
            continue
        path = entry.get("path")
        repo = entry.get("repo")
        if kind == "builtin":
            if not isinstance(path, str) or not path:
                logger.warning(
                    "plugin catalog: dropping %r -- a builtin entry needs a path",
                    plugin_id,
                )
                continue
            repo = None
        else:
            if not isinstance(repo, str) or not _REPO_RE.match(repo):
                logger.warning(
                    "plugin catalog: dropping %r -- a github entry needs "
                    "repo = 'owner/repo', got %r",
                    plugin_id,
                    repo,
                )
                continue
            if path is not None:
                logger.warning(
                    "plugin catalog: dropping %r -- a github entry cannot also "
                    "declare a path",
                    plugin_id,
                )
                continue
        entries[plugin_id] = CatalogEntry(
            id=plugin_id,
            name=_text(entry.get("name")) or plugin_id,
            description=_text(entry.get("description")),
            kind=kind,
            path=path if kind == "builtin" else None,
            repo=repo,
            ref=_text(entry.get("ref")),
            homepage=_text(entry.get("homepage")),
            tags=_string_tuple(entry.get("tags")),
            chapters=_string_tuple(entry.get("chapters")),
            official=bool(entry.get("official", False)),
        )
    return entries


def catalog_entries() -> dict[str, CatalogEntry]:
    """Every well-formed catalog entry this install has, keyed by id."""
    return validate_catalog(load_catalog())


def github_catalog_packs() -> list[CatalogEntry]:
    """The catalog's ``kind = "github"`` entries, sorted by id.

    Empty in a stock build: the catalog ships only in-repo packs today. It is
    here because the shape is already in the schema, and a Plugin Center that
    lists third-party packs must not have to re-derive what one looks like.
    """
    return sorted(
        (entry for entry in catalog_entries().values() if entry.kind == "github"),
        key=lambda entry: entry.id,
    )


def catalog_entry(plugin_id: str) -> CatalogEntry | None:
    """One entry by id, case-insensitively, or ``None``.

    Case-insensitive because that is how the id arrives -- typed at a prompt
    or taken off a URL -- and every id in the catalog is lower-case by
    :data:`~app.core.plugins.manifest.PLUGIN_ID_RE`, so lowering the query is
    the whole of the match.
    """
    return catalog_entries().get(plugin_id.lower())
