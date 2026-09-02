"""What the Plugin Center draws: one row per plugin, installed or not.

The panel asks one question -- "what can I have, and what do I have?" -- and
the two halves of the answer come from two documents that do not know about
each other. ``plugins/registry.json`` says what this build can install by
name; the lockfile says what this install actually did. Neither is a superset
of the other: a pack shipped by an update sits on disk uninstalled, and a
plugin fetched from a URL nobody put in the catalog is installed with no
catalog row at all.

So the merge happens HERE, once, and produces rows of one shape. Every row
carries every field, and a field a row cannot have is ``None`` or empty
rather than absent -- a card renderer should never have to ask whether the
key is there this time, and a TypeScript type should be able to say
``string | null`` instead of ``string | undefined``. The rules that decide
each field are in one file for the same reason the manifest rules are: the
CLI's ``plugin list`` and the panel must not be able to disagree about which
plugin is official, which one is missing its files, and which one the user
threw away on purpose.

What this module does NOT do: read a network, install anything, or format a
sentence. ``status`` is a word for a caller to translate, and the two node
sources -- the live registry for an installed pack, a source scan for one on
disk that is not installed -- are both answers about names, never about code
that has been imported.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.core import plugin_loader
from app.core.node_registry import NodeRegistry

from .catalog import CatalogEntry, catalog_entries
from .manifest import (
    REPO_RE,
    manifest_allowed_modules,
    manifest_capabilities,
    manifest_has_frontend,
    manifest_python_deps,
)

#: How a node announces its name. The same expression the catalog-honesty
#: test scans packs with, deliberately: what the Plugin Center promises a
#: pack contains and what that test proves a pack contains have to be the
#: same list, or one of them is lying to a student whose palette is empty.
_NODE_NAME_RE = re.compile(r"""NODE_NAME\s*=\s*["']([^"']+)["']""")

#: Scanned node names, keyed by ``(directory, newest mtime in it)``. The key
#: carries the mtime rather than a timestamp of its own so an edited pack is
#: a MISS instead of a stale hit -- which is what a plugin author reloading
#: their own pack needs. Small on purpose: this only ever holds the packs
#: this build ships, and it is cleared wholesale rather than evicted, because
#: an LRU for eight entries is more code than the scan it saves.
_NODE_SCAN_CACHE: dict[tuple[str, float], tuple[str, ...]] = {}
_NODE_SCAN_LIMIT = 32


def provider_token(plugin_id: str) -> str:
    """The ``cdui_plugins.<X>`` slot a pack's modules are imported under.

    kebab to snake, because a hyphen cannot appear in a Python module path:
    ``official-template`` is imported as ``cdui_plugins.official_template``.
    """
    return plugin_id.replace("-", "_")


def nodes_for_plugin(plugin_id: str, registry: NodeRegistry) -> list[str]:
    """The node names an installed plugin has REGISTERED, sorted.

    Read off the live registry rather than off disk, so it answers what the
    palette will actually offer: a disabled plugin has no nodes in there and
    correctly reports none, and a node whose file failed to import is absent
    from both.
    """
    prefix = f"cdui_plugins.{provider_token(plugin_id)}."
    return sorted(
        cls.NODE_NAME
        for cls in registry.nodes.values()
        if (cls.__module__ or "").startswith(prefix)
    )


def scan_node_names(nodes_dir: Path) -> list[str]:
    """The node names a pack's sources declare, without importing them.

    For a pack that is ON DISK but NOT INSTALLED there is nothing in the
    registry to ask, and importing it to find out is exactly what installing
    means -- the card has to say what the pack contains before the user
    agrees to run any of it. So the names are read as text.

    Never raises. An unreadable directory or a file in an encoding this
    cannot decode answers "no names", because a Plugin Center that traces
    back over one malformed pack is worse than one that undersells it.
    """
    try:
        files = sorted(nodes_dir.glob("*.py"))
        newest = max((path.stat().st_mtime for path in files), default=0.0)
    except OSError:
        return []

    key = (str(nodes_dir), newest)
    cached = _NODE_SCAN_CACHE.get(key)
    if cached is not None:
        return list(cached)

    names: set[str] = set()
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        names.update(_NODE_NAME_RE.findall(text))

    found = tuple(sorted(names))
    if len(_NODE_SCAN_CACHE) >= _NODE_SCAN_LIMIT:
        _NODE_SCAN_CACHE.clear()
    _NODE_SCAN_CACHE[key] = found
    return list(found)


def frontend_entry_url(
    plugin_id: str, plugin_dir: Path | None, manifest: dict[str, Any], *,
    enabled: bool,
) -> str | None:
    """Where the editor loads this plugin's browser code from, or ``None``.

    Three conditions, and all three are load-bearing: the manifest has to
    declare an entry that stays inside ``frontend/``
    (:func:`~app.core.plugin_loader.frontend_entry_rel` enforces that), the
    file has to be THERE -- a manifest can name a bundle the author forgot to
    build, and a URL to a missing file is a console error in the editor
    rather than a message about a plugin -- and the plugin has to be enabled,
    since a disabled plugin contributes no UI either.
    """
    if plugin_dir is None or not enabled:
        return None
    entry_rel = plugin_loader.frontend_entry_rel(manifest)
    if not entry_rel or not (plugin_dir / entry_rel).is_file():
        return None
    return f"/plugins/{plugin_id}/{entry_rel}"


def is_official(row: CatalogEntry | None, entry: dict[str, Any] | None) -> bool:
    """Does CodefyUI vouch for THIS plugin -- not just for this id?

    The badge is a claim about provenance, so matching the catalog by id is
    not enough to earn it. An id is only reserved against a foreign install
    when this build owns it outright (a route, or a pack that ships here);
    the ids of the catalog's ``github`` rows are deliberately NOT reserved,
    because the author of an official plugin has to be able to install their
    own repository under its own id. That leaves a hole an id-only rule falls
    straight into: install ``mallory/evil``, whose manifest says
    ``id = "self-learning"``, and the row would wear the official badge over
    a ``repo`` field reading ``mallory/evil``.

    So a row is official when the catalog vouches for it AND the install in
    front of us really is the thing the catalog named:

    * nothing installed under the id -- the row IS the catalog's, so the
      badge describes what installing it would get you;
    * a ``builtin`` row installed as ``source_kind = "builtin"`` -- activated
      in place from this release's own files, which is the only way a
      built-in pack can arrive;
    * a ``github`` row installed with the recorded ``catalog_id``, which
      ``cdui plugin install <name>`` writes only when the install really came
      from that row;
    * a ``github`` row whose repository is the one the install recorded --
      case-insensitively, because GitHub owners and repositories are. This is
      the free-text install of the catalog's own repository
      (``cdui plugin install CodefyUI/CodefyUI-Plugin-Self-Learning``), which
      is the same code by a longer road.

    Everything else is false, including every ``external`` row, and including
    a plugin LINKED from a local directory under a catalog id: a working tree
    is whatever the author has in it right now, and nothing about it can be
    checked against the repository the catalog names.
    """
    if row is None:
        return False
    # A ``github`` row the catalog does not mark ``official`` is a third
    # party the catalog merely lists -- it never earns the badge, however it
    # was installed.
    if not (row.kind == "builtin" or row.official):
        return False
    if entry is None:
        return True
    if row.kind == "builtin":
        return entry.get("source_kind") == "builtin"
    if _text(entry.get("catalog_id")) == row.id:
        return True
    installed = _repo_of(entry)
    return bool(row.repo) and installed is not None and (
        installed.lower() == row.repo.lower()
    )


def catalog_id_for(
    plugin_id: str,
    entry: dict[str, Any] | None,
    catalog: dict[str, CatalogEntry],
) -> str | None:
    """The catalog row an installed plugin came from, or ``None``.

    The recorded ``catalog_id`` wins: ``cdui plugin install <name>`` writes
    it precisely so a later reader can tell the catalog's own pack from free
    text that happens to carry the same id. A built-in install records none
    (its ``source`` IS the catalog id), so the id is matched against the
    catalog instead -- which is safe because the reserved-id rule refuses an
    install that claims a catalog id from anywhere but the repository the
    catalog names.
    """
    recorded = (entry or {}).get("catalog_id")
    if isinstance(recorded, str) and recorded:
        return recorded
    return plugin_id if plugin_id in catalog else None


def declared_capabilities(
    entry: dict[str, Any] | None, manifest: dict[str, Any]
) -> list[str]:
    """The capabilities this plugin has been GRANTED, or asks for.

    The lockfile wins for an installed plugin: it records what the user
    agreed to, and a plugin that rewrote its manifest afterwards must not be
    shown as though the new list were consented to. The manifest is the
    fallback for a plugin nobody has installed yet -- and for a lockfile
    written before capabilities were recorded at all.

    A RECORDED empty list is an answer, not a miss: "you granted this plugin
    nothing" is exactly what an empty grant means, and falling through to the
    manifest there would show an ungranted capability as though it had been
    agreed to. Only a missing (or malformed) field asks the manifest.
    """
    granted = (entry or {}).get("capabilities")
    if isinstance(granted, list):
        return [item for item in granted if isinstance(item, str)]
    return list(manifest_capabilities(manifest))


def declared_trusted_modules(
    entry: dict[str, Any] | None, manifest: dict[str, Any]
) -> list[str]:
    """The imports the AST gate was told to allow. Same rule as the
    capabilities above: a recorded list wins, empty or not; only a missing
    one asks the manifest."""
    trusted = (entry or {}).get("trusted_modules")
    if isinstance(trusted, list):
        return [item for item in trusted if isinstance(item, str)]
    return manifest_allowed_modules(manifest)


def installed_facts(
    plugin_id: str,
    entry: dict[str, Any],
    manifest: dict[str, Any],
    catalog: dict[str, CatalogEntry],
) -> dict[str, Any]:
    """The catalog-shaped fields ``GET /api/plugins`` reports per plugin.

    Six keys the installed-plugin listing did not have and the Plugin Center
    needs on every row it draws, computed by the same functions the catalog
    listing uses -- so the panel cannot be told a plugin is official on one
    route and unofficial on the other.
    """
    return {
        "official": is_official(catalog.get(plugin_id), entry),
        "catalog_id": catalog_id_for(plugin_id, entry, catalog),
        "capabilities": declared_capabilities(entry, manifest),
        "trusted_modules": declared_trusted_modules(entry, manifest),
        "python_deps": manifest_python_deps(manifest),
        "has_frontend": manifest_has_frontend(manifest),
    }


def catalog_listing(
    lockfile: dict[str, Any],
    *,
    registry: NodeRegistry,
    active_job: dict[str, Any] | None = None,
    remote_install_allowed: bool,
    generation: int,
) -> dict[str, Any]:
    """Every plugin this install can have or does have, in one payload.

    ``entries`` is the catalog in the order ``registry.json`` writes it --
    that order is curated, teaching packs before tools -- followed by every
    lockfile entry the catalog does not know about, sorted by id. The three
    envelope fields are passed in rather than read here: whether a remote
    install is allowed is a question about the ROUTER's gate, the reload
    generation is the counter the editor polls, and the active job belongs to
    whatever is running the install. This module answers about plugins.
    """
    catalog = catalog_entries()
    installed = lockfile.get("plugins")
    installed = installed if isinstance(installed, dict) else {}
    tombstones = plugin_loader.removed_ids(lockfile)
    dirs = dict(
        plugin_loader.iter_plugin_dirs(
            plugin_loader.plugins_builtin_root(),
            plugin_loader.plugins_user_root(),
            lockfile,
            include_disabled=True,
        )
    )

    entries = [
        _entry_payload(
            plugin_id,
            row=row,
            entry=_entry_of(installed, plugin_id),
            plugin_dir=dirs.get(plugin_id),
            tombstoned=plugin_id in tombstones,
            registry=registry,
            active_job=active_job,
        )
        for plugin_id, row in catalog.items()
    ]
    entries.extend(
        _entry_payload(
            plugin_id,
            row=None,
            entry=_entry_of(installed, plugin_id),
            plugin_dir=dirs.get(plugin_id),
            tombstoned=plugin_id in tombstones,
            registry=registry,
            active_job=active_job,
        )
        for plugin_id in sorted(set(installed) - set(catalog))
    )

    return {
        "entries": entries,
        "active_job": active_job,
        "remote_install_allowed": remote_install_allowed,
        "generation": generation,
    }


def _entry_of(installed: dict[str, Any], plugin_id: str) -> dict[str, Any] | None:
    """This install's lockfile record for *plugin_id*, or ``None``."""
    entry = installed.get(plugin_id)
    return entry if isinstance(entry, dict) else None


def _entry_payload(
    plugin_id: str,
    *,
    row: CatalogEntry | None,
    entry: dict[str, Any] | None,
    plugin_dir: Path | None,
    tombstoned: bool,
    registry: NodeRegistry,
    active_job: dict[str, Any] | None,
) -> dict[str, Any]:
    """One row. See the module docstring for what the shape promises."""
    manifest = _manifest_for(plugin_id, row=row, plugin_dir=plugin_dir)
    plugin_meta = _table(manifest.get("plugin"))
    if not plugin_meta and entry is not None:
        # A plugin whose files are gone still has the ``[plugin]`` table the
        # install recorded. Showing that beats showing the bare id for a row
        # whose whole point is to say "this is missing".
        plugin_meta = _table(entry.get("manifest"))
    lessons_meta = _table(manifest.get("lessons"))

    job = _job_for(plugin_id, active_job)
    enabled = plugin_loader.is_enabled(entry) if entry is not None else False
    nodes = _nodes(plugin_id, row=row, entry=entry, registry=registry)
    # What was installed beats what the catalog pins: a plugin installed off
    # the default branch must not report the tag the catalog has moved on to.
    # "" is the default branch on both sides, so an empty recorded ref is an
    # answer rather than a miss.
    ref = _text(entry.get("ref")) if entry is not None else (row.ref if row else "")

    return {
        "id": plugin_id,
        "name": _text(plugin_meta.get("name")) or (row.name if row else plugin_id),
        "description": (
            _text(plugin_meta.get("description"))
            or (row.description if row else "")
        ),
        "kind": row.kind if row else "external",
        "official": is_official(row, entry),
        "status": _status(
            entry=entry, plugin_dir=plugin_dir, tombstoned=tombstoned, job=job
        ),
        "source_kind": _recorded(entry, "source_kind"),
        "source": _source(plugin_id, row=row, entry=entry),
        "repo": _repo(row=row, entry=entry),
        "ref": ref,
        "sha": _recorded(entry, "sha"),
        "url": _url(row=row, entry=entry),
        "homepage": (
            _text(plugin_meta.get("homepage")) or (row.homepage if row else "")
        ),
        "version": _text(plugin_meta.get("version")) or None,
        "installed_at": _recorded(entry, "installed_at"),
        "enabled": enabled,
        "chapters": (
            _strings(lessons_meta.get("chapters"))
            or list(row.chapters if row else ())
        ),
        "lessons": _strings(lessons_meta.get("lessons")),
        "tags": list(row.tags) if row else [],
        "nodes": nodes,
        "node_count": len(nodes),
        "capabilities": declared_capabilities(entry, manifest),
        "trusted_modules": declared_trusted_modules(entry, manifest),
        "python_deps": manifest_python_deps(manifest),
        "has_frontend": manifest_has_frontend(manifest),
        "consent_required": _consent_required(row=row, entry=entry),
        "frontend_entry": frontend_entry_url(
            plugin_id, plugin_dir, manifest, enabled=enabled
        ),
        "job": job,
    }


def _status(
    *,
    entry: dict[str, Any] | None,
    plugin_dir: Path | None,
    tombstoned: bool,
    job: dict[str, Any] | None,
) -> str:
    """The one word the card's pill shows.

    Order is the whole rule. A job in flight outranks everything, because
    what the row is DOING is more useful than what it was. Then the lockfile:
    an entry with no directory is ``missing_files`` -- the state a moved
    checkout or a half-finished uninstall leaves, and the one that used to
    render as a normal installed row whose nodes had all silently vanished.
    Only for a plugin with no entry at all does the tombstone matter, and
    there it is the difference between "you removed this" and "you have never
    had this", which is the difference between a Reinstall button and an
    Install one.
    """
    if job is not None:
        return "installing"
    if entry is None:
        return "removed" if tombstoned else "available"
    if plugin_dir is None:
        return "missing_files"
    return "installed" if plugin_loader.is_enabled(entry) else "disabled"


def _manifest_for(
    plugin_id: str, *, row: CatalogEntry | None, plugin_dir: Path | None
) -> dict[str, Any]:
    """The manifest behind a row, or ``{}`` when there is nothing to read.

    An installed plugin's manifest is read where it is installed. A built-in
    pack that is NOT installed still ships in this release, so its manifest
    is read from the built-in root -- that is what lets an uninstalled pack's
    card show its version, its lessons and what it would install. A ``github``
    row that is not installed has nothing on this disk, and nothing may be
    fetched to find out: a listing that reached the network would turn
    drawing a panel into eight HTTP requests.
    """
    if plugin_dir is not None:
        return plugin_loader.read_manifest_safe(plugin_dir)
    if row is not None and row.kind == "builtin":
        return plugin_loader.read_manifest_safe(
            plugin_loader.plugins_builtin_root() / plugin_id
        )
    return {}


def _nodes(
    plugin_id: str,
    *,
    row: CatalogEntry | None,
    entry: dict[str, Any] | None,
    registry: NodeRegistry,
) -> list[str]:
    """What this plugin puts in the palette, by whichever means can say."""
    if entry is not None:
        return nodes_for_plugin(plugin_id, registry)
    if row is not None and row.kind == "builtin":
        return scan_node_names(
            plugin_loader.plugins_builtin_root() / plugin_id / "nodes"
        )
    return []


def _source(
    plugin_id: str, *, row: CatalogEntry | None, entry: dict[str, Any] | None
) -> str:
    """What this plugin was (or would be) installed FROM.

    The lockfile's own word for an installed plugin. For one that is not
    installed it is what the install would record: the id for a built-in
    pack, ``owner/repo`` (with ``@ref`` when the catalog pins one) for a
    repository -- so a row's ``source`` reads the same before and after.
    """
    if entry is not None:
        return _text(entry.get("source")) or plugin_id
    if row is not None and row.kind == "github" and row.repo:
        return f"{row.repo}@{row.ref}" if row.ref else row.repo
    return plugin_id


def _repo(*, row: CatalogEntry | None, entry: dict[str, Any] | None) -> str | None:
    """``owner/repo``: what this install came from, else what the catalog says.

    In that order, and the same order :func:`_url` uses, so the two fields
    cannot describe two different repositories on one row. They do disagree
    in real lockfiles -- a plugin installed from a fork, or from an owner the
    project has since moved away from -- and when they do, the repository the
    files actually came from is the one worth showing.
    """
    installed = _repo_of(entry)
    if installed is not None:
        return installed
    return row.repo if row is not None and row.kind == "github" else None


def _url(
    *, row: CatalogEntry | None, entry: dict[str, Any] | None
) -> str | None:
    """The repository this came from, when it came from one."""
    if entry is not None and _text(entry.get("url")):
        return _text(entry.get("url"))
    if row is not None and row.kind == "github" and row.repo:
        return f"https://github.com/{row.repo}"
    return None


def _repo_of(entry: dict[str, Any] | None) -> str | None:
    """``owner/repo`` as a lockfile entry recorded it, or ``None``.

    Derived from what the install wrote down, and only when it really is that
    shape: the ``url`` first, because ``https://github.com/alice/extras`` can
    only be read one way, then ``source``, which is ``alice/extras@v1`` for a
    repository install and an absolute path for a linked directory -- and an
    absolute path is not ``owner/repo`` in either operating system's
    spelling. Anything else answers ``None`` rather than a guess.
    """
    if entry is None:
        return None
    url = _text(entry.get("url"))
    if url:
        tail = url.rstrip("/").removesuffix(".git")
        owner_repo = "/".join(tail.split("/")[-2:])
        if REPO_RE.match(owner_repo):
            return owner_repo
    source = _text(entry.get("source")).split("@", 1)[0]
    return source if REPO_RE.match(source) else None


def _consent_required(
    *, row: CatalogEntry | None, entry: dict[str, Any] | None
) -> bool:
    """Would installing this ask the user to trust a third party?

    True for anything that comes out of a repository, false for a pack that
    ships in this release and for a directory the user linked themselves --
    the same split ``inspect`` makes, and for the same reason: a built-in
    pack arrived through a pull request here, and a local link is the
    author's own working tree.
    """
    if row is not None and row.kind == "github":
        return True
    return entry is not None and entry.get("source_kind") == "github_url"


def _job_for(
    plugin_id: str, active_job: dict[str, Any] | None
) -> dict[str, Any] | None:
    """The running job's progress, but only on the row it belongs to."""
    if not active_job or active_job.get("plugin_id") != plugin_id:
        return None
    return {
        "job_id": active_job.get("job_id"),
        "status": active_job.get("status"),
        "current_step": active_job.get("current_step"),
    }


def _recorded(entry: dict[str, Any] | None, key: str) -> str | None:
    """What the lockfile wrote down under *key*, or ``None``.

    One spelling for the three fields only an installed plugin has, so that
    "not installed" and "installed but the field is empty" arrive as the same
    ``None`` instead of as ``None`` in one field and ``""`` in the next. Not
    used for ``ref``, where ``""`` is a real answer -- the default branch.
    """
    if entry is None:
        return None
    return _text(entry.get(key)) or None


def _table(value: Any) -> dict[str, Any]:
    """*value* when it is a table, else an empty one."""
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    """*value* when it is a string, else the empty string."""
    return value if isinstance(value, str) else ""


def _strings(value: Any) -> list[str]:
    """The string members of *value* when it is a list, else empty."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]
