"""Plugin discovery and loader for CodefyUI.

Reuses :class:`NodeRegistry.discover` unchanged by exposing each installed
plugin's ``nodes/`` directory under a synthetic ``cdui_plugins.<id>``
namespace package registered in :data:`sys.modules`. That avoids
``uv pip install -e`` per plugin (which would pollute the venv and
complicate uninstall) while letting ``pkgutil.walk_packages`` find plugin
nodes the same way it finds built-ins.

Layout::

    <REPO>/plugins/<id>/                 ← built-in (first-party) packs
    <USER_DATA>/plugins/<id>/            ← downloaded (third-party) packs
    <USER_DATA>/plugins/installed.json   ← lockfile
"""

from __future__ import annotations

import json
import os
import sys
import types
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # 3.10 backport — same API.

from platformdirs import user_data_dir

LOCKFILE_SCHEMA = 1
MANIFEST_FILENAME = "cdui.plugin.toml"
NAMESPACE_PACKAGE = "cdui_plugins"

# Bumped on every full re-discovery (plugin/node reload, enable/disable). The
# editor polls this in dev mode — when a linked plugin is present — to know when
# to re-activate plugin frontends without a manual browser refresh.
_reload_generation = 0


def reload_generation() -> int:
    """Current reload generation; increases by one on each ``rediscover_all``."""
    return _reload_generation


def plugins_builtin_root() -> Path:
    """``<REPO>/plugins/`` — packs shipped with the CodefyUI distribution.

    Resolved from this file's location: ``backend/app/core/plugin_loader.py``
    → up 3 = ``backend/``'s parent = repo root.
    """
    return Path(__file__).resolve().parents[3] / "plugins"


def plugins_user_root() -> Path:
    """``<USER_DATA>/plugins/`` — where downloaded packs and the lockfile live.

    Honors the ``CODEFYUI_USER_DATA_DIR`` environment variable so a dev clone
    can pin the lockfile inside the project directory (``.codefyui_dev/``)
    instead of sharing the production user data dir across every clone on the
    machine. ``scripts/dev.py dev-install`` / ``start`` set this automatically.
    """
    override = os.environ.get("CODEFYUI_USER_DATA_DIR")
    base = Path(override) if override else Path(user_data_dir("codefyui", appauthor=False))
    return base / "plugins"


def lockfile_path() -> Path:
    return plugins_user_root() / "installed.json"


def empty_lockfile() -> dict[str, Any]:
    return {"schema": LOCKFILE_SCHEMA, "plugins": {}}


def load_lockfile() -> dict[str, Any]:
    p = lockfile_path()
    if not p.exists():
        return empty_lockfile()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return empty_lockfile()
    if not isinstance(data, dict) or "plugins" not in data:
        return empty_lockfile()
    return data


def save_lockfile(data: dict[str, Any]) -> None:
    """Write the lockfile, atomically.

    A temp file in the SAME directory and then ``os.replace``, because this
    file is the only record of what is installed: a crash, a full disk or a
    killed process partway through a ``write_text`` leaves half a JSON
    document, and half a JSON document is an empty lockfile to
    :func:`load_lockfile` -- every plugin the user has, gone, with their
    files still on disk. ``os.replace`` is atomic on both platforms and
    overwrites an existing file on Windows too, so a reader sees either the
    old document or the new one and never the write in progress.

    The temp name carries the pid so two processes writing at once (the CLI
    and the server both do) cannot land on each other's partial file. That is
    not a substitute for a lock -- two writers can still lose one edit
    between a read and a write -- but it does mean neither of them can ever
    read a torn file.
    """
    p = lockfile_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f"{p.suffix}.tmp-{os.getpid()}")
    try:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
    except BaseException:
        # Whatever went wrong, the caller's failure is the one worth
        # reporting -- and a temp file left in the plugins directory would
        # outlive it silently.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def read_manifest_safe(plugin_dir: Path) -> dict[str, Any]:
    """Parse a plugin's manifest, returning {} on any read/parse failure.

    ``ValueError`` covers two of those failures rather than one:
    ``TOMLDecodeError`` is a subclass, and so is the ``UnicodeDecodeError``
    that a manifest with a byte that is not UTF-8 raises out of
    ``read_text``. That one used to escape, and every reader here is a
    listing -- ``GET /api/plugins``, ``GET /api/plugins/catalog``,
    ``cdui plugin list`` -- so one unreadable pack took the whole list down
    with it instead of appearing as a pack with no metadata.
    """
    try:
        return tomllib.loads(
            (plugin_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return {}


def _py_id(plugin_id: str) -> str:
    """Convert kebab-case plugin id to a valid Python module identifier."""
    return plugin_id.replace("-", "_")


def _resolve_plugin_dir(
    plugin_id: str,
    entry: dict[str, Any],
    builtin_root: Path,
    user_root: Path,
) -> Path:
    source_kind = entry.get("source_kind")
    if source_kind == "builtin":
        return builtin_root / plugin_id
    if source_kind == "local":
        # Linked dev plugins (`cdui plugin link <path>`) record an absolute
        # path that lives outside both roots — the author's own checkout, loaded
        # in place with no copy. A malformed entry without a path falls back to
        # user_root (where it has no manifest and is skipped) rather than
        # crashing discovery on a KeyError.
        path = entry.get("path")
        return Path(path) if path else user_root / plugin_id
    return user_root / plugin_id


def is_enabled(entry: dict[str, Any]) -> bool:
    """Whether a lockfile entry is currently activated.

    Missing field defaults to ``True`` so lockfiles from before the
    enabled/disabled feature keep working without migration. New entries
    written by ``cmd_install`` always set the field explicitly.
    """
    return bool(entry.get("enabled", True))


def removed_ids(lockfile: dict[str, Any]) -> set[str]:
    """Pack ids the user uninstalled on purpose — the #175 "tombstones".

    ``cdui plugin uninstall`` used to only pop the entry, which made "this
    install has never heard of the pack" and "the user threw it away"
    the same state — and those are the only two states a catch-up command has
    to tell apart. So the removal is recorded in a top-level ``removed`` map
    rather than as a retained ``plugins`` entry: everything that walks
    ``plugins`` (discovery, the plugin list API, presets, examples) keeps its
    exact meaning, and an uninstalled pack cannot come back to life through a
    field one of those readers forgets to check.

    A missing or malformed field reads as "no tombstones", so a lockfile
    written before #175 loads unchanged — the same legacy-tolerant default as
    :func:`is_enabled`.
    """
    removed = lockfile.get("removed")
    if not isinstance(removed, dict):
        return set()
    return set(removed)


def mark_removed(
    lockfile: dict[str, Any],
    plugin_id: str,
    *,
    source_kind: str | None = None,
) -> None:
    """Record ``plugin_id`` as deliberately removed. The caller saves."""
    removed = lockfile.get("removed")
    if not isinstance(removed, dict):
        removed = {}
        lockfile["removed"] = removed
    entry: dict[str, Any] = {"removed_at": now_iso()}
    if source_kind:
        entry["source_kind"] = source_kind
    removed[plugin_id] = entry


def clear_removed(lockfile: dict[str, Any], plugin_id: str) -> bool:
    """Forget a tombstone; returns whether there was one. The caller saves.

    Installing a pack by name is the undo for having uninstalled it, so the
    record of the removal has to go with it — otherwise ``sync`` would keep
    skipping a pack that is now installed, and a later uninstall/install cycle
    would read as "still removed".
    """
    removed = lockfile.get("removed")
    if not isinstance(removed, dict) or plugin_id not in removed:
        return False
    del removed[plugin_id]
    if not removed:
        # Don't leave an empty map behind: a lockfile that records no removals
        # should look exactly like one written before this field existed.
        lockfile.pop("removed", None)
    return True


def frontend_entry_rel(manifest: dict[str, Any]) -> str | None:
    """Validated ``[frontend].entry`` path from a plugin manifest, or ``None``.

    The entry must be a relative POSIX-style path that stays inside the
    plugin's ``frontend/`` directory -- anything else (traversal, absolute
    paths, other directories) is treated as "no frontend" rather than an
    error, so a malformed third-party manifest can't break startup.
    """
    fe = manifest.get("frontend")
    if not isinstance(fe, dict):
        return None
    entry = fe.get("entry")
    if not isinstance(entry, str) or not entry:
        return None
    p = PurePosixPath(entry.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts:
        return None
    if p.parts[:1] != ("frontend",) or len(p.parts) < 2:
        return None
    return str(p)


class PluginNamespace(NamedTuple):
    """One installed plugin's nodes, ready to hand to ``NodeRegistry.discover``.

    Carries THREE things, and the third is the one that used to go missing.
    ``package_name`` and ``plugin_id`` are two different spellings of the same
    pack and only one of them is the real name: ``official-template`` is
    imported as ``cdui_plugins.official_template`` because a hyphen cannot
    appear in a Python module path, but ``official-template`` is what the
    manifest says, what the install directory is called, what
    ``cdui plugin list`` prints, and what a saved graph's
    ``"type": "official-template:HelloPlugin"`` has to say. Rebuilding the id
    from the package name recovers only the snake_case spelling, so it is
    carried rather than re-derived.

    A named tuple rather than a bare 3-tuple precisely because two of the
    three fields are strings: positional unpacking is exactly how the id was
    dropped in the first place.
    """

    nodes_dir: Path
    package_name: str
    plugin_id: str


def install_plugin_finder(
    builtin_root: Path,
    user_root: Path,
    lockfile: dict[str, Any],
) -> list[PluginNamespace]:
    """Register the synthetic namespace and return one entry per loadable pack.

    The returned entries are ready to pass straight to
    :meth:`NodeRegistry.discover` — see :func:`discover_plugin_nodes`, which
    is what every caller in the app actually uses. Plugins whose manifest is
    missing, whose ``nodes/`` directory is absent, **or whose ``enabled`` flag
    is false** are skipped silently — the caller is responsible for surfacing
    those.
    """
    pkg = sys.modules.get(NAMESPACE_PACKAGE)
    if pkg is None:
        pkg = types.ModuleType(NAMESPACE_PACKAGE)
        pkg.__path__ = []  # namespace package
        sys.modules[NAMESPACE_PACKAGE] = pkg

    found: list[PluginNamespace] = []
    for plugin_id, entry in lockfile.get("plugins", {}).items():
        if not is_enabled(entry):
            continue
        plugin_dir = _resolve_plugin_dir(plugin_id, entry, builtin_root, user_root)
        if not (plugin_dir / MANIFEST_FILENAME).exists():
            continue

        py = _py_id(plugin_id)
        sub_name = f"{NAMESPACE_PACKAGE}.{py}"
        sub = sys.modules.get(sub_name)
        if sub is None:
            sub = types.ModuleType(sub_name)
            sub.__path__ = [str(plugin_dir)]
            sys.modules[sub_name] = sub
            setattr(pkg, py, sub)
        else:
            sub.__path__ = [str(plugin_dir)]

        nodes_dir = plugin_dir / "nodes"
        if nodes_dir.exists():
            found.append(PluginNamespace(nodes_dir, f"{sub_name}.nodes", plugin_id))

    return found


def discover_plugin_nodes(
    registry: Any,
    builtin_root: Path,
    user_root: Path,
    lockfile: dict[str, Any],
    *,
    force_reload: bool = False,
) -> tuple[int, int]:
    """Register every enabled pack's nodes; returns ``(nodes, packs)``.

    The single place that turns installed packs into registry entries. The
    server lifespan, ``POST /api/plugins/reload`` (via :func:`rediscover_all`)
    and ``scripts/project.py`` all go through here instead of writing the
    ``install_plugin_finder`` → ``discover`` loop themselves, so there is
    exactly one line that has to remember to pass the manifest id along —
    and three copies of it cannot drift apart, which is how plugin nodes came
    to be registered under ``official_template:`` while every other subsystem
    said ``official-template``.

    Both counts are returned because the lifespan log reports "N nodes from M
    plugins", and M is not derivable from N.
    """
    namespaces = install_plugin_finder(builtin_root, user_root, lockfile)
    node_count = sum(
        registry.discover(
            ns.nodes_dir,
            ns.package_name,
            plugin_id=ns.plugin_id,
            force_reload=force_reload,
        )
        for ns in namespaces
    )
    return node_count, len(namespaces)


def purge_plugin_modules(plugin_id: str) -> None:
    """Remove a plugin's namespace from :data:`sys.modules` so reload sees new code."""
    py = _py_id(plugin_id)
    prefix = f"{NAMESPACE_PACKAGE}.{py}"
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            del sys.modules[name]


def purge_all_plugin_modules() -> None:
    """Remove every ``cdui_plugins.*`` entry — used by hot-reload."""
    for name in list(sys.modules):
        if name == NAMESPACE_PACKAGE or name.startswith(NAMESPACE_PACKAGE + "."):
            del sys.modules[name]


def iter_plugin_dirs(
    builtin_root: Path,
    user_root: Path,
    lockfile: dict[str, Any],
    *,
    include_disabled: bool = False,
) -> list[tuple[str, Path]]:
    """Return ``(plugin_id, plugin_dir)`` for every installed plugin with a manifest.

    Skips disabled plugins by default so examples / asset routes / preset
    discovery all silently respect the ``enabled`` flag. Pass
    ``include_disabled=True`` to enumerate every entry regardless of state
    — the plugin list API uses this to render greyed-out rows.
    """
    out: list[tuple[str, Path]] = []
    for plugin_id, entry in lockfile.get("plugins", {}).items():
        if not include_disabled and not is_enabled(entry):
            continue
        plugin_dir = _resolve_plugin_dir(plugin_id, entry, builtin_root, user_root)
        if (plugin_dir / MANIFEST_FILENAME).exists():
            out.append((plugin_id, plugin_dir))
    return out


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def rediscover_all(
    registry: Any,
    preset_registry: Any,
    *,
    nodes_dir: Path,
    custom_nodes_dir: Path,
    presets_dir: Path,
    builtin_root: Path,
    user_root: Path,
) -> dict[str, int]:
    """Clear and re-discover every node + preset source.

    Shared by ``POST /api/nodes/reload``, the custom-nodes upload/toggle
    handlers, and ``POST /api/plugins/reload``. Built-ins don't get
    ``force_reload`` because their class objects are stable for the server
    lifetime; custom nodes and plugins do, because their files can change
    between calls.
    """
    registry.clear()
    builtin = registry.discover(nodes_dir, "app.nodes")
    custom = registry.discover(custom_nodes_dir, "app.custom_nodes", force_reload=True)

    lockfile = load_lockfile()
    plugin_count, _packs = discover_plugin_nodes(
        registry, builtin_root, user_root, lockfile, force_reload=True
    )

    preset_registry.clear()
    preset_count = preset_registry.discover(presets_dir, registry)
    for _plugin_id, plugin_dir in iter_plugin_dirs(builtin_root, user_root, lockfile):
        preset_count += preset_registry.discover(plugin_dir / "presets", registry)

    global _reload_generation
    _reload_generation += 1

    return {
        "builtin": builtin,
        "custom": custom,
        "plugins": plugin_count,
        "presets": preset_count,
        "total": builtin + custom + plugin_count,
    }
