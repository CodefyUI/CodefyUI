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
from pathlib import Path
from typing import Any

from platformdirs import user_data_dir

LOCKFILE_SCHEMA = 1
MANIFEST_FILENAME = "cdui.plugin.toml"
NAMESPACE_PACKAGE = "cdui_plugins"


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
    p = lockfile_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _py_id(plugin_id: str) -> str:
    """Convert kebab-case plugin id to a valid Python module identifier."""
    return plugin_id.replace("-", "_")


def _resolve_plugin_dir(
    plugin_id: str,
    entry: dict[str, Any],
    builtin_root: Path,
    user_root: Path,
) -> Path:
    if entry.get("source_kind") == "builtin":
        return builtin_root / plugin_id
    return user_root / plugin_id


def is_enabled(entry: dict[str, Any]) -> bool:
    """Whether a lockfile entry is currently activated.

    Missing field defaults to ``True`` so lockfiles from before the
    enabled/disabled feature keep working without migration. New entries
    written by ``cmd_install`` always set the field explicitly.
    """
    return bool(entry.get("enabled", True))


def install_plugin_finder(
    builtin_root: Path,
    user_root: Path,
    lockfile: dict[str, Any],
) -> list[tuple[Path, str]]:
    """Register the synthetic namespace and return ``(nodes_dir, package_name)`` pairs.

    The returned pairs are ready to pass straight to
    :meth:`NodeRegistry.discover`. Plugins whose manifest is missing,
    whose ``nodes/`` directory is absent, **or whose ``enabled`` flag is
    false** are skipped silently — the caller is responsible for surfacing
    those.
    """
    pkg = sys.modules.get(NAMESPACE_PACKAGE)
    if pkg is None:
        pkg = types.ModuleType(NAMESPACE_PACKAGE)
        pkg.__path__ = []  # namespace package
        sys.modules[NAMESPACE_PACKAGE] = pkg

    pairs: list[tuple[Path, str]] = []
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
            pairs.append((nodes_dir, f"{sub_name}.nodes"))

    return pairs


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
    pairs = install_plugin_finder(builtin_root, user_root, lockfile)
    plugin_count = 0
    for plug_nodes_dir, pkg_name in pairs:
        plugin_count += registry.discover(plug_nodes_dir, pkg_name, force_reload=True)

    preset_registry.clear()
    preset_count = preset_registry.discover(presets_dir, registry)
    for _plugin_id, plugin_dir in iter_plugin_dirs(builtin_root, user_root, lockfile):
        preset_count += preset_registry.discover(plugin_dir / "presets", registry)

    return {
        "builtin": builtin,
        "custom": custom,
        "plugins": plugin_count,
        "presets": preset_count,
        "total": builtin + custom + plugin_count,
    }
