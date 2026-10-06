"""Headless CodefyUI runtime bootstrap.

The server, CLI tooling, and exported Python runners all need the same
built-in, custom, plugin, and preset discovery rules. Keep the exported
runner on the existing central re-discovery path -- one call, whose seven
arguments live in :func:`~app.core.plugins.reload.rediscover_now` -- rather
than growing a second partial registry bootstrap.

An exported script calls :func:`initialize_export_runtime`: that same call,
then the plugin packs bundled with CodefyUI that the script names and the
lockfile left out, loaded through
:func:`~app.core.plugin_loader.discover_plugin_nodes` like every other pack.
"""

from __future__ import annotations

from collections.abc import Iterable

from . import plugin_loader
from .node_registry import registry
from .plugins.manifest import PLUGIN_ID_RE
from .plugins.reload import rediscover_now


def initialize_runtime() -> dict[str, int]:
    """Reset and discover every executable node and preset source."""

    return rediscover_now()


def use_preset_copies(definitions: Iterable[dict]) -> None:
    """Register the preset copies an exported script carries, each in place
    of any preset of its name that discovery found here.

    Map runs the preset its ``subgraph`` setting names, looked up in the
    preset registry when it runs. Discovery fills that registry from this
    installation alone -- its built-in presets, the user presets folder and
    its plugin packs -- and a grader's machine has none of the presets a
    student saved. So the exporter copies each preset the graph's Map nodes
    run into the script, as the canvas had it
    (``codegen._presets_run_by_name``), and the script hands the copies here
    after discovery: wherever it runs, it runs what the canvas ran.
    """
    from ..schemas.models import PresetDefinition
    from .preset_registry import preset_registry

    # Copies of INSTALLED presets, so not a graph's embedded preset
    # definitions (its ``presets[]``), which Map never reads.
    for raw in definitions:
        preset_registry.add(PresetDefinition.model_validate(raw))


def initialize_export_runtime(required_types: Iterable[str]) -> list[str]:
    """:func:`initialize_runtime` for an exported script, which lists its types.

    Discovery loads plugin packs only from the lockfile in the user data dir,
    and a grader runs ``python <file>.py`` with an empty one: a graph using
    ``edu:FFNLayer`` could not run although the pack sat in this
    installation's ``plugins/`` directory. So each of *required_types* still
    unknown after discovery that names a pack shipped there loads that pack.
    Only the packs named load, from the read-only built-in root, and nothing
    is written.

    A pack loads even when the lockfile turns it off or records it as
    uninstalled: the lockfile governs the editor, and loading from the
    built-in root changes nothing on disk. A third-party pack the lockfile
    turns off is never loaded, since it may be off for being broken or
    untrusted, and its line gives the command that enables it.

    Returns one line per type still unknown, sorted, each beginning
    ``Unknown node type: <type>`` (graders and tests match that prefix) and,
    when the type names a pack, saying what is wrong with it. An empty list
    means every type resolved.
    """
    initialize_runtime()
    unresolved = sorted({t for t in required_types if registry.get(t) is None})
    if not unresolved:
        return []

    builtin_root = plugin_loader.plugins_builtin_root()
    bundled = {
        pack
        for pack in map(_pack_of, unresolved)
        if pack is not None
        and (builtin_root / pack / plugin_loader.MANIFEST_FILENAME).is_file()
    }
    if bundled:
        plugin_loader.discover_plugin_nodes(
            registry,
            builtin_root,
            plugin_loader.plugins_user_root(),
            {"plugins": {
                pack: {"source_kind": "builtin", "enabled": True}
                for pack in sorted(bundled)
            }},
        )

    loaded = {key.split(":", 1)[0] for key in registry.nodes if ":" in key}
    # Installed here, with files on disk, and turned off in the lockfile.
    lockfile = plugin_loader.load_lockfile()
    user_root = plugin_loader.plugins_user_root()
    installed = plugin_loader.iter_plugin_dirs(
        builtin_root, user_root, lockfile, include_disabled=True)
    enabled = plugin_loader.iter_plugin_dirs(builtin_root, user_root, lockfile)
    disabled = {pack for pack, _ in installed} - {pack for pack, _ in enabled}
    return [
        _unknown_type(node_type, loaded, bundled, disabled)
        for node_type in unresolved
        if registry.get(node_type) is None
    ]


def _pack_of(node_type: str) -> str | None:
    """The pack a qualified type names (``edu`` for ``edu:FFNLayer``), or None.

    Checked whole against the ids a manifest may declare because the id is
    joined onto the built-in root as a directory name: ``../x:Y`` names no
    pack and is never looked up on disk. A bare name names no pack either;
    the canvas saves plugin types qualified.
    """
    pack, sep, name = node_type.partition(":")
    if sep and name and PLUGIN_ID_RE.fullmatch(pack):
        return pack
    return None


def _unknown_type(
    node_type: str, loaded: set[str], bundled: set[str], disabled: set[str],
) -> str:
    """The line for one type still unknown after the bundled packs loaded."""
    line = f"Unknown node type: {node_type}"
    pack = _pack_of(node_type)
    if pack is None:
        return line
    if pack in loaded:
        name = node_type.split(":", 1)[1]
        return (
            f"{line} -- the plugin pack '{pack}' here has no node '{name}' "
            "(a different CodefyUI version, or one of its modules failed to "
            "load; see the log above)"
        )
    if pack in bundled:
        return (
            f"{line} -- the plugin pack '{pack}' ships with this CodefyUI but "
            "none of its nodes could be loaded"
        )
    if pack in disabled:
        # After `bundled`: a bundled pack loads however the lockfile has it,
        # so enabling it would change nothing.
        return (
            f"{line} -- it comes from the plugin pack '{pack}', which is "
            "installed here but disabled. Enable it with: "
            f"cdui plugin enable {pack}"
        )
    return (
        f"{line} -- it comes from the plugin pack '{pack}', which this "
        "CodefyUI installation does not have. Install it with: "
        f"cdui plugin install {pack}"
    )
