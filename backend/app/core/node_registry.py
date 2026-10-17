from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.util
import inspect
import json
import logging
import os
import pkgutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Type

from .node_base import BaseNode

logger = logging.getLogger(__name__)


_PLUGIN_NS_PREFIX = "cdui_plugins."  # synthetic namespace, see plugin_loader

#: The package ``backend/app/custom_nodes`` is, and the name every custom-node
#: discovery hands to :meth:`NodeRegistry.discover`: the server's startup,
#: every reload, ``cdui project`` and the tests.
CUSTOM_NODES_PACKAGE = "app.custom_nodes"

#: What the files of a custom nodes directory anywhere else are imported as:
#: ``CODEFYUI_CUSTOM_NODES_DIR`` pointed outside the install (#519).
#:
#: A package of its own because ``app.custom_nodes.<file>`` cannot name them.
#: Python finds that name through the real package's ``__path__``, which is
#: ``backend/app/custom_nodes`` -- so discovery walked the configured
#: directory and every import read the repo's: each file that was not also
#: in the repo failed to import, and each one that was loaded the repo's
#: copy. A separate name also keeps a module read from one directory from
#: ever answering for a same-named file in the other.
#:
#: Directly under ``app`` because the shipped example imports
#: ``from ..core.node_base``, which resolves only one package below ``app``.
#: Beginning with ``app.custom_nodes`` because that prefix is how
#: ``node_registry.provider_for`` tells a custom node from a built-in.
CUSTOM_NODES_DIR_PACKAGE = "app.custom_nodes_dir"


def _same_directory(path: Path, other: str) -> bool:
    """Whether *path* and *other* are one existing directory, however spelled."""
    try:
        return os.path.samefile(path, other)
    except OSError:
        return False


def _repo_package_locations() -> list[str]:
    """Where ``backend/app/custom_nodes`` is, found WITHOUT importing it.

    ``find_spec`` asks the finders and runs none of the package's code. An
    import here would run its ``__init__.py``, and a broken one would then
    stop the server from starting and fail every reload after the registry
    was cleared; left alone, it fails each module under it in
    :meth:`NodeRegistry.discover`'s own ``except``, logged, as it always did.
    A package that cannot be found at all (an install that removed it) is no
    location.
    """
    try:
        spec = importlib.util.find_spec(CUSTOM_NODES_PACKAGE)
    except (ImportError, ValueError):
        return []
    if spec is None or spec.submodule_search_locations is None:
        return []
    return list(spec.submodule_search_locations)


def _custom_nodes_package(directory: Path) -> str:
    """The package the custom nodes in *directory* are imported under (#519).

    The repo's own directory keeps :data:`CUSTOM_NODES_PACKAGE`, so its
    classes stay the ones a direct import returns. Any other directory is
    imported through :data:`CUSTOM_NODES_DIR_PACKAGE`, pointed at it -- as is
    every directory of an install whose ``backend/app/custom_nodes`` is gone.
    """
    if any(_same_directory(directory, entry)
           for entry in _repo_package_locations()):
        return CUSTOM_NODES_PACKAGE
    _point_custom_nodes_dir_package(directory)
    return CUSTOM_NODES_DIR_PACKAGE


def _point_custom_nodes_dir_package(directory: Path) -> None:
    """Make :data:`CUSTOM_NODES_DIR_PACKAGE` the package of *directory*.

    Pointed at the directory it already has, it is left alone: its modules
    stay imported and ``force_reload`` decides whether they are read again,
    exactly as for the repo's package. Pointed anywhere else, every module
    imported through it is dropped first -- each came from the old directory,
    and ``import_module`` would hand it back for a same-named file in the new
    one. Compared resolved, so another spelling of the same directory is the
    same directory.
    """
    location = str(directory.resolve())
    current = sys.modules.get(CUSTOM_NODES_DIR_PACKAGE)
    if current is not None and list(getattr(current, "__path__", ())) == [location]:
        return
    stale = CUSTOM_NODES_DIR_PACKAGE + "."
    for name in [name for name in sys.modules
                 if name == CUSTOM_NODES_DIR_PACKAGE or name.startswith(stale)]:
        del sys.modules[name]
    spec = importlib.machinery.ModuleSpec(CUSTOM_NODES_DIR_PACKAGE, None,
                                          is_package=True)
    spec.submodule_search_locations = [location]
    package = importlib.util.module_from_spec(spec)
    sys.modules[CUSTOM_NODES_DIR_PACKAGE] = package
    parent, _, child = CUSTOM_NODES_DIR_PACKAGE.rpartition(".")
    setattr(importlib.import_module(parent), child, package)


class _DeriveFromPackage:
    """Type of the "caller said nothing" default for ``discover(plugin_id=)``.

    A distinct sentinel rather than ``None`` because ``None`` is already a
    meaningful value here: :func:`qualify` reads it as "builtin, keep the
    bare name". So ``None`` cannot also mean "work it out yourself".
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return "<derive from package name>"


_DERIVE_FROM_PACKAGE = _DeriveFromPackage()


def _plugin_id_from_package(package_name: str) -> str | None:
    """Return ``c2`` for ``cdui_plugins.c2.nodes`` etc., else ``None``.

    Builtin nodes are discovered with ``package_name="app.nodes"`` and don't
    match — they keep their bare ``NODE_NAME``. Plugin nodes are discovered
    under ``cdui_plugins.<plugin_id>.nodes`` and get the prefix.

    LOSSY, and only a fallback. A plugin id may be kebab-case, and the
    synthetic package name it is loaded under cannot be: ``official-template``
    is imported as ``cdui_plugins.official_template``, and nothing in that
    string records that the underscore used to be a hyphen. Callers that know
    the manifest id — every plugin call site does, see
    ``plugin_loader.discover_plugin_nodes`` — pass it to :meth:`discover`
    explicitly instead of letting it be guessed from here. What is left for
    this function is the builtin case (``app.nodes`` → ``None``) and ad-hoc
    discoveries in tests.
    """
    if not package_name.startswith(_PLUGIN_NS_PREFIX):
        return None
    rest = package_name[len(_PLUGIN_NS_PREFIX):]  # "c2.nodes" or "c2"
    return rest.split(".", 1)[0] or None


def qualify(plugin_id: str | None, node_name: str) -> str:
    """Compose the registry key for a node.

    Builtins stay bare so existing graphs and frontend palette labels are
    untouched. Plugin nodes get a ``<plugin_id>:`` prefix to (a) prevent
    two plugins from colliding on the same ``NODE_NAME``, and (b) make the
    saved graph JSON self-documenting — readers see ``"type": "c2:EduKNN"``
    and know immediately which plugin pack the node ships in.
    """
    return f"{plugin_id}:{node_name}" if plugin_id else node_name


class NodeRegistry:
    def __init__(self) -> None:
        self._nodes: dict[str, Type[BaseNode]] = {}
        #: Bumped on every change to the set of nodes, so a value derived
        #: from it (:func:`node_catalog`) can be cached until the next one.
        self.generation = 0

    @property
    def nodes(self) -> dict[str, Type[BaseNode]]:
        return dict(self._nodes)

    def register(
        self,
        node_cls: Type[BaseNode],
        *,
        plugin_id: str | None = None,
    ) -> str:
        """Register a node class under its qualified name.

        Returns the qualified name actually used (``"<plugin_id>:NODE_NAME"``
        for plugin nodes, bare ``NODE_NAME`` for builtins) so callers can log
        / report it.
        """
        name = node_cls.NODE_NAME
        if not name:
            raise ValueError(f"{node_cls.__name__} has no NODE_NAME")
        qualified = qualify(plugin_id, name)
        self._nodes[qualified] = node_cls
        self.generation += 1
        return qualified

    def get(self, name: str) -> Type[BaseNode] | None:
        """Look up a node class by registry key.

        Exact match wins. When the caller passes a bare name like
        ``"EduKNN"`` but the only registered entry is qualified
        (``"c2:EduKNN"``), fall back to a suffix scan so old graphs from
        before the namespacing scheme keep loading. When two plugins both
        export ``EduKNN`` and the lookup is ambiguous, the bare form picks
        the alphabetically-first plugin id and logs a warning — graphs
        that need the other one must use the qualified type.
        """
        if name in self._nodes:
            return self._nodes[name]
        if ":" not in name:
            matches = [k for k in self._nodes if k.endswith(f":{name}")]
            if len(matches) == 1:
                return self._nodes[matches[0]]
            if len(matches) > 1:
                matches.sort()
                logger.warning(
                    "Ambiguous bare lookup %r matched %d plugins (%s); using %s. "
                    "Update the graph to use the qualified type to silence this.",
                    name,
                    len(matches),
                    ", ".join(matches),
                    matches[0],
                )
                return self._nodes[matches[0]]
        return None

    def discover(
        self,
        package_path: Path,
        package_name: str,
        *,
        plugin_id: str | None | _DeriveFromPackage = _DERIVE_FROM_PACKAGE,
        force_reload: bool = False,
    ) -> int:
        """Walk *package_path* and register every ``BaseNode`` subclass.

        ``force_reload=True`` re-runs ``importlib.reload`` on any module that
        is already cached in ``sys.modules`` so that edits to the file on
        disk (typically a custom node a teacher just tweaked) actually take
        effect when ``POST /api/nodes/reload`` is called. Default is False
        because reloading at first-discovery time would replace already-
        imported class objects with fresh ones, breaking ``is`` identity for
        any caller that imported the class directly.

        Plugin-namespace discoveries (``package_name`` starts with
        ``cdui_plugins.``) auto-prefix every registered node with the
        plugin id, so ``EduKNN`` from ``cdui_plugins.c2.nodes`` lands in
        the registry as ``c2:EduKNN``. Builtin discoveries (``app.nodes``,
        ``app.custom_nodes``) keep bare names.

        ``plugin_id`` is how a caller that KNOWS the id says so. It must be
        the id the plugin's manifest declares, hyphens and all, because that
        is the id the rest of the system uses — the examples route's
        ``plugin:<id>`` prefix, the install directory, ``cdui plugin list``
        and the ``"type"`` string in saved graphs. Left unset, the id is
        derived from ``package_name``, which is right for builtins (no
        prefix) but can only ever return the snake_case spelling of a plugin
        id; see :func:`_plugin_id_from_package`.

        A discovery of ``app.custom_nodes`` reads *package_path* even when
        that is not the package's own directory: a ``CODEFYUI_CUSTOM_NODES_DIR``
        outside the install is imported as :data:`CUSTOM_NODES_DIR_PACKAGE`
        (#519). Handled here so every caller gets it unchanged.
        """
        count = 0
        if not package_path.exists():
            return count
        if package_name == CUSTOM_NODES_PACKAGE:
            package_name = _custom_nodes_package(package_path)
        if isinstance(plugin_id, _DeriveFromPackage):
            plugin_id = _plugin_id_from_package(package_name)
        for importer, modname, ispkg in pkgutil.walk_packages(
            [str(package_path)], prefix=package_name + "."
        ):
            try:
                if force_reload and modname in sys.modules:
                    module = importlib.reload(sys.modules[modname])
                else:
                    module = importlib.import_module(modname)
            except Exception as e:
                logger.warning("Failed to import %s: %s", modname, e)
                continue
            for _, obj in inspect.getmembers(module, inspect.isclass):
                try:
                    is_node_cls = issubclass(obj, BaseNode)
                except TypeError:
                    continue
                if (
                    is_node_cls
                    and obj is not BaseNode
                    and obj.NODE_NAME
                ):
                    self.register(obj, plugin_id=plugin_id)
                    count += 1
        return count

    def clear(self) -> None:
        self._nodes.clear()
        self.generation += 1


registry = NodeRegistry()


# ── node definitions (``GET /api/nodes``) and the node catalog (#781) ─────


def provider_for(qualified_name: str, cls: type[BaseNode]) -> str:
    """Classify a node by where it was loaded from.

    The module path says WHICH KIND of source a node came from, and that is
    all it is used for here. For a plugin node the id itself comes from the
    registry key, because that key is the plugin's manifest id — hyphens
    included — threaded through discovery by
    ``plugin_loader.discover_plugin_nodes``. The module path cannot supply
    it: ``official-template`` is imported as ``cdui_plugins.official_template``
    because a hyphen is not legal in a Python module name, and reading the id
    back off that string returns the snake_case spelling no other part of the
    API uses. ``/api/plugins`` says ``official-template`` and
    ``/api/examples`` tags graphs ``plugin:official-template``; ``provider``
    has to say the same thing about the same pack.
    """
    mod = cls.__module__ or ""
    if mod.startswith("app.nodes"):
        return "builtin"
    if mod.startswith("app.custom_nodes"):
        return "custom"
    if mod.startswith("cdui_plugins."):
        plugin_id, sep, _node = qualified_name.partition(":")
        if sep:
            return f"plugin:{plugin_id}"
        # No prefix on the key: the node was registered bare even though it
        # lives under the plugin namespace (only reachable by discovering a
        # plugin package with the prefix explicitly suppressed). Fall back to
        # the import spelling — wrong for a kebab-case id, but it is the only
        # signal left, and it beats reporting the node's own name as the pack.
        parts = mod.split(".")
        if len(parts) >= 2:
            return f"plugin:{parts[1]}"
    return "builtin"


def param_definition(p: Any, *, served: bool = True) -> Any:
    """The wire shape of one param (``ParamDefinitionSchema``).

    *served* (the node API): a ``device`` param's option list and its
    default fallback come from one ``device_options`` call: the served list
    is what this machine offers, and a saved default this machine lacks
    (``cuda`` on a CPU box) is served as ``cpu`` so the dropdown never shows
    a value it cannot run. ``served=False`` keeps both as the node declares
    them, which is what :func:`node_catalog` hashes: the same node on a CPU
    box and on a CUDA box must hash the same.
    """
    from ..schemas import ParamDefinitionSchema
    from .device_utils import device_options

    options = p.options
    default = p.default
    if served:
        options = device_options(p.name, p.options)
        if p.name == "device" and default != "auto" and default not in options:
            default = "cpu"
    return ParamDefinitionSchema(
        name=p.name,
        param_type=p.param_type.value,
        default=default,
        description=p.description,
        options=options,
        min_value=p.min_value,
        max_value=p.max_value,
        visible_when=p.visible_when,
        advanced=p.advanced,
        option_packs=p.option_packs,
    )


def node_definition(qualified_name: str, cls: type[BaseNode], *,
                    served: bool = True) -> Any:
    """Build the wire-shape definition (``NodeDefinition``) for one node.

    ``qualified_name`` is the registry key — bare ``"Linear"`` for builtin
    nodes, ``"c2:EduKNN"`` for plugin-provided nodes. The frontend palette
    uses this as the display label and as the value it writes into saved
    graph JSON, so two plugins shipping the same bare ``NODE_NAME`` no
    longer collide and the graph JSON becomes self-documenting. *served*:
    see :func:`param_definition`.
    """
    from ..schemas import NodeDefinition, PortDefinitionSchema

    def ports(defined: list[Any]) -> list[Any]:
        return [PortDefinitionSchema(
            name=p.name, data_type=p.data_type.value,
            description=p.description, optional=p.optional) for p in defined]

    return NodeDefinition(
        node_name=qualified_name,
        category=cls.CATEGORY,
        description=cls.DESCRIPTION,
        # ``getattr`` for the same reason as ``requires_pack`` below: a class
        # in the registry that never inherited this BaseNode should list
        # without the field rather than 500.
        details=getattr(cls, "DETAILS", ""),
        provider=provider_for(qualified_name, cls),
        # ``getattr`` rather than ``cls.REQUIRES_PACK``: the registry is a
        # plain dict anything can be written into, and a node class that
        # never inherited this BaseNode should still list, not 500.
        requires_pack=getattr(cls, "REQUIRES_PACK", None),
        inputs=ports(cls.define_inputs()),
        outputs=ports(cls.define_outputs()),
        params=[param_definition(p, served=served)
                for p in cls.define_params()],
    )


@dataclass(frozen=True)
class NodeCatalog:
    """Which node types a process can execute, and a hash of how (#781)."""

    #: sha256 hex over every type's name and definition (:func:`node_catalog`).
    hash: str
    #: The registered node types, sorted.
    node_types: tuple[str, ...]


_catalog_cache: tuple[int, int, NodeCatalog] | None = None


def node_catalog(reg: NodeRegistry | None = None) -> NodeCatalog:
    """This process's node catalog: its node types and their hash (#781).

    The hash is the sha256 of one canonical JSON document (sorted keys, no
    whitespace, UTF-8): ``[[type, definition], ...]`` in type order, each
    definition being what ``GET /api/nodes`` serves for that type, except
    that param options and defaults are as declared rather than narrowed to
    this machine's devices (:func:`param_definition`, ``served=False``).
    Two processes with the same CodefyUI version, plugins, packs and custom
    nodes get the same hash on any hardware; a missing plugin, an extra
    custom node or a plugin at another version gets a different one.

    A node whose definition cannot be built (its ``define_params`` raises)
    hashes as ``{"error": "<exception type>"}`` rather than failing the
    catalog. Cached per registry generation, so calling it on every worker
    handshake costs a compare until a node is registered.
    """
    global _catalog_cache
    reg = registry if reg is None else reg
    key = (id(reg), reg.generation)
    if _catalog_cache is not None and _catalog_cache[:2] == key:
        return _catalog_cache[2]
    nodes = reg.nodes
    entries = []
    for name in sorted(nodes):
        try:
            definition: Any = node_definition(
                name, nodes[name], served=False).model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001 -- a broken node, not a broken catalog
            definition = {"error": type(exc).__name__}
        entries.append([name, definition])
    raw = json.dumps(entries, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, default=str).encode("utf-8")
    catalog = NodeCatalog(hash=hashlib.sha256(raw).hexdigest(),
                          node_types=tuple(name for name, _ in entries))
    _catalog_cache = (key[0], key[1], catalog)
    return catalog
