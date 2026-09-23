from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import inspect
import logging
import os
import pkgutil
import sys
from pathlib import Path
from typing import Type

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
#: ``routes_nodes._provider_for`` tells a custom node from a built-in.
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


registry = NodeRegistry()
