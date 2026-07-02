from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
import sys
from pathlib import Path
from typing import Type

from .node_base import BaseNode

logger = logging.getLogger(__name__)


_PLUGIN_NS_PREFIX = "cdui_plugins."  # synthetic namespace, see plugin_loader


def _plugin_id_from_package(package_name: str) -> str | None:
    """Return ``c2`` for ``cdui_plugins.c2.nodes`` etc., else ``None``.

    Builtin nodes are discovered with ``package_name="app.nodes"`` and don't
    match — they keep their bare ``NODE_NAME``. Plugin nodes are discovered
    under ``cdui_plugins.<plugin_id>.nodes`` and get the prefix.
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
        """
        count = 0
        if not package_path.exists():
            return count
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
