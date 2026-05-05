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


class NodeRegistry:
    def __init__(self) -> None:
        self._nodes: dict[str, Type[BaseNode]] = {}

    @property
    def nodes(self) -> dict[str, Type[BaseNode]]:
        return dict(self._nodes)

    def register(self, node_cls: Type[BaseNode]) -> None:
        name = node_cls.NODE_NAME
        if not name:
            raise ValueError(f"{node_cls.__name__} has no NODE_NAME")
        self._nodes[name] = node_cls

    def get(self, name: str) -> Type[BaseNode] | None:
        return self._nodes.get(name)

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
        """
        count = 0
        if not package_path.exists():
            return count
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
                if (
                    issubclass(obj, BaseNode)
                    and obj is not BaseNode
                    and obj.NODE_NAME
                ):
                    self.register(obj)
                    count += 1
        return count

    def clear(self) -> None:
        self._nodes.clear()


registry = NodeRegistry()
