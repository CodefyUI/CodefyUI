"""pytest conftest — makes the plugin importable during local development.

When installed via ``cdui plugin install``, CodefyUI registers this plugin under
the ``cdui_plugins.{{plugin_snake}}.*`` synthetic namespace. For local pytest we
set that up by hand so
``from cdui_plugins.{{plugin_snake}}.nodes.example_node import ExampleNode``
works. If you rename the plugin id in cdui.plugin.toml, update PLUGIN_ID below.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

PLUGIN_ID = "{{plugin_id}}"
_PY_ID = PLUGIN_ID.replace("-", "_")
_REPO_ROOT = Path(__file__).resolve().parents[1]

_pkg = sys.modules.get("cdui_plugins")
if _pkg is None:
    _pkg = types.ModuleType("cdui_plugins")
    _pkg.__path__ = []
    sys.modules["cdui_plugins"] = _pkg

_sub_name = f"cdui_plugins.{_PY_ID}"
if _sub_name not in sys.modules:
    _sub = types.ModuleType(_sub_name)
    _sub.__path__ = [str(_REPO_ROOT)]
    sys.modules[_sub_name] = _sub
    setattr(_pkg, _PY_ID, _sub)
