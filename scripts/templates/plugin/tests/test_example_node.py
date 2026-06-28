"""Starter test for ExampleNode. Run from the CodefyUI backend venv:

    uv run --directory path/to/CodefyUI/backend pytest path/to/{{plugin_id}}/tests/
"""

from __future__ import annotations

from cdui_plugins.{{plugin_snake}}.nodes.example_node import ExampleNode


def test_example_node_greets_by_name():
    out = ExampleNode().execute({}, {"name": "CodefyUI"})
    assert out["greeting"] == "Hello, CodefyUI!"


def test_example_node_defaults_to_world():
    out = ExampleNode().execute({}, {})
    assert "world" in out["greeting"]
