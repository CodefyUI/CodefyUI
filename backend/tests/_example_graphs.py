"""One reader for the shipped example graphs, shared by the example suites.

Every suite that opens an ``examples/**/graph.json`` or a
``plugins/*/examples/**/graph.json`` asks the same question of it: which of
these nodes are the graph? A note is not -- it carries no ``type`` the
registry knows, no ``data.params`` to index, and nothing to execute -- so a
test that walks the raw list either reads through a note and raises
KeyError, or compares its type against the palette and fails.

The examples are about to carry notes, so the filter lives here rather than
in each suite: the next example that explains itself must not be an edit to
four test files.
"""

from __future__ import annotations

from app.core.graph_engine import is_note_node


def graph_nodes(graph: dict) -> list[dict]:
    """The executable nodes of *graph*, in file order, notes removed."""
    return [node for node in graph.get("nodes", []) if not is_note_node(node)]
