"""Three prose tables claim to list every built-in node. This checks them.

``README.md``, ``docs/docs/usage/node-reference.md`` and its zh-TW mirror
each carry a Category / Nodes / Count table, and a reader treats all three
as the answer to "what ships?". None of them is generated, so each drifts
on its own schedule: the README's table summed to 110 against its own
"152 Built-in Nodes" headline four lines from the top of the file, and had
no VLA row at all, while the two docs pages happened to be exact.

The registry is the authority (``app.main`` counts the same call for the
``nodes_loaded`` it reports at startup), so every number and every name in
all three tables is compared against it -- counts per category, the node
names inside each cell, the total, and the README's headline sentence.

The parser is deliberately strict. A silently skipped row is how a table
like this rots in the first place, so a malformed row raises rather than
being passed over, and a page whose table cannot be found at all fails.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Type

import pytest

from app.core.node_base import BaseNode
from app.core.node_registry import NodeRegistry

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NODES_DIR = _REPO_ROOT / "backend" / "app" / "nodes"
_DOCS = _REPO_ROOT / "docs" / "docs" / "usage" / "node-reference.md"
_DOCS_ZH = (_REPO_ROOT / "docs" / "i18n" / "zh-TW"
            / "docusaurus-plugin-content-docs" / "current" / "usage"
            / "node-reference.md")
_README = _REPO_ROOT / "README.md"

#: How each page separates the node names inside one cell. The zh-TW page
#: uses the ideographic comma, which is why this is per-page rather than one
#: split shared by all three.
_SEPARATORS = {_README: ", ", _DOCS: ", ", _DOCS_ZH: "、"}

#: ``| **Category** | Node, Node | 12 |`` -- the bold category is what tells
#: a table row apart from the header and the alignment rule.
_ROW = re.compile(r"^\|\s*\*\*(?P<category>[^*]+)\*\*\s*\|"
                  r"(?P<nodes>[^|]*)\|\s*(?P<count>\d+)\s*\|\s*$")

#: Anything else that starts and ends a line with a pipe. Only the header and
#: the alignment rule are allowed to match this without matching ``_ROW``.
_ANY_ROW = re.compile(r"^\|.*\|\s*$")

_HEADLINE = re.compile(r"\*\*(?P<nodes>\d+) Built-in Nodes\*\* across "
                       r"(?P<categories>\d+) categories")

#: The zh-TW page translates the category label and keeps the registry's own
#: name in parentheses after it (``資料 (Data)``). That parenthesised name is
#: the identity being compared, so it is what the parser reads.
_TRANSLATED_CATEGORY = re.compile(r"^.+\((?P<name>[^()]+)\)$")


def _category_name(label: str) -> str:
    match = _TRANSLATED_CATEGORY.match(label.strip())
    return match["name"].strip() if match else label.strip()


@pytest.fixture(scope="module")
def builtins() -> dict[str, Type[BaseNode]]:
    """Every built-in node, discovered the way ``app.main`` discovers them.

    Into a FRESH registry rather than the process-wide singleton, for two
    reasons: the singleton has whatever a conftest fixture or an earlier
    test registered on it (custom nodes, plugin stand-ins), which is not
    what these tables claim to list; and discovering into it here would
    leave those built-ins behind for every test that runs afterwards.
    """
    isolated = NodeRegistry()
    isolated.discover(_NODES_DIR, "app.nodes")
    nodes = isolated.nodes
    assert nodes, "discovered no built-in nodes"
    return nodes


@pytest.fixture(scope="module")
def expected(builtins) -> dict[str, list[str]]:
    """``{category: [NodeName, ...]}`` -- the shape the tables are in."""
    grouped: dict[str, list[str]] = {}
    for name, cls in builtins.items():
        grouped.setdefault(cls.CATEGORY, []).append(name)
    return grouped


def _parse_table(path: Path) -> dict[str, tuple[list[str], int]]:
    """The page's table as ``{category: ([node, ...], stated_count)}``.

    Reads the FIRST table whose rows carry a bold category, which is the one
    every three of these pages leads with. Raises on a line that looks like a
    table row and is not one -- skipping it silently is exactly how a row
    goes missing without anybody noticing.
    """
    separator = _SEPARATORS[path]
    table: dict[str, tuple[list[str], int]] = {}
    seen_any = False

    for line in path.read_text(encoding="utf-8").splitlines():
        match = _ROW.match(line)
        if match is None:
            if seen_any and _ANY_ROW.match(line):
                raise AssertionError(
                    f"{path.name}: malformed table row inside the node "
                    f"table: {line!r}")
            if seen_any:
                break          # the table ended at the first non-row line
            continue
        seen_any = True
        names = [n.strip() for n in match["nodes"].split(separator)]
        assert all(names), f"{path.name}: empty node name in {line!r}"
        key = _category_name(match["category"])
        assert key not in table, f"{path.name}: {key} listed twice"
        table[key] = (names, int(match["count"]))

    assert table, f"{path.name}: no node table found"
    return table


@pytest.mark.parametrize("path", [_README, _DOCS, _DOCS_ZH],
                         ids=["README", "node-reference", "node-reference.zh"])
def test_the_table_lists_exactly_the_registry(path, expected):
    table = _parse_table(path)

    assert set(table) == set(expected), (
        f"{path.name}: the table's categories and the registry's disagree")

    for category, (names, stated) in table.items():
        assert set(names) == set(expected[category]), (
            f"{path.name}: the {category} cell does not name the "
            f"{category} nodes")
        assert len(names) == stated, (
            f"{path.name}: the {category} cell lists {len(names)} nodes and "
            f"claims {stated}")
        assert stated == len(expected[category]), (
            f"{path.name}: {category} says {stated}, the registry has "
            f"{len(expected[category])}")


@pytest.mark.parametrize("path", [_README, _DOCS, _DOCS_ZH],
                         ids=["README", "node-reference", "node-reference.zh"])
def test_the_counts_sum_to_the_whole_registry(path, builtins):
    table = _parse_table(path)
    assert sum(count for _, count in table.values()) == len(builtins)


def test_the_readme_headline_matches_its_own_table(builtins, expected):
    """Four lines from the top, and the first number anyone reads."""
    match = _HEADLINE.search(_README.read_text(encoding="utf-8"))
    assert match is not None, "README: no '**N Built-in Nodes**' headline"

    assert int(match["nodes"]) == len(builtins)
    assert int(match["categories"]) == len(expected)


def test_a_category_listed_twice_raises(tmp_path, monkeypatch):
    """A repeated category would otherwise overwrite the first row silently.

    The sum test only notices when the row that was lost had a count of its
    own to contribute, and the name test not at all when both rows name the
    same nodes, so the parser is where a doubled category has to be caught.
    """
    page = tmp_path / "doubled.md"
    page.write_text(
        "| Category | Nodes | Count |\n"
        "|---|---|---|\n"
        "| **LLM** | Tokenizer | 1 |\n"
        "| **LLM** | WordVector | 1 |\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(_SEPARATORS, page, ", ")

    with pytest.raises(AssertionError, match="LLM listed twice"):
        _parse_table(page)
