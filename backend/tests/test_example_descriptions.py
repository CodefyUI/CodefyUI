"""An example's description is one line.

``description`` in ``graph.json`` is what the gallery card, the sidebar row
and the modal's detail pane all show, and it had turned into the place the
whole example was explained: 71 of 71 shipped descriptions ran past the
card, the longest to 1,272 columns. Everything past the cut is text nobody
reads, in the one field whose job is to say what a graph is for. The long
explanation belongs in on-canvas notes, where the reader is looking at the
nodes it describes. Every one of them now says its one line, so the rule
below holds over both roots with no exceptions.

The rule is the node palette's rule (``MAX_DESCRIPTION_CHARS`` in
``test_api_nodes.py``) measured for this field: one line, at most 56
columns. Columns, not ``len`` -- a Chinese description mixes ideographs with
Latin tokens like ``SequentialModel`` and ``MNIST``, and a cap counted in
code points would let it render twice as wide as the English it replaces.
"""

from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _REPO_ROOT / "examples"
_PLUGINS_ROOT = _REPO_ROOT / "plugins"

#: One line, this wide. The node palette caps its summary at 56 characters
#: for the same reason and calls it ``MAX_DESCRIPTION_CHARS``; this one is
#: columns because the text it measures is bilingual.
MAX_DESCRIPTION_COLUMNS = 56

#: Widths both implementations of the measurement agree on.
#:
#: ``display_width`` below is a port of ``displayWidth`` in
#: ``frontend/src/utils/localizeExamples.ts``, which the cards use to cut a
#: description and ``exampleLocales/zh-TW.test.ts`` uses to check the
#: Chinese half of this same rule. Two implementations of one measurement
#: drift silently, so both read this file and assert against it.
_WIDTH_VECTORS = (_REPO_ROOT / "frontend" / "src" / "utils"
                  / "displayWidth.vectors.json")


def display_width(text: str) -> int:
    """Roughly how many Latin characters wide *text* renders.

    A port of ``displayWidth`` in ``frontend/src/utils/localizeExamples.ts``,
    range for range. CJK ideographs, kana, Hangul and full-width punctuation
    take two Latin columns each.
    """
    width = 0
    for ch in text:
        cp = ord(ch)
        wide = (
            0x1100 <= cp <= 0x115F      # Hangul Jamo
            or 0x2E80 <= cp <= 0x303E   # CJK radicals, kangxi, CJK punctuation
            or 0x3041 <= cp <= 0x33FF   # kana, Hangul compat, CJK compat
            or 0x3400 <= cp <= 0x4DBF   # CJK ext A
            or 0x4E00 <= cp <= 0x9FFF   # CJK unified
            or 0xA000 <= cp <= 0xA4CF   # Yi
            or 0xAC00 <= cp <= 0xD7A3   # Hangul syllables
            or 0xF900 <= cp <= 0xFAFF   # CJK compat ideographs
            or 0xFE30 <= cp <= 0xFE6F   # CJK compat forms
            or 0xFF01 <= cp <= 0xFF60   # full-width forms
            or 0xFFE0 <= cp <= 0xFFE6
            or 0x20000 <= cp <= 0x3FFFD  # CJK ext B and beyond
        )
        width += 2 if wide else 1
    return width


def _shipped_descriptions() -> dict[str, str]:
    """Every shipped example's description, keyed as ``/api/examples/list``
    keys it: the relative path for a built-in, ``plugin:<id>/<rest>`` for one
    a pack ships."""
    out: dict[str, str] = {}
    for graph_file in sorted(_EXAMPLES_ROOT.rglob("graph.json")):
        key = graph_file.parent.relative_to(_EXAMPLES_ROOT).as_posix()
        out[key] = json.loads(
            graph_file.read_text(encoding="utf-8")).get("description", "")
    for pack_dir in sorted(_PLUGINS_ROOT.glob("*")):
        examples_root = pack_dir / "examples"
        if not examples_root.is_dir():
            continue
        for graph_file in sorted(examples_root.rglob("graph.json")):
            rel = graph_file.parent.relative_to(examples_root).as_posix()
            key = f"plugin:{pack_dir.name}/{rel}"
            out[key] = json.loads(
                graph_file.read_text(encoding="utf-8")).get("description", "")
    return out


_DESCRIPTIONS = _shipped_descriptions()


def test_the_scan_finds_the_examples_at_all():
    """A broken scan would make the rule below hold over nothing."""
    assert len(_DESCRIPTIONS) > 50
    assert all(text for text in _DESCRIPTIONS.values())


def test_the_two_width_implementations_agree():
    """The frontend cuts cards with its own copy of this measurement."""
    vectors = json.loads(
        _WIDTH_VECTORS.read_text(encoding="utf-8"))["vectors"]
    assert len(vectors) > 5, "the shared vectors went missing"
    measured = {
        case["text"]: display_width(case["text"]) for case in vectors}
    assert measured == {case["text"]: case["width"] for case in vectors}


def test_every_description_is_one_line_within_the_cap():
    too_wide = sorted(
        f"{key} ({display_width(text)} columns)"
        for key, text in _DESCRIPTIONS.items()
        if display_width(text) > MAX_DESCRIPTION_COLUMNS or "\n" in text)

    assert not too_wide, (
        f"an example description is one line of at most "
        f"{MAX_DESCRIPTION_COLUMNS} columns -- what the card shows before it "
        f"cuts. Put the explanation in a note on the canvas, beside the nodes "
        f"it is about: {too_wide}")
