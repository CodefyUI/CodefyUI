"""An example's name is a label and its description is one line.

``name`` and ``description`` in ``graph.json`` are the two fields every
surface that lists examples has to render: the gallery card, the sidebar
row, the empty-canvas card and the modal's detail pane. Both had turned
into the place the whole example was explained. 71 of 71 descriptions once
ran past the card, the longest to 1,272 columns; the name had no rule at
all, and reached 58 columns and eleven words.

The card decides the widths. It is 13rem across, and the root font-size is
``clamp(16px, 0.35vw + 11px, 19px)`` (``frontend/src/App.css``), so the type
inside the card is at its largest exactly on the wide screen that has the
most room around the card and none inside it. That is how all 38 titles the
gallery listed came to be ellipsized at once, measured on a 2560px window.
A title that does not fit is a title nobody reads, so the name is a label
-- one line, at most five words and 34 columns -- and the description is
the one line under it, at most 40 columns, which is twenty Chinese
characters. The explanation goes in a note on the canvas, beside the nodes
it is about, where the reader is looking at the thing being explained.

Names are also unique. It is the one label the gallery, the sidebar and the
empty-canvas overlay all print, with nothing beside it to tell two examples
apart, so a repeat is a card the reader cannot choose between.

Columns, not ``len`` -- a Chinese description mixes ideographs with Latin
tokens like ``SequentialModel`` and ``MNIST``, and a cap counted in code
points would let it render twice as wide as the English it replaces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _REPO_ROOT / "examples"
_PLUGINS_ROOT = _REPO_ROOT / "plugins"

#: One line, this wide -- twenty Chinese characters, the budget a 13rem card
#: has for the line under the title. ``CARD_DESC_COLUMNS`` in
#: ``frontend/src/components/Canvas/EmptyCanvasOverlay.tsx`` cuts at the same
#: number, so nothing that obeys this rule is ever cut on screen. Columns
#: rather than characters because the text this measures is bilingual.
MAX_DESCRIPTION_COLUMNS = 40

#: A name is a label, not a sentence. Two caps because either one alone lets
#: the other through: five words of ``Convolutional`` and ``Normalization``
#: overflow the card anyway, and 34 columns of monosyllables still read as
#: prose. Both are set against the two lines the card gives a title at
#: ``--fs-md``, which a 34-column name fills at most, so the title arrives
#: whole instead of clamped.
MAX_NAME_WORDS = 5
MAX_NAME_COLUMNS = 34

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


class Example(NamedTuple):
    """The two labels every surface that lists examples has to render."""

    name: str
    description: str


def _shipped_examples() -> dict[str, Example]:
    """Every shipped example, keyed as ``/api/examples/list`` keys it: the
    relative path for a built-in, ``plugin:<id>/<rest>`` for one a pack ships.

    One walk for both fields rather than one per rule. The rules differ but
    the files they hold over are the same files, and a second copy of this
    walk is a second place for a root to be left out of.
    """
    def read(graph_file: Path) -> Example:
        data = json.loads(graph_file.read_text(encoding="utf-8"))
        return Example(data.get("name", ""), data.get("description", ""))

    out: dict[str, Example] = {}
    for graph_file in sorted(_EXAMPLES_ROOT.rglob("graph.json")):
        key = graph_file.parent.relative_to(_EXAMPLES_ROOT).as_posix()
        out[key] = read(graph_file)
    for pack_dir in sorted(_PLUGINS_ROOT.glob("*")):
        examples_root = pack_dir / "examples"
        if not examples_root.is_dir():
            continue
        for graph_file in sorted(examples_root.rglob("graph.json")):
            rel = graph_file.parent.relative_to(examples_root).as_posix()
            out[f"plugin:{pack_dir.name}/{rel}"] = read(graph_file)
    return out


_EXAMPLES = _shipped_examples()


def test_the_scan_finds_the_examples_at_all():
    """A broken scan would make the rules below hold over nothing."""
    assert len(_EXAMPLES) > 50
    assert all(example.description for example in _EXAMPLES.values())
    assert all(example.name for example in _EXAMPLES.values())


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
        f"{key} ({display_width(example.description)} columns)"
        for key, example in _EXAMPLES.items()
        if display_width(example.description) > MAX_DESCRIPTION_COLUMNS
        or "\n" in example.description)

    assert not too_wide, (
        f"an example description is one line of at most "
        f"{MAX_DESCRIPTION_COLUMNS} columns -- what the card shows before it "
        f"cuts. Put the explanation in a note on the canvas, beside the nodes "
        f"it is about: {too_wide}")


def test_every_name_is_a_label_not_a_sentence():
    over = sorted(
        f"{key}: {example.name!r} "
        f"({len(example.name.split())} words, "
        f"{display_width(example.name)} columns)"
        for key, example in _EXAMPLES.items()
        if len(example.name.split()) > MAX_NAME_WORDS
        or display_width(example.name) > MAX_NAME_COLUMNS
        or "\n" in example.name)

    assert not over, (
        f"an example name is one line of at most {MAX_NAME_WORDS} words and "
        f"{MAX_NAME_COLUMNS} columns -- the title on a 13rem card, at type "
        f"that grows to 19px on a wide screen. What the graph does beyond "
        f"that goes in the description, and why it does it goes in a note on "
        f"the canvas: {over}")


def test_no_two_examples_share_a_name():
    by_name: dict[str, list[str]] = {}
    for key, example in _EXAMPLES.items():
        by_name.setdefault(example.name, []).append(key)
    repeated = sorted(
        f"{name!r}: {', '.join(keys)}"
        for name, keys in by_name.items() if len(keys) > 1)

    assert not repeated, (
        "two examples cannot ship the same name: it is the one label the "
        "gallery card, the sidebar row and the empty-canvas card all print, "
        "and none of them shows the path beside it, so the reader has no way "
        f"to tell which is which: {repeated}")
