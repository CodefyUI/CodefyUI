"""What the pack-backed example graphs promise, checked without a pack.

Two shipped examples cannot run on a fresh install: Word-Embedding-Analogy
switches to a 69 MB GloVe table for its real vectors, and
Sentence-Similarity-zhTW needs a 470 MB multilingual encoder before it does
anything at all. Neither is executed here -- CI has no pack cache, and a box
that HAS one should not load half a gigabyte of weights inside the fast
suite (which is why ``TextEmbedding`` sits in
``test_builtin_examples._SLOW_NODE_TYPES``). So this file asserts everything
about them that does not need the model: what the gallery card says, how
many sentences the example carries, and where the labels are wired.

Why the card is worth a test. ``EmptyCanvasOverlay.tsx`` renders an
example's description as ``description.slice(0, 80) + '...'`` with no
``title`` attribute, so those 80 characters are the whole of the warning a
learner reads before pressing Run on a graph that will otherwise stop at a
missing pack. "The requirement is somewhere in the description" is
therefore not the property worth holding; "the requirement is inside the
first 80 characters, and the cut does not land mid-word" is.

The real run lives next door in ``test_pack_examples_real.py``, opt-in
behind ``CODEFYUI_PACK_NETWORK_TESTS=1`` because it needs the download this
file is written to avoid. PR 4's RAG examples join the parametrised card
list below.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import settings
from app.core.node_registry import NodeRegistry

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _REPO_ROOT / "examples"

_SENTENCE_EXAMPLE = (
    _EXAMPLES_ROOT / "LLM" / "Sentence-Similarity-zhTW" / "graph.json")
_ANALOGY_EXAMPLE = (
    _EXAMPLES_ROOT / "LLM" / "Word-Embedding-Analogy" / "graph.json")

#: ``EmptyCanvasOverlay.tsx`` renders a gallery card's description as
#: ``description.slice(0, 80) + '...'``. Mirrors
#: ``test_builtin_examples._CARD_VISIBLE_CHARS``.
_CARD_VISIBLE_CHARS = 80

#: The sidebar ``TemplatesTab`` shows the full description in a native
#: ``title`` tooltip, and a tooltip of a thousand characters is its own kind
#: of unreadable. A generous ceiling rather than a style rule: the detail
#: belongs beside the graph, not on the card.
_DESCRIPTION_LIMIT = 600

#: Each pack-backed example and the phrase its card must show, because
#: without it the learner presses Run and gets an error instead of a
#: download. One entry today; PR 4 appends the RAG examples.
_PACK_BACKED_CARDS = [
    pytest.param(
        _SENTENCE_EXAMPLE,
        "sentence-embeddings pack",
        id="Sentence-Similarity-zhTW",
    ),
]


def _payload(graph_path: Path) -> dict:
    return json.loads(graph_path.read_text(encoding="utf-8"))


def _cuts_cleanly(description: str) -> bool:
    """Does the card's 80-character slice end between words?

    A boundary, not a length: the cut is clean when the text ends before it,
    or when whitespace sits on either side of it. Anything else splits a
    word, and the card appends its own ellipsis -- so "for real Glo..."
    reads as a bug rather than as a summary.
    """
    return (len(description) <= _CARD_VISIBLE_CHARS
            or description[_CARD_VISIBLE_CHARS].isspace()
            or description[_CARD_VISIBLE_CHARS - 1].isspace())


#: The CJK Unified Ideographs block, as codepoints rather than as the two
#: boundary characters: U+9FFF is a rare ideograph that a cp950 console
#: cannot print, so a literal would break the traceback on the Windows box
#: this suite is developed on -- and it reads as noise in source either way.
_CJK_FIRST, _CJK_LAST = 0x4E00, 0x9FFF


def _has_cjk(text: str) -> bool:
    """Any CJK ideograph, which is all "this line is Chinese" needs to mean."""
    return any(_CJK_FIRST <= ord(char) <= _CJK_LAST for char in text)


@pytest.fixture(scope="module")
def palette() -> set[str]:
    """The registry keys an install offers, built here rather than read.

    The session-wide singleton's contents depend on what earlier tests did
    to it (``test_plugin_api`` runs the app lifespan against a lockfile
    naming three packs, which drops the others), and a check about whether
    an example matches the palette must not pass or fail on collection
    order. Built-ins only, deliberately: a graph under ``examples/`` ships
    with the app and must never need a plugin node.
    """
    reg = NodeRegistry()
    reg.discover(settings.NODES_DIR, "app.nodes")
    keys = set(reg.nodes)
    assert "Start" in keys, (
        "the builtin scan found nothing -- the check would pass vacuously")
    return keys


# -- the gallery cards -----------------------------------------------------

@pytest.mark.parametrize("graph_path, requirement", _PACK_BACKED_CARDS)
def test_pack_example_cards_warn_inside_the_80_char_truncation(
    graph_path: Path, requirement: str
):
    """The pack requirement survives the card's cut, and reads as a sentence.

    An example that needs a download has exactly one chance to say so before
    the Run button is pressed, and it is these 80 characters. Asserted
    against the truncated string rather than the whole description, because
    the whole one is not what anybody reads.
    """
    description = _payload(graph_path)["description"]
    visible = description[:_CARD_VISIBLE_CHARS]

    assert requirement in visible, (
        f"{requirement!r} is not inside the {_CARD_VISIBLE_CHARS} characters "
        f"the canvas gallery card shows. The card renders {visible!r} and "
        f"nothing more, so a learner without the pack presses Run and gets an "
        f"error instead of a download -- move the requirement to the front of "
        f"the description.")

    assert _cuts_cleanly(description), (
        f"the card renders {visible!r} and nothing more, cutting a word in "
        f"half; end the opening sentence inside {_CARD_VISIBLE_CHARS} "
        f"characters")

    assert len(description) <= _DESCRIPTION_LIMIT, (
        f"the description is {len(description)} characters; the sidebar shows "
        f"it in full as a tooltip, so keep the detail beside the graph")


def test_analogy_card_says_it_runs_offline():
    """Both facts that decide whether the analogy runs survive the cut.

    That it works offline on ``demo-16d`` and that ``glove-50d`` is where the
    real vectors are: a learner without the word-vectors pack needs the
    first, and one wondering what the toy table is for needs the second. It
    is the mirror image of the test above -- this example runs WITHOUT a
    pack, and the card has to say so.

    Where the cut LANDS is a separate assertion because the two come apart: a
    description can carry both facts inside 80 characters and still be cut in
    the middle of the word after them.
    """
    description = _payload(_ANALOGY_EXAMPLE)["description"]
    visible = description[:_CARD_VISIBLE_CHARS]

    assert "offline" in visible
    assert "glove-50d" in visible
    assert _cuts_cleanly(description), (
        f"the card renders {visible!r} and nothing more, cutting a word in "
        f"half; end the opening sentence inside {_CARD_VISIBLE_CHARS} "
        f"characters")

    # And the opening sentence is what has to fit, not just any 80 characters
    # of it: the rest of the description is written for the sidebar tooltip.
    first_sentence = description[:description.index(". ") + 1]
    assert len(first_sentence) <= _CARD_VISIBLE_CHARS


# -- Sentence-Similarity-zhTW ---------------------------------------------

def test_sentence_similarity_has_eight_sentences_in_four_pairs():
    """Eight lines, seven of them Chinese and the last one English.

    The example's whole claim is that a pair written with almost no shared
    characters still comes back as each other's nearest neighbour, and the
    last pair makes that claim across two languages. Both are facts about
    the TEXT, invisible to ``validate_graph`` -- an edit that drops a line,
    or that translates the English one into Chinese, leaves a graph that
    validates and runs and no longer demonstrates anything.

    ``split_lines`` is checked in the same breath because it is what turns
    eight lines into eight vectors: with it off the whole block is embedded
    as one text, and every downstream assertion about pairs becomes
    meaningless rather than false.
    """
    payload = _payload(_SENTENCE_EXAMPLE)
    by_id = {node["id"]: node for node in payload["nodes"]}

    lines = [line for line in by_id["sentences"]["data"]["params"]["value"]
             .splitlines() if line.strip()]
    assert len(lines) == 8, (
        f"the example carries {len(lines)} sentences, not the four pairs its "
        f"description promises: {lines}")

    assert lines[-1].isascii(), (
        f"the last sentence is the English half of the cross-language pair, "
        f"and it is not ASCII: {lines[-1]!r}")
    for line in lines[:-1]:
        assert _has_cjk(line), (
            f"{line!r} carries no Chinese characters; seven of the eight "
            f"sentences are the zh-TW half of this example")

    assert by_id["embed"]["data"]["params"]["split_lines"] is True, (
        "split_lines is off, so the eight sentences are embedded as ONE "
        "vector and the pairs cannot be found at all")


def test_sentence_similarity_wires_labels_to_both_consumers(palette: set[str]):
    """The labels reach the top-k list AND the scatter, from the same node.

    Both consumers read ``labels`` positionally against a tensor they get
    from somewhere else (``CosineSimilarity`` pairs ``key_labels`` with
    ``keys``; ``EmbeddingScatter`` pairs ``labels`` with ``embeddings``), and
    both treat them as OPTIONAL -- so an unlabelled run does not fail. It
    prints a list of empty lists where the neighbouring sentences should be,
    and plots eight anonymous dots: the example minus the point of it.

    The node types are checked here too, and against a freshly built
    registry rather than the session-wide singleton: ``registry.get`` falls
    back to a suffix scan server-side while the canvas does an exact
    ``Map.get``, so a type this file did not catch would validate, execute,
    and still render as an empty box with no ports on the learner's screen.
    """
    payload = _payload(_SENTENCE_EXAMPLE)

    wiring = {
        (edge["source"], edge.get("sourceHandle"),
         edge["target"], edge.get("targetHandle"))
        for edge in payload["edges"]
        if edge.get("type", "data") != "trigger"
    }
    for consumer, port in (("similarity", "key_labels"), ("scatter", "labels")):
        assert ("embed", "labels", consumer, port) in wiring, (
            f"TextEmbedding.labels is not wired to {consumer}.{port}; the "
            f"port is optional, so the run still succeeds and shows blanks "
            f"where the sentences should be")

    unresolved = sorted({node["type"] for node in payload["nodes"]}
                        - palette)
    assert not unresolved, (
        f"{_SENTENCE_EXAMPLE.name} uses node types that are not registry "
        f"keys: {unresolved}. The canvas resolves a node type by exact "
        f"match, so each of these renders as an empty box with no ports.")


def test_sentence_similarity_runs_from_a_single_trigger():
    """One Start edge, into the one node that has no data input.

    ``TextInput`` is the only source in this graph; everything else is
    downstream of it through data edges, which is what the engine schedules
    on. A second trigger edge would therefore mark a node that already runs,
    and none at all would leave the whole chain unreachable from Start.
    """
    payload = _payload(_SENTENCE_EXAMPLE)
    triggers = [edge for edge in payload["edges"]
                if edge.get("type") == "trigger"]

    assert len(triggers) == 1, (
        f"expected exactly one trigger edge, got "
        f"{[edge.get('id') for edge in triggers]}")
    assert triggers[0]["sourceHandle"] == "trigger"
    assert (triggers[0]["source"], triggers[0]["target"]) == (
        "start", "sentences")
