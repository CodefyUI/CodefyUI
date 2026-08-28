"""The pack-backed gallery examples, executed against the real models.

``test_rag_examples.py`` holds these graphs to their shape; this file is the
only place that asks whether they still DO what their cards claim. Both
claims are about a downloaded model and nothing else can stand in for it: a
faked encoder returns whatever the fake was written to return, so a test
built on one proves the wiring and re-states the author's assumption about
the model instead of checking it.

That makes the suite OPT-IN twice over::

    CODEFYUI_PACK_NETWORK_TESTS=1 pytest tests/test_pack_examples_real.py -q -s

The variable is the outer gate, shared with ``test_packs_network.py`` so one
switch turns on everything that needs more than an offline runner. The inner
gate is per test: each one asks ``pack_available`` for the exact item it
needs and SKIPS when it is missing, because "the maintainer has the
sentence encoder" and "the maintainer has GloVe" are different facts and
one of them being false must not hide the other test's result.

Nothing here redirects ``CODEFYUI_USER_DATA_DIR``. Every other pack test
does, to keep bytes out of the developer's cache; these read that cache on
purpose, because the question is whether what a user INSTALLED runs the
examples that user is about to open. Nothing is downloaded either way -- a
graph run never fetches pack contents (``require_pack`` refuses instead), so
a missing model skips rather than pulling half a gigabyte.

Run with ``-s`` to see the timings each test prints; the encode is a few
seconds on CPU and the whole point of ``TextEmbedding`` being in
``test_builtin_examples._SLOW_NODE_TYPES``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from app.core.graph_engine import execute_graph
from app.core.packs import pack_available

pytestmark = pytest.mark.skipif(
    os.environ.get("CODEFYUI_PACK_NETWORK_TESTS") != "1",
    reason="set CODEFYUI_PACK_NETWORK_TESTS=1 to run the pack-backed "
           "examples for real")

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_EXAMPLES_ROOT = Path(__file__).resolve().parents[2] / "examples"

_SENTENCE_EXAMPLE = (
    _EXAMPLES_ROOT / "LLM" / "Sentence-Similarity-zhTW" / "graph.json")
_ANALOGY_EXAMPLE = (
    _EXAMPLES_ROOT / "LLM" / "Word-Embedding-Analogy" / "graph.json")

#: The pack item each example is written against, spelled the way the
#: catalog spells it -- ``pack_available`` answers False for an id it does
#: not know, so a typo here reads as "not installed" and skips forever.
_SENTENCE_PACK = ("sentence-embeddings",
                  "paraphrase-multilingual-MiniLM-L12-v2")
_VECTOR_PACK = ("word-vectors", "glove-50d")

#: Which sentence is which one's partner: four pairs, in the order the
#: TextInput lists them (weather, food, the stock market, and machine
#: learning across Chinese and English).
_PARTNER = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6}


def _require(pack_id: str, item_id: str) -> None:
    if not pack_available(pack_id, item_id):
        pytest.skip(
            f"{pack_id}:{item_id} is not installed -- get it from Package "
            f"Center, or `cdui packs install {pack_id} --items {item_id}`")


def _load(graph_path: Path) -> tuple[list[dict], list[dict]]:
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    return payload["nodes"], payload["edges"]


def _execute(nodes: list[dict], edges: list[dict], label: str) -> dict:
    """Run a graph the way ``test_builtin_examples`` does, and time it.

    ``os.chdir`` for the same reason it does: example graphs name data files
    relative to ``backend/``, and a contributor running pytest from the repo
    root would otherwise hit a FileNotFoundError in a test about embeddings.
    """
    previous = Path.cwd()
    os.chdir(_BACKEND_DIR)
    started = time.monotonic()
    try:
        results = asyncio.run(
            execute_graph(nodes, edges, error_mode="fail_fast"))
    finally:
        os.chdir(previous)
    print(f"\n[{label}] executed in {time.monotonic() - started:.1f}s")
    return results


def test_sentence_similarity_example_pairs_up_for_real():
    """Every sentence's nearest neighbour after itself is its partner.

    This is the example's entire claim, and it cannot be checked any other
    way: that 今天天氣很好 and 天氣晴朗 come back adjacent while sharing
    two characters, and that a Chinese sentence about training a model lands
    next to an English one, is a property of the multilingual encoder rather
    than of the graph. The graph only decides whether that property is
    visible.

    Rank 1 is asserted as well as rank 2. It is trivially true -- every row
    is its own nearest neighbour at cosine 1.0 -- which is exactly why it is
    worth one line: a run where row i's top hit is NOT i means queries and
    keys are no longer the same tensor, and every rank-2 assertion below
    would then be testing something other than what it says.
    """
    _require(*_SENTENCE_PACK)

    nodes, edges = _load(_SENTENCE_EXAMPLE)
    results = _execute(nodes, edges, "Sentence-Similarity-zhTW")

    labels = results["embed"]["labels"]
    top_indices = results["similarity"]["top_k_indices"]
    top_labels = results["similarity"]["top_k_labels"]
    assert len(top_indices) == len(_PARTNER) == len(labels), (
        f"expected one row per sentence, got {len(top_indices)} rows for "
        f"{len(labels)} labels")

    for row, neighbours in enumerate(top_indices):
        assert neighbours[0] == row, (
            f"sentence {row} ({labels[row]!r}) is not its own nearest "
            f"neighbour; queries and keys are not the same tensor")
        assert neighbours[1] == _PARTNER[row], (
            f"sentence {row} ({labels[row]!r}) pairs with "
            f"{labels[neighbours[1]]!r} instead of "
            f"{labels[_PARTNER[row]]!r} -- the model no longer groups this "
            f"example by meaning")
        assert top_labels[row][0] == labels[row], (
            "key_labels are not the embedding's labels, so the printed "
            "top-k names the wrong sentences")


def test_wordvector_glove_analogy_is_queen():
    """king - man + woman is queen on the real 400k-word table.

    The shipped graph runs on ``demo-16d``, where the analogy is exact by
    construction -- the toy table was BUILT so that it works, so running it
    proves nothing about GloVe. Overriding the backend is what turns the
    example into a claim about real vectors, and it is the claim the card
    makes ("on glove-50d queen still wins here").
    """
    _require(*_VECTOR_PACK)

    nodes, edges = _load(_ANALOGY_EXAMPLE)
    switched = [node["id"] for node in nodes if node["type"] == "WordVector"]
    for node in nodes:
        if node["type"] == "WordVector":
            node["data"]["params"]["backend"] = "glove-50d"
    assert switched, (
        "the analogy example has no WordVector nodes left, so this test "
        "would silently run the demo table it was written to bypass")

    results = _execute(nodes, edges, "Word-Embedding-Analogy (glove-50d)")

    top_labels = results["similarity"]["top_k_labels"]
    assert top_labels[0][0] == "queen", (
        f"the analogy's top hit on glove-50d is {top_labels[0][0]!r}, not "
        f"'queen'; the whole top-k was {top_labels[0]}")
