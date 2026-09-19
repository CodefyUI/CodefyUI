"""The gallery metadata: what a graph file may declare, and what the list serves.

A shipped example says where it belongs in the gallery through an optional
top-level ``gallery`` block::

    "gallery": { "section": "architectures", "family": "CNN", "order": 2 }

``GET /api/examples/list`` turns that into three keys that are ALWAYS
present -- ``section``, ``family``, ``order`` -- so the frontend groups on
one shape instead of on whether a key exists. Anything it cannot read
degrades to ``null`` for that field and for that field only: the list is
built by walking every installed pack's examples directory, and a
third-party pack with a typo in its metadata must not take the gallery down
with a 500.

Two halves, in one file because they are two ends of the same contract: the
route, checked against a temporary examples directory holding the shapes a
hand-written file can really have, and the 35 built-ins, checked against the
section table this wave assigned them. The table is written out literally
rather than derived -- it IS the contract, and a derivation would agree with
whatever the files happen to say.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.api.routes_examples import GALLERY_SECTIONS
from app.config import settings

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _REPO_ROOT / "examples"


# ── the route ─────────────────────────────────────────────────────────────

@pytest.fixture
def examples_root(tmp_path, monkeypatch) -> Path:
    """An examples directory of this test's own, in place of the shipped one."""
    root = tmp_path / "examples"
    root.mkdir()
    monkeypatch.setattr(settings, "EXAMPLES_DIR", root)
    return root


def _write_example(root: Path, rel: str, *, nodes=None, edges=None, **extra):
    """Write ``<root>/<rel>/graph.json``. ``extra`` lands at the top level."""
    directory = root / rel
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": rel,
        "description": "",
        **extra,
        "nodes": nodes if nodes is not None else [],
        "edges": edges if edges is not None else [],
    }
    (directory / "graph.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")


async def _builtin_listing(test_client) -> dict[str, dict]:
    """``GET /api/examples/list``, the built-in half, keyed by path.

    The installed packs' examples are filtered out rather than prevented:
    what they contribute is a separate question, and the answer must not
    depend on which packs the machine running the suite has.
    """
    response = await test_client.get("/api/examples/list")
    assert response.status_code == 200, response.text
    return {item["path"]: item
            for item in response.json() if item["source"] == "builtin"}


async def test_a_full_gallery_block_is_served_verbatim(
    test_client, examples_root
):
    _write_example(examples_root, "Arch/ResNet",
                   gallery={"section": "architectures", "family": "CNN",
                            "order": 2})

    entry = (await _builtin_listing(test_client))["Arch/ResNet"]
    assert (entry["section"], entry["family"], entry["order"]) == (
        "architectures", "CNN", 2)


async def test_the_three_keys_are_present_even_with_no_gallery_block(
    test_client, examples_root
):
    """Always present, so the frontend groups on values and not on keys."""
    _write_example(examples_root, "Plain/Example")

    entry = (await _builtin_listing(test_client))["Plain/Example"]
    assert entry["section"] is None
    assert entry["family"] is None
    assert entry["order"] is None


#: One unreadable ``gallery`` block per row, and the fields it costs.
#:
#: Every one of these is a shape a hand-edited file or a third-party pack
#: can really ship. The rule under test is that a bad field degrades to
#: ``null`` ALONE -- the block's other fields survive, the example stays in
#: the list, and nothing 500s.
_DEGRADES = [
    pytest.param(
        {"section": "advanced", "order": 1},
        {"section": None, "family": None, "order": 1},
        id="unknown-section",
    ),
    pytest.param(
        # ``True`` is an ``int`` in Python and would sort as 1; a graph file
        # saying ``"order": true`` means nothing, so it says nothing.
        {"section": "training", "order": True},
        {"section": "training", "family": None, "order": None},
        id="boolean-order",
    ),
    pytest.param(
        {"section": "training", "order": "3"},
        {"section": "training", "family": None, "order": None},
        id="string-order",
    ),
    pytest.param(
        {"section": "architectures", "family": 5, "order": 1},
        {"section": "architectures", "family": None, "order": 1},
        id="numeric-family",
    ),
    pytest.param(
        # Not a dict at all: the block is unreadable as a whole.
        [],
        {"section": None, "family": None, "order": None},
        id="list-instead-of-object",
    ),
    pytest.param(
        "training",
        {"section": None, "family": None, "order": None},
        id="string-instead-of-object",
    ),
    pytest.param(
        # An unhashable section: the membership test against the vocabulary
        # has to survive it, which a bare ``in`` on a set does not.
        {"section": ["training"], "order": 2},
        {"section": None, "family": None, "order": 2},
        id="list-section",
    ),
    pytest.param(
        {"section": None, "family": None, "order": None},
        {"section": None, "family": None, "order": None},
        id="explicit-nulls",
    ),
]


@pytest.mark.parametrize("block, expected", _DEGRADES)
async def test_an_unreadable_field_degrades_to_null_on_its_own(
    test_client, examples_root, block, expected
):
    _write_example(examples_root, "Odd/Example", gallery=block)

    entry = (await _builtin_listing(test_client))["Odd/Example"]
    assert {key: entry[key] for key in expected} == expected


async def test_node_count_does_not_count_the_notes(test_client, examples_root):
    """The card's "N nodes" is how much graph there is, not how much prose.

    A note is an annotation in the node list (see ``test_graph_notes.py``),
    and the examples are about to carry several each. Counting them would
    make the best-explained example look like the most complicated one.
    """
    _write_example(
        examples_root, "Counted/Example",
        nodes=[
            {"id": "start", "type": "Start", "data": {"params": {}}},
            {"id": "print", "type": "Print", "data": {"params": {}}},
            {"id": "why", "type": "note", "data": {"noteContent": "because"}},
            {"id": "how", "type": "note", "data": {"noteContent": "like so"}},
        ],
        edges=[
            {"id": "e", "source": "start", "target": "print",
             "sourceHandle": "trigger", "type": "trigger"},
        ],
    )

    entry = (await _builtin_listing(test_client))["Counted/Example"]
    assert entry["node_count"] == 2
    assert entry["edge_count"] == 1


def _write_raw(root: Path, rel: str, text: str) -> None:
    """Write ``<root>/<rel>/graph.json`` verbatim, JSON or not.

    ``_write_example`` can only produce files this route can already read.
    The shapes below are the ones a hand-edited file really has.
    """
    directory = root / rel
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "graph.json").write_text(text, encoding="utf-8")


#: A readable JSON file that is not a graph, and what the list does with it.
#:
#: ``None`` means the entry is dropped: nothing sensible can be said about a
#: file whose root is not an object, and inventing a name for it would put a
#: card in the gallery that opens onto nothing.
_UNREADABLE_GRAPHS = [
    pytest.param("[]", None, id="root-is-a-list"),
    pytest.param('"a graph"', None, id="root-is-a-string"),
    pytest.param("null", None, id="root-is-null"),
    pytest.param(
        '{"name": "Nulls", "description": "", "nodes": null, "edges": null}',
        (0, 0),
        id="null-node-and-edge-lists",
    ),
    pytest.param(
        '{"name": "Nulls", "description": "", "nodes": {}, "edges": "none"}',
        (0, 0),
        id="lists-that-are-not-lists",
    ),
]


@pytest.mark.parametrize("text, expected", _UNREADABLE_GRAPHS)
async def test_a_graph_file_that_is_not_a_graph_costs_only_its_own_card(
    test_client, examples_root, text, expected
):
    """One pack's typo must not take everybody's gallery down with a 500.

    The route walks every installed pack's examples directory into a single
    list, so a file it cannot read is not that pack's problem alone -- an
    exception here is the whole gallery, for everyone. Only the malformed
    entry is affected; the example beside it still lists.
    """
    _write_raw(examples_root, "Broken/Example", text)
    _write_example(examples_root, "Fine/Example",
                   gallery={"section": "training", "order": 1})

    listing = await _builtin_listing(test_client)
    assert listing["Fine/Example"]["section"] == "training"
    if expected is None:
        assert "Broken/Example" not in listing
    else:
        entry = listing["Broken/Example"]
        assert (entry["node_count"], entry["edge_count"]) == expected


# ── the 35 built-ins ──────────────────────────────────────────────────────

#: Where every shipped example belongs: ``path -> (section, order, family)``.
#:
#: Literal on purpose. This table is the classification decision itself --
#: purpose first, model family second -- so a change to a file has to be a
#: change here too, and the 35 rows are the only record of what was agreed.
#: ``family`` is ``None`` outside ``architectures``, the one section that is
#: sub-grouped.
#:
#: ``training`` starts at 2: order 1 is reserved for the Hugging Face
#: training example that lands with the example content.
_SHIPPED_SECTIONS: dict[str, tuple[str, int, str | None]] = {
    "Usage_Example/CNN-MNIST/TrainCNN-MNIST": ("quickstart", 1, None),
    "Usage_Example/CNN-MNIST/InferenceCNN-MNIST": ("quickstart", 2, None),
    "Usage_Example/Api-Function": ("quickstart", 3, None),
    "Usage_Example/ResNet-CIFAR10/TrainResNet-CIFAR10": ("training", 2, None),
    "Usage_Example/GPT-Mini/TrainGPT-Mini": ("training", 3, None),
    "Usage_Example/ResNet18-CIFAR10-Baseline": ("training", 4, None),
    "LLM/TrainCausalLM-TinyStories": ("training", 5, None),
    "VLA/TrainVLA-PushWorld": ("training", 6, None),
    "LLM/Word-Embedding-Analogy": ("llm", 1, None),
    "LLM/Sentence-Similarity-zhTW": ("llm", 2, None),
    "LLM/RAG-Local-Offline": ("llm", 3, None),
    "LLM/RAG-LLMChat-API": ("llm", 4, None),
    "Classical/Tabular-Iris-Pipeline": ("concepts", 1, None),
    "Classical/Iris-Sklearn-KNN": ("concepts", 2, None),
    "RNN/RNN-OneStep": ("concepts", 3, None),
    "Transformer/MoE-TopK-Routing": ("concepts", 4, None),
    "Diffusion/Forward-Process": ("concepts", 5, None),
    "Diffusion/Toy-Sampling": ("concepts", 6, None),
    "Diffusion/Mini-UNet-Compact": ("concepts", 7, None),
    "RL/RLHF-Reward-and-KL": ("concepts", 8, None),
    "Model_Architecture/ResNet-SkipConnection-CNN": ("architectures", 1, "CNN"),
    "Model_Architecture/UNet-Segmentation-CNN": ("architectures", 2, "CNN"),
    "Model_Architecture/EfficientNet-CNN": ("architectures", 3, "CNN"),
    "Model_Architecture/ConvNeXt-CNN": ("architectures", 4, "CNN"),
    "Model_Architecture/TimeSeries-LSTM-RNN": ("architectures", 1, "RNN"),
    "Model_Architecture/BiGRU-SpeechRecognition-RNN": ("architectures", 2, "RNN"),
    "Model_Architecture/Seq2Seq-Attention-RNN": ("architectures", 3, "RNN"),
    "Model_Architecture/GPT-DecoderOnly-Transformer":
        ("architectures", 1, "Transformer"),
    "Model_Architecture/BERT-Encoder-Transformer":
        ("architectures", 2, "Transformer"),
    "Model_Architecture/LLaMA-Decoder-Transformer":
        ("architectures", 3, "Transformer"),
    "Model_Architecture/ViT-ImageClassifier-Transformer":
        ("architectures", 4, "Transformer"),
    "Model_Architecture/SwinTransformer-Transformer":
        ("architectures", 5, "Transformer"),
    "Model_Architecture/DiT-Diffusion-Transformer":
        ("architectures", 1, "Diffusion"),
    "Model_Architecture/DQN-Atari-RL": ("architectures", 1, "RL"),
    "Model_Architecture/PPO-Robotics-RL": ("architectures", 2, "RL"),
}


def _shipped_gallery() -> dict[str, dict]:
    """Every built-in's ``gallery`` block, keyed by its API path."""
    return {
        path.parent.relative_to(_EXAMPLES_ROOT).as_posix():
            json.loads(path.read_text(encoding="utf-8")).get("gallery")
        for path in sorted(_EXAMPLES_ROOT.rglob("graph.json"))
    }


def test_the_section_vocabulary_is_the_five_the_contract_names():
    """Widening it is a frontend change too, so it cannot happen quietly."""
    assert GALLERY_SECTIONS == frozenset(
        {"quickstart", "training", "llm", "concepts", "architectures"})


def test_every_builtin_declares_the_section_the_table_gives_it():
    """The files and the table say the same thing, both ways round.

    A missing row is as much a failure as a missing block: an example added
    without a decision about where it belongs would otherwise land silently
    in "other", at the bottom of the gallery, which is exactly the outcome
    this wave exists to end.
    """
    shipped = _shipped_gallery()
    assert set(shipped) == set(_SHIPPED_SECTIONS), (
        f"only on disk: {sorted(set(shipped) - set(_SHIPPED_SECTIONS))}; "
        f"only in the table: {sorted(set(_SHIPPED_SECTIONS) - set(shipped))}")

    actual = {}
    for path, block in shipped.items():
        block = block if isinstance(block, dict) else {}
        actual[path] = (
            block.get("section"), block.get("order"), block.get("family"))
    assert actual == _SHIPPED_SECTIONS


def test_every_builtin_section_is_one_the_route_will_serve():
    """A typo here degrades to ``null`` at the route and shows up in "other"."""
    for path, block in _shipped_gallery().items():
        assert isinstance(block, dict), f"{path} has no gallery block"
        assert block.get("section") in GALLERY_SECTIONS, (
            f"{path} declares section {block.get('section')!r}, which the "
            f"list route does not serve")
        assert isinstance(block.get("order"), int), (
            f"{path} declares order {block.get('order')!r}")


def test_only_architectures_examples_carry_a_family():
    """``family`` is the sub-header inside ``architectures`` and nothing else.

    A family on an example in another section is invisible -- no surface
    reads it -- so it would sit in the file looking like it did something.
    """
    for path, block in _shipped_gallery().items():
        family = (block or {}).get("family")
        if (block or {}).get("section") == "architectures":
            assert isinstance(family, str) and family, (
                f"{path} is an architecture example with no family, so it "
                f"would render under no sub-header")
        else:
            assert family is None, f"{path} carries a family: {family!r}"


async def test_the_list_route_serves_what_the_shipped_files_declare(
    test_client,
):
    """The two halves above meet: real files, real route, no fixture.

    Everything else in this file reads the graphs off disk or the route out
    of a temporary directory, so a scan that mangled a nested path -- the
    only built-ins two directories deep are the two CNN-MNIST examples --
    would pass both.
    """
    listing = await _builtin_listing(test_client)
    served = {
        path: (entry["section"], entry["order"], entry["family"])
        for path, entry in listing.items()
    }
    assert served == _SHIPPED_SECTIONS


def test_no_two_builtins_claim_the_same_place():
    """``(section, family, order)`` is a position, and positions are unique.

    Ties are broken by server order, which is alphabetical by path -- a
    stable answer, but not one anybody chose. Two examples sharing a slot
    means the ordering of at least one of them is an accident.
    """
    seen: dict[tuple[str, str | None, int], str] = {}
    for path, (section, order, family) in sorted(_SHIPPED_SECTIONS.items()):
        slot = (section, family, order)
        assert slot not in seen, (
            f"{path} and {seen[slot]} both claim {slot}")
        seen[slot] = path
