"""The gallery metadata: what a graph file may declare, and what the list serves.

A shipped example says where it belongs in the gallery through an optional
top-level ``gallery`` block::

    "gallery": { "main_tag": "training", "tags": ["vision", "cnn"], "order": 30 }

``GET /api/examples/list`` turns that into keys that are ALWAYS present --
``main_tag``, ``tags``, ``order``, ``family`` and the deprecated ``section``
(equal to ``main_tag``) -- so the frontend groups on one shape instead of on
whether a key exists. Anything it cannot read degrades on its own: the list
is built by walking every installed pack's examples directory, and a
third-party pack with a typo in its metadata must not take the gallery down
with a 500. An example with no readable main tag is filed under ``plugin``
(pack) or ``other`` (built-in); the built-in table below keeps ``other``
empty.

Two halves, in one file because they are two ends of the same contract: the
route, checked against a temporary examples directory holding the shapes a
hand-written file can really have, and the built-ins, checked against the
table this phase assigned them. The table is written out literally rather
than derived -- it IS the contract, and a derivation would agree with
whatever the files happen to say.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import settings
from app.core.example_tags import (
    DECLARABLE_MAIN_TAGS,
    MAIN_TAGS,
    MAX_TAGS,
    is_valid_tag,
)

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


async def test_a_full_gallery_block_is_served_with_the_main_tag_first(
    test_client, examples_root
):
    _write_example(examples_root, "Arch/ResNet",
                   gallery={"main_tag": "architectures", "family": "CNN",
                            "tags": ["vision", "cnn"], "order": 20})

    entry = (await _builtin_listing(test_client))["Arch/ResNet"]
    assert (entry["main_tag"], entry["tags"], entry["family"], entry["order"]) == (
        "architectures", ["architectures", "vision", "cnn"], "CNN", 20)
    # Deprecated, kept for older frontends and plugins: equal to main_tag.
    assert entry["section"] == "architectures"


async def test_the_keys_are_present_even_with_no_gallery_block(
    test_client, examples_root
):
    """Always present, so the frontend groups on values and not on keys.

    A built-in with nothing readable lands in ``other``; the built-in table
    below is what keeps that category empty in the shipped gallery.
    """
    _write_example(examples_root, "Plain/Example")

    entry = (await _builtin_listing(test_client))["Plain/Example"]
    assert entry["main_tag"] == "other"
    assert entry["tags"] == ["other"]
    assert entry["section"] == "other"
    assert entry["family"] is None
    assert entry["order"] is None


async def test_a_file_with_only_the_old_section_key_still_sorts_into_it(
    test_client, examples_root
):
    """Third-party packs written before tags keep their category unchanged."""
    _write_example(examples_root, "Legacy/Example",
                   gallery={"section": "training", "order": 5})

    entry = (await _builtin_listing(test_client))["Legacy/Example"]
    assert (entry["main_tag"], entry["tags"], entry["section"], entry["order"]) == (
        "training", ["training"], "training", 5)


async def test_main_tag_wins_over_an_old_section(test_client, examples_root):
    _write_example(examples_root, "Both/Example",
                   gallery={"main_tag": "evaluation", "section": "concepts"})

    entry = (await _builtin_listing(test_client))["Both/Example"]
    assert entry["main_tag"] == "evaluation"


async def test_tags_are_deduplicated_validated_and_capped(test_client, examples_root):
    """A bad tag costs that tag alone; the main tag is never listed twice."""
    _write_example(examples_root, "Tagged/Example", gallery={
        "main_tag": "data",
        "tags": ["vision", "Vision", "data", "a b", "", 5, "x" * 25, "vision",
                 "one", "two", "three", "four", "five", "six"]})

    entry = (await _builtin_listing(test_client))["Tagged/Example"]
    assert entry["tags"] == ["data", "vision", "one", "two", "three", "four"]
    assert len(entry["tags"]) - 1 <= MAX_TAGS


#: One unreadable ``gallery`` block per row, and the fields it costs.
#:
#: Every one of these is a shape a hand-edited file or a third-party pack
#: can really ship. The rule under test is that a bad field degrades to
#: ``null`` ALONE -- the block's other fields survive, the example stays in
#: the list, and nothing 500s.
_DEGRADES = [
    pytest.param(
        {"section": "advanced", "order": 1},
        {"main_tag": "other", "family": None, "order": 1},
        id="unknown-section",
    ),
    pytest.param(
        # Server-assigned fallbacks are not values an author may declare.
        {"main_tag": "plugin", "order": 1},
        {"main_tag": "other", "family": None, "order": 1},
        id="declared-fallback",
    ),
    pytest.param(
        {"main_tag": "training", "tags": "vision", "order": 1},
        {"main_tag": "training", "tags": ["training"], "order": 1},
        id="tags-not-a-list",
    ),
    pytest.param(
        # ``True`` is an ``int`` in Python and would sort as 1; a graph file
        # saying ``"order": true`` means nothing, so it says nothing.
        {"section": "training", "order": True},
        {"main_tag": "training", "family": None, "order": None},
        id="boolean-order",
    ),
    pytest.param(
        {"section": "training", "order": "3"},
        {"main_tag": "training", "family": None, "order": None},
        id="string-order",
    ),
    pytest.param(
        {"main_tag": "architectures", "family": 5, "order": 1},
        {"main_tag": "architectures", "family": None, "order": 1},
        id="numeric-family",
    ),
    pytest.param(
        # Not a dict at all: the block is unreadable as a whole.
        [],
        {"main_tag": "other", "family": None, "order": None},
        id="list-instead-of-object",
    ),
    pytest.param(
        "training",
        {"main_tag": "other", "family": None, "order": None},
        id="string-instead-of-object",
    ),
    pytest.param(
        # An unhashable section: the membership test against the vocabulary
        # has to survive it, which a bare ``in`` on a set does not.
        {"section": ["training"], "order": 2},
        {"main_tag": "other", "family": None, "order": 2},
        id="list-section",
    ),
    pytest.param(
        {"main_tag": "other", "family": None, "order": None},
        {"main_tag": "other", "family": None, "order": None},
        id="explicit-nulls",
    ),
]


@pytest.mark.parametrize("block, expected", _DEGRADES)
async def test_an_unreadable_field_degrades_on_its_own(
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
    assert listing["Fine/Example"]["main_tag"] == "training"
    if expected is None:
        assert "Broken/Example" not in listing
    else:
        entry = listing["Broken/Example"]
        assert (entry["node_count"], entry["edge_count"]) == expected


# ── the built-ins ─────────────────────────────────────────────────────────

#: Where every shipped example belongs:
#: ``path -> (main_tag, order, tags, family)``.
#:
#: Literal on purpose. This table is the classification decision itself --
#: what somebody uses the example for, not which folder it sits in (the
#: folders are the examples' ids and do not move) -- so a change to a file
#: has to be a change here too. Approved in
#: plans/agentic-workbench/reviews/phase-3-plan.md. ``family`` is ``None``
#: outside ``architectures``, the one main tag that is sub-grouped.
_SHIPPED: dict[str, tuple[str, int, tuple[str, ...], str | None]] = {
    "Usage_Example/CNN-MNIST/TrainCNN-MNIST": ("quickstart", 10, ("vision", "cnn", "mnist"), None),
    "Usage_Example/CNN-MNIST/InferenceCNN-MNIST": ("quickstart", 20, ("vision", "cnn", "inference"), None),
    "Classical/Iris-Sklearn-KNN": ("quickstart", 30, ("tabular", "knn", "iris"), None),
    "Usage_Example/Api-Function": ("quickstart", 40, ("api",), None),
    "Usage_Example/HuggingFace-Dataset/TrainCNN-Beans": ("training", 10, ("vision", "cnn", "huggingface"), None),
    "Usage_Example/ResNet-CIFAR10/TrainResNet-CIFAR10": ("training", 20, ("vision", "resnet", "cifar10"), None),
    "Usage_Example/Resume-From-Checkpoint": ("training", 40, ("tabular", "checkpoint", "mlp"), None),
    "Usage_Example/Augment-Then-Train": ("training", 50, ("vision", "augmentation", "mnist"), None),
    "Usage_Example/ResNet18-CIFAR10-Baseline": ("training", 70, ("vision", "resnet", "cifar10", "baseline"), None),
    "Usage_Example/Segmentation-Synthetic-Shapes": ("training", 80, ("vision", "segmentation", "unet"), None),
    "Usage_Example/GPT-Mini/TrainGPT-Mini": ("training", 90, ("vision", "transformer", "attention"), None),
    "Diffusion/Train-DDPM-Blobs": ("training", 100, ("diffusion", "ddpm", "generation"), None),
    "RNN/LSTM-Recall-First-Token": ("training", 110, ("nlp", "lstm", "sequence"), None),
    "VLA/TrainVLA-PushWorld": ("training", 120, ("vla", "imitation", "robotics"), None),
    "LLM/TrainCausalLM-TinyStories": ("llm", 10, ("nlp", "pretraining", "transformer"), None),
    "LLM/RAG-Local-Offline": ("llm", 30, ("nlp", "rag", "retrieval"), None),
    "LLM/RAG-LLMChat-API": ("llm", 50, ("nlp", "rag", "api"), None),
    "LLM/Data-Mix-Ratios": ("llm", 60, ("nlp", "data-mix", "pretraining"), None),
    "LLM/Sentence-Similarity-zhTW": ("llm", 70, ("nlp", "embeddings", "zh-tw"), None),
    "LLM/Word-Embedding-Analogy": ("llm", 80, ("nlp", "embeddings"), None),
    "Classical/Tabular-Iris-Pipeline": ("data", 30, ("tabular", "preprocessing", "iris"), None),
    "Others/Image-Batch-IO": ("data", 40, ("vision", "io"), None),
    "Classical/KFold-Cross-Validation": ("evaluation", 50, ("tabular", "cross-validation", "regression"), None),
    "Classical/Classifier-Showdown-Moons": ("evaluation", 60, ("tabular", "sklearn", "decision-boundary"), None),
    "Model_Architecture/ResNet-SkipConnection-CNN": ("architectures", 10, ("vision", "cnn", "resnet"), "CNN"),
    "Model_Architecture/UNet-Segmentation-CNN": ("architectures", 20, ("vision", "cnn", "segmentation"), "CNN"),
    "Model_Architecture/EfficientNet-CNN": ("architectures", 30, ("vision", "cnn"), "CNN"),
    "Model_Architecture/ConvNeXt-CNN": ("architectures", 40, ("vision", "cnn"), "CNN"),
    "Model_Architecture/DiT-Diffusion-Transformer": ("architectures", 10, ("diffusion", "transformer"), "Diffusion"),
    "Model_Architecture/DQN-Atari-RL": ("architectures", 10, ("rl", "dqn", "cnn"), "RL"),
    "Model_Architecture/PPO-Robotics-RL": ("architectures", 20, ("rl", "ppo", "robotics"), "RL"),
    "Model_Architecture/TimeSeries-LSTM-RNN": ("architectures", 10, ("timeseries", "lstm"), "RNN"),
    "Model_Architecture/BiGRU-SpeechRecognition-RNN": ("architectures", 20, ("audio", "gru"), "RNN"),
    "Model_Architecture/Seq2Seq-Attention-RNN": ("architectures", 30, ("nlp", "rnn", "attention"), "RNN"),
    "Model_Architecture/GPT-DecoderOnly-Transformer": ("architectures", 10, ("nlp", "transformer", "decoder"), "Transformer"),
    "Model_Architecture/BERT-Encoder-Transformer": ("architectures", 20, ("nlp", "transformer", "encoder"), "Transformer"),
    "Model_Architecture/LLaMA-Decoder-Transformer": ("architectures", 30, ("nlp", "transformer", "decoder"), "Transformer"),
    "Model_Architecture/ViT-ImageClassifier-Transformer": ("architectures", 40, ("vision", "transformer"), "Transformer"),
    "Model_Architecture/SwinTransformer-Transformer": ("architectures", 50, ("vision", "transformer"), "Transformer"),
    "Others/Tensor-Ops-Tour": ("concepts", 10, ("vision", "tensors"), None),
    "Others/Reduce-And-Switch": ("concepts", 20, ("tabular", "dataflow"), None),
    "Others/One-Backward-Pass": ("concepts", 30, ("nlp", "gradients", "embeddings"), None),
    "RNN/RNN-OneStep": ("concepts", 40, ("nlp", "rnn"), None),
    "RNN/Conv1d-Sequence-Classifier": ("concepts", 50, ("nlp", "cnn", "sequence"), None),
    "Transformer/Causal-Attention-From-Scratch": ("concepts", 60, ("nlp", "attention", "transformer"), None),
    "Transformer/MoE-TopK-Routing": ("concepts", 70, ("nlp", "moe", "transformer"), None),
    "Diffusion/Forward-Process": ("concepts", 80, ("diffusion", "noise"), None),
    "Diffusion/Toy-Sampling": ("concepts", 90, ("diffusion", "sampling"), None),
    "Diffusion/Mini-UNet-Compact": ("concepts", 100, ("diffusion", "unet"), None),
    "RL/Discounted-Returns": ("concepts", 110, ("rl", "returns"), None),
    "RL/Reward-Model-Bradley-Terry": ("concepts", 120, ("rl", "rlhf", "reward-model"), None),
    "RL/RLHF-Reward-and-KL": ("concepts", 130, ("rl", "rlhf", "kl"), None),
    "RL/Reward-Hacking-Shortcut": ("concepts", 140, ("rl", "reward-model", "generalization"), None),
    "RL/GRPO-Step-GridWorld": ("concepts", 150, ("rl", "grpo"), None),
    "Papers/Dropout-Srivastava-2014": ("papers", 10, ("vision", "regularization", "dropout"), None),
    "Papers/BatchNorm-Ioffe-2015": ("papers", 20, ("vision", "normalization"), None),
    "Papers/ResNet-Degradation-He-2016": ("papers", 30, ("vision", "resnet", "depth"), None),
    "Papers/Adam-Kingma-2015": ("papers", 40, ("vision", "optimization"), None),
    "Papers/Label-Smoothing-Szegedy-2016": ("papers", 50, ("vision", "regularization"), None),
    "Papers/SGDR-Loshchilov-2017": ("papers", 60, ("vision", "lr-schedule"), None),
    "Papers/Vanishing-Gradients-Glorot-2010": ("papers", 70, ("vision", "initialization", "gradients"), None),
    "Papers/GroupNorm-Wu-2018": ("papers", 80, ("vision", "normalization"), None),
    "Papers/Gradient-Clipping-RNN": ("papers", 90, ("nlp", "rnn", "gradients"), None),
    "Papers/Autoencoder-MNIST-2D": ("papers", 100, ("vision", "autoencoder", "embeddings"), None),
    "Papers/LSTM-Forecast-Sine": ("papers", 110, ("timeseries", "lstm"), None),
    "Papers/Transformer-Sort": ("papers", 120, ("nlp", "transformer", "attention"), None),
    "Papers/Weight-Decay-Regression": ("papers", 130, ("tabular", "regularization", "regression"), None),
    "Papers/Early-Stopping-Regression": ("papers", 140, ("tabular", "early-stopping", "regression"), None),
    "Papers/REINFORCE-CartPole": ("papers", 150, ("rl", "policy-gradient"), None),
    "Papers/DQN-CartPole": ("papers", 160, ("rl", "dqn"), None),
    "Papers/PPO-CartPole": ("papers", 170, ("rl", "ppo"), None),
    "Papers/QLearning-GridWorld": ("papers", 180, ("rl", "q-learning"), None),
}

#: The domain tags; every built-in names at least one of them.
_DOMAINS = frozenset(
    {"vision", "nlp", "tabular", "rl", "diffusion", "audio", "timeseries", "vla"})

#: Built-ins exempt from the domain rule, with the reason. Api-Function
#: echoes a string through the run API and has no data domain.
_NO_DOMAIN = {"Usage_Example/Api-Function"}


def _shipped_gallery() -> dict[str, dict]:
    """Every built-in's ``gallery`` block, keyed by its API path."""
    return {
        path.parent.relative_to(_EXAMPLES_ROOT).as_posix():
            json.loads(path.read_text(encoding="utf-8")).get("gallery")
        for path in sorted(_EXAMPLES_ROOT.rglob("graph.json"))
    }


def test_the_main_tag_vocabulary_and_its_order():
    """The order is the gallery's order: practical first, papers and courses
    after every practical category. Changing it is a product decision."""
    assert MAIN_TAGS == (
        "quickstart", "training", "llm", "data", "evaluation", "research",
        "deploy", "architectures", "concepts", "papers", "courses",
        "plugin", "other")
    assert MAIN_TAGS.index("papers") > max(
        MAIN_TAGS.index(t) for t in
        ("quickstart", "training", "llm", "data", "evaluation", "research", "deploy"))


def test_every_builtin_declares_what_the_table_gives_it():
    """The files and the table say the same thing, both ways round.

    A missing row is as much a failure as a missing block: an example added
    without a decision about where it belongs would otherwise land silently
    in "other", at the bottom of the gallery.
    """
    shipped = _shipped_gallery()
    assert set(shipped) == set(_SHIPPED), (
        f"only on disk: {sorted(set(shipped) - set(_SHIPPED))}; "
        f"only in the table: {sorted(set(_SHIPPED) - set(shipped))}")

    actual = {}
    for path, block in shipped.items():
        block = block if isinstance(block, dict) else {}
        actual[path] = (block.get("main_tag"), block.get("order"),
                        tuple(block.get("tags") or ()), block.get("family"))
    assert actual == _SHIPPED


def test_every_builtin_uses_the_new_schema_only():
    """A declarable main tag, legal tags, an int order, and no ``section``."""
    for path, block in _shipped_gallery().items():
        assert isinstance(block, dict), f"{path} has no gallery block"
        assert "section" not in block, f"{path} still declares the old section"
        assert block.get("main_tag") in DECLARABLE_MAIN_TAGS, (
            f"{path} declares main_tag {block.get('main_tag')!r}")
        tags = block.get("tags")
        assert isinstance(tags, list) and 1 <= len(tags) <= MAX_TAGS, path
        assert all(is_valid_tag(tag) for tag in tags), (path, tags)
        assert len(set(tags)) == len(tags), (path, tags)
        assert not set(tags) & set(MAIN_TAGS), (
            f"{path} uses a main tag as a free tag: {set(tags) & set(MAIN_TAGS)}")
        assert isinstance(block.get("order"), int) and not isinstance(
            block.get("order"), bool), path


def test_every_builtin_names_a_domain():
    for path, block in _shipped_gallery().items():
        if path in _NO_DOMAIN:
            continue
        assert set(block["tags"]) & _DOMAINS, f"{path} names no domain tag: {block['tags']}"


def test_only_architectures_examples_carry_a_family():
    """``family`` is the sub-header inside ``architectures`` and nothing else."""
    for path, block in _shipped_gallery().items():
        family = (block or {}).get("family")
        if (block or {}).get("main_tag") == "architectures":
            assert isinstance(family, str) and family, (
                f"{path} is an architecture example with no family")
        else:
            assert family is None, f"{path} carries a family: {family!r}"


async def test_the_list_route_serves_what_the_shipped_files_declare(
    test_client,
):
    """Real files, real route, no fixture; ``other`` is empty."""
    listing = await _builtin_listing(test_client)
    served = {
        path: (entry["main_tag"], entry["order"], tuple(entry["tags"][1:]), entry["family"])
        for path, entry in listing.items()
    }
    assert served == _SHIPPED
    assert not [path for path, entry in listing.items() if entry["main_tag"] == "other"]


async def test_the_list_is_in_gallery_order(test_client):
    """Main-tag rank, then order; papers come after every practical example."""
    response = await test_client.get("/api/examples/list")
    entries = [e for e in response.json() if e["source"] == "builtin"]
    keys = [(MAIN_TAGS.index(e["main_tag"]), e["order"]) for e in entries]
    assert keys == sorted(keys)
    first_paper = next(i for i, e in enumerate(entries) if e["main_tag"] == "papers")
    assert all(e["main_tag"] not in {"quickstart", "training", "llm", "data",
                                     "evaluation", "research", "deploy"}
               for e in entries[first_paper:])


async def test_the_tags_route_counts_main_tags_and_free_tags(test_client):
    response = await test_client.get("/api/examples/tags")
    assert response.status_code == 200, response.text
    body = response.json()
    main = body["main_tags"]
    assert [m["id"] for m in main] == [t for t in MAIN_TAGS if t in {m["id"] for m in main}]
    assert [m["order"] for m in main] == sorted(m["order"] for m in main)
    assert all(m["count"] > 0 for m in main)
    assert "other" not in {m["id"] for m in main}
    listing = (await test_client.get("/api/examples/list")).json()
    counts = {}
    for entry in listing:
        counts[entry["main_tag"]] = counts.get(entry["main_tag"], 0) + 1
    assert {m["id"]: m["count"] for m in main} == counts
    tags = body["tags"]
    assert tags == sorted(tags, key=lambda t: (-t["count"], t["id"]))
    assert not {t["id"] for t in tags} & set(MAIN_TAGS)
    assert sum(t["count"] for t in tags) == sum(len(e["tags"]) - 1 for e in listing)


async def test_a_plugin_example_without_metadata_is_filed_under_plugin(
    test_client, tmp_path, monkeypatch
):
    """And one with only an old ``section`` sorts into that section."""
    from app.api import routes_examples

    pack = tmp_path / "pack"
    for rel, gallery in [("A/NoMeta", None), ("B/Legacy", {"section": "llm", "order": 1})]:
        directory = pack / "examples" / rel
        directory.mkdir(parents=True)
        payload = {"name": rel, "description": "", "nodes": [], "edges": []}
        if gallery:
            payload["gallery"] = gallery
        (directory / "graph.json").write_bytes(json.dumps(payload).encode("utf-8"))
    monkeypatch.setattr(routes_examples, "iter_plugin_dirs",
                        lambda *args, **kwargs: [("thirdparty", pack)])
    response = await test_client.get("/api/examples/list")
    by_path = {e["path"]: e for e in response.json()}
    assert by_path["plugin:thirdparty/A/NoMeta"]["main_tag"] == "plugin"
    assert by_path["plugin:thirdparty/B/Legacy"]["main_tag"] == "llm"


def test_no_two_builtins_claim_the_same_place():
    """``(main_tag, family, order)`` is a position, and positions are unique."""
    seen: dict[tuple[str, str | None, int], str] = {}
    for path, (main_tag, order, _tags, family) in sorted(_SHIPPED.items()):
        slot = (main_tag, family, order)
        assert slot not in seen, f"{path} and {seen[slot]} both claim {slot}"
        seen[slot] = path


def test_every_builtin_pack_example_is_a_course():
    """The packs shipped in this repository are course material: ``courses``,
    legal tags, an int order, and no old ``section``."""
    pack_graphs = sorted((_REPO_ROOT / "plugins").glob("*/examples/**/graph.json"))
    assert pack_graphs
    for path in pack_graphs:
        block = json.loads(path.read_text(encoding="utf-8")).get("gallery")
        assert isinstance(block, dict), f"{path} has no gallery block"
        assert block.get("main_tag") == "courses", path
        assert "section" not in block, path
        assert block.get("tags") and all(is_valid_tag(t) for t in block["tags"]), path
        assert not set(block["tags"]) & set(MAIN_TAGS), path
        assert isinstance(block.get("order"), int), path
