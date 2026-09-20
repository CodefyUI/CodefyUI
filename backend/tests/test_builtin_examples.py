"""Smoke-test every builtin example by validating and (when fast) executing it.

The empty-canvas gallery offers every ``examples/**/graph.json`` as a
one-click starting point, so a broken graph wastes the user's very first
run. This suite asserts each builtin graph parses and validates, and
executes the fast ones end-to-end without error.

Mirrors ``test_chapter_examples.py`` (the plugin-example twin): graphs
containing dataset/training/weight-IO nodes are validated structurally but
skipped for execution — they ship as fully-working examples that users run
manually (and the underlying nodes have their own unit tests).

Parametrised by glob so adding a new example directory is enough — no test
code changes required.

One example gets more than the structural check: the ResNet-18/CIFAR-10
baseline (core#138) is the project's accuracy claim, and the part of it most
able to rot silently — the 70-node layer graph inside ``SequentialModel`` —
is an opaque JSON string that ``validate_graph`` cannot see into. The
short-epoch test at the bottom of this file executes that graph for real,
against a generated image folder instead of the 170 MB download. See
``test_resnet18_cifar10_baseline_short_epoch``.

The TinyStories LM pretraining example (#292) gets the same treatment for the
same reason, minus the execution: its README quotes a parameter count, an
effective batch size, two token budgets and a step count, and every one of
those is a fact about *other* nodes' params that ``validate_graph`` is blind
to. It is also the one builtin needing both the network and a 16 GB GPU --
and neither of those is on its gallery card. A card says what the graph
shows; what an example needs before it runs is written in the note on its
canvas, in both languages, which is the rule the three checks under "Where a
requirement is written" hold over all 72 shipped examples. See
``test_tinystories_lm_example_names_what_it_needs_in_its_notes`` and
``test_tinystories_lm_example_still_describes_itself``.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from pathlib import Path

import pytest

from app.core.graph_engine import execute_graph, validate_graph

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _REPO_ROOT / "examples"

#: The second root the gallery draws cards from. Only the prose rules at the
#: bottom of this file read it: executing a pack's graphs is
#: ``test_chapter_examples.py``'s job and stays there.
_PLUGINS_ROOT = _REPO_ROOT / "plugins"


def _discover_builtin_graphs() -> list[Path]:
    return sorted(_EXAMPLES_ROOT.rglob("graph.json"))


_GRAPHS = _discover_builtin_graphs()
assert _GRAPHS, "builtin example smoke suite discovered no examples"

# Node types that pull a real dataset, train for multiple epochs, or load/save
# weights — longer than a few seconds or dependent on prior runs.
# test_chapter_examples.py carries the same list minus ``TextCorpusDataset``,
# which no plugin example uses. This correctly skips execution for eleven
# graphs: the seven training examples (CNN-MNIST, GPT-Mini, ResNet-CIFAR10,
# the ResNet-18 baseline, the HuggingFace beans CNN, the TinyStories LM and
# VLA-PushWorld), the MNIST inference example, which needs weights from a
# prior training run, and the three pack-backed LLM examples (zh-TW Sentence
# Similarity, RAG-Local-Offline and RAG-LLMChat-API), whose encoder and
# generator are pack downloads.
# Everything else — including all Model_Architecture graphs — must execute.
_SLOW_NODE_TYPES = {
    "Dataset",
    "DataLoader",
    "TrainingLoop",
    "preset:Training Pipeline",
    "ModelLoader",
    "ModelSaver",
    "HuggingFaceDataset",
    "KaggleDataset",
    "Inference",
    # Pulls a corpus off the Hugging Face Hub, and tokenising it is minutes of
    # CPU even once it is local. Listed even though the one graph using it also
    # trains — so an LLM example that only prepares data cannot start
    # downloading in CI by being added.
    "TextCorpusDataset",
    # Needs the sentence-embeddings pack; graphs are still validated. CI has
    # no pack cache, so an example using this node would fail at the gate
    # rather than run -- and a machine that DOES have the pack would load
    # half a gigabyte of weights inside the fast smoke suite.
    "TextEmbedding",
    # Reads a gigabyte of Qwen2.5 weights out of the rag pack and then decodes
    # on the CPU at a few tokens a second. Same gate as TextEmbedding above --
    # CI has no pack cache, so an example using this node would stop at the
    # missing pack rather than run -- and a machine that DOES have the pack
    # would spend minutes of the fast smoke suite generating an answer.
    "HFTextGenerate",
    # One API call per run: a network round trip, a key CI does not have, and
    # somebody's money. RAG-LLMChat-API wires it (to a local Ollama by
    # default, to a hosted provider once someone switches the dropdown), so
    # this entry is what keeps the smoke suite from opening that socket.
    "LLMChat",
}


def _is_slow(payload: dict) -> bool:
    return any(n.get("type") in _SLOW_NODE_TYPES for n in payload.get("nodes", []))


@pytest.mark.parametrize(
    "graph_path",
    _GRAPHS,
    ids=[p.relative_to(_EXAMPLES_ROOT).as_posix() for p in _GRAPHS],
)
def test_builtin_graph_executes(graph_path: Path):
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = payload["nodes"]
    edges = payload["edges"]

    errors = validate_graph(nodes, edges)
    assert not errors, f"validate_graph errors for {graph_path}: {errors}"

    if _is_slow(payload):
        pytest.skip(
            "Graph pulls a real dataset / trains a model / loads saved weights — "
            "validated structurally, manual run required for full execution."
        )

    # Some example graphs reference data files via paths relative to the
    # backend cwd (CSVReader's default is ``data/samples/iris.csv``).
    # ``cdui test`` always launches pytest from backend/, but a contributor
    # running ``pytest`` directly from the repo root would otherwise hit a
    # FileNotFoundError. Hop cwd just for this test to keep both invocations
    # working — restore afterwards so other tests aren't affected.
    backend_dir = Path(__file__).resolve().parents[1]
    prev_cwd = Path.cwd()
    os.chdir(backend_dir)
    try:
        asyncio.run(execute_graph(nodes, edges, error_mode="fail_fast"))
    finally:
        os.chdir(prev_cwd)


# ── ResNet-18 CIFAR-10 baseline: short-epoch execution (core#138) ───────────

_RESNET18_EXAMPLE = (
    _EXAMPLES_ROOT / "Usage_Example" / "ResNet18-CIFAR10-Baseline" / "graph.json"
)

#: Classes and images-per-class in the generated stand-in dataset. Two classes
#: is the minimum ImageFolder accepts; the model still has its real 10-way head,
#: so labels 0/1 simply never exercise the other eight logits.
_FIXTURE_CLASSES = ("class_a", "class_b")
_FIXTURE_PER_CLASS = 4


def _write_image_folder(root: Path) -> None:
    """Create ``root/{train,val}/<class>/*.png`` of 32x32 RGB noise.

    Deliberately not CIFAR-10: CI must not download 170 MB, and this test is
    about the graph wiring and the layer spec, not about accuracy. Images are
    32x32 so ``RandomCrop(32, padding=4)`` in the example's real augmentation
    chain is exercised at its true size.
    """
    from PIL import Image

    rng = random.Random(0)  # local instance: never touches the global RNG
    for split in ("train", "val"):
        for label, cls in enumerate(_FIXTURE_CLASSES):
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(_FIXTURE_PER_CLASS):
                # Bias each class's colour so the task is at least learnable.
                base = 40 if label == 0 else 200
                px = bytes(
                    max(0, min(255, base + rng.randint(-30, 30)))
                    for _ in range(32 * 32 * 3)
                )
                Image.frombytes("RGB", (32, 32), px).save(d / f"{i}.png")


#: The recipe the published 95.48% was measured with.
#:
#: Asserted against the shipped graph *before* :func:`_shrink_for_ci` rewrites
#: it. Execution alone cannot guard these: the shrink replaces both dataset
#: nodes wholesale and overrides the loop/scheduler params, so a graph whose
#: eval split had been repointed at ``train``, or whose optimizer recipe had
#: been replaced outright, would still execute happily. The number on the docs
#: page is only meaningful if the file that ships is the file that produced it.
_SHIPPED_RECIPE: dict[str, dict] = {
    "ds-train": {"name": "CIFAR10", "split": "train"},
    "ds-test": {"name": "CIFAR10", "split": "test"},
    "aug-crop": {"size": 32, "padding": 4},
    "aug-flip": {"p": 0.5},
    "aug-norm": {"preset": "CIFAR-10"},
    "ev-norm": {"preset": "CIFAR-10"},
    "dl-train": {"batch_size": 128, "shuffle": True},
    "dl-val": {"batch_size": 512, "shuffle": False},
    "opt": {
        "type": "SGD", "lr": 0.1, "momentum": 0.9,
        "weight_decay": 5e-4, "nesterov": True,
    },
    "loss": {"type": "CrossEntropyLoss", "label_smoothing": 0.0},
    "sched": {"type": "CosineAnnealingLR", "T_max": 200},
    "train": {"epochs": 200, "precision": "bf16", "early_stopping_patience": 0},
    # ``device`` is pinned here although the published number came from an
    # explicit ``cuda``: since #204 "auto" on this node means "follow the
    # run-level device", which is what ``TrainingLoop.device`` beside it
    # already says, so the two now agree instead of the evaluation naming a
    # device the training did not. Pinned rather than left free because
    # "auto" is the property -- a re-hardcoded device would go back to
    # failing on the machines that do not have it.
    "eval": {"batch_size": 512, "device": "auto"},
}


def _data_source(edges: list[dict], target: str, handle: str) -> str:
    """Id of the node feeding ``target.handle`` over a data edge."""
    for e in edges:
        if (
            e.get("type", "data") == "data"
            and e["target"] == target
            and e.get("sourceHandle") is not None
            and e.get("targetHandle") == handle
        ):
            return e["source"]
    raise AssertionError(f"no data edge feeds {target}.{handle}")


def _note_text(payload: dict, note_id: str) -> str:
    """The text of one on-canvas note, which is where the long prose lives.

    A description is one line now (``test_example_descriptions.py``), so a
    number that used to be asserted against the card is asserted against the
    note holding the explanation instead. Same guarantee, different surface:
    the figure is still derived from the params, so the prose cannot drift
    from the graph it sits on.
    """
    for node in payload["nodes"]:
        if node.get("id") == note_id:
            assert node.get("type") == "note", f"{note_id} is not a note"
            return node["data"]["noteContent"]
    raise AssertionError(f"the graph has no note {note_id!r}")


def _notes_of(payload: dict) -> list[str]:
    """The text of every note on one graph's canvas.

    The unnamed twin of :func:`_note_text`: that one goes after a note a
    test knows by id, this one after "what does this graph tell its reader
    anywhere on the canvas", which is the question the requirement rules at
    the bottom of the file ask of all 72 examples at once.
    """
    return [node["data"]["noteContent"] for node in payload.get("nodes", [])
            if node.get("type") == "note"]


def _assert_shipped_recipe(nodes: list[dict], edges: list[dict]) -> None:
    """The shipped graph still is the graph the published number came from.

    Three kinds of check, and the second is the one that matters most:

    * every hyperparameter the README and the docs page quote is still the
      value in the file;
    * evaluation still happens on the *held-out* split. The whole milestone
      rests on that one property, and it is exactly the property that is
      invisible to an execution smoke against a synthetic image folder;
    * the accuracy it measures reaches something that shows it.
    """
    by_id = {n["id"]: n for n in nodes}

    for node_id, expected in _SHIPPED_RECIPE.items():
        assert node_id in by_id, f"{node_id} is missing from the shipped graph"
        actual = by_id[node_id]["data"]["params"]
        for key, want in expected.items():
            assert actual.get(key) == want, (
                f"{node_id}.{key} is {actual.get(key)!r}, but the published "
                f"result was measured with {want!r}"
            )

    # Cosine annealing is stepped once per epoch and must reach zero exactly at
    # the end of training; a mismatch is silent accuracy loss (issue #205).
    assert (
        by_id["sched"]["data"]["params"]["T_max"]
        == by_id["train"]["data"]["params"]["epochs"]
    ), "LRScheduler.T_max must equal TrainingLoop.epochs"

    # No leakage: the evaluated dataset is the test split, the trained one is
    # the train split, and they are different nodes.
    eval_ds = _data_source(edges, "eval", "dataset")
    train_ds = _data_source(edges, "dl-train", "dataset")
    assert by_id[eval_ds]["data"]["params"]["split"] == "test", (
        f"EvaluateModel reads {eval_ds}, whose split is "
        f"{by_id[eval_ds]['data']['params']['split']!r} — the accuracy claim "
        f"requires the held-out split"
    )
    assert by_id[train_ds]["data"]["params"]["split"] == "train", train_ds
    assert eval_ds != train_ds, "training and evaluation share one Dataset node"

    # The random augmentation chain reaches a train_transform port and nothing
    # else — a test split must never be augmented.
    aug_targets = {
        e.get("targetHandle")
        for e in edges
        if e.get("type", "data") == "data" and e["source"] == "aug-norm"
    }
    assert aug_targets == {"train_transform"}, (
        f"the augmentation chain feeds {aug_targets}, not just train_transform"
    )

    # And the number reaches the screen. An EvaluateModel whose ``accuracy``
    # has no outgoing edge runs, logs its metric point, and shows the reader
    # nothing -- the one figure this whole example exists to produce.
    assert [
        e for e in edges
        if e.get("type", "data") == "data"
        and e["source"] == "eval" and e.get("sourceHandle") == "accuracy"
    ], "EvaluateModel.accuracy is computed and never displayed"


def _shrink_for_ci(nodes: list[dict], data_root: Path) -> None:
    """Rewrite the shipped graph in place into a 2-step CPU run.

    Only params and the two dataset node types change — every edge, the
    augmentation chain, the optimizer/scheduler/loss wiring and the whole
    ``SequentialModel`` layer spec are executed exactly as shipped.
    ``ImageFolderDataset`` is a drop-in for ``Dataset`` here because it
    declares the same ``train_transform``/``eval_transform`` inputs and the
    same ``dataset`` output.

    Both dataset nodes are replaced *wholesale*, params included, so nothing
    this function touches can be asserted afterwards — call
    :func:`_assert_shipped_recipe` first.
    """
    by_id = {n["id"]: n for n in nodes}

    for node_id, split in (("ds-train", "train"), ("ds-test", "val")):
        node = by_id[node_id]
        node["type"] = "ImageFolderDataset"
        node["data"]["params"] = {"path": str(data_root), "split": split}

    for node_id in ("dl-train", "dl-val"):
        by_id[node_id]["data"]["params"].update(
            batch_size=4, num_workers=0, persistent_workers=False, pin_memory=False
        )

    by_id["train"]["data"]["params"].update(
        epochs=1, max_steps=2, precision="fp32", device="cpu", tensorboard=False
    )
    by_id["sched"]["data"]["params"]["T_max"] = 1
    by_id["eval"]["data"]["params"].update(batch_size=4, device="cpu")
    by_id["ckpt"]["data"]["params"].update(path="smoke.pt", epoch=1)


def test_resnet18_cifar10_baseline_recipe_intact():
    """The shipped graph is still the graph the published 95.48% came from.

    Reads the file unmodified — no shrinking, no execution — so it sees the
    real dataset splits and the real hyperparameters.
    """
    payload = json.loads(_RESNET18_EXAMPLE.read_text(encoding="utf-8"))
    _assert_shipped_recipe(payload["nodes"], payload["edges"])


def test_resnet18_cifar10_baseline_short_epoch(tmp_path, monkeypatch):
    """Execute the shipped baseline graph for two optimizer steps.

    Guards the parts ``validate_graph`` is blind to: that the 70-node ResNet-18
    layer JSON still builds (11,173,962 parameters, the CIFAR variant's count),
    that the transform chain still composes, and that model/optimizer/scheduler/
    loss/checkpoint/evaluate still agree on port names.
    """
    from app.config import settings

    payload = json.loads(_RESNET18_EXAMPLE.read_text(encoding="utf-8"))
    nodes, edges = payload["nodes"], payload["edges"]

    # Before anything is rewritten: the recipe and the eval split are only
    # observable on the pristine file.
    _assert_shipped_recipe(nodes, edges)

    data_root = tmp_path / "images"
    _write_image_folder(data_root)

    # CheckpointSaver refuses to write outside the data root, so move the data
    # root rather than the path — keeps the test out of the user's real
    # backend/data/models (cf. #151).
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "models")

    _shrink_for_ci(nodes, data_root)
    assert not validate_graph(nodes, edges), "shrunk graph must still validate"

    results = asyncio.run(execute_graph(nodes, edges, error_mode="fail_fast"))

    # The layer spec really built the CIFAR ResNet-18, not something smaller.
    model = results["model"]["model"]
    assert sum(p.numel() for p in model.parameters()) == 11_173_962

    # Training actually stepped and produced a finite loss.
    losses = results["train"]["losses"]
    assert len(losses) == 1, f"expected one epoch of loss, got {losses}"
    assert losses[0] == losses[0], "training loss is NaN"

    metrics = results["train"]["metrics"]
    assert metrics["total_steps"] == 2, metrics
    assert metrics["stopped_at_max_steps"] is True, metrics

    # The two transform chains really are what they claim to be: the training
    # one randomises, the evaluation one does not. Asserted on the composed
    # pipelines rather than on node types, so a chain rewired to send an
    # augmentation down the eval side is caught by what it does.
    from app.nodes.data.transforms._base import pipeline_is_random

    assert pipeline_is_random(results["aug-norm"]["transform"]) is True
    assert pipeline_is_random(results["ev-norm"]["transform"]) is False

    # Evaluation ran end to end over the held-out split.
    assert results["eval"]["total"] == len(_FIXTURE_CLASSES) * _FIXTURE_PER_CLASS
    assert 0.0 <= results["eval"]["accuracy"] <= 1.0

    # The checkpoint landed where CheckpointSaver said it would.
    assert (tmp_path / "models" / "smoke.pt").is_file()


# ── TinyStories LM pretraining: the prose is the graph (#292) ───────────────

_LM_DIR = _EXAMPLES_ROOT / "LLM" / "TrainCausalLM-TinyStories"
_LM_EXAMPLE = _LM_DIR / "graph.json"
_LM_README = _LM_DIR / "README.md"

#: What the example's own prose claims, and where each claim comes from.
#:
#: Every one of these is a relationship BETWEEN nodes, which is why none of
#: them can be checked by looking at any single node: the model's context
#: length has to match the dataset's block length, the effective batch is a
#: product of two nodes' params, and the parameter count is a fact about the
#: model node quoted in prose beside the graph.
_LM_SEQ_LEN = 1024
_LM_MICRO_BATCH = 8
_LM_ACCUMULATE = 4
_LM_TRAIN_TOKEN_BUDGET = 20_000_000
_LM_VAL_TOKEN_BUDGET = 2_000_000

#: The reference shape epic #292 sizes its run against.
#: ``test_causal_lm_model_node.py::
#: test_the_advertised_default_size_matches_the_declared_defaults`` is what
#: ties this number to the node's declared defaults, term by term. Here it is
#: only checked that the graph still USES those defaults and still quotes the
#: same figure -- so a change to the architecture fails there, and a change to
#: this graph fails here.
_LM_PARAM_COUNT = 203_668_480

#: The two things that decide whether a reader CAN run this graph at all,
#: and the word each half of the bilingual note says them with.
#:
#: Neither is on the card. A description is one line of forty columns
#: (``test_example_descriptions.py``), and a line that spends half of itself
#: on "needs a 16 GB GPU, a download" has stopped saying what the graph is
#: for -- so the warning moved onto the canvas, beside the nodes it is
#: about, where there is room to say how big the download is and why the GPU
#: has to be that size. Everything past that is in README.md.
#:
#: ``(what, English, Traditional Chinese)``: a note is written twice inside
#: the one note, and there is no translation table behind it, so a warning
#: that survives only the English half is a warning half the readers of this
#: project never see.
_LM_NOTE_REQUIREMENTS = (
    ("a 16 GB GPU", "16 GB GPU", "16 GB GPU"),
    ("the corpus download", "download", "下載"),
)


def test_tinystories_lm_example_names_what_it_needs_in_its_notes():
    """The 16 GB GPU and the download are on the canvas, in both languages.

    This example is the one builtin that needs the network AND a 16 GB GPU,
    and the gallery card says neither: the card says what the graph shows.
    That moves the warning rather than dropping it -- onto the notes, which
    are what a reader is looking at while deciding whether to start an
    hour-long run, and which have room for the size and the reason.

    Every note is searched, not the overview alone, because the two facts do
    not have to sit in the same note. What they cannot do is sit in only one
    language.
    """
    payload = json.loads(_LM_EXAMPLE.read_text(encoding="utf-8"))
    notes = _notes_of(payload)
    assert notes, "the LM example carries no notes, so this checks nothing"

    # English paragraph, blank line, the same thing in Traditional Chinese --
    # the shape ``test_concept_examples.py`` holds every note to. Split the
    # same way here so a fact asserted "in the Chinese" cannot be satisfied
    # by the English copy of it sitting above.
    english = "\n".join(note.partition("\n\n")[0] for note in notes)
    chinese = "\n".join(note.partition("\n\n")[2] for note in notes)

    for what, in_english, in_chinese in _LM_NOTE_REQUIREMENTS:
        assert in_english in english, (
            f"the English half of this example's notes never names "
            f"{what} ({in_english!r}); the card does not carry it either, so "
            f"nothing in the product warns an English reader before they "
            f"press Run")
        assert in_chinese in chinese, (
            f"the Traditional Chinese half of this example's notes never "
            f"names {what} ({in_chinese!r}); a note is the only bilingual "
            f"surface this warning has, and half of it is missing")

    # A tooltip-length description is still worth keeping short: the sidebar
    # shows it in full as a `title`, and a native tooltip of several hundred
    # characters is its own kind of unreadable. Generous ceiling, not a style
    # rule -- the point is that the recipe lives in README.md and the
    # explanation on the canvas, not in the one field every row renders.
    description = payload["description"]
    assert len(description) <= 500, (
        f"the description is {len(description)} characters; the recipe belongs "
        f"in {_LM_README.name}, not on the card")

    assert _LM_README.is_file(), (
        f"{_LM_README} is missing -- the overview note's last sentence points "
        f"at it, and the docs pages point at it instead of at the card")


# ── Where a requirement is written: the note, not the card ────────────────
#
# The three rules below replace one that pointed the other way. It required
# every requirement an example stated to land inside the characters the
# gallery card shows, which was right while a description was the only prose
# an example carried. It is not any more: a description is one line of forty
# columns and an example explains itself in notes on its canvas, so the
# ruling is that the line under the title says what the graph does and what
# it is explaining, and a download, a GPU, a pack, an API key or a sibling
# example that has to be run first is named on the canvas and in the docs.
#
# The detector underneath is the same detector, pointed the other way: the
# vocabulary tables below were written against the way these examples
# actually phrase a requirement, including the two shapes they used to get
# wrong, and that work is worth as much to a rule that forbids a sentence as
# to one that demanded it. "CARD" survives in their names from when the card
# was the only surface they were read on.

#: The heavy requirements, and the words each one is actually written with
#: across the examples.
#:
#: Deliberately NOT a bare "network": a dozen examples say it about a neural
#: one ("the agent's own network", "a small gating network"), and a check
#: that fires on every example is a check nobody can read.
_CARD_REQUIREMENT_WORDS = {
    "a GPU": r"\bGPUs?\b|\bCUDA\b|\bVRAM\b",
    "a download": r"\bdownload(s|ed|ing)?\b",
    "a pack": r"\bpacks?\b|Package Center",
    "a service of the user's own": r"\bAPI key\b|\bOllama\b|\binternet\b|Hugging Face",
}

#: What turns a mention into a requirement.
#:
#: ``install`` is NOT here on purpose: several examples name a pack only to
#: say where a SIBLING example ships (``cdui plugin install foundations``),
#: which the example doing the naming does not need. The verbs below are the
#: ones that say "you cannot run this without".
#:
#: ``needed`` is in ``need(s|ed)?`` because English says a requirement in the
#: passive at least as often as in the active: "a GPU is needed" is the same
#: warning as "needs a GPU", and a check that heard only one of them would
#: let a card keep the other one, and would read a note that carries it as a
#: note that says nothing.
_CARD_REQUIREMENT_VERBS = (
    r"\bneed(s|ed)?\b|\brequires?\b|\brequired\b|\bprerequisite\b|\bmust\b")

#: What un-says one.
#:
#: "No GPU is required." matches a requirement verb and names a resource, and
#: means the opposite of both. Flagged, it would tell the author to move a
#: reassurance off the card and onto the canvas -- advice that makes both
#: surfaces worse, which is a sharper failure than the one the check exists
#: to catch.
#:
#: Judged per resource, against the text BEFORE the resource word: a sentence
#: reading "needs a GPU, no download" negates the download and not the GPU.
_CARD_REQUIREMENT_NEGATIONS = r"\bno\b|\bnot\b|\bwithout\b|\boptional\b"


def _requirements_stated_in(text: str) -> list[tuple[str, str]]:
    """The heavy resources *text* says the graph cannot run without.

    One ``(requirement, sentence)`` pair per resource actually required, so
    the caller can quote the sentence back at whoever has to move it.

    Reads a description and a note alike, because the rules below need the
    same sentence heard on both surfaces: once to say it does not belong on
    the card, once to say it does belong on the canvas. A note carries its
    Traditional Chinese half in the same string, which this ignores -- there
    are no English requirement verbs in it, and the bilingual half of the
    rule is asserted where it is specific enough to be worth asserting
    (``test_tinystories_lm_example_names_what_it_needs_in_its_notes``).
    """
    stated: list[tuple[str, str]] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if not re.search(_CARD_REQUIREMENT_VERBS, sentence, re.I):
            continue
        for requirement, pattern in _CARD_REQUIREMENT_WORDS.items():
            named = re.search(pattern, sentence, re.I)
            if not named:
                continue
            if re.search(
                    _CARD_REQUIREMENT_NEGATIONS, sentence[:named.start()], re.I):
                continue
            stated.append((requirement, sentence.strip()))
    return stated


def _every_shipped_example() -> dict[str, dict]:
    """Every example the gallery lists, keyed as ``/api/examples/list`` keys
    it: the relative path for a built-in, ``plugin:<id>/<rest>`` for one a
    pack ships.

    Both roots, unlike ``_GRAPHS`` at the top of this file. Running a pack's
    graphs is ``test_chapter_examples.py``'s job and stays there, but the
    rules below are about what a card says and what a note says, and the
    gallery draws a pack's examples on the same cards as the built-ins. A
    reader cannot tell which root a card came from, so a rule that covered
    one root would be a rule half the gallery does not follow. Six of the
    sixteen descriptions the rewritten rule first caught sat in ``plugins/``,
    which the old card check never opened.
    """
    shipped: dict[str, dict] = {}

    def read(graph_file: Path) -> dict:
        return json.loads(graph_file.read_text(encoding="utf-8"))

    for graph_file in sorted(_EXAMPLES_ROOT.rglob("graph.json")):
        key = graph_file.parent.relative_to(_EXAMPLES_ROOT).as_posix()
        shipped[key] = read(graph_file)
    for pack_dir in sorted(_PLUGINS_ROOT.glob("*")):
        pack_examples = pack_dir / "examples"
        if not pack_examples.is_dir():
            continue
        for graph_file in sorted(pack_examples.rglob("graph.json")):
            rel = graph_file.parent.relative_to(pack_examples).as_posix()
            shipped[f"plugin:{pack_dir.name}/{rel}"] = read(graph_file)
    return shipped


_SHIPPED = _every_shipped_example()
assert len(_SHIPPED) > 50, (
    f"the two-root scan found {len(_SHIPPED)} examples; the rules below "
    f"would hold over an almost empty gallery")


def test_no_example_description_states_a_requirement():
    """A card says what the graph shows, not what you install first.

    The old rule required a requirement to land inside the characters the
    card shows. This is that rule inverted, and the inversion is the whole
    ruling: a description is one line of forty columns
    (``test_example_descriptions.py``), and a line that spends half of
    itself on "needs a 1.5 GB download" has stopped saying what the graph is
    for -- which is the one thing only the card can say, because it is what
    a reader picks a card by.

    So a download, a GPU, a pack or an API key -- and anything else an
    example needs first, such as a sibling graph that has to be run before
    it, which the vocabulary above cannot hear -- goes in the note on the
    canvas, beside the nodes it is about, where there is room to say how big
    it is and where it comes from, and in
    ``docs/docs/usage/examples-gallery.md``, where the exceptions are listed
    in full.

    Fixed by moving the sentence onto the note -- never by trimming the
    vocabulary above until the sentence stops matching.
    """
    offenders: list[str] = []
    for key, payload in _SHIPPED.items():
        for requirement, sentence in _requirements_stated_in(
                payload.get("description", "")):
            offenders.append(
                f"{key} states {requirement} on its card -- {sentence!r}")

    assert not offenders, (
        f"{len(offenders)} example description(s) state a requirement:\n  "
        + "\n  ".join(offenders)
        + "\nThe line under the title says what the graph does and what it "
          "is explaining. What an example needs before it runs belongs in "
          "the note on its canvas, beside the nodes it is about, and in the "
          "exceptions list in docs/docs/usage/examples-gallery.md.")


def test_every_requirement_an_example_states_is_stated_in_its_notes():
    """A requirement moves onto the canvas; it does not just disappear.

    The rule above is satisfied as well by deleting a warning as by moving
    it, and deleting it is the worse outcome of the two: the reader then
    presses Run on a graph that stops at a missing pack, with nothing
    anywhere in the product having said so. So whatever heavy resource a
    graph still names as required in the prose that is NOT a note, one of
    its notes has to name as required too.

    "Prose that is not a note" is ``name`` and ``description`` -- the only
    other two text fields a graph file has. Once every description is clean
    this rule has nothing left to compare, which is exactly why the floor in
    the test below exists: the two are halves of one guarantee, and neither
    is worth much on its own.
    """
    offenders: list[str] = []
    for key, payload in _SHIPPED.items():
        notes = _notes_of(payload)
        on_the_card = {
            requirement for requirement, _ in _requirements_stated_in(
                f"{payload.get('name', '')}. {payload.get('description', '')}")
        }
        in_the_notes = {
            requirement
            for note in notes
            for requirement, _ in _requirements_stated_in(note)
        }
        for requirement in sorted(on_the_card - in_the_notes):
            offenders.append(
                f"{key} says it needs {requirement} in its name or its "
                f"description, and in none of its {len(notes)} note(s)")

    assert not offenders, (
        f"{len(offenders)} requirement(s) are stated only where there is no "
        f"room to act on them:\n  " + "\n  ".join(offenders)
        + "\nMove the sentence onto the note bound to the nodes it is about. "
          "Do not delete it: an unwarned download is worse than a crowded "
          "card.")


#: How many examples must still name a heavy resource in one of their notes.
#:
#: A floor, set well under what ships -- sixteen examples across the two
#: roots named a download, a pack, a GPU or a chat backend when the warnings
#: were moved off the cards -- so that retiring one heavy example is not a
#: test failure while gutting the warnings is.
_EXAMPLES_THAT_MUST_WARN = 10


def test_the_notes_still_warn_about_something():
    """A rule that has stopped matching anything has stopped being a rule.

    This is the ``assert covered`` that used to close the card check, moved
    to the surface the warnings moved to, and for the same reason. Without
    it the two rules above are satisfied perfectly by a rewrite that drops
    every warning from every example: the cards would be clean, nothing
    would contradict a note, and a learner would press Run on a graph that
    needs a 1.5 GB download with no warning anywhere on screen.

    If this trips, either the vocabulary in ``_CARD_REQUIREMENT_WORDS`` no
    longer matches the way the notes are written -- fix the vocabulary -- or
    the warnings really are gone, which is the failure it was written for.
    """
    warned = sorted(
        key for key, payload in _SHIPPED.items()
        if any(_requirements_stated_in(note) for note in _notes_of(payload)))

    assert len(warned) >= _EXAMPLES_THAT_MUST_WARN, (
        f"only {len(warned)} of {len(_SHIPPED)} shipped examples name a GPU, "
        f"a download, a pack or a service of the reader's own in a note: "
        f"{warned}. At least {_EXAMPLES_THAT_MUST_WARN} have to, because at "
        f"least that many cannot run on a fresh offline install.")


def test_the_card_requirement_reader_hears_both_ways_of_saying_it():
    """Two sentences no builtin happens to be written with today.

    The rules above can only be as good as the vocabulary underneath them,
    and that vocabulary is only exercised by whatever the shipped
    descriptions and notes happen to say. These two are the shapes it used
    to get wrong: a passive requirement it did not hear, and a reassurance
    it heard as a requirement.

    Both matter more now that the detector is a prohibition. A requirement
    it cannot hear is a requirement that stays on the card; a reassurance it
    mishears is an author told to move "no GPU required" off the card and
    onto the canvas, which takes the one sentence a hesitant reader wanted
    and hides it.
    """
    assert _requirements_stated_in("A GPU is needed for the training loop.") == [
        ("a GPU", "A GPU is needed for the training loop.")]

    assert _requirements_stated_in("Runs on CPU. No GPU is required.") == []


def test_tinystories_lm_example_still_describes_itself():
    """Every number the example's README quotes is still in its params.

    The README is where the recipe lives, because the card cannot hold it (see
    the test above). That makes the README the file most able to rot silently:
    nothing executes it, and every figure in it is a fact about two nodes at
    once. So each one is derived here from the params and required to appear.
    """
    payload = json.loads(_LM_EXAMPLE.read_text(encoding="utf-8"))
    by_id = {n["id"]: n for n in payload["nodes"]}
    readme = _LM_README.read_text(encoding="utf-8")

    def params(node_id: str) -> dict:
        assert node_id in by_id, f"{node_id} is missing from the LM example"
        return by_id[node_id]["data"]["params"]

    # The model is the epic's reference shape, unmodified.
    from app.nodes.llm.causal_lm_model_node import CausalLMModelNode

    declared = {p.name: p.default for p in CausalLMModelNode.define_params()}
    model_params = params("model")
    for name in ("vocab_size", "d_model", "n_layers", "n_heads", "d_ff",
                 "max_seq_len", "tie_embeddings", "positional", "norm"):
        assert model_params[name] == declared[name], (
            f"model.{name} is {model_params[name]!r}, not the declared default "
            f"{declared[name]!r} -- the example no longer builds the "
            f"{_LM_PARAM_COUNT:,}-parameter reference shape it advertises")
    assert f"{_LM_PARAM_COUNT:,} parameters" in readme, (
        f"the README no longer quotes {_LM_PARAM_COUNT:,} parameters")

    # A block longer than the model's positions is rejected at runtime rather
    # than truncated (CausalLMModel), so this equality is the difference
    # between a run and a ValueError on the first batch.
    for packer in ("pack-train", "pack-val"):
        assert params(packer)["seq_len"] == _LM_SEQ_LEN, packer
    assert model_params["max_seq_len"] == _LM_SEQ_LEN

    # The effective batch is a product of two nodes, and the README states
    # both it and the token count it implies.
    loader, loop = params("dl-train"), params("train")
    assert loader["batch_size"] == _LM_MICRO_BATCH
    assert loop["accumulate_steps"] == _LM_ACCUMULATE
    effective = _LM_MICRO_BATCH * _LM_ACCUMULATE
    assert f"effective batch {effective}" in readme
    assert f"{effective * _LM_SEQ_LEN:,} tokens per optimizer step" in readme

    # The recipe the README tabulates.
    assert loop["precision"] == "bf16"
    assert loop["epochs"] == 1
    assert loop["grad_clip_norm"] == 1.0
    optimizer = params("opt")
    assert optimizer["type"] == "AdamW"
    assert optimizer["lr"] == 3e-4
    assert optimizer["weight_decay"] == 0.1
    # The one value here that is not the Optimizer node's default: 0.999 is
    # the vision-training beta2, and LM pretraining has used 0.95 since GPT-2.
    # The README calls the deviation out, and this holds it to that.
    assert optimizer["betas"] == "0.9, 0.95"
    assert "betas 0.9, 0.95" in readme

    # The two budgets that decide how long "one epoch" takes.
    assert params("pack-train")["max_tokens"] == _LM_TRAIN_TOKEN_BUDGET
    assert params("pack-val")["max_tokens"] == _LM_VAL_TOKEN_BUDGET
    assert f"**{_LM_TRAIN_TOKEN_BUDGET:,}** tokens" in readme
    assert f"**{_LM_VAL_TOKEN_BUDGET:,}** for validation" in readme

    # The chain the README walks through, and the sentence on the card: how
    # many steps "one epoch" actually is. Derived rather than transcribed, so
    # editing seq_len, either budget, batch_size or accumulate_steps fails
    # here until the prose is updated with it.
    blocks = (_LM_TRAIN_TOKEN_BUDGET - 1) // _LM_SEQ_LEN
    assert loader["drop_last"] is True, (
        "drop_last is what makes the micro-batch count below a clean floor")
    micro_batches = blocks // _LM_MICRO_BATCH
    optimizer_steps = micro_batches // _LM_ACCUMULATE
    for number in (blocks, micro_batches, optimizer_steps):
        assert f"**{number:,}**" in readme, (
            f"the README's budget table does not derive {number:,}")
    # What "one epoch" costs used to be on the card. The card is one line
    # now, so it is on the overview note -- which is where a reader deciding
    # whether to start an hour-long run is actually looking.
    overview = _note_text(payload, "note-overview")
    assert f"{micro_batches:,} micro-batches" in overview
    assert f"about {optimizer_steps:,} optimizer steps" in overview
    # Validation is scored whole, so its block count is quoted too.
    val_blocks = (_LM_VAL_TOKEN_BUDGET - 1) // _LM_SEQ_LEN
    assert params("ppl")["max_batches"] == 0
    assert f"{val_blocks:,} blocks" in readme


# ── HuggingFace beans: the four things a reader re-points at their data ────

_BEANS_DIR = (
    _EXAMPLES_ROOT / "Usage_Example" / "HuggingFace-Dataset" / "TrainCNN-Beans"
)
_BEANS_EXAMPLE = _BEANS_DIR / "graph.json"

#: The repo the example is wired to, and the column names it needs. ``beans``
#: calls its label column ``labels``, not the node's default ``label`` --
#: which is the whole reason the example ships pointed at it rather than at a
#: dataset where every default already fits.
_BEANS_REPO = "AI-Lab-Makerere/beans"
_BEANS_IMAGE_COLUMN = "image"
_BEANS_LABEL_COLUMN = "labels"

#: One ``HuggingFaceDataset`` node per split, and the split each one reads.
_BEANS_SPLITS = {"ds-train": "train", "ds-val": "validation", "ds-test": "test"}

#: ``beans`` has three classes, so the model's last ``Linear`` has three
#: outputs. The stage note beside it tells the reader this is the one number
#: to change for their own dataset, which only stays true while it is true.
_BEANS_CLASSES = 3


def _beans_graph() -> dict:
    return json.loads(_BEANS_EXAMPLE.read_text(encoding="utf-8"))


def _nodes_of_type(nodes: list[dict], node_type: str) -> dict[str, dict]:
    return {n["id"]: n for n in nodes if n.get("type") == node_type}


def test_huggingface_beans_example_reads_three_splits_of_one_dataset():
    """Three loads of one repo, and the column names spelled the same way.

    The example exists to be re-pointed: a reader swaps ``dataset_name`` and
    the two column names for their own repo. A split that had drifted onto a
    different dataset, or a column name that matched on one node and not the
    others, would still validate and still train -- on the wrong data.
    """
    nodes = _beans_graph()["nodes"]
    sources = _nodes_of_type(nodes, "HuggingFaceDataset")

    assert set(sources) == set(_BEANS_SPLITS), sorted(sources)
    for node_id, split in _BEANS_SPLITS.items():
        params = sources[node_id]["data"]["params"]
        assert params["dataset_name"] == _BEANS_REPO, node_id
        assert params["split"] == split, node_id
        assert params["image_column"] == _BEANS_IMAGE_COLUMN, node_id
        assert params["label_column"] == _BEANS_LABEL_COLUMN, node_id


def test_huggingface_beans_example_preprocesses_every_split_alike():
    """One Transform recipe on all three splits, or the accuracy means nothing.

    ``HuggingFaceDataset`` has no ``train_transform``/``eval_transform`` ports
    to keep the two pipelines honest -- a ``Transform`` node is installed per
    split instead -- so nothing but this stops the evaluation split from being
    resized or normalized differently from what training saw. That skew is
    invisible: the graph runs, and only the accuracy is wrong.
    """
    nodes = _beans_graph()["nodes"]
    transforms = _nodes_of_type(nodes, "Transform")

    assert len(transforms) == len(_BEANS_SPLITS), sorted(transforms)
    recipes = {node_id: node["data"]["params"]
               for node_id, node in transforms.items()}
    assert len(set(map(str, map(sorted, (r.items() for r in recipes.values()))))) == 1, (
        f"the splits are preprocessed differently: {recipes}")

    only = next(iter(recipes.values()))
    assert only["resize"] > 0, "the 500x500 originals have to be resized"
    assert only["to_tensor"] is True
    assert only["normalize"] is True


def test_huggingface_beans_example_scores_the_split_it_did_not_train_on():
    """Train on ``train``, watch ``validation``, report ``test``.

    Each of the three is reached through its own ``Transform``, so the check
    walks two edges back rather than one.
    """
    payload = _beans_graph()
    nodes, edges = payload["nodes"], payload["edges"]
    splits = {n["id"]: n["data"]["params"].get("split")
              for n in nodes if n.get("type") == "HuggingFaceDataset"}

    def split_behind(target: str, handle: str) -> str:
        transform = _data_source(edges, target, handle)
        return splits[_data_source(edges, transform, "dataset")]

    train_loader = _data_source(edges, "train", "dataloader")
    val_loader = _data_source(edges, "train", "val_dataloader")
    assert split_behind(train_loader, "dataset") == "train"
    assert split_behind(val_loader, "dataset") == "validation"
    assert split_behind("eval", "dataset") == "test"

    # And the accuracy is shown. EvaluateModel returning a number nothing
    # reads is the failure this whole wave is about.
    accuracy_readers = {
        e["target"] for e in edges
        if e.get("type", "data") == "data"
        and e["source"] == "eval" and e.get("sourceHandle") == "accuracy"
    }
    assert accuracy_readers, "EvaluateModel.accuracy is computed and not shown"


def test_huggingface_beans_example_keeps_its_training_loop_on_the_canvas():
    """No ``preset:`` node: every stage is one the reader can open and change.

    The MNIST quick-start hides Dataset/DataLoader/Optimizer/Loss/TrainingLoop
    inside ``preset:Training Pipeline``, which is right for a first run and
    wrong for the example whose job is to show where your own dataset plugs
    in. Also pins the head width, because the stage note beside it says three
    outputs is the number to change.
    """
    payload = _beans_graph()
    nodes = payload["nodes"]

    presets = [n["id"] for n in nodes
               if str(n.get("type", "")).startswith("preset:")]
    assert not presets, f"the example hides stages inside a preset: {presets}"
    assert not payload.get("presets"), "the file carries a preset definition"

    model = _nodes_of_type(nodes, "SequentialModel")
    assert len(model) == 1, sorted(model)
    layers = json.loads(next(iter(model.values()))["data"]["params"]["layers"])
    heads = [n for n in layers["nodes"] if n["type"] == "Linear"]
    assert heads, "the model has no Linear head"
    assert heads[-1]["params"]["out_features"] == _BEANS_CLASSES, heads[-1]


def test_tinystories_lm_example_scores_a_split_it_did_not_train_on():
    """No leakage, and one tokenizer for the whole graph.

    Both are invisible to ``validate_graph``: a perplexity node pointed at the
    training blocks type-checks perfectly and reports a meaningless number,
    and a second tokenizer with a different encoding would give the same
    integer two different meanings without any port disagreeing.
    """
    payload = json.loads(_LM_EXAMPLE.read_text(encoding="utf-8"))
    nodes, edges = payload["nodes"], payload["edges"]
    by_id = {n["id"]: n for n in nodes}

    def split_behind(packer_id: str) -> str:
        corpus = _data_source(edges, packer_id, "dataset")
        return by_id[corpus]["data"]["params"]["split"]

    train_blocks = _data_source(edges, "dl-train", "dataset")
    eval_blocks = _data_source(edges, "ppl", "dataset")
    assert train_blocks != eval_blocks, (
        "PerplexityEvaluate reads the blocks the DataLoader trains on")
    assert split_behind(train_blocks) == "train", split_behind(train_blocks)
    assert split_behind(eval_blocks) == "validation", split_behind(eval_blocks)

    # One LMTokenizer feeds both packers and the generator, so the ids the
    # model trained on are the ids it is scored and sampled with.
    tokenizer_consumers = {
        (e["target"], e.get("targetHandle"))
        for e in edges
        if e.get("type", "data") == "data" and e["source"] == "tok"
    }
    assert tokenizer_consumers == {
        ("pack-train", "tokenizer"),
        ("pack-val", "tokenizer"),
        ("gen", "tokenizer"),
    }, tokenizer_consumers
    assert [n for n in nodes if n["type"] == "LMTokenizer"] == [by_id["tok"]], (
        "the example has more than one LMTokenizer")


# ── VLA on PushWorld: an hour of training has to end in numbers ────────────

_VLA_EXAMPLE = _EXAMPLES_ROOT / "VLA" / "TrainVLA-PushWorld" / "graph.json"


def test_vla_example_shows_everything_its_evaluation_measures():
    """Both evaluation nodes' outputs reach a GraphOutput.

    The two of them are the entire point of the hour this example costs, and
    every port they have was computed and then dropped: the success rate, the
    per-episode report and the action error existed only in the run log. The
    port names come from the node classes rather than a list here, so a node
    that gains an output cannot quietly go unshown.

    ``frames`` is the exception, and a real one: it is the rollout video,
    already wired to ``VideoWrite``, which is what showing a video means.
    """
    from app.nodes.vla.vla_action_eval_node import VLAActionEvalNode
    from app.nodes.vla.vla_rollout_node import VLARolloutNode

    payload = json.loads(_VLA_EXAMPLE.read_text(encoding="utf-8"))
    by_id = {n["id"]: n for n in payload["nodes"]}
    consumed = {
        (e["source"], e.get("sourceHandle")): e["target"]
        for e in payload["edges"] if e.get("type", "data") == "data"
    }

    expected = {
        "rollout": [p.name for p in VLARolloutNode.define_outputs()
                    if p.name != "frames"],
        "eval": [p.name for p in VLAActionEvalNode.define_outputs()],
    }
    assert expected["rollout"], "VLARollout declares no outputs"

    unshown = []
    for node_id, ports in expected.items():
        for port in ports:
            reader = consumed.get((node_id, port))
            if reader is None or by_id[reader]["type"] != "GraphOutput":
                unshown.append(f"{node_id}.{port} -> {reader}")
    assert not unshown, (
        f"these measurements do not reach a GraphOutput: {unshown}")

    # And the rollout video is still written, which is why frames is exempt.
    assert by_id[consumed[("rollout", "frames")]]["type"] == "VideoWrite"


# ── Quick Start: the graphs a test can only read ───────────────────────────

#: The three examples whose training runs inside ``preset:Training Pipeline``
#: and that grew an evaluation tail in this wave. Each contains a slow node
#: type, so ``test_builtin_graph_executes`` validates them and skips the run:
#: structure is all a test can see here, and structure is what rots. A
#: ``split`` flipped to ``train`` scores the model on the images it
#: memorised, and a dropped ``accuracy`` edge takes the number off the
#: screen again -- both still validate, and both still train.
_PRESET_TRAINERS = (
    "Usage_Example/CNN-MNIST/TrainCNN-MNIST",
    "Usage_Example/GPT-Mini/TrainGPT-Mini",
    "Usage_Example/ResNet-CIFAR10/TrainResNet-CIFAR10",
)


def _example(path: str) -> dict:
    return json.loads(
        (_EXAMPLES_ROOT / path / "graph.json").read_text(encoding="utf-8"))


def _readers(edges: list[dict], source: str, handle: str | None = None
             ) -> list[str]:
    """Ids of the nodes reading ``source``'s output over a data edge."""
    return [e["target"] for e in edges
            if e.get("type", "data") == "data" and e["source"] == source
            and (handle is None or e.get("sourceHandle") == handle)]


def _assert_argmax_is_printed(edges: list[dict], by_id: dict, model: str
                              ) -> None:
    """``model`` -> ``Argmax`` -> ``Print``: the label, not just the logits.

    A Print on the raw logits is ten numbers; the reader compares the digit
    against the image, and the digit is what the Argmax makes.
    """
    argmax = [n for n in _readers(edges, model, "output")
              if by_id[n]["type"] == "Argmax"]
    assert argmax, f"nothing turns {model}'s logits into a label"
    printed = [t for node in argmax for t in _readers(edges, node)
               if by_id[t]["type"] == "Print"]
    assert printed, f"{argmax} computes a label that never reaches a Print"


@pytest.mark.parametrize("example", _PRESET_TRAINERS)
def test_a_quickstart_trainer_scores_the_split_it_did_not_train_on(example):
    """Train on ``train``, report ``test``, and put the number on screen.

    The dataset the example trains on is read off the preset's own
    ``internalParams`` rather than listed here, so the two halves cannot
    drift apart: an example repointed at another dataset has to repoint its
    evaluation with it.
    """
    payload = _example(example)
    nodes, edges = payload["nodes"], payload["edges"]
    by_id = {n["id"]: n for n in nodes}

    pipeline = by_id[_data_source(edges, "eval", "model")]
    assert pipeline["type"] == "preset:Training Pipeline", pipeline["type"]
    trained_on = pipeline["data"]["internalParams"]["dataset"]
    assert trained_on["split"] == "train", trained_on

    assert [
        e for e in edges
        if e.get("type", "data") == "data"
        and e["source"] == pipeline["id"] and e.get("sourceHandle") == "model"
        and e["target"] == "eval" and e.get("targetHandle") == "model"
    ], "EvaluateModel is scored on something other than the trained model"

    scored_on = by_id[_data_source(edges, "eval", "dataset")]
    assert scored_on["type"] == "Dataset", scored_on["type"]
    assert scored_on["data"]["params"]["name"] == trained_on["name"], (
        f"the example trains on {trained_on['name']} and scores itself on "
        f"{scored_on['data']['params']['name']}")
    assert scored_on["data"]["params"]["split"] == "test", (
        f"{scored_on['id']} reads the {scored_on['data']['params']['split']!r} "
        f"split -- an accuracy measured on the images the loop trained on is "
        f"not an accuracy")

    shown = {by_id[e["target"]]["type"] for e in edges
             if e.get("type", "data") == "data" and e["source"] == "eval"
             and e.get("sourceHandle") == "accuracy"}
    assert "Print" in shown, (
        f"EvaluateModel.accuracy reaches {sorted(shown)} -- the one number "
        f"the evaluation tail exists to produce has to reach a Print")


def _default_normalize():
    """The ``Normalize`` step ``Dataset`` applies when nothing is wired in.

    Read off the app's own transform rather than written out as 0.5/0.5, so
    the two examples below follow a changed default instead of silently
    disagreeing with it.
    """
    from torchvision import transforms

    from app.nodes.data._batched_vision import DefaultVisionTransform

    steps = DefaultVisionTransform().transforms
    assert isinstance(steps[-1], transforms.Normalize), steps
    return steps[-1]


#: Everything between the PNG and the tensor ``Inference`` reads. The model
#: half is left out on purpose: ``ModelLoader`` wants weights from a training
#: run, which is why the smoke above cannot execute this graph at all.
_INFERENCE_EXAMPLE = "Usage_Example/CNN-MNIST/InferenceCNN-MNIST"
_INFERENCE_PREPROCESSING = (
    "start-1", "test-image", "norm-mean", "centre", "scale", "add-batch-dim")


def test_the_mnist_inference_example_feeds_the_model_the_range_it_trained_on():
    """Executed, not read: the tensor has to match, not the node types.

    Training normalises every image through ``DefaultVisionTransform``, so
    the saved weights expect [-1, 1]; ``ImageReader`` hands over [0, 1]. The
    ``Add``/``ScalarMultiply`` pair in between is the whole of that fix, and
    a graph with the pair deleted validates, executes, and classifies a
    tensor unlike anything the weights ever saw -- a wrong answer with no
    error anywhere. So the preprocessing half is run for real here and its
    output compared with the transform the training split went through.
    """
    import torch

    payload = _example(_INFERENCE_EXAMPLE)
    edges = payload["edges"]
    by_id = {n["id"]: n for n in payload["nodes"]}
    assert _data_source(edges, "inference", "input") == "add-batch-dim", (
        "the chain measured below is no longer what Inference reads")
    _assert_argmax_is_printed(edges, by_id, "inference")

    kept = set(_INFERENCE_PREPROCESSING)
    nodes = [n for n in payload["nodes"] if n["id"] in kept]
    assert {n["id"] for n in nodes} == kept, sorted(n["id"] for n in nodes)
    sub_edges = [e for e in edges
                 if e["source"] in kept and e["target"] in kept]

    # ``ImageReader`` resolves a bare filename under the images directory,
    # which is backend-relative; the smoke test above hops cwd for the same
    # reason.
    backend_dir = Path(__file__).resolve().parents[1]
    prev_cwd = Path.cwd()
    os.chdir(backend_dir)
    try:
        results = asyncio.run(
            execute_graph(nodes, sub_edges, error_mode="fail_fast"))
    finally:
        os.chdir(prev_cwd)

    raw = results["test-image"]["tensor"]
    got = results["add-batch-dim"]["tensor"]

    assert tuple(got.shape) == (1, 1, 28, 28), tuple(got.shape)
    assert 0.0 <= float(raw.min()) and float(raw.max()) <= 1.0, (
        "ImageReader no longer hands over [0, 1]; the shift and scale below "
        "were chosen for that range")
    assert torch.allclose(got[0], _default_normalize()(raw), atol=1e-6), (
        f"the graph feeds Inference [{float(got.min()):.4f}, "
        f"{float(got.max()):.4f}], not what Dataset's default transform "
        f"produces -- the training/inference skew is back")


def test_the_mnist_quickstart_shows_the_digits_it_normalised():
    """The image grid is the picture, not the tensor the model reads.

    ``Dataset`` hands every image over normalised to [-1, 1] and
    ``Visualize`` clips floats to [0, 1] before drawing, so the two together
    erase every pixel at or below mid-grey -- roughly a third of the ink in
    an MNIST digit, drawn pure black. The multiply and add on the way to the
    Visualize node undo exactly the ``Normalize`` the dataset applied, and
    both numbers are read off that transform here rather than written out,
    so they cannot be right today and wrong after somebody changes it.

    ``Inference`` keeps reading the normalised batch: de-normalising what
    the model sees would be the same skew as the inference example's, in
    the other direction.
    """
    normalize = _default_normalize()
    mean, std = float(normalize.mean[0]), float(normalize.std[0])

    payload = _example("Usage_Example/CNN-MNIST/TrainCNN-MNIST")
    edges = payload["edges"]
    by_id = {n["id"]: n for n in payload["nodes"]}

    shift = by_id[_data_source(edges, "show-digits", "data")]
    assert shift["type"] == "Add", shift["type"]
    assert shift["data"]["params"]["alpha"] == 1.0, shift["data"]["params"]

    scale = by_id[_data_source(edges, shift["id"], "tensor_a")]
    assert scale["type"] == "ScalarMultiply", scale["type"]
    assert scale["data"]["params"]["scalar"] == std, (
        f"{scale['id']} multiplies by {scale['data']['params']['scalar']}, "
        f"but Dataset divided by {std}")
    assert _data_source(edges, scale["id"], "tensor") == "sample-batch", (
        "the tiles are drawn from something other than the batch the "
        "predictions were made on")

    offset = by_id[_data_source(edges, shift["id"], "tensor_b")]
    assert offset["type"] == "TensorCreate", offset["type"]
    assert offset["data"]["params"]["fill"] == "full", offset["data"]["params"]
    assert offset["data"]["params"]["value"] == mean, (
        f"{offset['id']} adds {offset['data']['params']['value']}, but "
        f"Dataset subtracted {mean}")

    assert _data_source(edges, "predict", "input") == "sample-batch", (
        "Inference must read the normalised batch, not the de-normalised one")

    # Both halves of the claim the note beside them makes: 16 tiles to look
    # at, and 16 labels to compare them with.
    _assert_argmax_is_printed(edges, by_id, "predict")
