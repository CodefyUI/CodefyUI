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
to. It is also the only builtin needing the network and a 16 GB GPU, and the
canvas gallery card truncates a description at 80 characters with no tooltip --
so where the warning sits in the string is itself an invariant. See
``test_tinystories_lm_example_warns_inside_the_card_truncation`` and
``test_tinystories_lm_example_still_describes_itself``.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path

import pytest

from app.core.graph_engine import execute_graph, validate_graph

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _REPO_ROOT / "examples"


def _discover_builtin_graphs() -> list[Path]:
    return sorted(_EXAMPLES_ROOT.rglob("graph.json"))


_GRAPHS = _discover_builtin_graphs()
assert _GRAPHS, "builtin example smoke suite discovered no examples"

# Node types that pull a real dataset, train for multiple epochs, or load/save
# weights — longer than a few seconds or dependent on prior runs.
# test_chapter_examples.py carries the same list minus ``TextCorpusDataset``,
# which no plugin example uses. This correctly skips execution for eight
# graphs: the six training examples (CNN-MNIST, GPT-Mini, ResNet-CIFAR10, the
# ResNet-18 baseline, the TinyStories LM and VLA-PushWorld), the MNIST
# inference example, which needs weights from a prior training run, and the
# zh-TW Sentence Similarity example, whose encoder is a pack download.
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
    # somebody's money. No shipped example wires it today; it is listed so one
    # added tomorrow cannot start billing from the smoke suite.
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


# ── ResNet-18 / CIFAR-10 baseline: short-epoch execution (core#138) ─────────

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
    "eval": {"batch_size": 512},
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


def _assert_shipped_recipe(nodes: list[dict], edges: list[dict]) -> None:
    """The shipped graph still is the graph the published number came from.

    Two kinds of check, and the second is the one that matters most:

    * every hyperparameter the README and the docs page quote is still the
      value in the file;
    * evaluation still happens on the *held-out* split. The whole milestone
      rests on that one property, and it is exactly the property that is
      invisible to an execution smoke against a synthetic image folder.
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

#: ``EmptyCanvasOverlay.tsx`` renders an example card's description as
#: ``description.slice(0, 80) + '...'`` with no ``title`` attribute. The
#: empty-canvas gallery is the surface a user reads BEFORE pressing Run, and it
#: is the only one that shows a description without being hovered (the sidebar
#: ``TemplatesTab`` puts the full text in a tooltip), so anything past this
#: many characters is not a warning -- it is a footnote nobody sees.
_CARD_VISIBLE_CHARS = 80

#: What must be legible in those 80 characters: the two things that decide
#: whether a user CAN run this graph at all. Everything else is in README.md.
_LM_CARD_REQUIREMENTS = ("16 GB GPU", "download")


def test_tinystories_lm_example_warns_inside_the_card_truncation():
    """The requirements survive the gallery card's 80-character cut.

    This example is the only builtin that needs the network and a 16 GB GPU,
    and a requirement that appears at character 900 of a 1,100-character
    description is a requirement the canvas card silently drops. Asserted
    against the truncated string rather than the whole one, because the whole
    one is not what anybody reads.
    """
    payload = json.loads(_LM_EXAMPLE.read_text(encoding="utf-8"))
    description = payload["description"]
    visible = description[:_CARD_VISIBLE_CHARS]

    for requirement in _LM_CARD_REQUIREMENTS:
        assert requirement in visible, (
            f"{requirement!r} is not inside the {_CARD_VISIBLE_CHARS} "
            f"characters the canvas gallery card shows. The card renders "
            f"{visible!r} and nothing more -- move the requirement to the "
            f"front of the description, not into README.md.")

    # A tooltip-length description is still worth keeping short: the sidebar
    # shows it in full as a `title`, and a native tooltip of several hundred
    # characters is its own kind of unreadable. Generous ceiling, not a style
    # rule -- the point is that the detail lives in README.md.
    assert len(description) <= 500, (
        f"the description is {len(description)} characters; the recipe belongs "
        f"in {_LM_README.name}, not on the card")

    assert _LM_README.is_file(), (
        f"{_LM_README} is missing -- the card's last sentence points at it, and "
        f"the docs pages point at it instead of at the card")


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
    description = payload["description"]
    assert f"{micro_batches:,} micro-batches" in description
    assert f"about {optimizer_steps:,} optimizer steps" in description
    # Validation is scored whole, so its block count is quoted too.
    val_blocks = (_LM_VAL_TOKEN_BUDGET - 1) // _LM_SEQ_LEN
    assert params("ppl")["max_batches"] == 0
    assert f"{val_blocks:,} blocks" in readme


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
