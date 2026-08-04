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
"""

from __future__ import annotations

import asyncio
import json
import os
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
# weights — longer than a few seconds or dependent on prior runs. Same list as
# test_chapter_examples.py. This correctly skips execution for the three
# training examples and the MNIST inference example (which needs weights from
# a prior training run); everything else — including all Model_Architecture
# graphs — must execute.
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

    rng = __import__("random").Random(0)
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


def _shrink_for_ci(nodes: list[dict], data_root: Path) -> None:
    """Rewrite the shipped graph in place into a 2-step CPU run.

    Only params and the two dataset node types change — every edge, the
    augmentation chain, the optimizer/scheduler/loss wiring and the whole
    ``SequentialModel`` layer spec are executed exactly as shipped.
    ``ImageFolderDataset`` is a drop-in for ``Dataset`` here because it
    declares the same ``train_transform``/``eval_transform`` inputs and the
    same ``dataset`` output.
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

    # Evaluation ran end to end over the held-out split.
    assert results["eval"]["total"] == len(_FIXTURE_CLASSES) * _FIXTURE_PER_CLASS
    assert 0.0 <= results["eval"]["accuracy"] <= 1.0

    # The checkpoint landed where CheckpointSaver said it would.
    assert (tmp_path / "models" / "smoke.pt").is_file()
