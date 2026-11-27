"""The practical examples (Phase 3): their notes, and the numbers the notes quote.

Each example runs once per session under the seed its file stores, the way
the canvas runs an opened example, with a temporary models directory. A
number a note quotes is checked against the run with a tolerance, not to the
last digit: the same seed gives slightly different floats on another CPU or
torch build, and the note only has to stay true. Where a note makes a claim
(transfer loses here, standardisation wins, the later checkpoint is better),
the claim itself is asserted.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import pytest

from app.core.graph_engine import execute_graph, is_note_node

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_EXAMPLES_ROOT = _BACKEND_DIR.parent / "examples"

PRACTICAL = (
    "Data/Dataset-Inspection",
    "Data/Split-And-Leak-Check",
    "Training/Early-Stopping-Best-Checkpoint",
    "Training/Transfer-Frozen-Backbone",
    "Evaluation/Multi-Seed-Mean-Std",
    "Evaluation/Checkpoint-Confusion-Matrix",
    "Evaluation/Compare-Two-Checkpoints",
    "Evaluation/Embedding-Projection-2D",
    "Research/LR-Range-Test",
    "Research/LR-BatchSize-Grid",
    "Research/Ablation-Input-Standardization",
    "Deploy/Classifier-Service/Train-And-Export",
    "Deploy/Classifier-Service/Serve-As-Function",
    "LLM/RAG-Retrieval-Hit-Rate",
    "LLM/Finetune-Char-LM-Custom-Text",
)

#: The six that use the bundled handwritten digits (review A).
DIGITS = (
    "Data/Dataset-Inspection",
    "Training/Early-Stopping-Best-Checkpoint",
    "Training/Transfer-Frozen-Backbone",
    "Evaluation/Multi-Seed-Mean-Std",
    "Evaluation/Checkpoint-Confusion-Matrix",
    "Evaluation/Embedding-Projection-2D",
)


def _graph(key: str) -> dict:
    return json.loads((_EXAMPLES_ROOT / key / "graph.json").read_text(encoding="utf-8"))


def _execute(key: str) -> dict:
    from app.core.execution_context import ExecutionContext

    graph = _graph(key)
    prev = Path.cwd()
    os.chdir(_BACKEND_DIR)
    try:
        return asyncio.run(execute_graph(
            graph["nodes"], graph["edges"], error_mode="fail_fast",
            subgraphs=graph.get("subgraphs"),
            context=ExecutionContext(device="cpu", seed=graph["settings"]["seed"])))
    finally:
        os.chdir(prev)


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory):
    """One models directory for the module: Serve reads what Train wrote."""
    from app.config import settings

    path = tmp_path_factory.mktemp("models")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(settings, "MODELS_DIR", path)
        yield path


_RESULTS: dict[str, dict] = {}


@pytest.fixture(scope="module")
def results(models_dir):
    """Run an example at most once per module, under its own seed."""
    def get(key: str) -> dict:
        if key not in _RESULTS:
            _RESULTS[key] = _execute(key)
        return _RESULTS[key]

    yield get
    _RESULTS.clear()


def _overview(key: str) -> str:
    """The English half of the overview note."""
    note = next(n for n in _graph(key)["nodes"] if n["id"] == "note-overview")
    return note["data"]["noteContent"].partition("\n\n")[0]


def _quoted(pattern: str, key: str) -> list[str]:
    match = re.search(pattern, _overview(key))
    assert match, f"{key}'s overview no longer says /{pattern}/"
    return list(match.groups())


def _notes(graph: dict) -> list[dict]:
    return [n for n in graph["nodes"] if is_note_node(n)]


# ── every practical example ───────────────────────────────────────────────

@pytest.mark.parametrize("key", PRACTICAL)
def test_a_practical_example_opens_with_an_overview_bound_to_start(key):
    graph = _graph(key)
    notes = _notes(graph)
    ids = [n["id"] for n in notes]
    assert len(ids) == len(set(ids)), ids
    overview = [n for n in notes if n["id"] == "note-overview"]
    assert overview, f"{key} has no overview note"
    start = [n for n in graph["nodes"] if n.get("type") == "Start"]
    assert overview[0]["data"]["boundToNodeId"] == start[0]["id"]
    assert isinstance(graph["settings"]["seed"], int)


@pytest.mark.parametrize("key", PRACTICAL)
def test_a_practical_note_sits_where_its_binding_says(key):
    graph = _graph(key)
    by_id = {n["id"]: n for n in graph["nodes"]}
    for note in _notes(graph):
        data = note["data"]
        anchor = by_id[data["boundToNodeId"]]
        assert not is_note_node(anchor)
        offset = data["boundOffset"]
        assert note["position"] == {"x": anchor["position"]["x"] + offset["x"],
                                    "y": anchor["position"]["y"] + offset["y"]}


@pytest.mark.parametrize("key", PRACTICAL)
def test_a_practical_note_is_written_in_both_languages(key):
    for note in _notes(_graph(key)):
        english, _, chinese = note["data"]["noteContent"].partition("\n\n")
        assert english.strip() and chinese.strip(), note["id"]
        assert any("一" <= ch <= "鿿" for ch in chinese), note["id"]
        assert not any("一" <= ch <= "鿿" for ch in english), note["id"]
        # The generator writes "?" where a measured number was missing.
        assert not re.search(r"[\s(]\?[\s).,;:]", english + " "), (
            f"{key}: {note['id']} still has an unfilled number")


@pytest.mark.parametrize("key", DIGITS)
def test_the_digits_examples_read_the_bundled_digits(key):
    graph = _graph(key)
    paths = [n["data"]["params"].get("path") for n in graph["nodes"] if n.get("type") == "CSVReader"]
    assert paths == ["data/samples/digits.csv"]
    assert "vision" in graph["gallery"]["tags"]
    assert (_BACKEND_DIR / "data" / "samples" / "digits.csv").is_file()


# ── data ──────────────────────────────────────────────────────────────────

def test_the_inspection_counts_balanced_digits_and_a_dead_border(results):
    key = "Data/Dataset-Inspection"
    lo, hi, mean, never = _quoted(
        r"\((\d+) to (\d+), so the classes.*?mean \(([\d.]+)\).*?(\d+) of the 64 pixels", key)
    r = results(key)
    counts = r["inspect"]["out1"]
    assert counts.sum().item() == 1797
    assert (int(counts.min()), int(counts.max())) == (int(lo), int(hi))
    assert r["stats"]["mean"] == pytest.approx(float(mean), abs=0.01)
    assert f"{never} of 64 pixels are 0 in every image" in r["inspect"]["out4"]
    assert tuple(r["inspect"]["out2"].shape) == (80, 40)  # 10 digits x 5 samples of 8x8


def test_the_leak_check_finds_the_repeated_iris_row_and_its_fix_removes_it(results):
    key = "Data/Split-And-Leak-Check"
    r = results(key)
    overview = _overview(key)
    assert r["leak-before"]["out1"] in overview
    assert r["leak-after"]["out1"] in overview
    assert "1 of 30 test rows" in r["leak-before"]["out1"]
    assert r["leak-after"]["out1"].startswith("0 of 30 validation rows and 0 of 30 test rows")
    assert r["dedup"]["out3"] == "150 rows, 1 exact repeat removed, 149 left"


# ── training ──────────────────────────────────────────────────────────────

def test_early_stopping_stops_and_keeps_the_best_epoch(results):
    key = "Training/Early-Stopping-Best-Checkpoint"
    best, stop, acc = _quoted(r"best epoch is (\d+) and training stops after (\d+);.*?at ([\d.]+)\.", key)
    r = results(key)
    metrics = r["train"]["metrics"]
    assert metrics["stopped_early"] is True
    assert metrics["total_epochs_run"] - metrics["best_epoch"] == 10  # the patience
    assert abs(metrics["best_epoch"] - int(best)) <= 6
    assert abs(metrics["total_epochs_run"] - int(stop)) <= 6
    assert r["eval"]["accuracy"] == pytest.approx(float(acc), abs=0.03)
    assert Path(r["save"]["path"]).name == "digits-best.pt"


def test_transfer_beats_the_baseline_on_twenty_labels(results):
    """Pretrained on all ten digits, the frozen backbone answers the new
    question from 20 labels better than the same model from scratch."""
    key = "Training/Transfer-Frozen-Backbone"
    test_n, transfer, scratch = _quoted(
        r"On the ([\d,]+) images neither run trained on, the frozen backbone scores ([\d.]+) "
        r"and the baseline ([\d.]+)\.", key)
    r = results(key)
    assert r["freeze"]["report"] == (
        "frozen: 4 tensors, 16,576 parameters; trainable: 2 tensors, 130 parameters")
    frozen_note = next(n for n in _graph(key)["nodes"] if n["id"] == "note-freeze")
    assert r["freeze"]["report"] in frozen_note["data"]["noteContent"]
    assert len(r["ds-pre"]["dataset"]) == 1000
    assert len(r["ds-few"]["dataset"]) == 20
    assert r["b-eval"]["total"] == int(test_n.replace(",", "")) == 1797 - 1000 - 20
    assert r["b-eval"]["accuracy"] == pytest.approx(float(transfer), abs=0.05)
    assert r["s-eval"]["accuracy"] == pytest.approx(float(scratch), abs=0.05)
    assert r["b-eval"]["accuracy"] - r["s-eval"]["accuracy"] > 0.05
    model = r["b-train"]["model"]
    assert all(p.requires_grad for p in model.layers["head"].parameters())
    for layer in ("backbone1", "backbone2"):
        assert not any(p.requires_grad for p in model.layers[layer].parameters())
    # The backbone the head trained on is the pretrained one, unchanged.
    pretrained = r["a-train"]["model"].state_dict()
    for name, value in model.state_dict().items():
        if not name.startswith("layers.head."):
            assert (value == pretrained[name]).all(), name


# ── evaluation ────────────────────────────────────────────────────────────

def test_five_seeds_give_five_different_accuracies(results):
    """The block instances get independent child seeds (review B)."""
    key = "Evaluation/Multi-Seed-Mean-Std"
    graph = _graph(key)
    assert [d["id"] for d in graph["subgraphs"]] == ["train-and-score"]
    instances = [n["id"] for n in graph["nodes"] if n.get("type") == "subgraph:train-and-score"]
    assert len(instances) == 5
    r = results(key)
    scores = [r[f"{i}/eval"]["accuracy"] for i in instances]
    assert len(set(scores)) > 1, scores
    mean, std = _quoted(r"mean accuracy ([\d.]+), standard deviation ([\d.]+)", key)
    assert r["summary"]["out1"] == pytest.approx(float(mean), abs=0.03)
    assert 0 < r["summary"]["out2"] < 0.05
    assert float(std) > 0


def test_the_confusion_matrix_covers_the_450_test_digits(results):
    key = "Evaluation/Checkpoint-Confusion-Matrix"
    acc, weak = _quoted(r"Accuracy is ([\d.]+), but.*?weakest digit is (\d) \(", key)
    r = results(key)
    matrix = r["confusion"]["out1"]
    assert tuple(matrix.shape) == (10, 10)
    assert matrix.sum().item() == 450
    assert matrix.diag().sum().item() / 450 == pytest.approx(float(acc), abs=0.03)
    assert 0 <= int(weak) <= 9
    assert Path(r["save"]["path"]).name == "digits-mlp.pt"


def test_the_later_checkpoint_beats_the_earlier_one(results):
    key = "Evaluation/Compare-Two-Checkpoints"
    acc1, acc2 = _quoted(r"Test accuracy goes from ([\d.]+) to ([\d.]+)\.", key)
    r = results(key)
    early, late = r["eval-early"]["accuracy"], r["eval-late"]["accuracy"]
    assert early == pytest.approx(float(acc1), abs=0.06)
    assert late == pytest.approx(float(acc2), abs=0.03)
    assert late > early + 0.1
    groups = _graph(key)["segmentGroups"]
    assert {(g["headNodeId"], g["tailNodeId"]) for g in groups} == {
        ("load-early", "infer-early"), ("load-late", "infer-late")}
    compare_note = next(n for n in _graph(key)["nodes"] if n["id"] == "note-compare")
    for node_id in ("load-early", "infer-early", "load-late", "infer-late"):
        assert node_id in compare_note["data"]["noteContent"]


def test_the_trained_layer_separates_digits_better_than_raw_pixels(results):
    key = "Evaluation/Embedding-Projection-2D"
    raw, trained = _quoted(r"same digit: ([\d.]+) for raw pixels, ([\d.]+) for the trained layer", key)
    r = results(key)
    assert r["sep-raw"]["out1"] == pytest.approx(float(raw), abs=0.03)
    assert r["sep-trained"]["out1"] == pytest.approx(float(trained), abs=0.06)
    assert r["sep-trained"]["out1"] > r["sep-raw"]["out1"]
    assert tuple(r["hidden-trained"]["out1"].shape) == (450, 8)


# ── research ──────────────────────────────────────────────────────────────

def test_the_range_test_picks_a_rate_the_confirming_run_trains_with(results):
    key = "Research/LR-Range-Test"
    r = results(key)
    suggested = r["pick"]["out2"]
    assert 0.01 < suggested < 0.2
    typed = _graph(key)["nodes"]
    confirm_lr = next(n for n in typed if n["id"] == "c-opt")["data"]["params"]["lr"]
    assert confirm_lr == pytest.approx(suggested, rel=0.5)
    assert r["c-eval"]["accuracy"] >= 0.9
    assert "suggested lr" in _overview(key)


def test_the_grid_ranks_lr_005_batch_8_first(results):
    key = "Research/LR-BatchSize-Grid"
    graph = _graph(key)
    assert sorted(d["id"] for d in graph["subgraphs"]) == ["train-lr-0001", "train-lr-005"]
    r = results(key)
    table = r["table"]["out1"]
    assert table.startswith("lr 0.05, batch 8:")
    quoted = dict(re.findall(r"(lr [\d.]+, batch \d+): (\d+\.\d+)", _overview(key)))
    measured = dict(re.findall(r"(lr [\d.]+, batch \d+): (\d+\.\d+)", table))
    assert quoted.keys() == measured.keys()
    for cell, value in quoted.items():
        assert float(measured[cell]) == pytest.approx(float(value), abs=0.05), cell


def test_standardisation_is_worth_what_the_ablation_note_says(results):
    key = "Research/Ablation-Input-Standardization"
    scaled, raw = _quoted(r"with standardisation ([\d.]+), without ([\d.]+)\.", key)
    r = results(key)
    assert r["print-scaled"]["value"] == pytest.approx(float(scaled), abs=0.05)
    assert r["print-raw"]["value"] == pytest.approx(float(raw), abs=0.05)
    assert r["print-scaled"]["value"] - r["print-raw"]["value"] > 0.3


# ── deploy ────────────────────────────────────────────────────────────────

def test_serve_without_the_weights_says_which_file_is_missing(tmp_path, monkeypatch):
    """The Serve note tells the reader what this error looks like."""
    from app.config import settings

    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="Weights file not found: .*iris-classifier.safetensors"):
        _execute("Deploy/Classifier-Service/Serve-As-Function")
    assert _overview("Deploy/Classifier-Service/Serve-As-Function").startswith(
        "Run Train-And-Export first")
    assert '"Weights file not found:"' in _overview("Deploy/Classifier-Service/Serve-As-Function")


def test_train_then_serve_answers_with_the_exported_weights(results, models_dir):
    train = results("Deploy/Classifier-Service/Train-And-Export")
    assert Path(train["save"]["path"]) == models_dir / "iris-classifier.safetensors"
    assert train["eval"]["accuracy"] >= 0.9
    key = "Deploy/Classifier-Service/Serve-As-Function"
    species, probs = _quoted(r"answers (\w+) \(([^)]*)\)", key)
    r = results(key)
    assert r["label"]["out1"] == species
    quoted = {name: float(value) for name, value in re.findall(r"(\w+) ([\d.]+)", probs)}
    for name, value in r["label"]["out2"].items():
        assert value == pytest.approx(quoted[name], abs=0.1), name
    graph = _graph(key)
    types = {n["type"] for n in graph["nodes"]}
    assert {"GraphInput", "GraphOutput", "ModelLoader"} <= types
    # Both graphs build the same layers, or the weights would not load.
    layers = {g: next(n for n in _graph(g)["nodes"] if n["type"] == "SequentialModel" and n["id"] == "model")
              ["data"]["params"]["layers"]
              for g in ("Deploy/Classifier-Service/Train-And-Export", key)}
    assert len(set(layers.values())) == 1


# ── llm ───────────────────────────────────────────────────────────────────

def test_the_retriever_hit_rate_is_the_one_the_note_quotes(results):
    key = "LLM/RAG-Retrieval-Hit-Rate"
    r = results(key)
    assert r["score"]["out3"] in _overview(key)
    assert str(r["chunk"]["count"]) in _overview(key)
    assert r["score"]["out2"] >= 0.8


def test_the_bag_of_words_note_says_what_it_is_not():
    note = next(n for n in _graph("LLM/RAG-Retrieval-Hit-Rate")["nodes"] if n["id"] == "note-bow")
    text = note["data"]["noteContent"]
    assert "not semantic retrieval" in text and "TextEmbedding" in text


def test_fine_tuning_memorises_the_facts_and_forgets_the_stories(results):
    key = "LLM/Finetune-Char-LM-Custom-Text"
    before, after = _quoted(r"facts falls from ([\d.]+) to ([\d.]+)", key)
    stories_before, stories_after = _quoted(r"stories rises from ([\d.]+) to ([\d.]+)", key)
    r = results(key)
    assert r["ppl-before"]["perplexity"] == pytest.approx(float(before), rel=0.3)
    assert r["ppl-after"]["perplexity"] < 1.5
    assert float(after) < 1.5
    assert r["ppl-stories"]["perplexity"] > 3 * r["ppl-stories-before"]["perplexity"]
    assert r["ppl-stories"]["perplexity"] == pytest.approx(float(stories_after), rel=0.3)
    assert r["ppl-stories-before"]["perplexity"] == pytest.approx(float(stories_before), rel=0.3)
    assert r["gen-after"]["text"].startswith("Water boils at 100 degrees")
