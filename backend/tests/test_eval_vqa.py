"""The eval pack's VQA path, offline, on CPU, without transformers (#139).

Image + question -> answer -> ExactMatch is the stand-in for a video-LM
evaluation. What is pinned here runs on every CI runner:

* Eval-JSONLDataset reads an item's image through ``image_field``, hashes it
  into the item and the set's identity, and refuses an image it cannot read
  or that lies outside the JSONL file's folder;
* Eval-BatchInference in ``vqa`` mode hands the processor images and
  questions together, records the image and its sha256 on every row, and
  folds the image into ``input_hash``, so CompareRuns refuses two runs that
  asked about different pictures;
* an unreadable image is ONE ``error`` row, and Stop keeps finished rows;
* the shipped example graph (``examples/Eval/VQA-Shapes``) runs end to end
  through the real engine on its bundled images, with the two Hugging Face
  loaders standing in a ViLT-shaped fake (see :class:`ColourVQA`).

``test_eval_hf_local.py`` repeats the model-facing half with a real (tiny,
random) ``ViltForQuestionAnswering`` when transformers is installed, and
runs the example against the real checkpoint behind
``CODEFYUI_PACK_NETWORK_TESTS=1``.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from app.core.execution_context import INTERRUPTED_KEY
from app.core.graph_engine import execute_graph, validate_graph
from cdui_plugins.eval.nodes import _eval_hub as hub
from cdui_plugins.eval.nodes._eval_core import (
    STATUS_CANCELLED,
    STATUS_ERROR,
    STATUS_OK,
    input_hash,
    item_input_hash,
    sha256_bytes,
    table_records,
)
from cdui_plugins.eval.nodes.eval_batch_inference_node import EvalBatchInferenceNode
from cdui_plugins.eval.nodes.eval_compare_runs_node import EvalCompareRunsNode
from cdui_plugins.eval.nodes.eval_exact_match_node import EvalExactMatchNode
from cdui_plugins.eval.nodes.eval_hf_processor_loader_node import EvalHFProcessorLoaderNode
from cdui_plugins.eval.nodes.eval_jsonl_dataset_node import EvalJSONLDatasetNode

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACK_DIR = _REPO_ROOT / "plugins" / "eval"
_SAMPLE = _PACK_DIR / "samples" / "shapes-vqa-12.jsonl"
_EXAMPLE = _PACK_DIR / "examples" / "Eval" / "VQA-Shapes" / "graph.json"

SHA = "d" * 40
COLOURS = ["red", "green", "blue"]


# ── a ViLT-shaped fake ───────────────────────────────────────────────────────

class ColourProcessor:
    """``processor(images=..., text=..., return_tensors="pt")`` like ViltProcessor.

    ``pixel_values`` is each image's mean RGB, ``input_ids`` one id per
    question. A question it has never seen raises, which is how a test makes
    one item of a batch fail.
    """

    def __init__(self, questions):
        self.vocab = {q: i for i, q in enumerate(questions)}
        self.calls: list[int] = []

    def __call__(self, images=None, text=None, return_tensors=None, **_kwargs):
        assert images is not None and text is not None, "VQA needs images and text together"
        assert len(images) == len(text)
        self.calls.append(len(text))
        ids = torch.tensor([[self.vocab[t]] for t in text])
        pixels = torch.stack([
            torch.from_numpy(np.asarray(image, dtype=np.float32)).reshape(-1, 3).mean(dim=0)
            for image in images
        ])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "pixel_values": pixels}


class ColourVQA(nn.Module):
    """Answers every question with the image's most saturated colour.

    The logit of red is ``r - (g + b) / 2``, and so on, so a white
    background contributes nothing. Wrong for counting questions, which is
    the point: the test compares the pipeline with this rule applied
    directly, not with the right answers.
    """

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.config = SimpleNamespace(id2label=dict(enumerate(COLOURS)))

    def forward(self, input_ids, attention_mask=None, pixel_values=None):
        assert pixel_values is not None, "the model was not given the image"
        assert pixel_values.dtype == self.weight.dtype
        r, g, b = pixel_values.unbind(-1)
        logits = torch.stack([r - (g + b) / 2, g - (r + b) / 2, b - (r + g) / 2], dim=-1)
        return SimpleNamespace(logits=logits)


def colour_of(path: Path) -> str:
    """:class:`ColourVQA`'s answer for one image, computed directly."""
    with Image.open(path) as image:
        pixels = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32)).reshape(-1, 3)
    r, g, b = pixels.mean(dim=0).tolist()
    scores = [r - (g + b) / 2, g - (r + b) / 2, b - (r + g) / 2]
    return COLOURS[scores.index(max(scores))]


def _sample_rows() -> list[dict]:
    return [json.loads(line) for line in _SAMPLE.read_text(encoding="utf-8").splitlines()]


def _stamped(model, processor, *, repo="org/vilt", revision=SHA):
    hub.stamp(model, {"model_id": repo, "requested_revision": revision,
                      "model_revision": revision, "revision_source": "pinned",
                      "task": "vqa", "dtype": "float32", "device": "cpu"})
    hub.stamp(processor, {"processor_id": repo, "processor_revision": revision,
                          "revision_source": "pinned"})
    return model, processor


def _load(path, **params):
    defaults = {"input_field": "question", "label_field": "answer", "id_field": "id",
                "image_field": "image"}
    return EvalJSONLDatasetNode().execute({}, {"path": str(path), **defaults, **params})


def _infer(dataset, *, processor=None, context=None, **params):
    questions = [item["input"] for item in dataset["items"]]
    model, processor = _stamped(ColourVQA(), processor or ColourProcessor(questions))
    return EvalBatchInferenceNode().execute(
        {"model": model, "processor": processor, "dataset": dataset},
        {"task": "auto", "batch_size": 4, **params},
        context=context,
    )


def _copy_sample(tmp_path: Path) -> Path:
    """The bundled sample, copied so a test can change an image."""
    shutil.copy(_SAMPLE, tmp_path / _SAMPLE.name)
    shutil.copytree(_SAMPLE.parent / "shapes-vqa", tmp_path / "shapes-vqa")
    return tmp_path / _SAMPLE.name


# ── the dataset ──────────────────────────────────────────────────────────────

def test_the_bundled_sample_has_twelve_questions_about_seven_images():
    out = _load("eval:samples/shapes-vqa-12.jsonl")
    items = out["dataset"]["items"]
    assert [i["id"] for i in items] == [f"q{n:02d}" for n in range(1, 13)]
    assert len({i["image"] for i in items}) == 7
    for item in items:
        path = _PACK_DIR / "samples" / item["image"]
        assert item["image_sha256"] == sha256_bytes(path.read_bytes())
    meta = out["dataset"]["meta"]
    assert meta["fields"]["image"] == "image"
    assert len(meta["images_sha256"]) == 64
    assert "one image each" in out["__log__"]


def test_without_image_field_no_image_is_read(tmp_path):
    out = EvalJSONLDatasetNode().execute({}, {
        "path": "eval:samples/shapes-vqa-12.jsonl", "input_field": "question",
        "label_field": "answer"})
    assert "image_sha256" not in out["dataset"]["items"][0]
    assert "images_sha256" not in out["dataset"]["meta"]


def test_an_image_that_changed_changes_the_sets_identity_but_not_the_files(tmp_path):
    path = _copy_sample(tmp_path)
    before = _load(path)["dataset"]["meta"]
    Image.new("RGB", (8, 8), (0, 0, 0)).save(tmp_path / "shapes-vqa" / "red-ball.png")
    after = _load(path)["dataset"]["meta"]
    assert after["sha256"] == before["sha256"]  # the JSONL did not change
    assert after["images_sha256"] != before["images_sha256"]


def test_limit_keeps_the_identity_of_every_image(tmp_path):
    full = _load("eval:samples/shapes-vqa-12.jsonl")["dataset"]["meta"]
    some = _load("eval:samples/shapes-vqa-12.jsonl", limit=3)["dataset"]
    assert len(some["items"]) == 3
    assert some["meta"]["images_sha256"] == full["images_sha256"]


@pytest.mark.parametrize("image", ["../secret.png", "/etc/hosts", "C:/x.png", "a/../../b.png",
                                   "./x.png", ""])
def test_an_image_outside_the_jsonl_folder_is_refused(tmp_path, image):
    path = tmp_path / "set.jsonl"
    path.write_text(json.dumps({"id": "1", "question": "q?", "answer": "a",
                                "image": image}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        _load(path)


def test_a_symlink_out_of_the_jsonl_folder_is_refused(tmp_path):
    """`link/secret.png` passes the string check; `link` points outside."""
    outside = tmp_path / "outside"
    outside.mkdir()
    Image.new("RGB", (4, 4)).save(outside / "secret.png")
    folder = tmp_path / "set"
    folder.mkdir()
    try:
        (folder / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this platform cannot create a symbolic link here")
    path = folder / "set.jsonl"
    path.write_text(json.dumps({"id": "1", "question": "q?", "answer": "a",
                                "image": "link/secret.png"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1: .* resolves outside"):
        _load(path)
    # A link that stays inside the folder is fine.
    (folder / "img").mkdir()
    Image.new("RGB", (4, 4)).save(folder / "img" / "ok.png")
    (folder / "alias").symlink_to(folder / "img", target_is_directory=True)
    path.write_text(json.dumps({"id": "1", "question": "q?", "answer": "a",
                                "image": "alias/ok.png"}) + "\n", encoding="utf-8")
    assert _load(path)["dataset"]["items"][0]["image"] == "alias/ok.png"


def test_vqa_caps_max_length_at_the_models_text_length():
    """ViLT reads at most 40 text tokens; the default max_length (256) used
    to make every batch fail on a long question."""
    dataset = _load("eval:samples/shapes-vqa-12.jsonl")["dataset"]
    questions = [item["input"] for item in dataset["items"]]
    seen: list[int] = []

    class Recording(ColourProcessor):
        def __call__(self, *args, max_length=None, **kwargs):
            seen.append(max_length)
            return super().__call__(*args, **kwargs)

    model = ColourVQA()
    model.config.max_position_embeddings = 40
    model, processor = _stamped(model, Recording(questions))
    out = EvalBatchInferenceNode().execute(
        {"model": model, "processor": processor, "dataset": dataset},
        {"task": "auto", "batch_size": 4})
    assert set(seen) == {40}
    assert out["predictions"]["meta"]["inference"]["max_length"] == 40
    out = EvalBatchInferenceNode().execute(
        {"model": model, "processor": processor, "dataset": dataset},
        {"task": "auto", "batch_size": 4, "max_length": 16})
    assert out["predictions"]["meta"]["inference"]["max_length"] == 16


def test_a_missing_image_is_refused_with_its_line(tmp_path):
    path = tmp_path / "set.jsonl"
    path.write_text(
        json.dumps({"id": "1", "question": "q?", "answer": "a", "image": "img/none.png"}) + "\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 1: cannot read image 'img/none.png'"):
        _load(path)


# ── inference ────────────────────────────────────────────────────────────────

def test_vqa_rows_record_the_image_and_fold_it_into_the_input_hash():
    dataset = _load("eval:samples/shapes-vqa-12.jsonl")["dataset"]
    out = _infer(dataset)
    table = out["predictions"]
    assert table["columns"][-4:] == ["score", "topk", "image", "image_sha256"]
    assert table["meta"]["inference"]["task"] == "vqa"
    assert table["meta"]["dataset"]["images_sha256"] == dataset["meta"]["images_sha256"]
    assert table["meta"]["reproducibility"]["reproducible"]
    for item, row in zip(dataset["items"], table_records(table)):
        assert row["status"] == STATUS_OK
        assert row["image"] == item["image"]
        assert row["image_sha256"] == item["image_sha256"]
        assert row["input_hash"] == item_input_hash(item["input"], item["image_sha256"])
        assert row["input_hash"] != input_hash(item["input"])
        assert row["raw_answer"] == colour_of(_PACK_DIR / "samples" / item["image"])


def test_one_question_about_two_pictures_has_two_identities():
    rows = {r["id"]: r for r in table_records(
        _infer(_load("eval:samples/shapes-vqa-12.jsonl")["dataset"])["predictions"])}
    # q02 and q03 ask "How many balls are there?" about two different pictures
    # (one red ball, two red balls): same text, two identities.
    assert rows["q02"]["input"] == rows["q03"]["input"] == "How many balls are there?"
    assert rows["q02"]["image"] != rows["q03"]["image"]
    assert rows["q02"]["input_hash"] != rows["q03"]["input_hash"]
    # And one picture asked two questions gives two identities too.
    assert rows["q01"]["image"] == rows["q02"]["image"]
    assert rows["q01"]["input_hash"] != rows["q02"]["input_hash"]


def test_compare_runs_refuses_two_runs_over_different_pictures(tmp_path):
    path = _copy_sample(tmp_path)
    run_a = _infer(_load(path)["dataset"])["predictions"]
    Image.new("RGB", (64, 64), (0, 200, 0)).save(tmp_path / "shapes-vqa" / "red-ball.png")
    run_b = _infer(_load(path)["dataset"])["predictions"]
    with pytest.raises(ValueError, match="q01"):
        EvalCompareRunsNode().execute({"table_a": run_a, "table_b": run_b}, {})
    same = EvalCompareRunsNode().execute({"table_a": run_a, "table_b": run_a}, {})
    assert same["agreement"] == 1.0


def test_an_unreadable_image_is_one_error_and_its_batch_mates_finish(tmp_path):
    path = _copy_sample(tmp_path)
    dataset = _load(path)["dataset"]
    # Corrupt it after the dataset hashed it: the predictor opens it per batch.
    (tmp_path / "shapes-vqa" / "orange-ball.png").write_bytes(b"not a png")
    rows = table_records(_infer(dataset)["predictions"])
    broken = [r for r in rows if r["status"] == STATUS_ERROR]
    assert [r["id"] for r in broken] == ["q09"]
    assert broken[0]["answer"] is None and "UnidentifiedImageError" in broken[0]["error"]
    assert all(r["status"] == STATUS_OK for r in rows if r["id"] != "q09")


def test_an_item_without_an_image_is_an_error_naming_image_field():
    dataset = EvalJSONLDatasetNode().execute({}, {
        "path": "eval:samples/shapes-vqa-12.jsonl", "input_field": "question",
        "label_field": "answer"})["dataset"]
    rows = table_records(_infer(dataset, task="vqa")["predictions"])
    assert {r["status"] for r in rows} == {STATUS_ERROR}
    assert "image_field" in rows[0]["error"]


class _StopAfter:
    def __init__(self, n):
        self.n, self.checks = n, 0

    def should_stop(self):
        self.checks += 1
        return self.checks > self.n


def test_stop_keeps_the_finished_vqa_rows():
    dataset = _load("eval:samples/shapes-vqa-12.jsonl")["dataset"]
    out = _infer(dataset, context=_StopAfter(2))
    rows = table_records(out["predictions"])
    assert [r["status"] for r in rows] == [STATUS_OK] * 8 + [STATUS_CANCELLED] * 4
    assert out.get(INTERRUPTED_KEY)
    exact = EvalExactMatchNode().execute({"predictions": out["predictions"]}, {})
    assert exact["table"].tolist()[0][1:] == [8.0, 4.0]


def test_the_processor_loader_refuses_a_bare_tokenizer_for_a_vqa_model():
    model, _ = _stamped(ColourVQA(), ColourProcessor([]))
    with pytest.raises(ValueError, match="kind` to auto"):
        EvalHFProcessorLoaderNode().execute({"model": model}, {"kind": "tokenizer"})


# ── the example graph, end to end, offline ───────────────────────────────────

def _example() -> dict:
    return json.loads(_EXAMPLE.read_text(encoding="utf-8"))


def test_the_example_validates_and_pins_its_checkpoint():
    graph = _example()
    assert validate_graph(graph["nodes"], graph["edges"]) == []
    loader = next(n for n in graph["nodes"] if n.get("type") == "eval:Eval-HFModelLoader")
    assert loader["data"]["params"]["model_id"] == "dandelin/vilt-b32-finetuned-vqa"
    assert hub.is_commit_sha(loader["data"]["params"]["revision"])
    proc = next(n for n in graph["nodes"] if n.get("type") == "eval:Eval-HFProcessorLoader")
    assert proc["data"]["params"]["kind"] == "auto"
    data = next(n for n in graph["nodes"] if n.get("type") == "eval:Eval-JSONLDataset")
    assert data["data"]["params"]["path"] == "eval:samples/shapes-vqa-12.jsonl"
    assert data["data"]["params"]["image_field"] == "image"


def test_the_example_runs_end_to_end_with_a_vilt_shaped_stand_in(monkeypatch):
    """The shipped graph, the real engine and the bundled images; only the
    two Hugging Face loads are replaced, so nothing is downloaded."""
    questions = [r["question"] for r in _sample_rows()]
    loads: list[dict] = []

    def resolve(model_id, revision):
        return {"model_id": model_id, "requested_revision": revision or SHA,
                "revision": revision or SHA, "source": "pinned", "note": "",
                "files": None, "cache_only": True}

    def load_model(resolved, *, task, dtype, device):
        loads.append({"model": resolved["model_id"], "task": task})
        return ColourVQA(), "vqa"

    def load_processor(resolved, *, kind):
        loads.append({"processor": resolved["model_id"], "kind": kind})
        return ColourProcessor(questions)

    monkeypatch.setattr(hub, "resolve_revision", resolve)
    monkeypatch.setattr(hub, "load_model", load_model)
    monkeypatch.setattr(hub, "load_processor", load_processor)

    graph = _example()
    results = asyncio.run(execute_graph(graph["nodes"], graph["edges"], error_mode="fail_fast"))

    assert {"processor": "dandelin/vilt-b32-finetuned-vqa", "kind": "auto"} in loads
    rows = table_records(results["infer"]["predictions"])
    assert [r["status"] for r in rows] == [STATUS_OK] * 12
    expected = [colour_of(_PACK_DIR / "samples" / r["image"]) for r in _sample_rows()]
    assert [r["answer"] for r in rows] == expected
    right = sum(e == r["answer"] for e, r in zip(expected, _sample_rows()))
    assert results["em"]["value"] == pytest.approx(right / 12)
    assert results["top3"]["value"] >= results["em"]["value"]
    producer = results["infer"]["predictions"]["meta"]["producer"]
    assert producer["model_revision"] == producer["processor_revision"]
