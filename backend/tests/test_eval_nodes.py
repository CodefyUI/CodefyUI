"""The eval pack's nodes, offline, on CPU, without transformers.

Everything here runs on a CI runner that has never installed transformers:
the inference loop talks to its model and processor only through the call
shapes a Hugging Face model and tokenizer have, so a lookup-table model and
a dict tokenizer stand in for them. ``test_eval_hf_local.py`` repeats the
model-facing half against real (tiny, randomly initialised) transformers
models when the library is installed.

What is pinned, by issue #139's acceptance list:

* metric parity with scikit-learn on fixtures (accuracy, micro/macro F1,
  top-k accuracy);
* CompareRuns aligns reordered rows by id, and refuses duplicate ids,
  mismatched id sets, different inputs and different labels;
* a stop between batches keeps the finished items ``ok`` and marks the rest
  ``cancelled``; one item that raises is ``error`` and the others carry on;
* the provenance a prediction table records, and when it is reproducible.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest
import torch
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from torch import nn

from app.core import loop_control
from app.core.execution_context import INTERRUPTED_KEY
from cdui_plugins.eval.nodes import _eval_hub as hub
from cdui_plugins.eval.nodes._eval_core import (
    PREDICTION_COLUMNS,
    STATUS_CANCELLED,
    STATUS_ERROR,
    STATUS_OK,
    input_hash,
    make_dataset,
    make_table,
    table_records,
)
from cdui_plugins.eval.nodes._eval_infer import GenerationPredictor, item_seed
from cdui_plugins.eval.nodes.eval_batch_inference_node import EvalBatchInferenceNode
from cdui_plugins.eval.nodes.eval_compare_runs_node import EvalCompareRunsNode
from cdui_plugins.eval.nodes.eval_hf_model_loader_node import EvalHFModelLoaderNode
from cdui_plugins.eval.nodes.eval_jsonl_dataset_node import EvalJSONLDatasetNode
from cdui_plugins.eval.nodes.eval_exact_match_node import EvalExactMatchNode
from cdui_plugins.eval.nodes.eval_f1_score_node import EvalF1ScoreNode
from cdui_plugins.eval.nodes.eval_topk_accuracy_node import EvalTopKAccuracyNode
from cdui_plugins.eval.nodes.eval_prompt_template_node import EvalPromptTemplateNode

SHA_A = "a" * 40
SHA_B = "b" * 40


# ── fakes ────────────────────────────────────────────────────────────────────

class LookupTokenizer:
    """``tokenizer(texts, ...)`` -> one id per text, looked up in *vocab*.

    An unknown text raises, which is how a test makes ONE item of a batch
    fail: the batch call raises, and the loop retries item by item.
    """

    eos_token_id = None

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.calls: list[int] = []

    def __call__(self, texts, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        self.calls.append(len(texts))
        for text in texts:
            if text not in self.vocab:
                raise KeyError(f"unknown text {text!r}")
        ids = torch.tensor([[self.vocab[t]] for t in texts])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class LookupClassifier(nn.Module):
    """Row *i* of a fixed logit table is the output for input id *i*."""

    def __init__(self, logits, id2label: dict[int, str]):
        super().__init__()
        self.table = nn.Parameter(torch.tensor(logits, dtype=torch.float32), requires_grad=False)
        self.config = SimpleNamespace(id2label=id2label)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.table[input_ids[:, 0]])


class StopAfter:
    """An execution context whose Stop is pressed after *n* batches."""

    def __init__(self, n: int):
        self.n = n
        self.checks = 0
        self.metrics: list[tuple[str, float, int]] = []

    def should_stop(self) -> bool:
        self.checks += 1
        return self.checks > self.n

    def log_metric(self, name, value, step, node_id=None):
        self.metrics.append((name, float(value), int(step)))


def _dataset(texts: list[str], labels: list[str | None], *, sha: str = "f" * 64):
    items = [
        {"id": f"s{i:02d}", "input": t, "label": y, "line": i + 1, "fields": {"text": t, "label": y}}
        for i, (t, y) in enumerate(zip(texts, labels))
    ]
    return make_dataset(items, {
        "path": "fixture.jsonl", "sha256": sha, "rows": len(items), "used": len(items),
        "fields": {"input": "text", "label": "label", "id": "id"}, "template": None,
    })


def _sentiment_fixture(n: int = 10):
    """*n* texts, a classifier that gets the odd ones wrong, and the labels."""
    texts = [f"text {i}" for i in range(n)]
    labels = ["positive" if i % 2 == 0 else "negative" for i in range(n)]
    # Even items: logits favour positive (right). Odd items: also positive (wrong).
    logits = [[0.0, 3.0] for _ in range(n)]
    model = LookupClassifier(logits, {0: "NEGATIVE", 1: "POSITIVE"})
    tokenizer = LookupTokenizer({t: i for i, t in enumerate(texts)})
    return model, tokenizer, _dataset(texts, labels)


def _stamped(model, tokenizer, *, model_rev=SHA_A, proc_rev=SHA_A, repo="org/model"):
    hub.stamp(model, {"model_id": repo, "requested_revision": "main",
                      "model_revision": model_rev, "revision_source": "hub",
                      "task": "classification", "dtype": "float32", "device": "cpu"})
    hub.stamp(tokenizer, {"processor_id": repo, "processor_revision": proc_rev,
                          "revision_source": "pinned"})
    return model, tokenizer


def _infer(model, tokenizer, dataset, context=None, progress=None, **params):
    return EvalBatchInferenceNode().execute(
        {"model": model, "processor": tokenizer, "dataset": dataset},
        {"task": "classification", "batch_size": 4, **params},
        progress,
        context=context,
    )


def _prediction_table(rows: list[dict], meta: dict | None = None) -> dict:
    """A prediction table from partial row dicts, defaults filled in."""
    full = []
    for r in rows:
        text = r.get("input", f"input of {r['id']}")
        full.append([
            r["id"], text, r.get("input_hash", input_hash(text)), r.get("answer"),
            r.get("answer"), r.get("label"), r.get("status", STATUS_OK), r.get("error"),
        ])
    return make_table(PREDICTION_COLUMNS, full, meta or {})


# ── JSONLDataset ─────────────────────────────────────────────────────────────

def _write(tmp_path, lines: list[str], name: str = "set.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _load(path, **params):
    return EvalJSONLDatasetNode().execute({}, {"path": str(path), **params})


def test_jsonl_items_carry_ids_labels_and_the_files_identity(tmp_path):
    path = _write(tmp_path, [
        json.dumps({"id": "q1", "text": "first", "label": 1}),
        "",
        json.dumps({"text": "second", "label": "neg"}),
        json.dumps({"id": "q3", "text": "third"}),
    ])
    out = _load(path)
    items = out["dataset"]["items"]
    assert [i["id"] for i in items] == ["q1", "line-3", "q3"]
    assert [i["label"] for i in items] == ["1", "neg", None]
    assert out["labels"] == ["1", "neg", None]
    meta = out["dataset"]["meta"]
    assert meta["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (meta["rows"], meta["used"]) == (3, 3)


def test_jsonl_refuses_duplicate_ids_naming_them(tmp_path):
    path = _write(tmp_path, [json.dumps({"id": i, "text": "x"}) for i in ("a", "b", "a", "c", "b")])
    with pytest.raises(ValueError, match=r"duplicate ids .*\['a', 'b'\]"):
        _load(path)


@pytest.mark.parametrize(("line", "message"), [
    ("{not json", "line 2 is not valid JSON"),
    ("[1, 2]", "line 2 is a JSON list"),
    (json.dumps({"id": "z", "text": "  "}), "line 2 has no text"),
])
def test_jsonl_names_the_line_it_cannot_use(tmp_path, line, message):
    path = _write(tmp_path, [json.dumps({"id": "ok", "text": "fine"}), line])
    with pytest.raises(ValueError, match=message):
        _load(path)


def test_the_bundled_sample_resolves_inside_the_pack():
    out = EvalJSONLDatasetNode().execute({}, {"path": "eval:samples/sst2-mini-20.jsonl"})
    items = out["dataset"]["items"]
    assert len(items) == 20
    assert {i["label"] for i in items} == {"negative", "positive"}
    assert sum(i["label"] == "positive" for i in items) == 10


@pytest.mark.parametrize("path", ["eval:../registry.json", "eval:", "eval:samples/../../x"])
def test_a_pack_path_cannot_leave_the_pack(path):
    with pytest.raises(ValueError, match="does not name a file inside the eval pack"):
        EvalJSONLDatasetNode().execute({}, {"path": path})


def test_limit_keeps_the_first_items_but_the_identity_of_the_whole_file(tmp_path):
    path = _write(tmp_path, [json.dumps({"id": f"r{i}", "text": f"t{i}"}) for i in range(5)])
    out = _load(path, limit=2)
    assert [i["id"] for i in out["dataset"]["items"]] == ["r0", "r1"]
    assert (out["dataset"]["meta"]["rows"], out["dataset"]["meta"]["used"]) == (5, 2)


# ── PromptTemplate ───────────────────────────────────────────────────────────

def test_the_template_rewrites_inputs_and_joins_the_identity():
    dataset = _dataset(["good film"], ["positive"])
    dataset["items"][0]["fields"]["genre"] = "drama"
    out = EvalPromptTemplateNode().execute(
        {"dataset": dataset}, {"template": "Review ({genre}): {input}\nSentiment:"})
    item = out["dataset"]["items"][0]
    assert item["input"] == "Review (drama): good film\nSentiment:"
    assert item["source_input"] == "good film"
    template = out["dataset"]["meta"]["template"]
    assert template["sha256"] == hashlib.sha256(
        b"Review ({genre}): {input}\nSentiment:").hexdigest()


@pytest.mark.parametrize(("template", "message"), [
    ("Answer: {label}", "the label field"),
    ("{input.__class__}", "not a plain field name"),
    ("{input} {missing}", r"no field\(s\) \['missing'\]"),
    ("{input", "malformed"),
])
def test_the_template_refuses_what_it_cannot_render_honestly(template, message):
    with pytest.raises(ValueError, match=message):
        EvalPromptTemplateNode().execute(
            {"dataset": _dataset(["x"], ["positive"])}, {"template": template})


# ── BatchInference ───────────────────────────────────────────────────────────

def test_a_prediction_table_has_the_fixed_columns_and_one_row_per_item():
    model, tokenizer, dataset = _sentiment_fixture(6)
    out = _infer(*_stamped(model, tokenizer), dataset)
    table = out["predictions"]
    assert table["columns"][: len(PREDICTION_COLUMNS)] == list(PREDICTION_COLUMNS)
    assert table["columns"][len(PREDICTION_COLUMNS):] == ["score", "topk"]
    rows = table_records(table)
    assert [r["id"] for r in rows] == [f"s{i:02d}" for i in range(6)]
    assert {r["status"] for r in rows} == {STATUS_OK}
    assert {r["raw_answer"] for r in rows} == {"POSITIVE"}
    assert {r["answer"] for r in rows} == {"positive"}
    assert rows[0]["input_hash"] == hashlib.sha256(b"text 0").hexdigest()[:16]
    assert rows[0]["topk"] == ["positive", "negative"]
    assert out["answers"] == ["positive"] * 6
    assert out["labels"] == ["positive", "negative"] * 3
    json.dumps(table, allow_nan=False)  # rides the run store's encoder


def test_the_table_records_where_every_row_came_from():
    model, tokenizer, dataset = _sentiment_fixture(4)
    meta = _infer(*_stamped(model, tokenizer), dataset)["predictions"]["meta"]
    producer = meta["producer"]
    assert producer["model_id"] == "org/model"
    assert producer["model_revision"] == SHA_A
    assert producer["processor_revision"] == SHA_A
    assert meta["dataset"]["sha256"] == "f" * 64
    assert meta["inference"] == {"task": "classification", "batch_size": 4,
                                 "max_length": 256, "top_k": 5}
    assert meta["counts"] == {"ok": 4, "error": 0, "cancelled": 0}
    assert meta["reproducibility"] == {"reproducible": True, "reasons": []}


def test_a_model_id_alone_is_not_called_reproducible():
    """A model with no pinned revision -- e.g. not from Eval-HFModelLoader."""
    model, tokenizer, dataset = _sentiment_fixture(2)
    meta = _infer(model, tokenizer, dataset)["predictions"]["meta"]
    assert meta["reproducibility"]["reproducible"] is False
    assert "model revision is not pinned to a commit or content hash" in (
        meta["reproducibility"]["reasons"])


def test_a_processor_from_another_commit_of_the_same_repo_is_refused():
    model, tokenizer, dataset = _sentiment_fixture(2)
    _stamped(model, tokenizer, model_rev=SHA_A, proc_rev=SHA_B)
    with pytest.raises(ValueError, match="same repository"):
        _infer(model, tokenizer, dataset)


def test_stop_between_batches_keeps_finished_items_and_cancels_the_rest():
    model, tokenizer, dataset = _sentiment_fixture(10)
    context = StopAfter(2)  # two batches of four run, then Stop
    out = _infer(*_stamped(model, tokenizer), dataset, context=context)
    rows = table_records(out["predictions"])
    assert [r["status"] for r in rows] == [STATUS_OK] * 8 + [STATUS_CANCELLED] * 2
    assert all(r["answer"] == "positive" for r in rows[:8])
    assert all(r["answer"] is None and r["error"] for r in rows[8:])
    assert out[INTERRUPTED_KEY] == {"batch": 2, "completed": 8}
    assert out["predictions"]["meta"]["counts"] == {"ok": 8, "error": 0, "cancelled": 2}
    assert out["predictions"]["meta"]["interrupted_at_batch"] == 2
    # The cancelled rows are still inspectable downstream, and skipped there.
    metric = EvalExactMatchNode().execute({"predictions": out["predictions"]}, {})
    assert metric["table"].tolist() == [[0.5, 8.0, 2.0]]


def test_one_failing_item_is_an_error_and_its_batch_mates_still_finish():
    model, tokenizer, dataset = _sentiment_fixture(8)
    del tokenizer.vocab["text 5"]
    out = _infer(*_stamped(model, tokenizer), dataset)
    rows = table_records(out["predictions"])
    statuses = [r["status"] for r in rows]
    assert statuses == [STATUS_OK] * 5 + [STATUS_ERROR] + [STATUS_OK] * 2
    assert "unknown text 'text 5'" in rows[5]["error"]
    assert rows[5]["answer"] is None
    # First batch whole; second batch tried whole, then item by item.
    assert tokenizer.calls == [4, 4, 1, 1, 1, 1]


def test_an_item_without_input_is_an_error_not_a_crash():
    model, tokenizer, dataset = _sentiment_fixture(3)
    dataset["items"][1]["input"] = None
    rows = table_records(_infer(*_stamped(model, tokenizer), dataset)["predictions"])
    assert [r["status"] for r in rows] == [STATUS_OK, STATUS_ERROR, STATUS_OK]


def test_progress_is_reported_per_batch(monkeypatch):
    monkeypatch.setattr(loop_control, "PROGRESS_MIN_INTERVAL_S", 0.0)
    model, tokenizer, dataset = _sentiment_fixture(10)
    frames: list[dict] = []
    _infer(*_stamped(model, tokenizer), dataset, progress=frames.append)
    assert [(f["batch"], f["total_batches"], f["completed"]) for f in frames] == [
        (1, 3, 4), (2, 3, 8), (3, 3, 10)]
    assert {f["event"] for f in frames} == {"batch"}


def test_label_map_renames_raw_answers_before_comparison():
    texts = ["a", "b"]
    model = LookupClassifier([[2.0, 0.0], [0.0, 2.0]], {})  # no names: LABEL_i
    tokenizer = LookupTokenizer({"a": 0, "b": 1})
    out = _infer(model, tokenizer, _dataset(texts, ["negative", "positive"]),
                 label_map="LABEL_0=negative, LABEL_1=positive")
    rows = table_records(out["predictions"])
    assert [r["raw_answer"] for r in rows] == ["LABEL_0", "LABEL_1"]
    assert [r["answer"] for r in rows] == ["negative", "positive"]


# ── generation, with a fake causal LM ────────────────────────────────────────

class CountingLM(nn.Module):
    """A causal LM whose next-token distribution depends on the last token.

    Vocabulary of 8; token 0 is eos and is never the most likely, so greedy
    decoding runs to max_new_tokens. ``past_key_values`` counts the tokens
    seen, which proves the cache was threaded through.
    """

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=True):
        seen = (past_key_values or 0) + input_ids.shape[1]
        assert attention_mask.shape[1] == seen
        last = input_ids[0, -1].item()
        logits = torch.full((1, input_ids.shape[1], 8), 0.5)
        logits[0, -1, (last % 7) + 1] = 2.0
        logits[0, -1, 0] = -5.0
        return SimpleNamespace(logits=logits, past_key_values=seen)


class CharTokenizer:
    eos_token_id = 0

    def __call__(self, text, return_tensors=None, **_kwargs):
        ids = torch.tensor([[(ord(c) % 7) + 1 for c in text]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids, skip_special_tokens=True):
        return "".join("_abcdefg"[i] for i in ids)


def _generate(dataset, **params):
    model, tokenizer = CountingLM(), CharTokenizer()
    hub.stamp(model, {"model_id": "org/lm", "model_revision": SHA_A, "task": "generation"})
    hub.stamp(tokenizer, {"processor_id": "org/lm", "processor_revision": SHA_A})
    return EvalBatchInferenceNode().execute(
        {"model": model, "processor": tokenizer, "dataset": dataset},
        {"task": "auto", "batch_size": 2, "max_new_tokens": 5, **params},
    )


def test_greedy_generation_decodes_with_the_cache_and_records_its_config():
    out = _generate(_dataset(["ab"], ["x"]))
    row = table_records(out["predictions"])[0]
    # 'b' is ord 98 -> id 1; each next token is the last + 1: 2, 3, 4, 5, 6.
    assert row["raw_answer"] == "bcdef"
    assert row["new_tokens"] == 5
    generation = out["predictions"]["meta"]["generation"]
    assert generation["decoding"] == "greedy"
    assert generation["max_new_tokens"] == 5
    assert generation["temperature"] is None


def test_sampling_is_seeded_per_item_and_never_touches_the_global_rng():
    texts = [f"item{i}" for i in range(6)]
    dataset = _dataset(texts, [None] * 6)
    before = torch.get_rng_state()
    first = _generate(dataset, do_sample=True, temperature=5.0, seed=7, batch_size=2)
    after = torch.get_rng_state()
    assert torch.equal(before, after), "sampling consumed the global RNG"

    again = _generate(dataset, do_sample=True, temperature=5.0, seed=7, batch_size=5)
    answers = [r["raw_answer"] for r in table_records(first["predictions"])]
    assert answers == [r["raw_answer"] for r in table_records(again["predictions"])]
    other = _generate(dataset, do_sample=True, temperature=5.0, seed=8)
    assert answers != [r["raw_answer"] for r in table_records(other["predictions"])]
    assert first["predictions"]["meta"]["generation"]["seed"] == 7


def test_the_per_item_seed_depends_on_the_seed_and_the_id_only():
    assert item_seed(1, "s01") == item_seed(1, "s01")
    assert item_seed(1, "s01") != item_seed(1, "s02")
    assert item_seed(1, "s01") != item_seed(2, "s01")


def test_top_p_keeps_at_least_the_most_likely_token():
    predictor = GenerationPredictor(
        CountingLM(), CharTokenizer(), max_new_tokens=1, do_sample=True,
        temperature=1.0, top_p=0.01, seed=0)
    generator = torch.Generator().manual_seed(0)
    logits = torch.tensor([0.0, 1.0, 9.0, 1.0])
    assert {predictor._choose(logits, generator) for _ in range(20)} == {2}


# ── generation settings: node or the model's generation_config (#139, #380) ──

class _GenerationConfig:
    """The ``to_dict`` half of transformers' GenerationConfig."""

    def __init__(self, **fields):
        self.fields = fields

    def to_dict(self):
        return dict(self.fields)


def _penalized(logits: torch.Tensor, seen: list[int], penalty: float) -> torch.Tensor:
    """transformers' RepetitionPenaltyLogitsProcessor, written out longhand."""
    out = logits.clone()
    for token in set(seen):
        value = logits[token]
        out[token] = value * penalty if value < 0 else value / penalty
    return out


def test_the_default_settings_are_this_nodes_and_record_the_models_suggestion():
    model, tokenizer = CountingLM(), CharTokenizer()
    model.generation_config = _GenerationConfig(do_sample=True, temperature=0.7, top_p=0.8,
                                                top_k=20, repetition_penalty=1.1)
    hub.stamp(model, {"model_id": "org/lm", "model_revision": SHA_A, "task": "generation"})
    hub.stamp(tokenizer, {"processor_id": "org/lm", "processor_revision": SHA_A})
    out = EvalBatchInferenceNode().execute(
        {"model": model, "processor": tokenizer, "dataset": _dataset(["ab"], ["x"])},
        {"task": "auto", "max_new_tokens": 5})
    generation = out["predictions"]["meta"]["generation"]
    # Unchanged from before the option: greedy, no penalty -- the graph's
    # params decide, and the model's suggestion is recorded beside them.
    assert generation["source"] == "node"
    assert generation["decoding"] == "greedy"
    assert generation["repetition_penalty"] == 1.0
    assert table_records(out["predictions"])[0]["raw_answer"] == "bcdef"
    assert generation["model_generation_config"] == {
        "do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
        "repetition_penalty": 1.1, "unset": []}


def test_generation_config_model_takes_the_models_settings_and_fills_unset_ones():
    model, tokenizer = CountingLM(), CharTokenizer()
    model.generation_config = _GenerationConfig(do_sample=True, temperature=0.7, top_p=None,
                                                repetition_penalty=1.1)
    hub.stamp(model, {"model_id": "org/lm", "model_revision": SHA_A, "task": "generation"})
    hub.stamp(tokenizer, {"processor_id": "org/lm", "processor_revision": SHA_A})
    out = EvalBatchInferenceNode().execute(
        {"model": model, "processor": tokenizer, "dataset": _dataset(["ab"], ["x"])},
        {"task": "auto", "max_new_tokens": 3, "generation_config": "model",
         "do_sample": False, "temperature": 3.0})
    generation = out["predictions"]["meta"]["generation"]
    assert generation["source"] == "model"
    assert generation["decoding"] == "sample"
    assert generation["temperature"] == 0.7  # the model's, not the node's 3.0
    assert generation["top_p"] == 1.0        # unset: transformers' default
    assert generation["top_k"] == 50         # unset: transformers' default
    assert generation["repetition_penalty"] == 1.1
    assert generation["model_generation_config"]["unset"] == ["top_p", "top_k"]


def test_generation_config_model_refuses_a_model_without_one():
    with pytest.raises(ValueError, match="no generation_config"):
        _generate(_dataset(["ab"], ["x"]), generation_config="model")


def test_the_repetition_penalty_is_transformers_rule():
    predictor = GenerationPredictor(
        CountingLM(), CharTokenizer(), max_new_tokens=1, do_sample=False,
        temperature=1.0, top_p=1.0, seed=0, repetition_penalty=1.5)
    logits = torch.tensor([2.0, -1.0, 0.5, 3.0, -2.0])
    seen = [0, 1, 1, 4]
    assert torch.equal(predictor._penalize(logits, seen), _penalized(logits, seen, 1.5))
    # Greedy picks token 3 either way here, but a penalty on 3 itself flips it.
    assert predictor._choose(logits, None, seen) == 3
    assert predictor._choose(torch.tensor([0.0, 2.9, 0.0, 3.0]), None, [3]) == 1


def test_a_repetition_penalty_changes_greedy_output_and_is_recorded():
    # "ba" is ids [1, 7]; CountingLM's favourite after 7 is 1, already in the
    # prompt. Penalised by 5 its logit 2.0 drops to 0.4, below the flat 0.5
    # of the tokens not seen yet, so greedy takes the first of those (2).
    plain = table_records(_generate(_dataset(["ba"], ["x"]), max_new_tokens=1)["predictions"])
    out = _generate(_dataset(["ba"], ["x"]), max_new_tokens=1, repetition_penalty=5.0)
    penalized = table_records(out["predictions"])
    assert plain[0]["raw_answer"] == "a"
    assert penalized[0]["raw_answer"] == "b"
    assert out["predictions"]["meta"]["generation"]["repetition_penalty"] == 5.0


def test_top_k_keeps_only_the_k_most_likely_tokens_when_sampling():
    predictor = GenerationPredictor(
        CountingLM(), CharTokenizer(), max_new_tokens=1, do_sample=True,
        temperature=1.0, top_p=1.0, seed=0, top_k=2)
    generator = torch.Generator().manual_seed(0)
    logits = torch.tensor([3.0, 2.9, 0.0, 0.1, 0.2])
    assert {predictor._choose(logits, generator) for _ in range(200)} == {0, 1}


# ── metrics: parity with scikit-learn ────────────────────────────────────────

def _random_table(seed: int, classes: list[str], n: int, *, broken: int = 0):
    """Answers and labels drawn at random, plus *broken* error/cancelled rows."""
    rng = random.Random(seed)
    rows = [{"id": f"r{i}", "answer": rng.choice(classes), "label": rng.choice(classes)}
            for i in range(n)]
    for i in range(broken):
        rows.append({"id": f"x{i}", "answer": None, "label": rng.choice(classes),
                     "status": STATUS_ERROR if i % 2 else STATUS_CANCELLED, "error": "boom"})
    return _prediction_table(rows), rows[:n]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("classes", [["neg", "pos"], ["a", "b", "c", "d", "e"], ["3", "10", "2"]])
def test_exact_match_and_f1_agree_with_sklearn(seed, classes):
    table, scored = _random_table(seed, classes, 40, broken=3)
    answers = [r["answer"] for r in scored]
    labels = [r["label"] for r in scored]

    exact = EvalExactMatchNode().execute({"predictions": table}, {})
    assert exact["value"] == pytest.approx(accuracy_score(labels, answers))
    assert exact["table"].tolist()[0][1:] == [40.0, 3.0]
    for average in ("macro", "micro"):
        out = EvalF1ScoreNode().execute({"predictions": table}, {"average": average})
        expected = f1_score(labels, answers, average=average, zero_division=0)
        assert out["value"] == pytest.approx(expected), average
        assert out["columns"] == [f"f1_{average}", "counted", "skipped"]


def test_macro_f1_counts_a_class_only_ever_predicted():
    rows = [{"id": "1", "answer": "a", "label": "a"}, {"id": "2", "answer": "z", "label": "b"}]
    out = EvalF1ScoreNode().execute({"predictions": _prediction_table(rows)}, {"average": "macro"})
    assert out["value"] == pytest.approx(
        f1_score(["a", "b"], ["a", "z"], average="macro", zero_division=0))


def test_top_k_accuracy_agrees_with_sklearn():
    gen = torch.Generator().manual_seed(3)
    n, classes = 30, ["c0", "c1", "c2", "c3"]
    logits = torch.randn(n, 4, generator=gen).tolist()
    truth = torch.randint(0, 4, (n,), generator=gen).tolist()
    texts = [f"t{i}" for i in range(n)]
    model = LookupClassifier(logits, dict(enumerate(classes)))
    tokenizer = LookupTokenizer({t: i for i, t in enumerate(texts)})
    table = _infer(model, tokenizer, _dataset(texts, [classes[y] for y in truth]))["predictions"]
    for k in (1, 2, 3):
        out = EvalTopKAccuracyNode().execute({"predictions": table}, {"k": k})
        expected = top_k_accuracy_score(truth, logits, k=k, labels=[0, 1, 2, 3])
        assert out["value"] == pytest.approx(expected), k
    assert out["columns"][0] == "top3_accuracy"


def test_top_k_refuses_a_table_without_candidates():
    table = _prediction_table([{"id": "1", "answer": "a", "label": "a"}])
    with pytest.raises(ValueError, match="classification mode only"):
        EvalTopKAccuracyNode().execute({"predictions": table}, {"k": 2})


def test_a_metric_logs_itself_as_a_run_metric_and_names_its_producer():
    rows = [{"id": "1", "answer": "a", "label": "a"}, {"id": "2", "answer": "a", "label": "b"}]
    table = _prediction_table(rows, {"producer": {"model_id": "org/m", "model_revision": SHA_A}})
    context = StopAfter(0)
    out = EvalExactMatchNode().execute(
        {"predictions": table}, {"metric_name": "exact_match_a", "step": 3}, context=context)
    assert context.metrics == [("exact_match_a", 0.5, 3)]
    assert out["row_labels"] == ["org/m@aaaaaaa"]
    meta = out["metrics"]["meta"]
    assert meta["metric"]["node"] == "Eval-ExactMatch"
    assert meta["metric"]["name"] == "exact_match"
    assert meta["producer"]["model_revision"] == SHA_A


def test_a_metric_with_nothing_to_score_says_so():
    rows = [{"id": "1", "answer": None, "label": "a", "status": STATUS_CANCELLED}]
    with pytest.raises(ValueError, match="nothing to score"):
        EvalExactMatchNode().execute({"predictions": _prediction_table(rows)}, {})


# ── CompareRuns ──────────────────────────────────────────────────────────────

def _ab():
    labels = {"s1": "pos", "s2": "neg", "s3": "pos", "s4": "neg"}
    a = {"s1": "pos", "s2": "neg", "s3": "neg", "s4": "pos"}
    b = {"s1": "pos", "s2": "pos", "s3": "pos", "s4": "pos"}
    rows_a = [{"id": i, "answer": a[i], "label": labels[i]} for i in labels]
    rows_b = [{"id": i, "answer": b[i], "label": labels[i]} for i in labels]
    return rows_a, rows_b


def _compare(rows_a, rows_b, **params):
    return EvalCompareRunsNode().execute(
        {"table_a": _prediction_table(rows_a), "table_b": _prediction_table(rows_b)}, params)


def test_reordered_rows_are_aligned_by_id_not_by_position():
    rows_a, rows_b = _ab()
    in_order = _compare(rows_a, rows_b)
    shuffled = _compare(rows_a, list(reversed(rows_b)))
    assert shuffled["diff"]["rows"] == in_order["diff"]["rows"]
    diff = {r["id"]: r for r in table_records(shuffled["diff"])}
    assert diff["s2"] == {
        "id": "s2", "answer_a": "neg", "answer_b": "pos", "label": "neg", "agree": 0,
        "a_correct": 1, "b_correct": 0, "status_a": "ok", "status_b": "ok"}
    summary = shuffled["summary"]
    assert summary["outcomes"] == {"both_correct": 1, "only_a_correct": 1,
                                   "only_b_correct": 1, "both_wrong": 1}
    assert shuffled["agreement"] == pytest.approx(0.5)
    assert (summary["accuracy_a"], summary["accuracy_b"]) == (0.5, 0.5)
    assert shuffled["summary_table"].tolist() == [[1, 0.25]] * 4
    assert shuffled["summary_rows"] == [
        "both_correct", "only_a_correct", "only_b_correct", "both_wrong"]


def test_duplicate_ids_are_refused_by_name():
    rows_a, rows_b = _ab()
    rows_b.append(dict(rows_b[1]))
    with pytest.raises(ValueError, match=r"table_b has duplicate id values \['s2'\]"):
        _compare(rows_a, rows_b)


def test_mismatched_item_sets_are_refused_naming_the_first_offenders():
    rows_a, rows_b = _ab()
    rows_a.append({"id": "only-a", "answer": "pos", "label": "pos"})
    rows_b = [r for r in rows_b if r["id"] != "s3"]
    with pytest.raises(ValueError) as caught:
        _compare(rows_a, rows_b)
    message = str(caught.value)
    assert "2 id(s) only in table_a (first: ['s3', 'only-a'])" in message
    assert "0 only in table_b" in message


def test_different_inputs_for_one_id_are_refused_unless_allowed():
    rows_a, rows_b = _ab()
    rows_b[2]["input"] = "a different prompt"
    with pytest.raises(ValueError, match=r"different inputs .*\['s3'\]"):
        _compare(rows_a, rows_b)
    assert _compare(rows_a, rows_b, require_same_input=False)["summary"]["compared"] == 4


def test_different_labels_for_one_id_are_refused():
    rows_a, rows_b = _ab()
    rows_b[0]["label"] = "neg"
    with pytest.raises(ValueError, match=r"different labels .*\['s1'\]"):
        _compare(rows_a, rows_b)


def test_items_one_run_did_not_finish_stay_visible_and_uncompared():
    rows_a, rows_b = _ab()
    rows_b[3].update(status=STATUS_CANCELLED, answer=None)
    rows_a[0].update(status=STATUS_ERROR, answer=None, error="boom")
    out = _compare(rows_a, rows_b)
    diff = {r["id"]: r for r in table_records(out["diff"])}
    assert (diff["s4"]["agree"], diff["s4"]["status_b"]) == (None, STATUS_CANCELLED)
    assert (diff["s1"]["agree"], diff["s1"]["status_a"]) == (None, STATUS_ERROR)
    assert out["summary"]["compared"] == 2
    assert out["summary"]["skipped"] == 2
    values = out["diff_table"].tolist()
    assert values[1] == [0.0, 1.0, 0.0]
    assert all(v != v for v in values[0])  # NaN: not compared
    json.dumps(out["chart"], allow_nan=False)
    json.dumps(out["summary"], allow_nan=False)


def test_the_agreement_matrix_counts_a_against_b():
    rows_a, rows_b = _ab()
    chart = _compare(rows_a, rows_b)["chart"]
    assert chart["kind"] == "heatmap"
    assert chart["row_labels"] == ["neg", "pos"]
    # A neg: s2 (B pos), s3 (B pos). A pos: s1 (B pos), s4 (B pos).
    assert chart["matrix"] == [[0.0, 2.0], [0.0, 2.0]]


def test_too_many_distinct_answers_fall_back_to_an_outcome_bar_chart():
    rows = [{"id": f"r{i}", "answer": f"answer {i}", "label": f"answer {i}"} for i in range(25)]
    chart = _compare(rows, rows)["chart"]
    assert chart["kind"] == "bar"
    assert "25 distinct answers" in chart["note"]


def test_compare_refuses_something_that_is_not_a_table():
    rows_a, _ = _ab()
    with pytest.raises(ValueError, match="must be an eval record table"):
        EvalCompareRunsNode().execute(
            {"table_a": _prediction_table(rows_a), "table_b": [1, 2, 3]}, {})


# ── the loaders, without the network and without transformers ───────────────

def test_a_local_directory_is_identified_by_its_files(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights v1")
    first = hub.resolve_revision(str(tmp_path), "main")
    assert first["source"] == "local"
    assert first["revision"].startswith("local-sha256:")
    (tmp_path / "model.safetensors").write_bytes(b"weights v2")
    assert hub.resolve_revision(str(tmp_path), "main")["revision"] != first["revision"]


def test_a_full_commit_sha_offline_is_used_as_given(monkeypatch):
    monkeypatch.setattr(hub, "hub_offline", lambda: True)
    monkeypatch.setattr(hub, "_hub_info", lambda *a: pytest.fail("asked the Hub"))
    resolved = hub.resolve_revision("org/m", SHA_A)
    assert (resolved["revision"], resolved["source"], resolved["cache_only"]) == (
        SHA_A, "pinned", True)


def test_online_resolution_asks_the_hub(monkeypatch):
    monkeypatch.setattr(hub, "hub_offline", lambda: False)
    monkeypatch.setattr(hub, "_hub_info", lambda repo, rev: (SHA_B, ["config.json"]))
    resolved = hub.resolve_revision("org/m", "main")
    assert (resolved["revision"], resolved["source"], resolved["requested_revision"]) == (
        SHA_B, "hub", "main")
    assert (resolved["files"], resolved["cache_only"]) == (["config.json"], False)


def test_only_one_weights_format_is_downloaded():
    files = ["config.json", "vocab.txt", "pytorch_model.bin", "model.safetensors",
             "flax_model.msgpack", "training_args.bin", "onnx/model.onnx", "README.md"]
    assert hub._wanted_files(files) == ["config.json", "vocab.txt", "model.safetensors"]
    bin_only = ["config.json", "pytorch_model.bin", "training_args.bin", "tokenizer.json"]
    assert hub._wanted_files(bin_only) == ["config.json", "pytorch_model.bin", "tokenizer.json"]


def test_offline_resolution_uses_the_cached_snapshot(monkeypatch):
    monkeypatch.setattr(hub, "hub_offline", lambda: True)
    monkeypatch.setattr(hub, "_hub_info", lambda *a: pytest.fail("went online"))
    monkeypatch.setattr(hub, "cached_sha", lambda repo, rev, cache: SHA_A)
    resolved = hub.resolve_revision("org/m", "main")
    assert (resolved["revision"], resolved["source"]) == (SHA_A, "cache")
    assert "HF_HUB_OFFLINE" in resolved["note"]


def test_an_unreachable_hub_falls_back_to_the_cache_and_says_why(monkeypatch):
    class ConnectError(Exception):
        pass

    def unreachable(*_args):
        raise ConnectError("no route")

    monkeypatch.setattr(hub, "hub_offline", lambda: False)
    monkeypatch.setattr(hub, "_hub_info", unreachable)
    monkeypatch.setattr(hub, "cached_sha", lambda repo, rev, cache: SHA_A)
    resolved = hub.resolve_revision("org/m", "main")
    assert resolved["source"] == "cache"
    assert "unreachable (ConnectError)" in resolved["note"]


def test_an_unknown_repository_is_an_error_not_a_cache_lookup(monkeypatch):
    class RepositoryNotFoundError(Exception):
        pass

    def missing(*_args):
        raise RepositoryNotFoundError("404")

    monkeypatch.setattr(hub, "hub_offline", lambda: False)
    monkeypatch.setattr(hub, "_hub_info", missing)
    monkeypatch.setattr(hub, "cached_sha", lambda *a: pytest.fail("looked in the cache"))
    with pytest.raises(ValueError, match="does not resolve on the Hugging Face Hub"):
        hub.resolve_revision("org/nope", "main")


def test_offline_and_not_cached_is_an_error_that_says_so(monkeypatch, tmp_path):
    monkeypatch.setattr(hub, "hub_offline", lambda: True)
    monkeypatch.setattr(hub, "cached_sha", lambda *a: None)
    monkeypatch.setattr(hub, "hf_cache_dir", lambda: str(tmp_path))
    with pytest.raises(ValueError, match="HF_HUB_OFFLINE is set, so nothing is downloaded"):
        hub.resolve_revision("org/not-cached", "main")


def test_cached_sha_reads_the_snapshot_directory(monkeypatch, tmp_path):
    import huggingface_hub

    snapshot = tmp_path / "models--org--m" / "snapshots" / SHA_B / "config.json"
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda repo, name, cache_dir, revision: str(snapshot)
                        if name == "config.json" else None)
    assert hub.cached_sha("org/m", "main", str(tmp_path)) == SHA_B


def test_the_loader_says_how_to_get_transformers_when_it_is_missing(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ValueError, match="cdui plugin install eval"):
        EvalHFModelLoaderNode().execute(
            {}, {"model_id": str(tmp_path), "task": "classification", "device": "cpu"})


# ── transformers' lazy module, hit by two loaders at once ───────────────────

_HF_NAMES = ("AutoConfig", "AutoModelForSequenceClassification", "AutoModelForCausalLM",
             "AutoModelForVisualQuestionAnswering", "AutoTokenizer", "AutoProcessor")


class _SlowLazyTransformers(types.ModuleType):
    """A lazy module whose first attribute access is slow and not thread-safe.

    Like transformers' ``_LazyModule``: a missing attribute is imported from
    inside ``__getattr__``. A second thread asking while the first is still
    importing sees no attribute -- the failure the first run of the A/B
    example hit in a fresh server, where the engine starts both
    Eval-HFModelLoader nodes at once.
    """

    def __init__(self):
        super().__init__("transformers")
        self._importing: set[str] = set()
        self.resolutions = 0

    def __getattr__(self, name):
        if name not in _HF_NAMES:
            raise AttributeError(name)
        if name in self._importing:
            raise AttributeError(f"module 'transformers' has no attribute {name!r}")
        self._importing.add(name)
        time.sleep(0.05)
        value = type(name, (), {})
        setattr(self, name, value)
        self._importing.discard(name)
        self.resolutions += 1
        return value


def test_two_loaders_resolving_transformers_at_once_both_get_the_classes(monkeypatch):
    fake = _SlowLazyTransformers()
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(hub, "_CLASSES", {})
    barrier = threading.Barrier(2)
    results: list[dict] = []
    errors: list[BaseException] = []

    def resolve():
        barrier.wait()
        try:
            results.append(hub.hf_classes())
        except BaseException as exc:  # noqa: BLE001 - collected for the assert
            errors.append(exc)

    threads = [threading.Thread(target=resolve) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(results) == 2
    for name in _HF_NAMES:
        assert results[0][name] is results[1][name] is fake.__dict__[name]
    assert fake.resolutions == len(_HF_NAMES)  # each lazy import ran once


def test_a_different_transformers_module_is_resolved_afresh(monkeypatch):
    monkeypatch.setattr(hub, "_CLASSES", {})
    first, second = _SlowLazyTransformers(), _SlowLazyTransformers()
    monkeypatch.setitem(sys.modules, "transformers", first)
    assert hub.hf_classes()["AutoTokenizer"] is first.__dict__["AutoTokenizer"]
    monkeypatch.setitem(sys.modules, "transformers", second)
    assert hub.hf_classes()["AutoTokenizer"] is second.__dict__["AutoTokenizer"]
