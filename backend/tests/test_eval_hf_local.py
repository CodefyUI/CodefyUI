"""The eval pack against real transformers models, offline.

Two tiers, kept apart on purpose:

* **Local fixtures** (skipped when transformers is not installed, as on CI,
  where the backend is installed without it): a two-layer
  ``BertForSequenceClassification`` and a two-layer ``GPT2LMHeadModel``,
  randomly initialised and saved with ``save_pretrained`` into a temp
  directory, with a word-level tokenizer built in the test. Nothing is
  downloaded, and the pack's offline switch is forced on so a slip would
  fail rather than fetch.
* **The Hub smoke check** (opt-in, ``CODEFYUI_PACK_NETWORK_TESTS=1``, the
  switch ``test_pack_examples_real.py`` already uses): runs the pack's A/B
  example graph against the two real checkpoints it pins and checks the
  numbers its notes quote. Run it as::

      CODEFYUI_PACK_NETWORK_TESTS=1 pytest tests/test_eval_hf_local.py -q -k hub

Random weights make the predictions arbitrary, so the local tests compare
the pipeline with the same model called directly, not with a "right"
answer.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")

from app.core.graph_engine import execute_graph  # noqa: E402
from cdui_plugins.eval.nodes import _eval_hub as hub  # noqa: E402
from cdui_plugins.eval.nodes._eval_core import STATUS_OK, table_records  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE = (_REPO_ROOT / "plugins" / "eval" / "examples" / "Eval"
            / "AB-Text-Classification" / "graph.json")

WORDS = ["good", "great", "fine", "bad", "awful", "dull", "film", "plot", "very", "not"]


_NETWORK = os.environ.get("CODEFYUI_PACK_NETWORK_TESTS") == "1"
network = pytest.mark.skipif(
    not _NETWORK, reason="set CODEFYUI_PACK_NETWORK_TESTS=1 to download the two "
                         "checkpoints the eval example pins")


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """No local test may reach the network, and none may reuse a cached model.

    ``hub_offline`` is patched rather than ``HF_HUB_OFFLINE`` set:
    huggingface_hub reads that variable once, at import.
    """
    if not _NETWORK:
        monkeypatch.setattr(hub, "hub_offline", lambda: True)
    hub.clear_model_cache()
    yield
    hub.clear_model_cache()


def _save_tokenizer(directory: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[PAD]": 0, "[UNK]": 1, "<eos>": 2}
    vocab.update({w: i + 3 for i, w in enumerate(WORDS)})
    core = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=core, unk_token="[UNK]", pad_token="[PAD]", eos_token="<eos>")
    fast.save_pretrained(str(directory))


@pytest.fixture(scope="module")
def tiny_classifier(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("tiny-bert")
    config = transformers.BertConfig(
        vocab_size=len(WORDS) + 3, hidden_size=16, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=32,
        num_labels=3, id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2},
    )
    with torch.random.fork_rng():
        torch.manual_seed(0)
        transformers.BertForSequenceClassification(config).save_pretrained(str(directory))
    _save_tokenizer(directory)
    return directory


@pytest.fixture(scope="module")
def tiny_lm(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("tiny-gpt2")
    config = transformers.GPT2Config(
        vocab_size=len(WORDS) + 3, n_positions=64, n_embd=16, n_layer=2, n_head=2,
        bos_token_id=2, eos_token_id=2, pad_token_id=0,
    )
    with torch.random.fork_rng():
        torch.manual_seed(0)
        transformers.GPT2LMHeadModel(config).save_pretrained(str(directory))
    _save_tokenizer(directory)
    return directory


def _jsonl(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "set.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _items(n: int = 12) -> list[dict]:
    labels = ["negative", "neutral", "positive"]
    return [{"id": f"s{i:02d}", "text": " ".join(WORDS[(i + k) % len(WORDS)] for k in range(3)),
             "label": labels[i % 3]} for i in range(n)]


def _graph(model_dir: Path, data: Path, *, batch_size: int = 4, extra: dict | None = None,
           task: str = "auto", second_model: Path | None = None) -> tuple[list, list]:
    """JSONL -> loader -> processor -> inference -> metrics (-> compare)."""
    infer = {"task": "auto", "batch_size": batch_size, "normalize": "strip_lower",
             "label_map": "", "max_new_tokens": 4, **(extra or {})}
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "data", "type": "eval:Eval-JSONLDataset",
         "data": {"params": {"path": str(data)}}},
    ]
    edges = [{"id": "t-data", "source": "start", "target": "data",
              "sourceHandle": "trigger", "type": "trigger"}]
    sides = [("a", model_dir)] + ([("b", second_model)] if second_model else [])
    for side, directory in sides:
        nodes += [
            {"id": f"model_{side}", "type": "eval:Eval-HFModelLoader",
             "data": {"params": {"model_id": str(directory), "task": task, "device": "cpu"}}},
            {"id": f"proc_{side}", "type": "eval:Eval-HFProcessorLoader",
             "data": {"params": {}}},
            {"id": f"infer_{side}", "type": "eval:Eval-BatchInference",
             "data": {"params": infer}},
            {"id": f"em_{side}", "type": "eval:Eval-ExactMatch",
             "data": {"params": {"metric_name": f"exact_match_{side}"}}},
        ]
        edges += [
            {"id": f"t-{side}", "source": "start", "target": f"model_{side}",
             "sourceHandle": "trigger", "type": "trigger"},
            {"id": f"m-p-{side}", "source": f"model_{side}", "sourceHandle": "model",
             "target": f"proc_{side}", "targetHandle": "model"},
            {"id": f"m-i-{side}", "source": f"model_{side}", "sourceHandle": "model",
             "target": f"infer_{side}", "targetHandle": "model"},
            {"id": f"p-i-{side}", "source": f"proc_{side}", "sourceHandle": "processor",
             "target": f"infer_{side}", "targetHandle": "processor"},
            {"id": f"d-i-{side}", "source": "data", "sourceHandle": "dataset",
             "target": f"infer_{side}", "targetHandle": "dataset"},
            {"id": f"i-e-{side}", "source": f"infer_{side}", "sourceHandle": "predictions",
             "target": f"em_{side}", "targetHandle": "predictions"},
        ]
    if second_model:
        nodes.append({"id": "compare", "type": "eval:Eval-CompareRuns",
                      "data": {"params": {}}})
        edges += [
            {"id": "c-a", "source": "infer_a", "sourceHandle": "predictions",
             "target": "compare", "targetHandle": "table_a"},
            {"id": "c-b", "source": "infer_b", "sourceHandle": "predictions",
             "target": "compare", "targetHandle": "table_b"},
        ]
    return nodes, edges


def _run(nodes, edges) -> dict:
    return asyncio.run(execute_graph(nodes, edges, error_mode="fail_fast"))


# ── classification ───────────────────────────────────────────────────────────

def test_the_graph_classifies_exactly_as_the_model_does_directly(tiny_classifier, tmp_path):
    rows = _items()
    results = _run(*_graph(tiny_classifier, _jsonl(tmp_path, rows)))

    table = results["infer_a"]["predictions"]
    records = table_records(table)
    assert [r["id"] for r in records] == [r["id"] for r in rows]
    assert {r["status"] for r in records} == {STATUS_OK}

    model = transformers.AutoModelForSequenceClassification.from_pretrained(str(tiny_classifier))
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(tiny_classifier))
    with torch.no_grad():
        logits = model(**tokenizer([r["text"] for r in rows], padding=True,
                                   return_tensors="pt")).logits
    expected = [model.config.id2label[i] for i in logits.argmax(-1).tolist()]
    assert [r["raw_answer"] for r in records] == expected

    correct = sum(e == r["label"] for e, r in zip(expected, rows))
    assert results["em_a"]["value"] == pytest.approx(correct / len(rows))


def test_a_local_model_records_a_content_identity_shared_with_its_processor(
        tiny_classifier, tmp_path):
    results = _run(*_graph(tiny_classifier, _jsonl(tmp_path, _items(3))))
    model_prov = results["model_a"]["provenance"]
    assert model_prov["revision_source"] == "local"
    assert model_prov["model_revision"].startswith("local-sha256:")
    assert model_prov["task"] == "classification"
    producer = results["infer_a"]["predictions"]["meta"]["producer"]
    assert producer["processor_revision"] == model_prov["model_revision"]
    assert results["infer_a"]["predictions"]["meta"]["reproducibility"]["reproducible"]


def test_batch_size_does_not_change_a_single_answer(tiny_classifier, tmp_path):
    data = _jsonl(tmp_path, _items())
    one = table_records(_run(*_graph(tiny_classifier, data, batch_size=1))["infer_a"]["predictions"])
    five = table_records(_run(*_graph(tiny_classifier, data, batch_size=5))["infer_a"]["predictions"])
    assert [r["answer"] for r in one] == [r["answer"] for r in five]
    for a, b in zip(one, five):
        assert a["score"] == pytest.approx(b["score"], abs=1e-5)


def test_two_runs_of_one_model_agree_on_every_item(tiny_classifier, tmp_path):
    results = _run(*_graph(tiny_classifier, _jsonl(tmp_path, _items()),
                           second_model=tiny_classifier))
    assert results["compare"]["agreement"] == 1.0
    assert results["compare"]["summary"]["compared"] == 12


# ── generation ───────────────────────────────────────────────────────────────

def test_greedy_generation_matches_transformers_generate(tiny_lm, tmp_path):
    rows = _items(4)
    results = _run(*_graph(tiny_lm, _jsonl(tmp_path, rows)))
    records = table_records(results["infer_a"]["predictions"])
    assert results["model_a"]["provenance"]["task"] == "generation"

    model = transformers.AutoModelForCausalLM.from_pretrained(str(tiny_lm))
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(tiny_lm))
    for row, record in zip(rows, records):
        encoded = tokenizer(row["text"], return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**encoded, max_new_tokens=4, do_sample=False,
                                 pad_token_id=0, eos_token_id=2)
        new = out[0, encoded["input_ids"].shape[1]:].tolist()
        if 2 in new:
            new = new[:new.index(2)]
        assert record["raw_answer"] == tokenizer.decode(new, skip_special_tokens=True)
    generation = results["infer_a"]["predictions"]["meta"]["generation"]
    assert generation["decoding"] == "greedy"


def test_sampled_generation_is_reproducible_and_leaves_the_global_rng_alone(tiny_lm, tmp_path):
    data = _jsonl(tmp_path, _items(5))
    sample = {"do_sample": True, "temperature": 3.0, "seed": 11}
    before = torch.get_rng_state()
    first = _run(*_graph(tiny_lm, data, batch_size=2, extra=sample))
    assert torch.equal(torch.get_rng_state(), before)
    second = _run(*_graph(tiny_lm, data, batch_size=5, extra=sample))
    answers = [r["raw_answer"] for r in table_records(first["infer_a"]["predictions"])]
    assert answers == [r["raw_answer"] for r in table_records(second["infer_a"]["predictions"])]


def test_greedy_with_a_repetition_penalty_matches_transformers_generate(tiny_lm, tmp_path):
    rows = _items(4)
    results = _run(*_graph(tiny_lm, _jsonl(tmp_path, rows), extra={"repetition_penalty": 1.7}))
    records = table_records(results["infer_a"]["predictions"])
    model = transformers.AutoModelForCausalLM.from_pretrained(str(tiny_lm))
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(tiny_lm))
    for row, record in zip(rows, records):
        encoded = tokenizer(row["text"], return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**encoded, max_new_tokens=4, do_sample=False,
                                 repetition_penalty=1.7, pad_token_id=0, eos_token_id=2)
        new = out[0, encoded["input_ids"].shape[1]:].tolist()
        if 2 in new:
            new = new[:new.index(2)]
        assert record["raw_answer"] == tokenizer.decode(new, skip_special_tokens=True)
    assert results["infer_a"]["predictions"]["meta"]["generation"]["repetition_penalty"] == 1.7


@pytest.fixture(scope="module")
def tiny_lm_with_config(tiny_lm, tmp_path_factory) -> Path:
    """The tiny LM with a generation_config.json asking for a penalty."""
    directory = tmp_path_factory.mktemp("tiny-gpt2-config")
    for item in tiny_lm.iterdir():
        (directory / item.name).write_bytes(item.read_bytes())
    transformers.GenerationConfig(do_sample=False, repetition_penalty=1.4, bos_token_id=2,
                                  eos_token_id=2, pad_token_id=0).save_pretrained(str(directory))
    return directory


def test_generation_config_model_decodes_as_generate_does_with_no_arguments(
        tiny_lm_with_config, tmp_path):
    rows = _items(4)
    results = _run(*_graph(tiny_lm_with_config, _jsonl(tmp_path, rows),
                           extra={"generation_config": "model"}))
    records = table_records(results["infer_a"]["predictions"])
    model = transformers.AutoModelForCausalLM.from_pretrained(str(tiny_lm_with_config))
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(tiny_lm_with_config))
    for row, record in zip(rows, records):
        encoded = tokenizer(row["text"], return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**encoded, max_new_tokens=4)
        new = out[0, encoded["input_ids"].shape[1]:].tolist()
        if 2 in new:
            new = new[:new.index(2)]
        assert record["raw_answer"] == tokenizer.decode(new, skip_special_tokens=True)
    generation = results["infer_a"]["predictions"]["meta"]["generation"]
    assert generation["source"] == "model"
    assert generation["repetition_penalty"] == 1.4
    assert generation["model_generation_config"]["repetition_penalty"] == 1.4


# ── vqa ──────────────────────────────────────────────────────────────────────

VQA_WORDS = ["what", "color", "is", "the", "ball", "how", "many", "?"]
VQA_ANSWERS = {0: "red", 1: "green", 2: "blue"}


@pytest.fixture(scope="module")
def tiny_vilt(tmp_path_factory) -> Path:
    """A two-layer random ViltForQuestionAnswering and its processor, saved."""
    directory = tmp_path_factory.mktemp("tiny-vilt")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + VQA_WORDS
    (directory / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(directory / "vocab.txt"))
    images = transformers.ViltImageProcessor(size={"shortest_edge": 32}, size_divisor=8)
    transformers.ViltProcessor(image_processor=images, tokenizer=tokenizer).save_pretrained(
        str(directory))
    config = transformers.ViltConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=32, image_size=32, patch_size=8, max_position_embeddings=16,
        num_labels=3, id2label=VQA_ANSWERS, label2id={v: k for k, v in VQA_ANSWERS.items()})
    with torch.random.fork_rng():
        torch.manual_seed(0)
        transformers.ViltForQuestionAnswering(config).save_pretrained(str(directory))
    return directory


def _vqa_set(tmp_path: Path) -> tuple[Path, list[dict]]:
    from PIL import Image

    (tmp_path / "img").mkdir()
    colours = [(220, 30, 30), (30, 160, 60), (20, 60, 230), (240, 200, 20), (90, 90, 90)]
    rows = []
    for i, colour in enumerate(colours):
        Image.new("RGB", (40 + 8 * i, 32), colour).save(tmp_path / "img" / f"{i}.png")
        for j, question in enumerate(["what color is the ball ?", "how many ?"]):
            rows.append({"id": f"v{i}{j}", "q": question, "a": VQA_ANSWERS[i % 3],
                         "img": f"img/{i}.png"})
    path = tmp_path / "vqa.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path, rows


def _vqa_graph(model_dir: Path, data: Path, *, batch_size: int) -> tuple[list, list]:
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "data", "type": "eval:Eval-JSONLDataset", "data": {"params": {
            "path": str(data), "input_field": "q", "label_field": "a", "image_field": "img"}}},
        {"id": "model", "type": "eval:Eval-HFModelLoader",
         "data": {"params": {"model_id": str(model_dir), "device": "cpu"}}},
        {"id": "proc", "type": "eval:Eval-HFProcessorLoader", "data": {"params": {"kind": "auto"}}},
        {"id": "infer", "type": "eval:Eval-BatchInference",
         "data": {"params": {"batch_size": batch_size, "max_length": 16}}},
        {"id": "em", "type": "eval:Eval-ExactMatch", "data": {"params": {}}},
    ]
    edges = [
        {"id": "t-d", "source": "start", "target": "data", "sourceHandle": "trigger",
         "type": "trigger"},
        {"id": "t-m", "source": "start", "target": "model", "sourceHandle": "trigger",
         "type": "trigger"},
        {"id": "m-p", "source": "model", "sourceHandle": "model", "target": "proc",
         "targetHandle": "model"},
        {"id": "m-i", "source": "model", "sourceHandle": "model", "target": "infer",
         "targetHandle": "model"},
        {"id": "p-i", "source": "proc", "sourceHandle": "processor", "target": "infer",
         "targetHandle": "processor"},
        {"id": "d-i", "source": "data", "sourceHandle": "dataset", "target": "infer",
         "targetHandle": "dataset"},
        {"id": "i-e", "source": "infer", "sourceHandle": "predictions", "target": "em",
         "targetHandle": "predictions"},
    ]
    return nodes, edges


def test_vqa_answers_exactly_as_the_model_does_directly(tiny_vilt, tmp_path):
    from PIL import Image

    data, rows = _vqa_set(tmp_path)
    results = _run(*_vqa_graph(tiny_vilt, data, batch_size=3))
    assert results["model"]["provenance"]["task"] == "vqa"
    records = table_records(results["infer"]["predictions"])
    assert {r["status"] for r in records} == {STATUS_OK}

    model = transformers.AutoModelForVisualQuestionAnswering.from_pretrained(str(tiny_vilt))
    processor = transformers.AutoProcessor.from_pretrained(str(tiny_vilt))
    expected = []
    for row in rows:
        with Image.open(tmp_path / row["img"]) as image:
            encoded = processor(images=image.convert("RGB"), text=row["q"], return_tensors="pt")
        with torch.no_grad():
            expected.append(model.config.id2label[int(model(**encoded).logits.argmax(-1))])
    assert [r["raw_answer"] for r in records] == expected
    assert [r["image"] for r in records] == [r["img"] for r in rows]
    correct = sum(e == r["a"] for e, r in zip(expected, rows))
    assert results["em"]["value"] == pytest.approx(correct / len(rows))


def test_vqa_batch_size_does_not_change_an_answer(tiny_vilt, tmp_path):
    data, _ = _vqa_set(tmp_path)
    one = table_records(_run(*_vqa_graph(tiny_vilt, data, batch_size=1))["infer"]["predictions"])
    four = table_records(_run(*_vqa_graph(tiny_vilt, data, batch_size=4))["infer"]["predictions"])
    assert [r["answer"] for r in one] == [r["answer"] for r in four]
    for a, b in zip(one, four):
        assert a["score"] == pytest.approx(b["score"], abs=1e-5)


# ── against the Hub (opt-in) ─────────────────────────────────────────────────

_VQA_EXAMPLE = _EXAMPLE.parent.parent / "VQA-Shapes" / "graph.json"


@network
def test_hub_vqa_example_reproduces_the_numbers_its_notes_quote():
    payload = json.loads(_VQA_EXAMPLE.read_text(encoding="utf-8"))
    results = _run(payload["nodes"], payload["edges"])
    assert f"{results['em']['value']:.4f}" == "0.5000"
    assert f"{results['top3']['value']:.4f}" == "0.8333"
    answers = {r["id"]: r["answer"] for r in table_records(results["infer"]["predictions"])}
    assert answers == {
        "q01": "red", "q02": "1", "q03": "2", "q04": "orange", "q05": "triangle", "q06": "yes",
        "q07": "yellow", "q08": "4", "q09": "orange", "q10": "yes", "q11": "orange", "q12": "2"}
    notes = " ".join(n["data"]["noteContent"] for n in payload["nodes"] if n.get("type") == "note")
    for quoted in ("0.5000", "0.8333", "6 of 12", "10 of 12", "3,129", "470 MB"):
        assert quoted in notes

@network
def test_hub_example_graph_reproduces_the_numbers_its_notes_quote():
    payload = json.loads(_EXAMPLE.read_text(encoding="utf-8"))
    results = _run(payload["nodes"], payload["edges"])
    assert results["em_a"]["value"] == pytest.approx(1.0)
    assert results["em_b"]["value"] == pytest.approx(0.95)
    assert f"{results['f1_b']['value']:.4f}" == "0.9499"
    compare = results["compare"]
    assert compare["agreement"] == pytest.approx(0.95)
    disagree = [r["id"] for r in table_records(compare["diff"]) if r["agree"] == 0]
    assert disagree == ["s15"]
    assert compare["summary"]["outcomes"] == {
        "both_correct": 19, "only_a_correct": 1, "only_b_correct": 0, "both_wrong": 0}
    notes = " ".join(n["data"]["noteContent"] for n in payload["nodes"] if n.get("type") == "note")
    for quoted in ("1.0000", "0.9500", "0.9499", "s15"):
        assert quoted in notes


@network
def test_hub_a_branch_downloaded_online_resolves_offline_from_the_cache(monkeypatch):
    from cdui_plugins.eval.nodes.eval_hf_model_loader_node import EvalHFModelLoaderNode

    params = {"model_id": "philschmid/tiny-bert-sst2-distilled", "revision": "main",
              "task": "auto", "device": "cpu"}
    online = EvalHFModelLoaderNode().execute({}, params)["provenance"]
    assert online["revision_source"] == "hub"

    hub.clear_model_cache()
    monkeypatch.setattr(hub, "hub_offline", lambda: True)
    offline = EvalHFModelLoaderNode().execute({}, params)["provenance"]
    assert offline["revision_source"] == "cache"
    assert offline["model_revision"] == online["model_revision"]
