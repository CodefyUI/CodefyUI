"""Tests for PerplexityEvaluateNode (#291).

The load-bearing property is that the reported ``val_loss`` is the
TOKEN-WEIGHTED mean over the whole dataset -- not the mean of per-batch means,
which would make the number depend on ``batch_size``. Every arithmetic test
here pins the node against a single whole-dataset ``F.cross_entropy`` over the
same data, computed independently, with a deliberately RAGGED final batch so
the two disagree unless the weighting is right.

No network anywhere: the datasets are hand-built tensors and the models are
tiny real modules.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from app.core.execution_context import INTERRUPTED_KEY
from app.nodes.llm.perplexity_evaluate_node import PerplexityEvaluateNode

VOCAB = 11
SEQ_LEN = 4


class BlockDataset(Dataset):
    """``(input_ids, labels)`` pairs, exactly what LMTokenizedDataset yields."""

    def __init__(self, inputs: torch.Tensor, labels: torch.Tensor) -> None:
        self.inputs = inputs
        self.labels = labels

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.labels[idx]


class TinyLM(nn.Module):
    """A fixed (batch, seq_len) -> (batch, seq_len, vocab) module.

    Real enough to be scored -- an embedding and a linear head, so the logits
    depend on the ids the way a language model's do -- and seeded, so the
    reference computation below and the node see the same numbers.
    """

    def __init__(self, vocab_size: int = VOCAB, d_model: int = 8,
                 seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_seq_len = 32

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(input_ids))


class RecordingContext:
    """Records every ``log_metric`` call. No device set, so nodes get "cpu"."""

    def __init__(self) -> None:
        self.metrics: list[tuple[str, float, int]] = []

    def log_metric(self, name, value, step, node_id=None) -> None:
        self.metrics.append((name, float(value), int(step)))


class StopAfter(RecordingContext):
    """A context whose ``should_stop`` turns True after *n* checks."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self._left = n

    def should_stop(self) -> bool:
        if self._left > 0:
            self._left -= 1
            return False
        return True


def _blocks(n_blocks: int = 5, *, seed: int = 7) -> BlockDataset:
    """*n_blocks* blocks of ids in [1, VOCAB). 0 is left free for a pad label."""
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(
        1, VOCAB, (n_blocks, SEQ_LEN + 1), generator=generator)
    # ``clone()``, unlike the real dataset's views: the tests below write -100
    # into ``labels``, and a view would write it straight back into ``inputs``
    # -- where it is not a mask but an out-of-range embedding index.
    return BlockDataset(ids[:, :-1].clone(), ids[:, 1:].clone())


def _reference_loss(model: nn.Module, dataset: BlockDataset) -> tuple[float, int]:
    """Whole-dataset mean cross-entropy, computed in ONE call.

    The independent oracle: no batching at all, so it cannot share a bug with
    the node's accumulation. ``reduction="mean"`` over every non-ignored
    position IS the token-weighted mean by definition.
    """
    with torch.no_grad():
        logits = model(dataset.inputs)
    targets = dataset.labels.reshape(-1)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets,
        ignore_index=-100, reduction="mean")
    return float(loss.item()), int((targets != -100).sum().item())


def _run(model, dataset, *, context=None, progress=None, **params) -> dict:
    return PerplexityEvaluateNode().execute(
        {"model": model, "dataset": dataset}, params, progress, context=context)


# ── metadata ────────────────────────────────────────────────────────────


def test_node_metadata():
    assert PerplexityEvaluateNode.NODE_NAME == "PerplexityEvaluate"
    assert PerplexityEvaluateNode.CATEGORY == "LLM"
    assert [p.name for p in PerplexityEvaluateNode.define_inputs()] == [
        "model", "dataset"]
    assert [p.name for p in PerplexityEvaluateNode.define_outputs()] == [
        "val_loss", "perplexity", "tokens"]
    params = {p.name: p for p in PerplexityEvaluateNode.define_params()}
    assert params["batch_size"].default == 8
    assert (params["batch_size"].min_value, params["batch_size"].max_value) == (1, 256)
    assert params["max_batches"].default == 0
    assert params["device"].default == "auto"
    assert params["device"].options == ["auto", "cpu", "cuda"]
    assert params["precision"].default == "bf16"
    assert params["precision"].options == ["fp32", "bf16", "fp16"]


def test_the_node_is_not_cacheable():
    # It logs metrics no output carries, and it measures WEIGHTS the cache key
    # cannot describe. Both are enforced registry-wide by
    # test_cache_live_handle_nodes.py; this is the local statement of intent.
    assert PerplexityEvaluateNode.cacheable is False


def test_the_description_says_the_number_is_per_token_and_dataset_specific():
    text = PerplexityEvaluateNode.DESCRIPTION
    assert "PER TOKEN" in text
    assert "dataset" in text


# ── the arithmetic ──────────────────────────────────────────────────────


def test_perplexity_is_exactly_exp_of_val_loss():
    res = _run(TinyLM(), _blocks(), batch_size=2, precision="fp32")
    assert res["perplexity"] == math.exp(res["val_loss"])


def test_val_loss_is_the_whole_dataset_mean_not_the_mean_of_batch_means():
    """5 blocks at batch_size=2 -> batches of 2, 2 and 1.

    The ragged final batch is the whole point: its 4 tokens must carry 4/20 of
    the weight, not 1/3 of it. A mean-of-batch-means implementation lands about
    a percent off here, which is well outside the tolerance below.
    """
    model = TinyLM()
    dataset = _blocks(5)
    expected, expected_tokens = _reference_loss(model, dataset)

    res = _run(model, dataset, batch_size=2, precision="fp32")
    assert res["tokens"] == expected_tokens == 20
    assert res["val_loss"] == pytest.approx(expected, rel=1e-6)


def test_batch_size_does_not_change_the_result():
    model = TinyLM()
    dataset = _blocks(5)
    losses = {
        size: _run(model, dataset, batch_size=size, precision="fp32")["val_loss"]
        for size in (1, 2, 3, 5, 64)
    }
    for size, value in losses.items():
        assert value == pytest.approx(losses[1], rel=1e-6), size


def test_ignored_positions_contribute_neither_loss_nor_count():
    model = TinyLM()
    dataset = _blocks(4)
    # Mask the second half of every block, the shape an instruction dataset
    # has when only the answer is scored.
    dataset.labels[:, SEQ_LEN // 2:] = -100
    expected, expected_tokens = _reference_loss(model, dataset)

    res = _run(model, dataset, batch_size=3, precision="fp32")
    assert expected_tokens == 4 * (SEQ_LEN // 2) == 8
    assert res["tokens"] == 8
    assert res["val_loss"] == pytest.approx(expected, rel=1e-6)


def test_a_masked_position_actually_changes_the_number():
    """Guards the test above from being vacuously true."""
    model = TinyLM()
    dataset = _blocks(4)
    full = _run(model, dataset, batch_size=4, precision="fp32")
    dataset.labels[:, 0] = -100
    masked = _run(model, dataset, batch_size=4, precision="fp32")
    assert masked["tokens"] == full["tokens"] - 4
    assert masked["val_loss"] != full["val_loss"]


def test_max_batches_caps_the_tokens_scored():
    model = TinyLM()
    dataset = _blocks(5)
    capped = _run(model, dataset, batch_size=2, max_batches=1, precision="fp32")
    # One batch of 2 blocks x 4 positions.
    assert capped["tokens"] == 8

    head = BlockDataset(dataset.inputs[:2], dataset.labels[:2])
    expected, _ = _reference_loss(model, head)
    assert capped["val_loss"] == pytest.approx(expected, rel=1e-6)
    # And it really is a subset, not the whole thing.
    assert capped["tokens"] < _run(
        model, dataset, batch_size=2, precision="fp32")["tokens"]


def test_max_batches_beyond_the_dataset_scores_everything():
    model = TinyLM()
    dataset = _blocks(5)
    assert _run(model, dataset, batch_size=2, max_batches=999,
                precision="fp32")["tokens"] == 20


def test_a_capped_run_is_not_reported_as_interrupted():
    # A run that reached its OWN cap finished; marking it interrupted would
    # withhold the metric point it earned.
    context = RecordingContext()
    res = _run(TinyLM(), _blocks(5), batch_size=2, max_batches=1,
               precision="fp32", context=context)
    assert INTERRUPTED_KEY not in res
    assert [name for name, _v, _s in context.metrics] == ["val_loss", "perplexity"]


# ── logged metrics ──────────────────────────────────────────────────────


def test_both_metrics_are_logged_once_at_step_zero():
    context = RecordingContext()
    res = _run(TinyLM(), _blocks(), batch_size=2, precision="fp32",
               context=context)
    assert context.metrics == [
        ("val_loss", res["val_loss"], 0),
        ("perplexity", res["perplexity"], 0),
    ]


def test_without_a_context_nothing_is_logged_and_the_node_still_runs():
    # How the export runner and most unit tests call it.
    res = _run(TinyLM(), _blocks(), batch_size=2, precision="fp32")
    assert res["tokens"] == 20


# ── progress and stop ───────────────────────────────────────────────────


def test_progress_reports_batches_done_out_of_total():
    from app.core.loop_control import EVENT_BATCH

    frames: list[dict] = []
    _run(TinyLM(), _blocks(5), batch_size=2, precision="fp32",
         progress=frames.append)
    assert frames, "no progress frame was emitted"
    assert frames[0]["event"] == EVENT_BATCH
    assert frames[0]["batch"] == 1
    assert frames[0]["total_batches"] == 3
    assert frames[0]["perplexity"] > 0


def test_progress_total_batches_respects_max_batches():
    frames: list[dict] = []
    _run(TinyLM(), _blocks(5), batch_size=1, max_batches=2, precision="fp32",
         progress=frames.append)
    assert frames[0]["total_batches"] == 2


def test_a_stopped_run_returns_the_partial_average_and_does_not_raise():
    context = StopAfter(2)
    res = _run(TinyLM(), _blocks(5), batch_size=1, precision="fp32",
               context=context)
    # Two blocks got through before the third check said stop.
    assert res["tokens"] == 2 * SEQ_LEN
    assert res[INTERRUPTED_KEY]["batch"] == 2
    # An incomplete pass is deliberately NOT filed as a measurement.
    assert context.metrics == []


def test_a_run_stopped_before_the_first_batch_reports_zero_rather_than_dividing():
    res = _run(TinyLM(), _blocks(5), batch_size=1, precision="fp32",
               context=StopAfter(0))
    assert res["tokens"] == 0
    assert res["val_loss"] == 0.0
    assert res["perplexity"] == 1.0


# ── model handling ──────────────────────────────────────────────────────


def test_the_models_training_flag_survives_the_measurement():
    model = TinyLM()
    model.train()
    _run(model, _blocks(), batch_size=2, precision="fp32")
    # The same module flows on to whatever is wired next -- most likely another
    # training phase, which must not silently run with dropout disabled.
    assert model.training is True

    model.eval()
    _run(model, _blocks(), batch_size=2, precision="fp32")
    assert model.training is False


def test_the_forward_pass_runs_in_eval_mode():
    seen: list[bool] = []

    class ModeProbe(TinyLM):
        def forward(self, input_ids):
            seen.append(self.training)
            return super().forward(input_ids)

    model = ModeProbe()
    model.train()
    _run(model, _blocks(2), batch_size=2, precision="fp32")
    assert seen == [False]


def test_the_measurement_leaves_no_gradients_behind():
    model = TinyLM()
    _run(model, _blocks(), batch_size=2, precision="fp32")
    assert all(p.grad is None for p in model.parameters())


def test_a_missing_model_names_the_node_to_wire():
    with pytest.raises(ValueError, match="CausalLMModel"):
        PerplexityEvaluateNode().execute({"dataset": _blocks()}, {})


def test_a_missing_dataset_names_the_node_to_wire():
    with pytest.raises(ValueError, match="LMTokenizedDataset"):
        PerplexityEvaluateNode().execute({"model": TinyLM()}, {})


def test_a_classifier_shaped_model_is_refused_by_name():
    class Classifier(nn.Module):
        def forward(self, input_ids):
            return torch.zeros(input_ids.shape[0], 3)

    with pytest.raises(ValueError, match="EvaluateModel"):
        _run(Classifier(), _blocks(), batch_size=2)


def test_a_raw_text_corpus_names_the_node_that_should_be_between():
    # The likeliest wiring mistake in this chain. Without the check the failure
    # surfaces as torch.as_tensor refusing a str, naming neither the node nor
    # the fix.
    from app.nodes.llm.text_corpus_dataset_node import TextRowDataset

    with pytest.raises(ValueError, match="LMTokenizedDataset"):
        _run(TinyLM(), TextRowDataset(["ab", "cd"]), batch_size=2)


def test_labels_of_the_wrong_length_say_which_shapes_disagree():
    dataset = _blocks(2)
    dataset.labels = dataset.labels[:, :-1]
    with pytest.raises(ValueError, match="LMTokenizedDataset"):
        _run(TinyLM(), dataset, batch_size=2)


# ── device and precision wiring ─────────────────────────────────────────


def test_device_resolves_through_resolve_node_device(monkeypatch):
    """"auto" must mean "follow the run-level device" (#204's lesson).

    The spy delegates to the real implementation, so this proves the wiring
    and that real resolution still runs.
    """
    from app.core import device_utils

    real = device_utils.resolve_node_device
    seen: dict[str, object] = {}

    def spy(value, context):
        seen["value"] = value
        seen["context"] = context
        return real(value, context)

    monkeypatch.setattr(device_utils, "resolve_node_device", spy)
    context = RecordingContext()
    _run(TinyLM(), _blocks(2), batch_size=2, precision="fp32", device="auto",
         context=context)
    assert seen["value"] == "auto"
    assert seen["context"] is context


def test_bf16_precision_is_honoured_on_the_cpu_and_the_loss_is_still_finite():
    """CPU autocast is about numerical parity, not speed -- and it runs in CI.

    The forward pass really is 16-bit (the probe reads the dtype off its own
    output), and the summed loss still comes back a sensible float because it
    is accumulated in fp32.
    """
    class DtypeProbe(TinyLM):
        seen_dtype = None

        def forward(self, input_ids):
            out = super().forward(input_ids)
            DtypeProbe.seen_dtype = out.dtype
            return out

    res = _run(DtypeProbe(), _blocks(), batch_size=2, precision="bf16")
    assert DtypeProbe.seen_dtype == torch.bfloat16
    assert math.isfinite(res["val_loss"])
    assert res["perplexity"] == math.exp(res["val_loss"])


def test_bf16_and_fp32_agree_to_about_two_decimals():
    # Not bit-identical and not expected to be; the point is that the fp32
    # accumulation keeps a 16-bit forward pass from drifting into a different
    # answer.
    model = TinyLM()
    dataset = _blocks(5)
    fp32 = _run(model, dataset, batch_size=2, precision="fp32")["val_loss"]
    bf16 = _run(model, dataset, batch_size=2, precision="bf16")["val_loss"]
    assert bf16 == pytest.approx(fp32, abs=0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_an_explicit_cuda_device_scores_on_the_gpu():
    model = TinyLM()
    res = _run(model, _blocks(), batch_size=2, device="cuda")
    assert math.isfinite(res["val_loss"])
    assert next(model.parameters()).device.type == "cuda"


# ── the extreme end of the scale ────────────────────────────────────────


def test_an_astronomical_loss_reports_infinite_perplexity_rather_than_raising():
    """A model and a dataset that disagree about what the ids mean.

    ``math.exp`` raises above about 709, and a mismatched tokenizer is exactly
    how a loss gets there. Infinity is the honest reading; a failed run is not.
    """
    class Diverged(nn.Module):
        def forward(self, input_ids):
            logits = torch.zeros(
                *input_ids.shape, VOCAB, dtype=torch.float32)
            # Everything except the true class is a certainty.
            logits[..., 0] = 5000.0
            return logits

    dataset = _blocks(2)
    dataset.labels.fill_(1)
    res = _run(Diverged(), dataset, batch_size=2, precision="fp32")
    assert res["val_loss"] > 709
    assert res["perplexity"] == float("inf")
