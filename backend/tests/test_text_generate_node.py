"""Tests for TextGenerateNode (#291).

Two kinds of test, deliberately:

* Against a REAL ``CausalLMModel`` overfitted on one memorized sequence, for
  the property that only a real model can demonstrate -- greedy decoding
  reproduces what the model was taught.
* Against ``StubLM``, a module with the same forward contract and logits this
  file chooses, for the properties that are about TextGenerate's OWN loop:
  seeded sampling, the top-k / top-p filters, the sliding context window and
  the EOS stop. A randomly initialised model's argmax is an arbitrary token,
  so those assertions would be flaky rather than meaningful against one.

No network and no tiktoken: the tokenizer is a fake whose ids are letter
positions, so every expected string can be read off the source text.

The wired end-to-end test at the bottom is #291's acceptance criterion and
covers BOTH new nodes: a real corpus trains a real model through TrainingLoop,
and the trained module -- the same Python object, with no checkpoint round trip
-- flows straight into PerplexityEvaluate and TextGenerate.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.execution_context import INTERRUPTED_KEY
from app.nodes.llm.text_generate_node import TextGenerateNode

#: The fake vocabulary: 0 unused, 1..14 are the letters a..n, 15 is EOS.
VOCAB = 16
EOS_ID = 15

#: Score for a token the stub wants to make unreachable. Not ``-inf``: the
#: filters under test insert their own ``-inf``, and a stub that also used one
#: could hide a filter that did nothing.
_BLOCKED = -1e9


class FakeTokenizer:
    """The duck-typed contract, with ids that are readable by eye.

    ``encode`` maps a..n to 1..14 and ``decode`` maps back, so a generated
    string is exactly the ids the loop appended. ``eos_id = 15`` decodes to
    "o", which no test text contains -- an EOS leaking into the output is
    therefore visible rather than plausible.
    """

    name = "fake"
    vocab_size = VOCAB
    eos_id = EOS_ID

    def encode(self, text: str) -> list[int]:
        return [ord(c) - 96 for c in text]

    def decode(self, ids) -> str:
        return "".join(chr(int(i) + 96) for i in ids)


class StubLM(nn.Module):
    """CausalLMModel's forward contract, with logits this file chooses.

    *scores* is one list of per-token scores, or a list of such lists consumed
    one per forward call with the last one repeating (which is how the EOS test
    schedules an end-of-text on the third step).

    ``max_seq_len`` is published only when given, and a longer sequence RAISES
    -- mirroring ``CausalLMModule``, so a test of the sliding window fails
    loudly if the window is not applied instead of quietly producing text.
    """

    def __init__(self, scores, *, max_seq_len: int | None = None) -> None:
        super().__init__()
        rows = scores if isinstance(scores[0], (list, tuple)) else [scores]
        self.rows = [torch.tensor(row, dtype=torch.float32) for row in rows]
        self.calls = 0
        self.longest_context = 0
        if max_seq_len is not None:
            self.max_seq_len = int(max_seq_len)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = int(input_ids.shape[1])
        self.longest_context = max(self.longest_context, seq_len)
        limit = getattr(self, "max_seq_len", None)
        if limit is not None and seq_len > limit:
            raise ValueError(
                f"StubLM: got {seq_len} tokens but max_seq_len is {limit}.")
        row = self.rows[min(self.calls, len(self.rows) - 1)]
        self.calls += 1
        return row.expand(int(input_ids.shape[0]), seq_len, VOCAB).clone()


class StopAfter:
    """A context whose ``should_stop`` turns True after *n* checks."""

    def __init__(self, n: int) -> None:
        self._left = n

    def should_stop(self) -> bool:
        if self._left > 0:
            self._left -= 1
            return False
        return True


def _scores(*, favour: int | None = None, over: range | None = None) -> list[float]:
    """A score row: everything blocked except *favour*, or flat over *over*."""
    row = [_BLOCKED] * VOCAB
    if favour is not None:
        row[favour] = 1.0
    for token in over or ():
        row[token] = 0.0
    return row


def _run(model, *, tokenizer=None, prompt_input=None, progress=None,
         context=None, **params) -> dict:
    inputs = {
        "model": model,
        "tokenizer": FakeTokenizer() if tokenizer is None else tokenizer,
    }
    if prompt_input is not None:
        inputs["prompt"] = prompt_input
    return TextGenerateNode().execute(inputs, params, progress, context=context)


# ── metadata ────────────────────────────────────────────────────────────


def test_node_metadata():
    assert TextGenerateNode.NODE_NAME == "TextGenerate"
    assert TextGenerateNode.CATEGORY == "LLM"
    ports = {p.name: p for p in TextGenerateNode.define_inputs()}
    assert list(ports) == ["model", "tokenizer", "prompt"]
    assert ports["prompt"].optional is True
    assert [p.name for p in TextGenerateNode.define_outputs()] == [
        "text", "token_count"]

    params = {p.name: p for p in TextGenerateNode.define_params()}
    assert params["prompt"].default == "Once upon a time"
    assert params["max_new_tokens"].default == 200
    assert (params["max_new_tokens"].min_value,
            params["max_new_tokens"].max_value) == (1, 4096)
    assert params["temperature"].default == 0.8
    assert (params["temperature"].min_value, params["temperature"].max_value) == (0.0, 2.0)
    assert params["top_k"].default == 50
    assert (params["top_k"].min_value, params["top_k"].max_value) == (0, 1000)
    assert params["top_p"].default == 0.95
    assert (params["top_p"].min_value, params["top_p"].max_value) == (0.0, 1.0)
    assert params["seed"].default == 0
    assert params["device"].default == "auto"
    assert params["device"].options == ["auto", "cpu", "cuda"]


def test_the_node_is_not_cacheable():
    # Sampling is not a function of the params, and the streamed tokens are
    # the point of running it -- a cache hit would return the final string
    # having emitted nothing.
    assert TextGenerateNode.cacheable is False


# ── greedy decoding against a real, overfitted model ────────────────────


def _tiny_model(*, seed: int = 0):
    """A real, untrained CausalLMModel small enough to train in a test."""
    from app.nodes.llm.causal_lm_model_node import CausalLMModelNode

    return CausalLMModelNode().execute({}, {
        "vocab_size": VOCAB,
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 2,
        "d_ff": 64,
        "max_seq_len": 32,
        "tie_embeddings": False,
        "seed": seed,
    })["model"]


def _memorize(sequence: str, *, steps: int = 400, seed: int = 0):
    """Train a real tiny CausalLMModel until it has memorized *sequence*.

    Deliberately not through TrainingLoop: this is a fixture, and a plain loop
    keeps it fast and its convergence checkable right here. The wired test at
    the bottom is the one that goes through the real training node.
    """
    model = _tiny_model(seed=seed)
    ids = torch.tensor([FakeTokenizer().encode(sequence)], dtype=torch.int64)
    inputs, labels = ids[:, :-1], ids[:, 1:]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model.train()
    loss = torch.tensor(float("nan"))
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(
            model(inputs).reshape(-1, VOCAB), labels.reshape(-1))
        loss.backward()
        optimizer.step()
    # The test below is only meaningful if the model really did memorize it.
    assert float(loss.item()) < 0.05, "the fixture did not converge"
    return model


def test_greedy_decoding_reproduces_what_the_model_memorized():
    model = _memorize("abcdefghijklmn")
    res = _run(model, prompt="abcd", max_new_tokens=5, temperature=0.0)
    assert res["text"] == "abcdefghi"
    assert res["token_count"] == 5


def test_greedy_decoding_is_reproducible_and_ignores_the_seed():
    model = _memorize("abcdefghijklmn")
    first = _run(model, prompt="abcd", max_new_tokens=5, temperature=0.0, seed=1)
    second = _run(model, prompt="abcd", max_new_tokens=5, temperature=0.0, seed=99)
    assert first["text"] == second["text"]


# ── sampling: the seed, the temperature and the two filters ─────────────


def _uniform_stub() -> StubLM:
    """Flat scores over the 14 letters; the EOS and id 0 unreachable.

    Sampling from this is a pure function of the generator, so a difference
    between two runs can only come from the seed -- and an EOS can never end a
    run early and make two runs equal by accident.
    """
    return StubLM(_scores(over=range(1, 15)))


def test_the_same_seed_generates_the_same_text_twice():
    first = _run(_uniform_stub(), prompt="ab", max_new_tokens=20,
                 temperature=1.0, top_k=0, top_p=1.0, seed=1234)
    second = _run(_uniform_stub(), prompt="ab", max_new_tokens=20,
                  temperature=1.0, top_k=0, top_p=1.0, seed=1234)
    assert first["text"] == second["text"]
    assert first["token_count"] == second["token_count"] == 20


def test_a_different_seed_generates_different_text():
    first = _run(_uniform_stub(), prompt="ab", max_new_tokens=20,
                 temperature=1.0, top_k=0, top_p=1.0, seed=1)
    second = _run(_uniform_stub(), prompt="ab", max_new_tokens=20,
                  temperature=1.0, top_k=0, top_p=1.0, seed=2)
    # 20 draws from 14 equally likely tokens: a collision is a 14**-20 event,
    # so this is a real assertion rather than a hopeful one.
    assert first["text"] != second["text"]


def test_the_node_does_not_disturb_the_global_rng():
    # A node that called torch.manual_seed would silently change every other
    # node's randomness (dropout masks, a DataLoader's shuffle).
    torch.manual_seed(4321)
    before = torch.rand(3)
    torch.manual_seed(4321)
    _run(_uniform_stub(), prompt="ab", max_new_tokens=10, temperature=1.0,
         seed=7)
    assert torch.equal(torch.rand(3), before)


def test_sampling_never_reaches_a_token_the_model_ruled_out():
    res = _run(_uniform_stub(), prompt="ab", max_new_tokens=40,
               temperature=1.0, top_k=0, top_p=1.0, seed=5)
    # Only a..n are reachable; "o" is the EOS id and "`" would be id 0.
    assert set(res["text"][2:]) <= set("abcdefghijklmn")


def test_top_k_of_one_collapses_sampling_onto_the_most_likely_token():
    """Proves the top-k filter is actually applied.

    The scores make id 3 best and id 5 a close second, so an unfiltered draw at
    temperature 1 would take id 5 a good fraction of the time.
    """
    row = _scores(over=range(1, 15))
    row[3], row[5] = 2.0, 1.9
    res = _run(StubLM(row), prompt="ab", max_new_tokens=12, temperature=1.0,
               top_k=1, top_p=1.0, seed=3)
    assert res["text"] == "ab" + "c" * 12


def test_top_p_of_zero_keeps_the_single_most_likely_token():
    """The shifted mask, tested at its edge.

    Without the one-place shift in ``_filter_top_p``, top_p = 0 would remove
    EVERY token -- softmax over all -inf is nan and multinomial raises. Keeping
    the token that crosses the threshold is what makes the degenerate setting
    read as "greedy".
    """
    row = _scores(over=range(1, 15))
    row[7] = 3.0
    res = _run(StubLM(row), prompt="ab", max_new_tokens=6, temperature=1.0,
               top_k=0, top_p=0.0, seed=11)
    assert res["text"] == "ab" + "g" * 6


def test_temperature_zero_takes_the_argmax_even_with_filters_on():
    row = _scores(over=range(1, 15))
    row[9] = 5.0
    res = _run(StubLM(row), prompt="ab", max_new_tokens=3, temperature=0.0)
    assert res["text"] == "abiii"


def test_a_non_numeric_temperature_says_which_param_is_wrong():
    with pytest.raises(ValueError, match="temperature"):
        _run(_uniform_stub(), prompt="ab", temperature="warm")


def test_a_non_numeric_top_k_says_which_param_is_wrong():
    with pytest.raises(ValueError, match="top_k"):
        _run(_uniform_stub(), prompt="ab", top_k="lots")


def test_a_null_top_k_reads_as_unset_rather_than_as_disabled():
    """The falsiness trap, in the direction it actually bites.

    A hand-built graph.json with ``"top_k": null`` means "I did not set this",
    and the default (50) is what the editor would have written. Reading it as 0
    would silently disable a filter nobody turned off; reading a genuine 0 as
    50 would silently enable one nobody turned on. Both cases are pinned here.
    """
    row = _scores(over=range(1, 15))
    row[3], row[5] = 2.0, 1.9

    unset = _run(StubLM(row), prompt="ab", max_new_tokens=12, temperature=1.0,
                 top_k=None, top_p=1.0, seed=3)
    explicit = _run(StubLM(row), prompt="ab", max_new_tokens=12,
                    temperature=1.0, top_k=50, top_p=1.0, seed=3)
    assert unset["text"] == explicit["text"]

    disabled = _run(StubLM(row), prompt="ab", max_new_tokens=12,
                    temperature=1.0, top_k=0, top_p=1.0, seed=3)
    # top_k=50 is wider than this 16-token vocabulary, so it removes nothing
    # and the two agree -- which is the point: 0 and 50 mean the same thing
    # here, and NEITHER may be confused with the other's provenance.
    assert disabled["text"] == explicit["text"]


def test_a_max_seq_len_that_is_not_a_usable_number_is_not_treated_as_one():
    """A module publishing something odd gets no window, not a nonsense one.

    ``max_seq_len = True`` is the trap: bool is an int subclass, so a naive
    ``isinstance(raw, int)`` would slide a ONE-token window and happily
    generate from a model that never asked for a window at all. The whole
    prompt therefore reaches the model, and the model's own length check is
    what speaks -- which is exactly the outcome asserted here, because a
    one-token context would have passed that check silently.
    """
    model = StubLM(_scores(favour=1))
    model.max_seq_len = True
    with pytest.raises(ValueError, match="max_seq_len is True"):
        _run(model, prompt="ab", max_new_tokens=6, temperature=0.0)
    assert model.longest_context == 2


# ── the end-of-text token ───────────────────────────────────────────────


def test_eos_stops_generation_early_and_is_not_written_out():
    model = StubLM([_scores(favour=3), _scores(favour=3), _scores(favour=EOS_ID)])
    res = _run(model, prompt="ab", max_new_tokens=50, temperature=0.0)
    assert res["text"] == "abcc"
    assert res["token_count"] == 2
    # The EOS id decodes to "o". It is punctuation for the machine; writing it
    # out would put a literal <|endoftext|> in a real tokenizer's output.
    assert "o" not in res["text"]
    # And it stopped: 3 forward passes, not 50.
    assert model.calls == 3


def test_a_run_that_ends_on_eos_is_not_reported_as_interrupted():
    model = StubLM([_scores(favour=3), _scores(favour=EOS_ID)])
    res = _run(model, prompt="ab", max_new_tokens=50, temperature=0.0)
    assert INTERRUPTED_KEY not in res


def test_an_immediate_eos_returns_the_prompt_unchanged():
    res = _run(StubLM(_scores(favour=EOS_ID)), prompt="ab", max_new_tokens=9,
               temperature=0.0)
    assert res["text"] == "ab"
    assert res["token_count"] == 0


# ── the sliding context window ──────────────────────────────────────────


def test_generation_past_the_context_length_slides_the_window():
    """max_seq_len=16, 40 new tokens: 52 ids exist, 16 are ever forwarded.

    StubLM raises on a longer sequence exactly as CausalLMModel does, so this
    fails loudly if the window is not applied rather than quietly returning
    short text.
    """
    model = StubLM(_scores(favour=1), max_seq_len=16)
    res = _run(model, prompt="abcdefghijkl", max_new_tokens=40, temperature=0.0)
    assert res["token_count"] == 40
    assert res["text"].startswith("abcdefghijkl")
    assert len(res["text"]) == 12 + 40
    assert model.longest_context == 16


def test_a_model_without_a_max_seq_len_is_given_the_whole_context():
    """No window is guessed for a module that does not publish one.

    A guessed window would silently truncate a model that could have handled
    the whole context; the model's own length check is the authority.
    """
    model = StubLM(_scores(favour=1))
    assert not hasattr(model, "max_seq_len")
    res = _run(model, prompt="ab", max_new_tokens=10, temperature=0.0)
    assert res["token_count"] == 10
    assert model.longest_context == 11


def test_a_context_that_outgrows_the_model_still_raises_from_the_model():
    """The fallback above does not swallow the model's own refusal."""
    class NoLimitAttribute(StubLM):
        def forward(self, input_ids):
            if int(input_ids.shape[1]) > 5:
                raise ValueError("model: sequence too long")
            return super().forward(input_ids)

    with pytest.raises(ValueError, match="too long"):
        _run(NoLimitAttribute(_scores(favour=1)), prompt="ab",
             max_new_tokens=10, temperature=0.0)


# ── the prompt port ─────────────────────────────────────────────────────


def test_a_connected_prompt_overrides_the_param():
    res = _run(StubLM(_scores(favour=3)), prompt_input="ab", prompt="mnmn",
               max_new_tokens=1, temperature=0.0)
    assert res["text"] == "abc"


def test_the_param_is_used_when_nothing_is_connected():
    res = _run(StubLM(_scores(favour=3)), prompt="mn", max_new_tokens=1,
               temperature=0.0)
    assert res["text"] == "mnc"


def test_an_empty_prompt_says_what_to_do_about_it():
    with pytest.raises(ValueError, match="prompt"):
        _run(StubLM(_scores(favour=3)), prompt="")


def test_an_empty_connected_prompt_is_refused_rather_than_falling_back():
    # A connected port that carries "" is a wired-up upstream node producing
    # nothing, not an unset param -- falling back to the param would generate
    # from text the user cannot see in the graph.
    with pytest.raises(ValueError, match="prompt"):
        _run(StubLM(_scores(favour=3)), prompt_input="", prompt="abcd")


# ── progress and stop ───────────────────────────────────────────────────


def test_progress_streams_the_text_so_far():
    from app.core.loop_control import EVENT_BATCH

    frames: list[dict] = []
    _run(StubLM(_scores(favour=3)), prompt="ab", max_new_tokens=5,
         temperature=0.0, progress=frames.append)
    assert frames, "no progress frame was emitted"
    first = frames[0]
    # Tagged as liveness so run_service does not mine "tokens" into a chart
    # series -- that would be a chart of how fast the machine is.
    assert first["event"] == EVENT_BATCH
    assert first["text"] == "abc"
    assert first["tokens"] == 1
    assert first["total_tokens"] == 5


def test_a_stopped_run_returns_the_partial_text_and_does_not_raise():
    res = _run(StubLM(_scores(favour=3)), prompt="ab", max_new_tokens=50,
               temperature=0.0, context=StopAfter(3))
    assert res["text"] == "abccc"
    assert res["token_count"] == 3
    assert res[INTERRUPTED_KEY]["tokens"] == 3


def test_a_run_stopped_before_the_first_token_returns_the_prompt():
    model = StubLM(_scores(favour=3))
    res = _run(model, prompt="ab", max_new_tokens=50, temperature=0.0,
               context=StopAfter(0))
    assert res["text"] == "ab"
    assert res["token_count"] == 0
    assert res[INTERRUPTED_KEY]["tokens"] == 0
    # And it did not pay for a forward pass on the way out.
    assert model.calls == 0


# ── the model handle ────────────────────────────────────────────────────


def test_the_models_training_flag_survives_the_generation():
    model = StubLM(_scores(favour=3))
    model.train()
    _run(model, prompt="abc", max_new_tokens=2, temperature=0.0)
    # The same module flows on to whatever is wired next; a second training
    # phase must not silently run with dropout disabled.
    assert model.training is True

    model.eval()
    _run(model, prompt="abc", max_new_tokens=2, temperature=0.0)
    assert model.training is False


def test_the_forward_pass_runs_in_eval_mode():
    seen: list[bool] = []

    class ModeProbe(StubLM):
        def forward(self, input_ids):
            seen.append(self.training)
            return super().forward(input_ids)

    model = ModeProbe(_scores(favour=3))
    model.train()
    _run(model, prompt="ab", max_new_tokens=2, temperature=0.0)
    assert seen == [False, False]


def test_generation_leaves_no_gradients_behind():
    model = _tiny_model()
    _run(model, prompt="abc", max_new_tokens=3, temperature=0.0)
    assert all(p.grad is None for p in model.parameters())


def test_a_missing_model_names_the_node_to_wire():
    with pytest.raises(ValueError, match="CausalLMModel"):
        TextGenerateNode().execute({"tokenizer": FakeTokenizer()}, {})


def test_a_missing_tokenizer_names_the_node_to_wire():
    with pytest.raises(ValueError, match="LMTokenizer"):
        TextGenerateNode().execute({"model": StubLM(_scores(favour=3))}, {})


def test_a_non_conforming_tokenizer_lists_what_it_is_missing():
    class NoDecode:
        vocab_size = VOCAB
        eos_id = EOS_ID

        def encode(self, text):
            return [1]

    with pytest.raises(ValueError, match="decode"):
        _run(StubLM(_scores(favour=3)), tokenizer=NoDecode(), prompt="ab")


def test_a_classifier_shaped_model_is_refused_by_name():
    class Classifier(nn.Module):
        def forward(self, input_ids):
            return torch.zeros(input_ids.shape[0], 3)

    with pytest.raises(ValueError, match="CausalLMModel"):
        _run(Classifier(), prompt="ab", max_new_tokens=2)


# ── device wiring ───────────────────────────────────────────────────────


def test_device_resolves_through_resolve_node_device(monkeypatch):
    """"auto" must mean "follow the run-level device" (#204's lesson)."""
    from app.core import device_utils

    real = device_utils.resolve_node_device
    seen: dict[str, object] = {}

    def spy(value, context):
        seen["value"] = value
        seen["context"] = context
        return real(value, context)

    monkeypatch.setattr(device_utils, "resolve_node_device", spy)
    context = StopAfter(99)
    _run(StubLM(_scores(favour=3)), prompt="ab", max_new_tokens=2,
         temperature=0.0, device="auto", context=context)
    assert seen["value"] == "auto"
    assert seen["context"] is context


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_generating_on_cuda_gives_the_same_text_as_on_the_cpu():
    # The sampling arithmetic runs on the CPU precisely so a seed means the
    # same text everywhere; this is that promise.
    model = _memorize("abcdefghijklmn")
    on_cpu = _run(model, prompt="abcd", max_new_tokens=5, temperature=0.9,
                  seed=17, device="cpu")
    on_gpu = _run(model, prompt="abcd", max_new_tokens=5, temperature=0.9,
                  seed=17, device="cuda")
    assert on_gpu["text"] == on_cpu["text"]


# ── wired end to end (the #291 acceptance criterion) ────────────────────


def test_a_trained_model_flows_into_perplexity_and_generation():
    """corpus -> packed blocks -> DataLoader -> TrainingLoop -> both new nodes.

    All real nodes, and the MODEL that reaches PerplexityEvaluate and
    TextGenerate is the very object TrainingLoop mutated -- no checkpoint round
    trip. The perplexity dropping across training is what proves it: a fresh
    module would score the same as it did before.

    The tokenizer is the fake one rather than a real gpt2 encoding, so this
    stays offline and the model stays small enough to actually converge in a
    test. #290's acceptance test already covers the real tiktoken path.
    """
    from app.nodes.data.dataloader_node import DataLoaderNode
    from app.nodes.llm.causal_lm_model_node import CausalLMModelNode
    from app.nodes.llm.lm_cross_entropy_loss_node import LMCrossEntropyLossNode
    from app.nodes.llm.lm_tokenized_dataset_node import LMTokenizedDatasetNode
    from app.nodes.llm.perplexity_evaluate_node import PerplexityEvaluateNode
    from app.nodes.llm.text_corpus_dataset_node import TextRowDataset
    from app.nodes.training.optimizer_node import OptimizerNode
    from app.nodes.training.training_loop_node import TrainingLoopNode

    seq_len = 8
    tokenizer = FakeTokenizer()
    corpus = TextRowDataset(["abcdefghijklmn"] * 12)

    packed = LMTokenizedDatasetNode().execute(
        {"dataset": corpus, "tokenizer": tokenizer},
        {"seq_len": seq_len, "cache": False},
    )
    assert packed["num_blocks"] >= 8

    loader = DataLoaderNode().execute(
        {"dataset": packed["dataset"]}, {"batch_size": 4, "shuffle": False})
    built = CausalLMModelNode().execute({}, {
        "vocab_size": VOCAB,
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 2,
        "d_ff": 64,
        "max_seq_len": 16,
        "seed": 0,
    })
    model = built["model"]
    optimizer = OptimizerNode().execute(
        {"model": model}, {"type": "AdamW", "lr": 0.01})
    loss_fn = LMCrossEntropyLossNode().execute({}, {})

    def measure() -> dict:
        return PerplexityEvaluateNode().execute(
            {"model": model, "dataset": packed["dataset"]},
            {"batch_size": 4, "device": "cpu", "precision": "fp32"},
        )

    before = measure()
    assert before["tokens"] == packed["num_blocks"] * seq_len

    trained = TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": loader["dataloader"],
            "optimizer": optimizer["optimizer"],
            "loss_fn": loss_fn["loss_fn"],
        },
        {"epochs": 10, "device": "cpu"},
    )
    # The live handle, not a copy: this is what "no checkpoint round trip"
    # means, and it is why the measurement below moves at all.
    assert trained["model"] is model
    assert torch.isfinite(trained["losses"]).all()
    mode_after_training = model.training

    after = measure()
    assert after["perplexity"] < before["perplexity"]
    assert after["perplexity"] == pytest.approx(math.exp(after["val_loss"]))
    assert after["tokens"] == before["tokens"]

    generated = TextGenerateNode().execute(
        {"model": model, "tokenizer": tokenizer},
        {"prompt": "abcd", "max_new_tokens": 6, "temperature": 0.0,
         "device": "cpu"},
    )
    assert generated["text"].startswith("abcd")
    # Self-consistent: one character per generated id, and the model may have
    # chosen to stop early on the end-of-text token it was trained with.
    assert len(generated["text"]) == 4 + generated["token_count"]
    assert 0 <= generated["token_count"] <= 6
    # Neither new node latched the module into eval(), so it is still in the
    # mode training left it in and a second training phase downstream would
    # run with dropout exactly as configured.
    assert model.training is mode_after_training
