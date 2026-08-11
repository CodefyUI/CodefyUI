"""Tests for LMTokenizedDatasetNode (#290).

No tiktoken and no network for the packing tests: a FAKE tokenizer whose
``encode`` is "one id per character" makes every expected number computable by
hand, which is the point -- the packing math (EOS joining, block boundaries,
the shift by one, the dropped remainder) is what the rest of the LM chain is
built on, so it is pinned against arithmetic rather than against itself.

The wired end-to-end test at the bottom is the issue's acceptance criterion and
uses the REAL nodes, including a real gpt2 LMTokenizer.
"""

from __future__ import annotations

import pytest
import torch

from app.config import settings
from app.nodes.llm.lm_tokenized_dataset_node import LMTokenizedDatasetNode


class FakeTokenizer:
    """The duck-typed contract, with a countable ``encode``.

    ``encode`` is ``ord()`` per character, so a two-character row is exactly
    two ids and every expected stream in this file can be read off the source
    text. ``eos_id = 0`` is not a character any test uses, so an EOS in the
    stream is unambiguous.
    """

    name = "fake"
    vocab_size = 256
    eos_id = 0

    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, text: str) -> list[int]:
        self.encode_calls += 1
        return [ord(c) for c in text]

    def decode(self, ids) -> str:
        return "".join(chr(int(i)) for i in ids)


class StopAfter:
    """A context whose ``should_stop`` turns True after *n* checks."""

    def __init__(self, n: int) -> None:
        self._left = n
        self.checks = 0

    def should_stop(self) -> bool:
        self.checks += 1
        if self._left > 0:
            self._left -= 1
            return False
        return True


@pytest.fixture(autouse=True)
def data_root_in_tmp(tmp_path, monkeypatch):
    """Keep the disk cache out of the real backend/data for every test here."""
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "models")
    return tmp_path


def _rows(*texts: str):
    from app.nodes.llm.text_corpus_dataset_node import TextRowDataset

    return TextRowDataset(list(texts))


def _run(dataset, tokenizer, **params) -> dict:
    return LMTokenizedDatasetNode().execute(
        {"dataset": dataset, "tokenizer": tokenizer}, params)


# ── metadata ────────────────────────────────────────────────────────────


def test_node_metadata():
    assert LMTokenizedDatasetNode.NODE_NAME == "LMTokenizedDataset"
    assert LMTokenizedDatasetNode.CATEGORY == "LLM"
    assert [p.name for p in LMTokenizedDatasetNode.define_inputs()] == [
        "dataset", "tokenizer"]
    assert [p.name for p in LMTokenizedDatasetNode.define_outputs()] == [
        "dataset", "total_tokens", "num_blocks"]
    params = {p.name: p for p in LMTokenizedDatasetNode.define_params()}
    assert params["seq_len"].default == 1024
    assert params["append_eos"].default is True
    assert params["cache"].default is True
    assert params["cache_dir"].visible_when == {"cache": True}


def test_the_node_is_not_cacheable():
    assert LMTokenizedDatasetNode.cacheable is False


# ── packing math ────────────────────────────────────────────────────────
#
# corpus = "ab", "cde"; ord()-per-character; eos_id = 0
#   with EOS:    [97, 98, 0, 99, 100, 101, 0]   -> 7 tokens
#   without EOS: [97, 98, 99, 100, 101]         -> 5 tokens


def test_total_tokens_and_num_blocks_are_hand_computable():
    res = _run(_rows("ab", "cde"), FakeTokenizer(), seq_len=3, cache=False)
    assert res["total_tokens"] == 7
    # (7 - 1) // 3: the last token can only ever be a LABEL, and the remainder
    # is dropped rather than padded.
    assert res["num_blocks"] == 2
    assert len(res["dataset"]) == 2


def test_block_contents_match_the_raw_stream():
    res = _run(_rows("ab", "cde"), FakeTokenizer(), seq_len=3, cache=False)
    dataset = res["dataset"]

    inputs0, labels0 = dataset[0]
    assert inputs0.tolist() == [97, 98, 0]
    assert labels0.tolist() == [98, 0, 99]

    # Block 1 starts where block 0's INPUTS end (at 1 * seq_len), so the
    # windows overlap by exactly the one token the shift needs.
    inputs1, labels1 = dataset[1]
    assert inputs1.tolist() == [99, 100, 101]
    assert labels1.tolist() == [100, 101, 0]


def test_labels_are_inputs_shifted_by_one():
    res = _run(_rows("abcdefghij"), FakeTokenizer(), seq_len=4, cache=False)
    inputs, labels = res["dataset"][0]
    assert labels[:-1].tolist() == inputs[1:].tolist()


def test_blocks_are_int64_of_exactly_seq_len():
    res = _run(_rows("abcdefghij"), FakeTokenizer(), seq_len=4, cache=False)
    for inputs, labels in (res["dataset"][0], res["dataset"][1]):
        assert inputs.dtype == torch.int64 and labels.dtype == torch.int64
        assert inputs.shape == labels.shape == (4,)


def test_append_eos_off_removes_the_document_separators():
    res = _run(_rows("ab", "cde"), FakeTokenizer(),
               seq_len=3, append_eos=False, cache=False)
    assert res["total_tokens"] == 5
    assert res["num_blocks"] == 1
    inputs, labels = res["dataset"][0]
    assert inputs.tolist() == [97, 98, 99]
    assert labels.tolist() == [98, 99, 100]


def test_max_tokens_truncates_the_stream():
    res = _run(_rows("ab", "cde"), FakeTokenizer(),
               seq_len=3, max_tokens=5, cache=False)
    assert res["total_tokens"] == 5
    assert res["num_blocks"] == 1
    inputs, _labels = res["dataset"][0]
    assert inputs.tolist() == [97, 98, 0]


def test_max_tokens_stops_tokenizing_early():
    tokenizer = FakeTokenizer()
    _run(_rows("ab", "cde", "fgh", "ijk"), tokenizer,
         seq_len=2, max_tokens=4, cache=False)
    # Row 1 already fills the budget, so rows 2 and 3 are never encoded --
    # the cap has to save the work, not just discard it afterwards.
    assert tokenizer.encode_calls == 2


def test_the_remainder_is_dropped_not_padded():
    # 10 characters + one EOS = 11 tokens; (11 - 1) // 4 = 2 blocks, covering
    # stream positions 0..8. The last two tokens are dropped, not padded.
    res = _run(_rows("abcdefghij"), FakeTokenizer(), seq_len=4, cache=False)
    assert res["total_tokens"] == 11
    assert res["num_blocks"] == 2


def test_indexing_past_the_end_raises_index_error():
    res = _run(_rows("ab", "cde"), FakeTokenizer(), seq_len=3, cache=False)
    with pytest.raises(IndexError):
        res["dataset"][2]


def test_negative_indexing_works_like_a_sequence():
    res = _run(_rows("ab", "cde"), FakeTokenizer(), seq_len=3, cache=False)
    assert res["dataset"][-1][0].tolist() == res["dataset"][1][0].tolist()


def test_a_corpus_too_short_for_one_block_says_so():
    with pytest.raises(ValueError, match="seq_len"):
        _run(_rows("ab"), FakeTokenizer(), seq_len=64, cache=False)


def test_an_empty_corpus_says_so():
    with pytest.raises(ValueError, match="seq_len"):
        _run(_rows(), FakeTokenizer(), seq_len=4, cache=False)


# ── the tokenizer contract ──────────────────────────────────────────────


def test_a_non_conforming_tokenizer_names_the_node_to_wire():
    with pytest.raises(ValueError, match="LMTokenizer"):
        _run(_rows("abcdefghij"), object(), seq_len=4, cache=False)


def test_a_tokenizer_missing_only_eos_id_is_still_rejected():
    class NoEos:
        vocab_size = 256

        def encode(self, text):
            return [ord(c) for c in text]

        def decode(self, ids):
            return ""

    with pytest.raises(ValueError, match="eos_id"):
        _run(_rows("abcdefghij"), NoEos(), seq_len=4, cache=False)


def test_eos_id_is_required_even_when_append_eos_is_off():
    # Nothing in this run would READ eos_id, and it is still refused: Task 3's
    # generation nodes read the same contract off the same wire, so a partial
    # object must fail at the node the user can see rather than three nodes
    # later.
    class NoEos:
        vocab_size = 256

        def encode(self, text):
            return [ord(c) for c in text]

        def decode(self, ids):
            return ""

    with pytest.raises(ValueError, match="eos_id"):
        _run(_rows("abcdefghij"), NoEos(), seq_len=4,
             append_eos=False, cache=False)


def test_a_dataset_without_a_length_is_refused():
    class Streaming:
        def __iter__(self):
            yield "ab"

    with pytest.raises(ValueError, match="length"):
        _run(Streaming(), FakeTokenizer(), seq_len=4, cache=False)


# ── disk cache ──────────────────────────────────────────────────────────


def _cache_files(tmp_path):
    return sorted((tmp_path / "cache" / "lm_blocks").glob("*.pt"))


def test_a_cache_hit_skips_retokenization_and_returns_the_same_tensors(tmp_path):
    corpus = _rows("ab", "cde")
    first_tok = FakeTokenizer()
    first = _run(corpus, first_tok, seq_len=3)
    assert first_tok.encode_calls == 2
    assert len(_cache_files(tmp_path)) == 1

    second_tok = FakeTokenizer()
    second = _run(corpus, second_tok, seq_len=3)
    assert second_tok.encode_calls == 0
    assert second["total_tokens"] == first["total_tokens"]
    assert second["num_blocks"] == first["num_blocks"]
    for i in range(first["num_blocks"]):
        assert torch.equal(second["dataset"][i][0], first["dataset"][i][0])
        assert torch.equal(second["dataset"][i][1], first["dataset"][i][1])


def test_cache_off_writes_nothing(tmp_path):
    _run(_rows("ab", "cde"), FakeTokenizer(), seq_len=3, cache=False)
    assert not (tmp_path / "cache" / "lm_blocks").exists()


def test_a_different_seq_len_is_a_different_cache_entry(tmp_path):
    corpus = _rows("abcdefghij")
    _run(corpus, FakeTokenizer(), seq_len=3)
    tokenizer = FakeTokenizer()
    _run(corpus, tokenizer, seq_len=4)
    assert tokenizer.encode_calls == 1
    assert len(_cache_files(tmp_path)) == 2


def test_a_different_corpus_is_a_different_cache_entry(tmp_path):
    _run(_rows("abcdefghij"), FakeTokenizer(), seq_len=3)
    tokenizer = FakeTokenizer()
    _run(_rows("klmnopqrst"), tokenizer, seq_len=3)
    assert tokenizer.encode_calls == 1
    assert len(_cache_files(tmp_path)) == 2


def test_a_different_tokenizer_is_a_different_cache_entry(tmp_path):
    corpus = _rows("abcdefghij")
    _run(corpus, FakeTokenizer(), seq_len=3)

    class OtherTokenizer(FakeTokenizer):
        name = "other"

    tokenizer = OtherTokenizer()
    _run(corpus, tokenizer, seq_len=3)
    # Same corpus, same params, different vocabulary -> a different stream. A
    # key without the tokenizer identity would have served gpt2's blocks to a
    # cl100k model.
    assert tokenizer.encode_calls == 1
    assert len(_cache_files(tmp_path)) == 2


def test_append_eos_and_max_tokens_are_part_of_the_key(tmp_path):
    corpus = _rows("abcdefghij")
    _run(corpus, FakeTokenizer(), seq_len=3)
    _run(corpus, FakeTokenizer(), seq_len=3, append_eos=False)
    _run(corpus, FakeTokenizer(), seq_len=3, max_tokens=6)
    assert len(_cache_files(tmp_path)) == 3


def test_a_corrupt_cache_file_is_rebuilt_rather_than_raised(tmp_path):
    corpus = _rows("ab", "cde")
    _run(corpus, FakeTokenizer(), seq_len=3)
    cached = _cache_files(tmp_path)[0]
    cached.write_bytes(b"not a torch file")

    tokenizer = FakeTokenizer()
    res = _run(corpus, tokenizer, seq_len=3)
    assert tokenizer.encode_calls == 2
    assert res["num_blocks"] == 2


def test_a_relative_cache_dir_nests_under_the_default(tmp_path):
    _run(_rows("ab", "cde"), FakeTokenizer(),
         seq_len=3, cache_dir="my_corpus")
    assert list((tmp_path / "cache" / "lm_blocks" / "my_corpus").glob("*.pt"))


def test_a_cache_dir_outside_the_data_directory_is_refused(tmp_path):
    outside = tmp_path.parent / "elsewhere"
    with pytest.raises(ValueError, match="data directory"):
        _run(_rows("ab", "cde"), FakeTokenizer(),
             seq_len=3, cache_dir=str(outside))


# ── progress and stop ───────────────────────────────────────────────────


def test_progress_reports_rows_done_out_of_total():
    from app.core.loop_control import EVENT_BATCH

    frames: list[dict] = []
    LMTokenizedDatasetNode().execute(
        {"dataset": _rows("abcde", "fghij"), "tokenizer": FakeTokenizer()},
        {"seq_len": 3, "cache": False},
        frames.append,
    )
    assert frames, "no progress frame was emitted"
    assert frames[0]["total_rows"] == 2
    assert frames[0]["rows"] >= 1
    # Tagged as a liveness frame so run_service does not mine "rows" and
    # "tokens" into chart series (see NON_METRIC_EVENTS).
    assert frames[0]["event"] == EVENT_BATCH


def test_a_stopped_run_returns_the_blocks_it_managed_and_does_not_raise(
        tmp_path):
    from app.core.execution_context import INTERRUPTED_KEY

    res = LMTokenizedDatasetNode().execute(
        {"dataset": _rows("ab", "cde", "fgh"), "tokenizer": FakeTokenizer()},
        {"seq_len": 2},
        context=StopAfter(1),
    )
    # One row got through: [97, 98, 0] -> (3 - 1) // 2 = 1 block.
    assert res["total_tokens"] == 3
    assert res["num_blocks"] == 1
    assert res[INTERRUPTED_KEY]["rows"] == 1
    # And nothing is cached: the key describes the WHOLE corpus, so writing a
    # partial stream under it would serve a truncated dataset to every later
    # run of the same graph.
    assert not _cache_files(tmp_path)


# ── wired end to end (the #290 acceptance criterion) ────────────────────


def test_a_real_corpus_trains_a_real_causal_lm(tmp_path):
    """local file -> tokenize -> DataLoader -> TrainingLoop, all real nodes."""
    from app.nodes.data.dataloader_node import DataLoaderNode
    from app.nodes.llm.causal_lm_model_node import CausalLMModelNode
    from app.nodes.llm.lm_cross_entropy_loss_node import LMCrossEntropyLossNode
    from app.nodes.llm.lm_tokenizer_node import LMTokenizerNode
    from app.nodes.llm.text_corpus_dataset_node import TextCorpusDatasetNode
    from app.nodes.training.optimizer_node import OptimizerNode
    from app.nodes.training.training_loop_node import TrainingLoopNode

    corpus = tmp_path / "stories.txt"
    corpus.write_text(
        "\n".join(
            f"Once upon a time there was a small robot named {name} who "
            f"learned to read every book in the quiet library."
            for name in ("Ana", "Bo", "Cy", "Dee", "Eli", "Fay", "Gus", "Hana",
                         "Ivo", "Jun", "Kai", "Lia", "Moe", "Ned", "Oli")
        ),
        encoding="utf-8",
    )

    seq_len = 64
    tokenizer = LMTokenizerNode().execute({}, {"encoding": "gpt2"})
    corpus_out = TextCorpusDatasetNode().execute({}, {
        "source": "local_file",
        "local_path": str(corpus),
        "split_lines": True,
    })
    assert corpus_out["num_rows"] == 15

    packed = LMTokenizedDatasetNode().execute(
        {"dataset": corpus_out["dataset"], "tokenizer": tokenizer["tokenizer"]},
        {"seq_len": seq_len},
    )
    assert packed["num_blocks"] >= 2

    loader = DataLoaderNode().execute(
        {"dataset": packed["dataset"]}, {"batch_size": 2, "shuffle": False})
    model = CausalLMModelNode().execute({}, {
        "vocab_size": tokenizer["vocab_size"],
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 2,
        "d_ff": 64,
        "max_seq_len": seq_len,
    })
    optimizer = OptimizerNode().execute(
        {"model": model["model"]}, {"type": "AdamW", "lr": 0.001})
    loss_fn = LMCrossEntropyLossNode().execute({}, {})

    res = TrainingLoopNode().execute(
        {
            "model": model["model"],
            "dataloader": loader["dataloader"],
            "optimizer": optimizer["optimizer"],
            "loss_fn": loss_fn["loss_fn"],
        },
        {"epochs": 2, "device": "cpu"},
    )

    losses = res["losses"]
    assert losses.shape == (2,)
    assert torch.isfinite(losses).all()
    # A fresh 50257-vocab model starts near ln(50257) ~= 10.8; anything wildly
    # off that means the labels are not the shifted inputs.
    assert 0.0 < float(losses[0]) < 20.0
