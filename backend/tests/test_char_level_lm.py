"""Character- and byte-level tokenizers for the LM stack (#692).

The end-to-end test at the bottom is the issue's acceptance criterion: a
char-level GPT built only from built-in nodes trains on a tiny corpus, its
loss falls, and TextGenerate decodes the sampled ids back to corpus text.
"""

from __future__ import annotations

import pytest
import torch

from app.config import settings
from app.core.graph_engine import validate_graph
from app.nodes.llm.lm_tokenizer_node import LMTokenizerNode
from app.nodes.llm.text_corpus_dataset_node import TextRowDataset

CORPUS = "sue finds a red ball. tom finds a blue cup."


@pytest.fixture(autouse=True)
def data_root_in_tmp(tmp_path, monkeypatch):
    """Keep LMTokenizedDataset's disk cache out of the real backend/data."""
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "models")
    return tmp_path


def _tokenizer(encoding: str, corpus=None) -> dict:
    inputs = {} if corpus is None else {"corpus": corpus}
    return LMTokenizerNode().execute(inputs, {"encoding": encoding})


# ── byte ────────────────────────────────────────────────────────────────


def test_byte_encoding_is_utf8_bytes_with_an_eos_after_them():
    result = _tokenizer("byte")
    tok = result["tokenizer"]
    assert result["vocab_size"] == tok.vocab_size == 257
    assert tok.eos_id == 256
    assert tok.encode("hé") == list("hé".encode("utf-8")) == [104, 195, 169]
    assert tok.decode(tok.encode("hé, 世界")) == "hé, 世界"


def test_byte_decode_skips_eos_and_survives_a_split_character():
    tok = _tokenizer("byte")["tokenizer"]
    assert tok.decode([104, 256, 105]) == "hi"
    # Half of a two-byte character, as a sampler may produce.
    assert tok.decode([104, 195]) == "h�"


# ── char ────────────────────────────────────────────────────────────────


def test_char_vocabulary_is_the_corpus_alphabet_plus_eos_and_unknown():
    corpus = TextRowDataset(["abca", "cab!"])
    result = _tokenizer("char", corpus)
    tok = result["tokenizer"]
    assert tok.chars == "!abc"
    assert result["vocab_size"] == tok.vocab_size == 6
    assert tok.eos_id == 4 and tok.unk_id == 5
    assert tok.encode("cab") == [3, 1, 2]
    assert tok.decode(tok.encode("ab!c")) == "ab!c"


def test_char_unknown_characters_and_eos_decode_predictably():
    tok = _tokenizer("char", TextRowDataset(["ab"]))["tokenizer"]
    assert tok.encode("az") == [0, tok.unk_id]
    assert tok.decode([0, tok.eos_id, 1, tok.unk_id]) == "ab�"


def test_char_needs_the_corpus_input():
    with pytest.raises(ValueError, match="corpus"):
        _tokenizer("char")


def test_char_name_differs_between_alphabets():
    """LMTokenizedDataset keys its disk cache on ``name``; two alphabets
    assign different ids to the same character, so they must not share it."""
    one = _tokenizer("char", TextRowDataset(["abc"]))["tokenizer"]
    two = _tokenizer("char", TextRowDataset(["abd"]))["tokenizer"]
    same = _tokenizer("char", TextRowDataset(["cba"]))["tokenizer"]
    assert one.name != two.name
    assert one.name == same.name


# ── vocab_size range ────────────────────────────────────────────────────


def test_a_small_vocab_size_passes_graph_validation():
    """The issue's repro: CausalLMModel(vocab_size=29) was refused with
    'value 29 is below minimum 256' before the run started."""
    nodes = [
        {"id": "start", "type": "Start", "data": {"params": {}}},
        {"id": "lm", "type": "CausalLMModel", "data": {"params": {
            "vocab_size": 29, "d_model": 64, "n_layers": 1, "n_heads": 2,
            "d_ff": 128, "max_seq_len": 32}}},
    ]
    edges = [{"id": "t", "source": "start", "target": "lm",
              "sourceHandle": "trigger", "type": "trigger"}]
    errors = validate_graph(nodes, edges)
    assert not [e for e in errors if "vocab_size" in str(e)], errors


# ── end to end ──────────────────────────────────────────────────────────


def test_a_char_level_gpt_trains_and_generates_corpus_text():
    from app.nodes.data.dataloader_node import DataLoaderNode
    from app.nodes.llm.causal_lm_model_node import CausalLMModelNode
    from app.nodes.llm.lm_cross_entropy_loss_node import LMCrossEntropyLossNode
    from app.nodes.llm.lm_tokenized_dataset_node import LMTokenizedDatasetNode
    from app.nodes.llm.text_generate_node import TextGenerateNode
    from app.nodes.training.optimizer_node import OptimizerNode
    from app.nodes.training.training_loop_node import TrainingLoopNode

    corpus = TextRowDataset([CORPUS] * 16)
    built_tokenizer = _tokenizer("char", corpus)
    tokenizer = built_tokenizer["tokenizer"]
    vocab_size = built_tokenizer["vocab_size"]
    assert vocab_size == len(set(CORPUS)) + 2 < 256

    packed = LMTokenizedDatasetNode().execute(
        {"dataset": corpus, "tokenizer": tokenizer},
        {"seq_len": 16, "cache": False},
    )
    loader = DataLoaderNode().execute(
        {"dataset": packed["dataset"]}, {"batch_size": 8, "shuffle": True})
    torch.manual_seed(0)
    model = CausalLMModelNode().execute({}, {
        "vocab_size": vocab_size,
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 2,
        "d_ff": 128,
        "max_seq_len": 16,
        "seed": 0,
    })["model"]
    optimizer = OptimizerNode().execute(
        {"model": model}, {"type": "AdamW", "lr": 0.005})
    loss_fn = LMCrossEntropyLossNode().execute({}, {})

    trained = TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": loader["dataloader"],
            "optimizer": optimizer["optimizer"],
            "loss_fn": loss_fn["loss_fn"],
        },
        {"epochs": 60, "device": "cpu"},
    )
    losses = trained["losses"]
    assert torch.isfinite(losses).all()
    assert losses[-1] < 0.5 * losses[0]

    # Prompt + new tokens fill exactly one 16-token block that starts at a
    # document start, a context the model saw in training. (Non-overlapping
    # blocks over a 44-token period only ever start at every 4th offset, so a
    # window slid past 16 tokens would read a context it never trained on.)
    generated = TextGenerateNode().execute(
        {"model": trained["model"], "tokenizer": tokenizer},
        {"prompt": "sue", "max_new_tokens": 13, "temperature": 0.0,
         "device": "cpu"},
    )
    assert generated["token_count"] == 13
    assert generated["text"] == "sue finds a red "
