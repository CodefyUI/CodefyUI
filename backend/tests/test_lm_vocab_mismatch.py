"""A CausalLMModel smaller than its tokenizer is refused, naming both sizes (#681).

On CPU the mismatch used to surface as a bare ``index out of range in self``
from inside TrainingLoop; on MPS it trained without an error, because an id
past the embedding table reads zeros there, and PerplexityEvaluate reported a
perplexity below what the data allows. No tiktoken here: the tokenizer is a
fake whose ids are ``ord(c)``, so its vocab_size of 256 is larger than the
64-id model on purpose.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from app.config import settings
from app.nodes.llm.causal_lm_model_node import CausalLMModelNode
from app.nodes.llm.lm_cross_entropy_loss_node import LMCrossEntropyLossNode
from app.nodes.llm.lm_tokenized_dataset_node import LMTokenizedDatasetNode
from app.nodes.llm.perplexity_evaluate_node import PerplexityEvaluateNode
from app.nodes.llm.text_corpus_dataset_node import TextRowDataset
from app.nodes.llm.text_generate_node import TextGenerateNode
from app.nodes.training.optimizer_node import OptimizerNode
from app.nodes.training.training_loop_node import TrainingLoopNode

mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

TOKENIZER_VOCAB = 256
SEQ_LEN = 8


class FakeTokenizer:
    name = "fake"
    vocab_size = TOKENIZER_VOCAB
    eos_id = 0

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids) -> str:
        return "".join(chr(int(i)) for i in ids)


@pytest.fixture(autouse=True)
def block_cache_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "models")


def _blocks():
    rows = TextRowDataset(["the quick brown fox jumps over the lazy dog"] * 4)
    return LMTokenizedDatasetNode().execute(
        {"dataset": rows, "tokenizer": FakeTokenizer()}, {"seq_len": SEQ_LEN},
    )["dataset"]


def _model(vocab_size: int):
    return CausalLMModelNode().execute({}, {
        "vocab_size": vocab_size, "d_model": 16, "n_layers": 1, "n_heads": 2,
        "d_ff": 32, "max_seq_len": SEQ_LEN, "seed": 0,
    })["model"]


def _train(model, loader, device="cpu"):
    optimizer = OptimizerNode().execute({"model": model}, {"type": "AdamW", "lr": 1e-3})["optimizer"]
    return TrainingLoopNode().execute(
        {"model": model, "dataloader": loader, "optimizer": optimizer,
         "loss_fn": LMCrossEntropyLossNode().execute({}, {})["loss_fn"]},
        {"epochs": 1, "device": device},
    )


_NAMES_BOTH = r"ids up to 255 but CausalLMModel\.vocab_size is 64"


def test_training_refuses_a_model_smaller_than_the_tokenizer():
    model = _model(64)
    before = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(ValueError, match=_NAMES_BOTH):
        _train(model, DataLoader(_blocks(), batch_size=2))
    # Refused before the first step, so nothing was trained.
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_a_split_of_the_blocks_is_checked_too():
    blocks = _blocks()
    with pytest.raises(ValueError, match=_NAMES_BOTH):
        _train(_model(64), DataLoader(Subset(blocks, range(2)), batch_size=2))


def test_a_model_as_large_as_the_tokenizer_trains_as_before():
    result = _train(_model(TOKENIZER_VOCAB), DataLoader(_blocks(), batch_size=2))
    assert torch.isfinite(result["losses"]).all()


def test_perplexity_evaluate_refuses_the_mismatched_pair():
    with pytest.raises(ValueError, match=_NAMES_BOTH):
        PerplexityEvaluateNode().execute(
            {"model": _model(64), "dataset": _blocks()}, {"device": "cpu"})


def test_text_generate_refuses_the_mismatched_pair():
    with pytest.raises(ValueError, match=_NAMES_BOTH):
        TextGenerateNode().execute(
            {"model": _model(64), "tokenizer": FakeTokenizer()},
            {"prompt": "a", "max_new_tokens": 2, "device": "cpu"})


def test_the_model_itself_names_an_id_past_its_table():
    """The backstop for a dataset that does not say which tokenizer packed it."""
    ids = torch.tensor([[1, 2, 100, 3]])
    with pytest.raises(ValueError, match=r"token id 100.*vocab_size is 64"):
        _model(64)(ids)


@pytest.mark.skipif(not mps_available, reason="needs an MPS device")
def test_training_on_mps_refuses_the_mismatch_instead_of_training():
    with pytest.raises(ValueError, match=_NAMES_BOTH):
        _train(_model(64), DataLoader(_blocks(), batch_size=2), device="mps")


@pytest.mark.skipif(not mps_available, reason="needs an MPS device")
def test_the_model_on_mps_names_an_id_past_its_table():
    model = _model(64).to("mps")
    with pytest.raises(ValueError, match=r"token id 100.*vocab_size is 64"):
        model(torch.tensor([[1, 2, 100, 3]], device="mps"))
