"""Tests for TextGenerateNode."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from app.nodes.llm.text_generate_node import TextGenerateNode

VOCAB = 64


class AlphaTokenizer:
    """A..Z <-> 0..25; eos configurable. Honors the LMTokenizer contract."""

    encoding_name = "alpha-test"

    def __init__(self, eos_id: int = 63):
        self.eos_id = eos_id

    def encode(self, text: str) -> list[int]:
        return [ord(ch) - 65 for ch in text if 65 <= ord(ch) <= 90]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(65 + i) if 0 <= i < 26 else "#" for i in ids)


class NextTokenModel(nn.Module):
    """Deterministic toy LM: always predicts (last_token + 1) % VOCAB."""

    max_seq_len = 16

    def __init__(self):
        super().__init__()
        # An unused parameter so .to(device) has something to move.
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, VOCAB)
        for b in range(batch):
            for t in range(seq_len):
                logits[b, t, (int(input_ids[b, t]) + 1) % VOCAB] = 10.0
        return logits + self.bias


class UniformModel(nn.Module):
    """All-zero logits -> uniform sampling distribution."""

    max_seq_len = 16

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        return torch.zeros(batch, seq_len, VOCAB) + self.bias


def generate(model, tokenizer, params):
    return TextGenerateNode().execute(
        {"model": model, "tokenizer": tokenizer},
        {"device": "cpu", **params},
    )


def test_node_metadata():
    assert TextGenerateNode.NODE_NAME == "TextGenerate"
    assert TextGenerateNode.CATEGORY == "LLM"
    assert TextGenerateNode.cacheable is False


def test_greedy_continuation_is_exact():
    result = generate(NextTokenModel(), AlphaTokenizer(), {
        "prompt": "AB", "max_new_tokens": 3, "temperature": 0.0,
    })
    assert result["text"] == "ABCDE"
    assert result["token_count"] == 3


def test_generation_stops_at_eos():
    result = generate(NextTokenModel(), AlphaTokenizer(eos_id=5), {
        "prompt": "AB", "max_new_tokens": 20, "temperature": 0.0,
    })
    # 1 -> generates 2,3,4,5(eos) and stops.
    assert result["token_count"] == 4
    assert result["text"].startswith("ABCDE")


def test_prompt_longer_than_window_slides():
    tokenizer = AlphaTokenizer()
    model = NextTokenModel()
    prompt = "".join(chr(65 + i) for i in range(20))  # 20 ids > window 16
    result = generate(model, tokenizer, {
        "prompt": prompt, "max_new_tokens": 2, "temperature": 0.0,
    })
    assert result["text"] == prompt + "UV"


def test_prompt_input_port_overrides_param():
    result = TextGenerateNode().execute(
        {"model": NextTokenModel(), "tokenizer": AlphaTokenizer(), "prompt": "XY"},
        {"prompt": "AB", "max_new_tokens": 1, "temperature": 0.0, "device": "cpu"},
    )
    assert result["text"] == "XYZ"


def test_seeded_sampling_is_reproducible():
    first = generate(UniformModel(), AlphaTokenizer(), {
        "prompt": "A", "max_new_tokens": 12, "temperature": 1.0,
        "top_k": 0, "top_p": 1.0, "seed": 7,
    })
    second = generate(UniformModel(), AlphaTokenizer(), {
        "prompt": "A", "max_new_tokens": 12, "temperature": 1.0,
        "top_k": 0, "top_p": 1.0, "seed": 7,
    })
    third = generate(UniformModel(), AlphaTokenizer(), {
        "prompt": "A", "max_new_tokens": 12, "temperature": 1.0,
        "top_k": 0, "top_p": 1.0, "seed": 8,
    })
    assert first["text"] == second["text"]
    assert first["text"] != third["text"]


def test_top_k_one_is_greedy_even_at_high_temperature():
    result = generate(NextTokenModel(), AlphaTokenizer(), {
        "prompt": "AB", "max_new_tokens": 3, "temperature": 1.5, "top_k": 1,
    })
    assert result["text"] == "ABCDE"


def test_tokenizer_contract_is_refused():
    class NoDecode:
        eos_id = 0

        def encode(self, text):
            return [1]

    with pytest.raises(ValueError, match="LMTokenizer"):
        TextGenerateNode().execute(
            {"model": NextTokenModel(), "tokenizer": NoDecode()}, {},
        )


def test_trained_causal_lm_reproduces_a_memorized_cycle():
    from app.nodes.llm.causal_lm_model_node import CausalLMModelNode
    from app.nodes.llm.lm_cross_entropy_loss_node import LMCrossEntropyLossNode

    model = CausalLMModelNode().execute({}, {
        "vocab_size": 256, "d_model": 32, "n_layers": 2, "n_heads": 2,
        "d_ff": 64, "max_seq_len": 32, "seed": 5,
    })["model"]
    loss_fn = LMCrossEntropyLossNode().execute({}, {})["loss_fn"]

    cycle = torch.arange(16, dtype=torch.int64)
    sequence = cycle.repeat(2)  # 0..15 0..15
    inputs = sequence[:-1].unsqueeze(0)
    targets = sequence[1:].unsqueeze(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    for _ in range(150):
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < 0.2  # memorized

    class IdTokenizer:
        encoding_name = "id-test"
        eos_id = 255

        def encode(self, text):
            return [int(piece) for piece in text.split() if piece.strip()]

        def decode(self, ids):
            return " ".join(str(i) for i in ids)

    result = TextGenerateNode().execute(
        {"model": model, "tokenizer": IdTokenizer()},
        {"prompt": "0 1 2 3 4 5", "max_new_tokens": 6,
         "temperature": 0.0, "device": "cpu"},
    )
    assert result["text"] == " ".join(str(i) for i in range(12))
