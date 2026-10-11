"""#688 remainder: the layer editor's PositionalEncoding layer and causal toggles.

The specs below are what the layer editor saves for
``Embedding -> PositionalEncoding -> MultiHeadAttention -> TransformerDecoder
-> Linear`` with both causal checkboxes ticked. Each model is checked against
the same network written in plain PyTorch with the editor model's weights.
"""

from __future__ import annotations

import json
import math

import pytest
import torch
import torch.nn as nn

from app.nodes.utility.sequential_node import SequentialModelNode

D, HEADS, FF, VOCAB, T = 16, 2, 32, 10, 8


def _spec(*, attn_causal=True, decoder_causal=True, learned=False, max_len=64):
    return {
        "version": 2,
        "nodes": [
            {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
            {"id": "emb", "type": "Embedding", "params": {"num_embeddings": VOCAB, "embedding_dim": D}},
            {"id": "pos", "type": "PositionalEncoding",
             "params": {"d_model": D, "max_len": max_len, "learned": learned}},
            {"id": "attn", "type": "MultiHeadAttention",
             "params": {"embed_dim": D, "num_heads": HEADS, "causal": attn_causal}},
            {"id": "dec", "type": "TransformerDecoder", "params": {
                "d_model": D, "nhead": HEADS, "num_layers": 2, "dim_feedforward": FF,
                "causal": decoder_causal}},
            {"id": "head", "type": "Linear", "params": {"in_features": D, "out_features": VOCAB}},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        "edges": [
            {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "emb"},
            {"id": "e2", "source": "emb", "target": "pos"},
            {"id": "e3", "source": "pos", "target": "attn"},
            {"id": "e4", "source": "attn", "target": "dec"},
            {"id": "e5", "source": "dec", "target": "head"},
            {"id": "e6", "source": "head", "target": "out", "targetHandle": "p_y"},
        ],
    }


def _build(**kw):
    torch.manual_seed(0)
    spec = _spec(**kw)
    return SequentialModelNode().execute({}, {"layers": json.dumps(spec)})["model"].eval()


def _sinusoid(seq: int, d: int) -> torch.Tensor:
    pos = torch.arange(seq, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, d, 2, dtype=torch.float32)
    angle = pos / torch.pow(10000.0, i / d)
    pe = torch.zeros(seq, d)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe


class _PlainTorchLM(nn.Module):
    """The same network, written directly against torch.nn."""

    def __init__(self, causal: bool):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D)
        self.attn = nn.MultiheadAttention(D, HEADS, batch_first=True)
        layer = nn.TransformerDecoderLayer(D, HEADS, dim_feedforward=FF, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=2)
        self.head = nn.Linear(D, VOCAB)
        self.causal = causal

    def forward(self, ids):
        x = self.emb(ids) + _sinusoid(ids.shape[1], D)
        mask = nn.Transformer.generate_square_subsequent_mask(ids.shape[1]) if self.causal else None
        x, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.dec(x, x, tgt_mask=mask, memory_mask=mask)
        return self.head(x)


def _reference_from(model, causal: bool) -> _PlainTorchLM:
    """A plain-PyTorch copy of the editor model's weights."""
    ref = _PlainTorchLM(causal).eval()
    layers = model.layers
    ref.emb.load_state_dict(layers["emb"].state_dict())
    ref.attn.load_state_dict(layers["attn"].attn.state_dict())
    ref.dec.load_state_dict(layers["dec"].decoder.state_dict())
    ref.head.load_state_dict(layers["head"].state_dict())
    return ref


IDS = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])


def test_the_editor_model_matches_plain_pytorch_causal_attention():
    model = _build()
    ref = _reference_from(model, causal=True)
    with torch.no_grad():
        torch.testing.assert_close(model(IDS), ref(IDS), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("position", [1, 4, 7])
def test_changing_a_later_token_leaves_every_earlier_output_unchanged(position):
    model = _build()
    changed = IDS.clone()
    changed[:, position] = 0
    with torch.no_grad():
        before, after = model(IDS), model(changed)
    # Earlier positions: identical. The changed position itself: different,
    # so the check is not passing because the model ignores its input.
    torch.testing.assert_close(before[:, :position], after[:, :position], atol=0.0, rtol=0.0)
    assert (before[:, position] - after[:, position]).abs().max() > 1e-3


def test_attention_without_causal_still_reads_later_tokens():
    """Guard: ``causal`` off on the attention layer keeps full attention."""
    model = _build(attn_causal=False, decoder_causal=False)
    ref = _reference_from(model, causal=False)
    changed = IDS.clone()
    changed[:, 7] = 0
    with torch.no_grad():
        torch.testing.assert_close(model(IDS), ref(IDS), atol=1e-5, rtol=1e-5)
        assert (model(IDS)[:, 0] - model(changed)[:, 0]).abs().max() > 1e-3


def test_a_causal_attention_layer_alone_hides_later_tokens():
    """Only the attention layer is causal; the decoder is off. Position 0
    must still be blind to position 7, so the toggle works on its own."""
    spec = _spec(attn_causal=True)
    spec["nodes"] = [n for n in spec["nodes"] if n["id"] != "dec"]
    spec["edges"] = [e for e in spec["edges"] if e["id"] not in ("e4", "e5")] + [
        {"id": "e45", "source": "attn", "target": "head"},
    ]
    torch.manual_seed(0)
    model = SequentialModelNode().execute({}, {"layers": json.dumps(spec)})["model"].eval()
    changed = IDS.clone()
    changed[:, 7] = 0
    with torch.no_grad():
        torch.testing.assert_close(model(IDS)[:, :7], model(changed)[:, :7], atol=0.0, rtol=0.0)


def test_positional_encoding_makes_the_model_see_token_order():
    """Without position information self-attention is permutation-equivariant:
    the output for a token is the same wherever the sequence puts it. With the
    layer, swapping two tokens changes the outputs at the unswapped positions."""
    model = _build(attn_causal=False, decoder_causal=False)
    swapped = IDS.clone()
    swapped[:, [2, 5]] = swapped[:, [5, 2]]
    with torch.no_grad():
        moved = (model(IDS)[:, 0] - model(swapped)[:, 0]).abs().max().item()
    assert moved > 1e-3


def test_sinusoidal_table_matches_the_vaswani_formula_and_the_canvas_node():
    from app.nodes.llm.positional_encoding_node import PositionalEncodingNode
    from app.nodes.utility.sequential_modules import PositionalEncodingBlock

    block = PositionalEncodingBlock(d_model=D, max_len=12)
    x = torch.zeros(3, 12, D)
    torch.testing.assert_close(block(x)[0], _sinusoid(12, D))
    torch.testing.assert_close(block(x)[0], PositionalEncodingNode._sinusoidal_pe(12, D))
    assert block(x[0]).shape == (12, D)  # unbatched [T, D]
    # A fixed table: nothing to train and nothing in the checkpoint.
    assert list(block.parameters()) == []
    assert block.state_dict() == {}
    assert math.isclose(block(x)[0, 1, 0].item(), math.sin(1.0), rel_tol=1e-6)


def test_learned_positional_encoding_is_trainable():
    from app.nodes.utility.sequential_modules import PositionalEncodingBlock

    torch.manual_seed(0)
    block = PositionalEncodingBlock(d_model=4, max_len=6, learned=True)
    assert [name for name, _ in block.named_parameters()] == ["pe"]
    loss = block(torch.zeros(2, 5, 4)).pow(2).sum()
    loss.backward()
    # Rows 0..4 were used and get a gradient; row 5 was not.
    assert block.pe.grad[:5].abs().sum() > 0
    assert block.pe.grad[5].abs().sum() == 0


def test_a_sequence_longer_than_max_len_names_the_fix():
    model = _build(max_len=4)
    with pytest.raises(ValueError, match="exceeds max_len=4"):
        model(IDS)


def test_the_model_round_trips_through_a_full_model_save(monkeypatch, tmp_path):
    """``full_model`` loading admits classes by name, so the new layer has
    to be on the loader's allowlist for this model to come back."""
    from app.config import settings
    from app.nodes.io.model_loader_node import ModelLoaderNode
    from app.nodes.io.model_saver_node import ModelSaverNode

    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "models")
    settings.MODELS_DIR.mkdir(parents=True)
    model = _build(learned=True)
    ModelSaverNode().execute(
        {"model": model}, {"path": "lm.pt", "save_mode": "full_model", "format": "pytorch"},
    )
    loaded = ModelLoaderNode().execute(
        {}, {"path": "lm.pt", "load_mode": "full_model", "device": "cpu"},
    )["model"].eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(IDS), model(IDS))
