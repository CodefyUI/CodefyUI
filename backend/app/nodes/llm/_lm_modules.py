"""Torch modules behind CausalLMModel / LMCrossEntropyLoss.

Module-scope classes on purpose (#283): ``ModelSaver(save_mode="full_model")``
pickles the whole ``nn.Module``, and pickle stores a class by module plus
qualified name — a class created inside a function has no name anyone can
import it back by. ``torch`` is imported at the top HERE, and the node modules
import this module inside ``execute``, so the node registry's metadata scan
never pays the torch import (the same split ``sequential_modules.py`` uses).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (RoPE) applied to q/k head dims.

    The rotation method is named ``rotate`` — NOT ``apply`` — because
    ``nn.Module.apply(fn)`` is the recursive initializer walk, and the
    model's weight-init pass visits this module through it.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(
                "positional='rope' needs an even head dimension "
                f"(d_model / n_heads), got {head_dim}"
            )
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        # (max_seq_len, head_dim/2) each; kept fp32 and cast at rotate time.
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, D). Rotate pairs (even, odd) by position angle.
        seq_len = x.shape[-2]
        cos = self.cos[:seq_len].to(dtype=x.dtype, device=x.device)
        sin = self.sin[:seq_len].to(dtype=x.dtype, device=x.device)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated = torch.empty_like(x)
        rotated[..., 0::2] = x1 * cos - x2 * sin
        rotated[..., 1::2] = x1 * sin + x2 * cos
        return rotated


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float,
                 rope: RotaryEmbedding | None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = dropout
        self.rope = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(d_model, dim=2)

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = heads(q), heads(k), heads(v)
        if self.rope is not None:
            q = self.rope.rotate(q)
            k = self.rope.rotate(k)
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation: str, dropout: float):
        super().__init__()
        self.fc_in = nn.Linear(d_model, d_ff, bias=True)
        self.fc_out = nn.Linear(d_ff, d_model, bias=True)
        self.act = {"gelu": nn.GELU(), "relu": nn.ReLU(), "silu": nn.SiLU()}[activation]
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc_out(self.act(self.fc_in(x))))


def make_norm(kind: str, d_model: int) -> nn.Module:
    return nn.RMSNorm(d_model) if kind == "rmsnorm" else nn.LayerNorm(d_model)


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(norm(x)); x + mlp(norm(x))."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float,
                 norm: str, activation: str, rope: RotaryEmbedding | None):
        super().__init__()
        self.norm_attn = make_norm(norm, d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, rope)
        self.norm_mlp = make_norm(norm, d_model)
        self.mlp = MLP(d_model, d_ff, activation, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm_attn(x)))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class CausalLMModule(nn.Module):
    """GPT-style decoder-only LM: token ids in, next-token logits out."""

    def __init__(self, *, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, d_ff: int, max_seq_len: int, dropout: float,
                 positional: str, norm: str, activation: str,
                 tie_embeddings: bool, gradient_checkpointing: bool,
                 init_std: float):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.positional = positional
        self.gradient_checkpointing = gradient_checkpointing

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        rope: RotaryEmbedding | None = None
        if positional == "learned":
            self.pos_emb: nn.Module | None = nn.Embedding(max_seq_len, d_model)
        elif positional == "sinusoidal":
            self.pos_emb = None
            position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32)
                * (-math.log(10000.0) / d_model)
            )
            pe = torch.zeros(max_seq_len, d_model)
            pe[:, 0::2] = torch.sin(position * div)
            pe[:, 1::2] = torch.cos(position * div)
            self.register_buffer("sin_pos", pe, persistent=False)
        else:  # rope
            self.pos_emb = None
            rope = RotaryEmbedding(d_model // n_heads, max_seq_len)
            self.rope = rope

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, dropout, norm, activation, rope)
            for _ in range(n_layers)
        ])
        self.norm_final = make_norm(norm, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # GPT-2-style init: normal(0, init_std) everywhere, residual output
        # projections scaled down by sqrt(2 * n_layers) so the residual
        # stream's variance stays stable at depth.
        self.apply(lambda m: _init_module(m, init_std))
        scaled = init_std / math.sqrt(2 * n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=scaled)
            nn.init.normal_(block.mlp.fc_out.weight, mean=0.0, std=scaled)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError(
                "CausalLM expects input_ids of shape (batch, seq_len), "
                f"got {tuple(input_ids.shape)}"
            )
        seq_len = input_ids.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len "
                f"{self.max_seq_len}"
            )
        x = self.tok_emb(input_ids)
        if self.positional == "learned":
            positions = torch.arange(seq_len, device=input_ids.device)
            x = x + self.pos_emb(positions)  # type: ignore[misc]
        elif self.positional == "sinusoidal":
            x = x + self.sin_pos[:seq_len].to(dtype=x.dtype)
        x = self.drop(x)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, use_reentrant=False,
                )
            else:
                x = block(x)
        x = self.norm_final(x)
        return self.lm_head(x)


def _init_module(module: nn.Module, init_std: float) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=init_std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=init_std)


class LMCrossEntropy(nn.Module):
    """Token-level cross-entropy that flattens (B, T, V) internally.

    Deliberately NOT a subclass of ``nn.CrossEntropyLoss``: TrainingLoop
    opens its val-accuracy argmax path for that isinstance, and an argmax
    over dim=1 of (B, T, V) logits is the time axis, not the vocabulary.
    """

    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.shape[-1])
            targets = targets.reshape(-1)
        return F.cross_entropy(
            logits,
            targets,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
