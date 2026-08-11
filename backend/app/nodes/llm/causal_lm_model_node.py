"""CausalLMModel node (#289) -- a trainable GPT-style decoder-only LM.

The node the LLM epic (#292) is built around: it hands the training nodes an
ordinary ``torch.nn.Module`` whose forward is ``input_ids (B, T) -> logits
(B, T, vocab_size)``, so ``Optimizer`` / ``TrainingLoop`` / ``ModelSaver``
need to know nothing about language models. Everything is plain torch --
``transformers`` is deliberately NOT a dependency of this project, and the
whole point of the node is that a learner can read the ~200 lines below and
see every piece of a modern decoder.

Architecture (pre-LN, the arrangement every GPT after GPT-1 uses)::

    input_ids (B, T)  int64
        -> token embedding                       (B, T, d_model)
        -> + position information (learned / sinusoidal / none, for RoPE)
        -> dropout
        -> n_layers x [ x + Attn(Norm(x)) ; x + MLP(Norm(x)) ]
        -> final Norm
        -> LM head (Linear d_model -> vocab, no bias)
                                                 (B, T, vocab_size)

Pre-LN rather than the post-LN of the original paper because post-LN needs a
warmup schedule to train at all at this depth; a learner who presses Run
with a flat learning rate should see a loss that goes down.

**Why the causal mask is not a parameter.** ``F.scaled_dot_product_attention
(..., is_causal=True)`` builds the mask inside the kernel, which is both
faster and the only version that reaches the fused/flash backends. A node
that materialised a ``(T, T)`` boolean mask would be teaching the shape of
the mask at the cost of the memory the mask costs; ``AttentionMask`` already
exists for teaching the shape.

**Why this node is stateful.** It OWNS trainable parameters and
``TrainingLoop`` mutates them in place, which is #253's lesson exactly:
carrying :class:`StatefulModuleMixin` is what makes "Persist weights between
runs" and "Reset all weights now" mean anything here, and what stops run 2
from being handed run 1's already-trained module out of ``ExecutionCache``.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from ...core.stateful_module import StatefulModuleMixin

logger = logging.getLogger(__name__)

#: How position gets into the residual stream.
POSITIONAL_MODES = ["learned", "sinusoidal", "rope"]
#: Which normaliser sits in front of each sub-layer.
NORM_TYPES = ["layernorm", "rmsnorm"]
#: The MLP's non-linearity.
ACTIVATIONS = ["gelu", "relu", "silu"]

#: The RoPE / sinusoidal frequency base. 10000 is the constant from
#: Vaswani et al. (2017) that RoFormer and every Llama since reused, so the
#: tables here match what a learner will find in any reference implementation.
_FREQUENCY_BASE = 10000.0

#: The two projections that write into the residual stream, and therefore the
#: two that get GPT-2's scaled init (see :func:`_init_module_weights`).
_RESIDUAL_PROJECTIONS = re.compile(r"(?:attn\.out_proj|mlp\.fc2)\.weight$")


def _make_norm(norm: str, d_model: int) -> nn.Module:
    """The pre-LN normaliser. ``rmsnorm`` has a gain but no bias and no mean
    subtraction, which is why Llama-class models pick it: one fewer reduction
    per sub-layer for no measured loss in quality."""
    if norm == "layernorm":
        return nn.LayerNorm(d_model)
    if norm == "rmsnorm":
        # torch's own RMSNorm rather than a hand-rolled one: it keeps the
        # reduction in float32 under autocast, which a naive
        # ``x * rsqrt(x.pow(2).mean(-1))`` does not.
        return nn.RMSNorm(d_model)
    raise ValueError(
        f"CausalLMModel: unknown norm {norm!r}; expected one of {NORM_TYPES}.")


def _make_activation(activation: str) -> nn.Module:
    if activation == "gelu":
        return nn.GELU()
    if activation == "relu":
        return nn.ReLU()
    if activation == "silu":
        return nn.SiLU()
    raise ValueError(
        f"CausalLMModel: unknown activation {activation!r}; expected one of "
        f"{ACTIVATIONS}.")


def _sinusoidal_table(max_seq_len: int, d_model: int) -> torch.Tensor:
    """The fixed ``(max_seq_len, d_model)`` table from Vaswani et al. (2017).

    Built in float32 and cast at the add, never at construction: hard-coding
    a dtype here is precisely what would break a bf16 ``torch.autocast`` run.
    """
    position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(_FREQUENCY_BASE) / d_model)
    )
    table = torch.zeros(max_seq_len, d_model, dtype=torch.float32)
    angles = position * frequency
    table[:, 0::2] = torch.sin(angles)[:, : (d_model + 1) // 2]
    # An odd d_model leaves the cos half one column shorter than the sin
    # half, so it is sliced rather than assumed square.
    table[:, 1::2] = torch.cos(angles)[:, : d_model // 2]
    return table


def _rope_tables(max_seq_len: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos, sin)`` of shape ``(max_seq_len, head_dim // 2)``.

    ``max_seq_len`` bounds the table, which is what makes it a cache rather
    than a per-forward computation -- the forward slices ``[:T]`` out of it.
    """
    inverse_frequency = 1.0 / (
        _FREQUENCY_BASE
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    angles = torch.outer(
        torch.arange(max_seq_len, dtype=torch.float32), inverse_frequency)
    return angles.cos(), angles.sin()


def _apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> torch.Tensor:
    """Rotate the per-head vectors in *x* ``(B, H, T, head_dim)`` by position.

    The half-split (GPT-NeoX / Llama) pairing rather than the interleaved one
    from the RoFormer paper: the two are equivalent up to a permutation of
    the head dimension, which the projections learn, and the half-split is
    what every checkpoint in the wild uses.

    ``cos``/``sin`` are cast to ``x``'s dtype at the multiply, so the same
    float32 table serves an fp32 run and a bf16 autocast run.
    """
    left, right = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return torch.cat(
        [left * cos - right * sin, left * sin + right * cos], dim=-1)


class _CausalSelfAttention(nn.Module):
    """Multi-head self-attention that can only look left."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # One fused projection for q, k and v: three separate Linears would
        # be numerically identical and three times as many kernel launches.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        # A float, not an nn.Dropout: SDPA takes the probability itself so
        # the dropout happens inside the fused kernel on the attention
        # weights, which are never materialised for us to drop out of.
        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        query, key, value = self.qkv(x).split(d_model, dim=2)
        heads = (batch, seq_len, self.n_heads, self.head_dim)
        query = query.view(heads).transpose(1, 2)
        key = key.view(heads).transpose(1, 2)
        value = value.view(heads).transpose(1, 2)

        # RoPE rotates q and k and leaves v alone: position enters through
        # the dot product, not through the values being averaged.
        if cos is not None and sin is not None:
            query = _apply_rope(query, cos, sin)
            key = _apply_rope(key, cos, sin)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).reshape(batch, seq_len, d_model)
        return self.resid_dropout(self.out_proj(attended))


class _FeedForward(nn.Module):
    """The position-wise MLP: widen to ``d_ff``, activate, project back."""

    def __init__(
        self, d_model: int, d_ff: int, activation: str, dropout: float,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = _make_activation(activation)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class _DecoderBlock(nn.Module):
    """One pre-LN transformer block: attention then MLP, both residual."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        norm: str,
        activation: str,
    ) -> None:
        super().__init__()
        self.norm1 = _make_norm(norm, d_model)
        self.attn = _CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = _make_norm(norm, d_model)
        self.mlp = _FeedForward(d_model, d_ff, activation, dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


def _init_module_weights(
    model: nn.Module, init_std: float, n_layers: int, seed: int,
) -> None:
    """GPT-2's initialisation, reproducibly.

    Everything with a matrix shape gets ``normal(0, init_std)``; biases go to
    zero; the norm gains keep torch's own 1.0. The two projections that write
    into the residual stream (attention's output projection and the MLP's
    down-projection) get ``init_std / sqrt(2 * n_layers)`` instead -- the
    residual stream accumulates ``2 * n_layers`` of those writes, and without
    the scaling the activations at the last block start out proportional to
    the depth, which is how a deep stack diverges in its first few steps.

    Seeded through a LOCAL ``torch.Generator`` rather than
    ``torch.manual_seed``: the global RNG belongs to the run (the DataLoader's
    shuffle, dropout masks, ``deterministic`` mode), and a node that reseeded
    it would silently change every other node's randomness. The parameter
    order is registration order, which is fixed, so the same seed gives the
    same weights.
    """
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    residual_std = init_std / math.sqrt(2 * max(1, n_layers))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                param.zero_()
            elif _RESIDUAL_PROJECTIONS.search(name):
                param.normal_(0.0, residual_std, generator=generator)
            elif param.dim() >= 2:
                param.normal_(0.0, init_std, generator=generator)
            # 1-D non-bias parameters are the norm gains. Left at torch's
            # 1.0: a normal(0, 0.02) gain would scale every activation to
            # near zero before the first step.


class CausalLMModule(nn.Module):
    """A decoder-only transformer language model.

    Module scope, not a closure inside the node, so ``torch.save`` can name
    the class -- #283 was exactly this defect one package over.

    Public attributes downstream nodes rely on (#289):

    ``max_seq_len``
        The longest sequence the position information covers.
        ``TextGenerate`` reads it to size its sliding context window.
    ``vocab_size``
        The width of the logits, i.e. how many tokens can be sampled.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float,
        positional: str,
        norm: str,
        activation: str,
        tie_embeddings: bool,
        gradient_checkpointing: bool,
        init_std: float,
        seed: int,
    ) -> None:
        super().__init__()
        if positional not in POSITIONAL_MODES:
            raise ValueError(
                f"CausalLMModel: unknown positional {positional!r}; expected "
                f"one of {POSITIONAL_MODES}.")
        if d_model % n_heads:
            raise ValueError(
                f"CausalLMModel: d_model={d_model} must be divisible by "
                f"n_heads={n_heads} so every head gets the same width. "
                f"Change d_model or n_heads.")
        head_dim = d_model // n_heads
        if positional == "rope" and head_dim % 2:
            raise ValueError(
                f"CausalLMModel: positional='rope' rotates each head's "
                f"dimensions in pairs, so d_model // n_heads must be even; "
                f"d_model={d_model} and n_heads={n_heads} give {head_dim}. "
                f"Change d_model or n_heads, or pick another positional.")

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.positional = positional
        self.gradient_checkpointing = gradient_checkpointing

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # Present only for ``learned``: the other two modes have no
        # parameters to learn, which is the whole trade they make.
        self.pos_emb = (
            nn.Embedding(max_seq_len, d_model)
            if positional == "learned" else None
        )
        if positional == "sinusoidal":
            self.register_buffer(
                "pos_table", _sinusoidal_table(max_seq_len, d_model),
                persistent=False)
        if positional == "rope":
            cos, sin = _rope_tables(max_seq_len, head_dim)
            # ``persistent=False`` here and on ``pos_table`` above: a position
            # table is a pure function of the config, so writing it into every
            # checkpoint would only make the file bigger and give
            # ``load_state_dict`` another way to complain. Non-persistent
            # buffers still follow ``.to(device)``.
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            _DecoderBlock(d_model, n_heads, d_ff, dropout, norm, activation)
            for _ in range(n_layers)
        ])
        self.norm_f = _make_norm(norm, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            # ONE matrix doing both jobs: "which vector means token i" and
            # "how much does token i score". Saves vocab_size * d_model
            # parameters, which at a 50k vocab is most of a small model.
            # ``parameters()`` de-duplicates shared tensors, so the tied
            # weight is counted once by ``param_count`` -- as it should be,
            # since the optimizer also sees it once.
            self.lm_head.weight = self.tok_emb.weight

        _init_module_weights(self, init_std, n_layers, seed)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError(
                f"CausalLMModel expects a tensor of token ids, got "
                f"{type(input_ids).__name__}.")
        if input_ids.dim() != 2:
            raise ValueError(
                f"CausalLMModel expects input_ids shaped (batch, seq_len), "
                f"got {tuple(input_ids.shape)}. A single sequence needs a "
                f"batch dimension: use ids.unsqueeze(0).")
        if input_ids.is_floating_point():
            raise ValueError(
                f"CausalLMModel expects integer token ids (int64), got "
                f"dtype {input_ids.dtype}. A float here usually means the "
                f"dataset handed over normalised features rather than ids.")
        seq_len = input_ids.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"CausalLMModel: got a sequence of {seq_len} tokens but "
                f"max_seq_len is {self.max_seq_len}. Shorten the sequence "
                f"(the dataset's seq_len) or raise max_seq_len.")

        # ``.long()`` rather than a rejection for int32/int16: nn.Embedding
        # only takes long, and widening an integer id is never lossy.
        ids = input_ids if input_ids.dtype == torch.long else input_ids.long()

        x = self.tok_emb(ids)
        if self.pos_emb is not None:
            positions = torch.arange(seq_len, device=ids.device)
            x = x + self.pos_emb(positions).unsqueeze(0)
        elif self.positional == "sinusoidal":
            x = x + self.pos_table[:seq_len].unsqueeze(0)
        x = self.drop(x)

        cos = sin = None
        if self.positional == "rope":
            cos, sin = self.rope_cos[:seq_len], self.rope_sin[:seq_len]

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                # Recompute the block's activations during backward instead
                # of keeping them: the memory a long context costs is
                # dominated by these, and the price is one extra forward.
                # ``use_reentrant=False`` is the supported implementation --
                # the reentrant one breaks under no-grad callers and is on
                # its way out of torch.
                x = checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)

        return self.lm_head(self.norm_f(x))


def _positive_int(params: dict[str, Any], name: str, default: int) -> int:
    """A size param, floored at 1 -- never clamped to the declared range.

    The ``min_value`` on each :class:`ParamDefinition` below is the editor's
    guard rail for a model somebody intends to train (a 32-dimensional LM
    learns nothing). Enforcing it HERE would silently rewrite the deliberately
    tiny configurations that teaching material and this file's tests are made
    of -- ``vocab_size=100, d_model=32`` would quietly become 256 and 64, and
    an analytic parameter count would stop matching the model. So execute
    enforces only what the module physically needs to be constructible.

    The coercion itself is for hand-built or Copilot-generated graph JSON;
    the INT widget cannot produce a string or a None. A value ``int()``
    cannot read is reported rather than guessed at -- this is not a port
    count, so there is no frontend ``parseInt`` whose numeric-prefix reading
    it has to agree with (see ``resolve_count_param`` for the case where
    there is).
    """
    raw = params.get(name, default)
    if raw is None or raw == "":
        raw = default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"CausalLMModel: {name} must be a whole number, got {raw!r}."
        ) from exc
    return max(1, value)


def _float_param(params: dict[str, Any], name: str, default: float) -> float:
    """A float param, defaulted on a missing/null/empty value.

    Deliberately NOT the ``float(params.get(name, d) or d)`` idiom the rest of
    this repo uses. That form reads FALSINESS as "not set", which is a no-op
    wherever the default is itself 0 (``dropout``, ``seed``) and silently
    wrong for ``init_std``, whose default is 0.02: an explicitly requested
    ``init_std=0`` -- a legal value, the declared minimum, and a lesson about
    what happens when every weight starts identical -- came back as 0.02.
    """
    raw = params.get(name, default)
    if raw is None or raw == "":
        raw = default
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"CausalLMModel: {name} must be a number, got {raw!r}."
        ) from exc


def _select(params: dict[str, Any], name: str, default: str,
            options: list[str]) -> str:
    value = str(params.get(name, default) or default)
    if value not in options:
        raise ValueError(
            f"CausalLMModel: unknown {name} {value!r}; expected one of "
            f"{options}.")
    return value


def _resolve_config(params: dict[str, Any]) -> dict[str, Any]:
    """Params -> constructor kwargs, validated. The single place that decides
    what a graph's JSON means, so ``execute`` can fail early with the same
    message the constructor would give."""
    dropout = _float_param(params, "dropout", 0.0)
    init_std = _float_param(params, "init_std", 0.02)
    config: dict[str, Any] = {
        "vocab_size": _positive_int(params, "vocab_size", 50257),
        "d_model": _positive_int(params, "d_model", 1024),
        "n_layers": _positive_int(params, "n_layers", 12),
        "n_heads": _positive_int(params, "n_heads", 16),
        "d_ff": _positive_int(params, "d_ff", 4096),
        "max_seq_len": _positive_int(params, "max_seq_len", 1024),
        # Clamped, unlike the sizes: p >= 1 zeroes every activation and torch
        # rejects p > 1 outright, so an out-of-range dropout has no reading
        # worth preserving.
        "dropout": min(0.9, max(0.0, dropout)),
        "positional": _select(params, "positional", "learned", POSITIONAL_MODES),
        "norm": _select(params, "norm", "layernorm", NORM_TYPES),
        "activation": _select(params, "activation", "gelu", ACTIVATIONS),
        "tie_embeddings": bool(params.get("tie_embeddings", True)),
        "gradient_checkpointing": bool(
            params.get("gradient_checkpointing", False)),
        # A negative standard deviation is not a distribution; 0 is (every
        # weight starts identical, which is a lesson of its own).
        "init_std": max(0.0, init_std),
        "seed": int(params.get("seed", 0) or 0),
    }
    if config["d_model"] % config["n_heads"]:
        raise ValueError(
            f"CausalLMModel: d_model={config['d_model']} must be divisible "
            f"by n_heads={config['n_heads']} so every head gets the same "
            f"width. Change d_model or n_heads.")
    return config


class CausalLMModelNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "CausalLMModel"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "A GPT-style decoder-only transformer you can actually train. Outputs "
        "a MODEL that maps token ids (batch, seq_len) to next-token logits "
        "(batch, seq_len, vocab_size) -- wire it to Optimizer and "
        "TrainingLoop like any other model, with LMCrossEntropyLoss as the "
        "loss. The defaults describe a ~350M-parameter model; shrink d_model "
        "and n_layers to something a laptop can train in a lesson."
    )

    # A MODEL output is a live handle, so the cache cannot describe it: the
    # key says how the module was BUILT and stops being true the moment
    # TrainingLoop takes a step in it. A hit would hand run 2 the module run
    # 1 already trained (#253, #254) -- StatefulModuleMixin sets this too,
    # and it is repeated here because the flag is the invariant
    # ``test_cache_live_handle_nodes.py`` reads off the class.
    cacheable = False

    # Every param below feeds the constructor, so every param belongs in the
    # structure hash. Editing any of them therefore discards the persisted
    # weights -- which is the honest outcome: the alternative is an edit that
    # appears on the form and does nothing to the model that runs. (Even
    # ``gradient_checkpointing``, which only changes the forward, is here
    # rather than patched onto a reused module: one rule is easier to trust
    # than two.)
    structural_params = (
        "vocab_size",
        "d_model",
        "n_layers",
        "n_heads",
        "d_ff",
        "max_seq_len",
        "dropout",
        "positional",
        "norm",
        "activation",
        "tie_embeddings",
        "gradient_checkpointing",
        "init_std",
        "seed",
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="model",
                data_type=DataType.MODEL,
                description=(
                    "The language model as an nn.Module: input_ids int64 "
                    "(batch, seq_len) in, logits (batch, seq_len, "
                    "vocab_size) out. Carries max_seq_len and vocab_size as "
                    "attributes for the generation nodes to read."
                ),
            ),
            PortDefinition(
                name="param_count",
                data_type=DataType.SCALAR,
                description=(
                    "Number of trainable parameters, counting a tied "
                    "embedding/head matrix once. The number people mean by "
                    "'a 350M model'."
                ),
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="vocab_size",
                param_type=ParamType.INT,
                default=50257,
                min_value=256,
                max_value=300000,
                description=(
                    "How many distinct tokens the model knows. Must match "
                    "the tokenizer feeding it -- 50257 is GPT-2's."
                ),
            ),
            ParamDefinition(
                name="d_model",
                param_type=ParamType.INT,
                default=1024,
                min_value=64,
                max_value=8192,
                description=(
                    "Width of the residual stream: the size of the vector "
                    "carrying each token through the network. Must divide "
                    "evenly by n_heads."
                ),
            ),
            ParamDefinition(
                name="n_layers",
                param_type=ParamType.INT,
                default=12,
                min_value=1,
                max_value=64,
                description=(
                    "How many transformer blocks are stacked. Depth is where "
                    "multi-step reasoning comes from, and it costs time "
                    "linearly."
                ),
            ),
            ParamDefinition(
                name="n_heads",
                param_type=ParamType.INT,
                default=16,
                min_value=1,
                max_value=64,
                description=(
                    "How many attention heads share each block's width. More "
                    "heads means more relationships tracked at once, each in "
                    "a narrower subspace (d_model / n_heads wide)."
                ),
            ),
            ParamDefinition(
                name="d_ff",
                param_type=ParamType.INT,
                default=4096,
                min_value=64,
                max_value=32768,
                description=(
                    "Hidden width of each block's MLP, conventionally 4x "
                    "d_model. Two thirds of the parameters live here."
                ),
            ),
            ParamDefinition(
                name="max_seq_len",
                param_type=ParamType.INT,
                default=1024,
                min_value=16,
                max_value=8192,
                description=(
                    "Longest sequence the model has positions for, in "
                    "tokens. A longer batch is rejected rather than "
                    "truncated; generation slides a window of this size."
                ),
            ),
            ParamDefinition(
                name="tie_embeddings",
                param_type=ParamType.BOOL,
                default=True,
                description=(
                    "Share one matrix between the input embedding and the "
                    "output head. Standard for small models: it saves "
                    "vocab_size x d_model parameters and usually helps."
                ),
            ),
            ParamDefinition(
                name="positional",
                param_type=ParamType.SELECT,
                default="learned",
                options=list(POSITIONAL_MODES),
                description=(
                    "How the model learns where a token is. learned = a "
                    "trained vector per position (GPT-2); sinusoidal = the "
                    "fixed sine/cosine table (Vaswani et al.); rope = rotate "
                    "queries and keys by position (Llama), which generalises "
                    "best to longer text."
                ),
                advanced=True,
            ),
            ParamDefinition(
                name="norm",
                param_type=ParamType.SELECT,
                default="layernorm",
                options=list(NORM_TYPES),
                description=(
                    "Normaliser in front of each sub-layer. rmsnorm drops "
                    "the mean subtraction and the bias -- slightly cheaper, "
                    "and what modern open models use."
                ),
                advanced=True,
            ),
            ParamDefinition(
                name="activation",
                param_type=ParamType.SELECT,
                default="gelu",
                options=list(ACTIVATIONS),
                description=(
                    "The MLP's non-linearity. gelu is the transformer "
                    "default; silu (a.k.a. swish) is the Llama choice; relu "
                    "is the cheapest."
                ),
                advanced=True,
            ),
            ParamDefinition(
                name="dropout",
                param_type=ParamType.FLOAT,
                default=0.0,
                min_value=0.0,
                max_value=0.9,
                description=(
                    "Fraction of activations zeroed during training "
                    "(0 = disabled). Large-corpus pretraining leaves this "
                    "off; raise it when fine-tuning on a small dataset."
                ),
                advanced=True,
            ),
            ParamDefinition(
                name="gradient_checkpointing",
                param_type=ParamType.BOOL,
                default=False,
                description=(
                    "Recompute each block during the backward pass instead "
                    "of storing its activations: much less memory for about "
                    "30% more time. Turn it on when a batch will not fit."
                ),
                advanced=True,
            ),
            ParamDefinition(
                name="init_std",
                param_type=ParamType.FLOAT,
                default=0.02,
                min_value=0.0,
                description=(
                    "Standard deviation of the normal distribution the "
                    "weights start from. 0.02 is GPT-2's; the projections "
                    "feeding the residual stream are scaled down further by "
                    "sqrt(2 x n_layers)."
                ),
                advanced=True,
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=0,
                description=(
                    "Seed for the weight initialisation. The same seed gives "
                    "the same starting model, so two runs differ only by "
                    "what you changed."
                ),
                advanced=True,
            ),
        ]

    def build_module(self, params: dict[str, Any]) -> nn.Module:
        return CausalLMModule(**_resolve_config(params))

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        # Resolved here as well as in ``build_module`` so a contradictory
        # configuration says so even when the persisted module would have
        # been reused -- and so the message names the params rather than
        # arriving from inside a state-store callback.
        config = _resolve_config(params)
        model = self.get_or_build_module(context, params)

        param_count = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "Built CausalLMModel: %d layers, d_model=%d, %s parameters",
            config["n_layers"], config["d_model"], f"{param_count:,}")

        # ``__log__`` is the one result key the canvas Log tab renders, and
        # dunder keys are filtered out of recorded outputs and port
        # summaries -- so this adds a line to the log and nothing else.
        note = (
            f"{param_count:,} trainable parameters: {config['n_layers']} "
            f"blocks of d_model={config['d_model']} / {config['n_heads']} "
            f"heads / d_ff={config['d_ff']}, {config['positional']} "
            f"positions, context {config['max_seq_len']} tokens"
            f"{', tied embeddings' if config['tie_embeddings'] else ''}."
        )
        return {"model": model, "param_count": param_count, "__log__": note}
