"""CausalLMModelNode — 建一個可訓練的 GPT-style decoder-only transformer。

LLM 分類原本只有前向示範節點（TransformerEncoder/Decoder 吃 TENSOR 吐 TENSOR，
不是可訓練的 MODEL），SequentialModel 的層編輯器也排不出帶殘差的注意力區塊。
這顆補上「語言模型本體」：輸出一個真正的 `MODEL`，可以直接接 Optimizer、
TrainingLoop、CheckpointSaver 與 LM 評估／生成節點，在畫布上訓練出一個真的
小語言模型（預設參數約 2.02 億，對應 epic #292 的 200M 驗收）。

Forward contract: ``input_ids`` int64 ``(B, T)`` → logits float ``(B, T, vocab)``.
Causal masking 在模型內部（SDPA ``is_causal=True``），不需要外接 AttentionMask。
純 torch 實作，不依賴 ``transformers``。Torch modules live in
``_lm_modules.py`` (module-scope for full_model pickling, #283); this module
imports it lazily so the registry's metadata scan stays torch-free.
"""

from __future__ import annotations

from typing import Any

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)

POSITIONAL_MODES = ["learned", "sinusoidal", "rope"]
NORM_KINDS = ["layernorm", "rmsnorm"]
ACTIVATIONS = ["gelu", "relu", "silu"]


class CausalLMModelNode(BaseNode):
    NODE_NAME = "CausalLMModel"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Build a trainable GPT-style decoder-only transformer language model. "
        "Outputs a real MODEL (input_ids (B,T) int64 -> logits (B,T,vocab)) for "
        "Optimizer / TrainingLoop / CheckpointSaver, plus its exact trainable "
        "parameter count. Causal masking is internal; pair with "
        "LMCrossEntropyLoss and packed LM data. Defaults are a ~202M-parameter "
        "shape (d_model 1024, 12 layers, gpt2 vocab, tied embeddings)."
    )

    # Owns trainable parameters and hands out a live MODEL handle (rules 1
    # and 2 of the cacheable contract): a cache hit would replay a handle
    # whose weights a later TrainingLoop already moved.
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="model",
                data_type=DataType.MODEL,
                description="Freshly initialized causal LM (torch nn.Module)",
            ),
            PortDefinition(
                name="param_count",
                data_type=DataType.SCALAR,
                description="Exact trainable-parameter count, so a graph can prove its model size",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="vocab_size", param_type=ParamType.INT, default=50257,
                min_value=256, max_value=300000,
                description="Vocabulary size (50257 = tiktoken gpt2; keep it matched to the LMTokenizer encoding)",
            ),
            ParamDefinition(
                name="d_model", param_type=ParamType.INT, default=1024,
                min_value=64, max_value=8192,
                description="Embedding / residual-stream width",
            ),
            ParamDefinition(
                name="n_layers", param_type=ParamType.INT, default=12,
                min_value=1, max_value=64,
                description="Number of transformer blocks",
            ),
            ParamDefinition(
                name="n_heads", param_type=ParamType.INT, default=16,
                min_value=1, max_value=64,
                description="Attention heads (d_model must divide evenly by n_heads)",
            ),
            ParamDefinition(
                name="d_ff", param_type=ParamType.INT, default=4096,
                min_value=64, max_value=32768,
                description="Feed-forward hidden width (typically 4x d_model)",
            ),
            ParamDefinition(
                name="max_seq_len", param_type=ParamType.INT, default=1024,
                min_value=16, max_value=8192,
                description="Maximum sequence length the model accepts",
            ),
            ParamDefinition(
                name="dropout", param_type=ParamType.FLOAT, default=0.0,
                min_value=0.0, max_value=0.9, advanced=True,
                description="Dropout on attention/MLP/embeddings (0 is standard for single-epoch LM pretraining)",
            ),
            ParamDefinition(
                name="positional", param_type=ParamType.SELECT, default="learned",
                options=POSITIONAL_MODES, advanced=True,
                description="Position encoding: learned embedding, fixed sinusoidal, or rotary (RoPE)",
            ),
            ParamDefinition(
                name="norm", param_type=ParamType.SELECT, default="layernorm",
                options=NORM_KINDS, advanced=True,
                description="Normalization layer (pre-norm blocks either way)",
            ),
            ParamDefinition(
                name="activation", param_type=ParamType.SELECT, default="gelu",
                options=ACTIVATIONS, advanced=True,
                description="Feed-forward activation",
            ),
            ParamDefinition(
                name="tie_embeddings", param_type=ParamType.BOOL, default=True,
                advanced=True,
                description="Share the token-embedding matrix with the output head (saves vocab*d_model params)",
            ),
            ParamDefinition(
                name="gradient_checkpointing", param_type=ParamType.BOOL,
                default=False, advanced=True,
                description="Recompute activations in backward to trade compute for memory",
            ),
            ParamDefinition(
                name="init_std", param_type=ParamType.FLOAT, default=0.02,
                min_value=0.001, max_value=0.2, advanced=True,
                description="Weight init standard deviation (residual projections are scaled down by sqrt(2*n_layers))",
            ),
            ParamDefinition(
                name="seed", param_type=ParamType.INT, default=0,
                min_value=0, advanced=True,
                description="Initialization seed — the same seed builds identical weights",
            ),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        import torch

        from ._lm_modules import CausalLMModule

        vocab_size = int(params.get("vocab_size", 50257))
        d_model = int(params.get("d_model", 1024))
        n_layers = int(params.get("n_layers", 12))
        n_heads = int(params.get("n_heads", 16))
        d_ff = int(params.get("d_ff", 4096))
        max_seq_len = int(params.get("max_seq_len", 1024))
        dropout = float(params.get("dropout", 0.0))
        positional = str(params.get("positional", "learned"))
        norm = str(params.get("norm", "layernorm"))
        activation = str(params.get("activation", "gelu"))
        tie_embeddings = bool(params.get("tie_embeddings", True))
        gradient_checkpointing = bool(params.get("gradient_checkpointing", False))
        init_std = float(params.get("init_std", 0.02))
        seed = int(params.get("seed", 0))

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must divide evenly by n_heads ({n_heads})"
            )
        if positional not in POSITIONAL_MODES:
            raise ValueError(f"Unknown positional mode: {positional!r}")
        if norm not in NORM_KINDS:
            raise ValueError(f"Unknown norm kind: {norm!r}")
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation!r}")

        # fork_rng so a fixed init seed never disturbs the run-level RNG
        # stream other nodes derive from (deterministic mode, DataLoader
        # shuffles). CPU-only: the model is built on CPU and moved to its
        # device later by Optimizer/TrainingLoop.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = CausalLMModule(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                dropout=dropout,
                positional=positional,
                norm=norm,
                activation=activation,
                tie_embeddings=tie_embeddings,
                gradient_checkpointing=gradient_checkpointing,
                init_std=init_std,
            )

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if progress_callback:
            progress_callback({
                "event": "built",
                "param_count": param_count,
                "layers": n_layers,
                "d_model": d_model,
            })
        return {"model": model, "param_count": int(param_count)}
