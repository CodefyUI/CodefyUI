"""VLAModel node (#312) -- a mini vision-language-action policy.

The policy side of the VLA epic (#309), shaped like the current literature
at canvas scale: a vision+language encoder trunk feeding a chunked action
expert, with the HEAD PARADIGM as a first-class research knob --

- ``head_type="flow_matching"``: the pi0 / SmolVLA family. Training noises
  the action chunk, the expert predicts the velocity field, inference
  integrates Euler steps from pure noise.
- ``head_type="regression"``: plain behavior cloning. The expert reads
  learned queries and predicts the chunk directly; the loss is MSE.

Both heads share the trunk, the data, and every other knob, so the two
dominant continuous-action approaches can be compared with everything else
held fixed. (Discretized action tokens, the OpenVLA family, would be a
third head -- an issue away, not a rewrite.)

**The loss comes out of a port.** A flow head trains on the residual
``v_pred - v_target`` against zero; a regression head trains on MSE against
the chunk. Wiring a generic loss to the wrong head would train the wrong
objective SILENTLY -- so this node emits the mode-matched ``loss_fn``
itself and there is no separate VLA loss node to mismatch.

**forward(data) carries the actions.** A training sample is
``((image, tokens, actions), actions)`` (see PushWorldDemos): flow matching
needs the target actions INSIDE forward to noise them, and TrainingLoop's
contract (``outputs = model(data); loss = loss_fn(outputs, targets)``)
stays untouched. The regression head ignores ``data[2]``.

**The vision stem defaults to a small conv pyramid.** A naive
``patch_size x patch_size`` patchify is the ViT lineage but localizes
poorly at 96px from scratch -- the "early convolutions help transformers
see" result reproduces at this scale (see the numbers in the node
description). ``vision_stem="patchify"`` keeps the classic stem available
for exactly that study.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from ...core.stateful_module import StatefulModuleMixin

logger = logging.getLogger(__name__)

HEAD_TYPES = ["flow_matching", "regression"]
VISION_STEMS = ["conv", "patchify"]
FLOW_TIME_DISTS = ["uniform", "beta"]

#: The conv stem's fixed downsampling (three stride-2 convs).
_CONV_STEM_FACTOR = 8


class _EncoderBlock(nn.Module):
    """Pre-LN self-attention + MLP, bidirectional -- the trunk unit."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(4 * d_model, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class _ExpertBlock(nn.Module):
    """Action-expert unit: self-attention over the chunk queries, then
    cross-attention into the trunk's vision+language features."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(4 * d_model, d_model))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        x = x + self.cross_attn(self.norm2(x), context, context,
                                need_weights=False)[0]
        return x + self.mlp(self.norm3(x))


class VLAModule(nn.Module):
    """The policy: (image, instruction bytes[, actions]) -> head output.

    ``forward`` is the TRAINING face (returns the residual for flow, the
    predicted chunk for regression); :meth:`predict_action` is the
    INFERENCE face both evaluation nodes call.
    """

    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        expert_layers: int,
        chunk: int,
        action_dim: int,
        max_text_len: int,
        head_type: str,
        vision_stem: str,
        dropout: float,
    ):
        super().__init__()
        if head_type not in HEAD_TYPES:
            raise ValueError(f"head_type must be one of {HEAD_TYPES}")
        if vision_stem not in VISION_STEMS:
            raise ValueError(f"vision_stem must be one of {VISION_STEMS}")
        if vision_stem == "conv":
            if image_size % _CONV_STEM_FACTOR:
                raise ValueError(
                    f"image_size must be divisible by {_CONV_STEM_FACTOR} "
                    f"for the conv stem, got {image_size}")
            n_vision_tokens = (image_size // _CONV_STEM_FACTOR) ** 2
            self.stem = nn.Sequential(
                nn.Conv2d(3, d_model // 4, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(d_model // 4, d_model // 2, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(d_model // 2, d_model, 3, stride=2, padding=1),
            )
        else:
            if image_size % patch_size:
                raise ValueError(
                    f"image_size must be divisible by patch_size, got "
                    f"{image_size} / {patch_size}")
            n_vision_tokens = (image_size // patch_size) ** 2
            self.stem = nn.Conv2d(
                3, d_model, kernel_size=patch_size, stride=patch_size)

        self.image_size = image_size
        self.chunk = chunk
        self.action_dim = action_dim
        self.max_text_len = max_text_len
        self.head_type = head_type
        self.d_model = d_model
        # Runtime knobs predict_action reads; the node refreshes them every
        # run so editing them never has to discard persisted weights.
        self.flow_steps = 10
        self.flow_time_dist = "uniform"

        self.vision_pos = nn.Parameter(
            torch.randn(1, n_vision_tokens, d_model) * 0.02)
        self.text_embedding = nn.Embedding(256, d_model)
        self.text_pos = nn.Parameter(
            torch.randn(1, max_text_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.trunk = nn.ModuleList(
            _EncoderBlock(d_model, n_heads, dropout) for _ in range(n_layers))
        self.trunk_norm = nn.LayerNorm(d_model)

        self.action_in = nn.Linear(action_dim, d_model)
        self.queries = nn.Parameter(torch.randn(1, chunk, d_model) * 0.02)
        self.query_pos = nn.Parameter(torch.randn(1, chunk, d_model) * 0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model))
        self.expert = nn.ModuleList(
            _ExpertBlock(d_model, n_heads, dropout)
            for _ in range(expert_layers))
        self.head = nn.Linear(d_model, action_dim)

    # -- shared pieces --------------------------------------------------

    def encode(self, image: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        if image.dim() != 4 or image.shape[1] != 3:
            raise ValueError(
                f"image must be (B, 3, {self.image_size}, {self.image_size}), "
                f"got {tuple(image.shape)}")
        # Zero-centered input: measured to matter for the from-scratch small
        # ViT (the slow-start plateau shortens visibly).
        vision = self.stem(image * 2.0 - 1.0)
        vision = vision.flatten(2).transpose(1, 2) + self.vision_pos
        text = self.text_embedding(tokens) + self.text_pos
        x = self.dropout(torch.cat([vision, text], dim=1))
        for block in self.trunk:
            x = block(x)
        return self.trunk_norm(x)

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.d_model // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * (-math.log(10000.0) / half))
        angles = t[:, None].float() * freqs[None, :] * 1000.0
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.time_mlp(embedding.to(self.queries.dtype))

    def _expert_pass(self, queries: torch.Tensor,
                     context: torch.Tensor) -> torch.Tensor:
        x = queries
        for block in self.expert:
            x = block(x, context)
        return self.head(x)

    def _sample_time(self, batch: int, device: torch.device) -> torch.Tensor:
        if self.flow_time_dist == "beta":
            # More mass near t=0 (nearly pure noise) -- the pi0-style
            # emphasis on the noisier end of the path.
            dist = torch.distributions.Beta(
                torch.tensor(1.0, device=device),
                torch.tensor(1.5, device=device))
            return dist.sample((batch,))
        return torch.rand(batch, device=device)

    # -- training face ---------------------------------------------------

    def forward(self, data: Any) -> torch.Tensor:
        if not isinstance(data, (tuple, list)) or len(data) < 2:
            raise TypeError(
                "VLAModel.forward expects (image, tokens, actions) as one "
                "tuple -- the PushWorldDemos sample shape. Got "
                f"{type(data).__name__}.")
        image, tokens = data[0], data[1]
        context = self.encode(image, tokens)
        batch = image.shape[0]

        if self.head_type == "regression":
            queries = self.queries.expand(batch, -1, -1) + self.query_pos
            return self._expert_pass(queries, context)

        if len(data) < 3 or data[2] is None:
            raise ValueError(
                "flow_matching training needs the action chunk inside data "
                "((image, tokens, actions)) so forward can noise it; wire "
                "PushWorldDemos' dataset, not a bare (image, tokens) pair.")
        actions = data[2]
        t = self._sample_time(batch, image.device)
        noise = torch.randn_like(actions)
        t_wide = t[:, None, None].to(actions.dtype)
        noised = (1 - t_wide) * noise + t_wide * actions
        velocity_target = actions - noise
        queries = (self.action_in(noised) + self.query_pos
                   + self._time_embedding(t)[:, None, :])
        velocity_pred = self._expert_pass(queries, context)
        # The residual, so the loss is mean(residual^2) with no second
        # forward output to thread through TrainingLoop.
        return velocity_pred - velocity_target

    # -- inference face --------------------------------------------------

    @torch.no_grad()
    def predict_action(self, image: torch.Tensor,
                       tokens: torch.Tensor) -> torch.Tensor:
        """(B, 3, S, S) + (B, L) -> (B, chunk, action_dim)."""
        was_training = self.training
        self.eval()
        try:
            context = self.encode(image, tokens)
            batch = image.shape[0]
            if self.head_type == "regression":
                queries = self.queries.expand(batch, -1, -1) + self.query_pos
                return self._expert_pass(queries, context)
            steps = max(1, int(self.flow_steps))
            x = torch.randn(
                batch, self.chunk, self.action_dim, device=image.device,
                dtype=self.queries.dtype)
            dt = 1.0 / steps
            for index in range(steps):
                t = torch.full((batch,), index * dt, device=image.device)
                queries = (self.action_in(x) + self.query_pos
                           + self._time_embedding(t)[:, None, :])
                x = x + self._expert_pass(queries, context) * dt
            return x
        finally:
            self.train(was_training)


class VLABehaviorLoss(nn.Module):
    """The mode-matched loss VLAModel emits on its ``loss_fn`` port.

    flow_matching: ``outputs`` IS the residual, so the loss is its squared
    mean and ``targets`` is deliberately unused. regression: plain MSE.
    Reductions in float32 so bf16 autocast cannot degrade them. Must never
    subclass CrossEntropyLoss/NLLLoss -- TrainingLoop's accuracy gate keys
    on that isinstance.
    """

    def __init__(self, head_type: str):
        super().__init__()
        self.head_type = head_type

    def forward(self, outputs: torch.Tensor,
                targets: torch.Tensor | None = None) -> torch.Tensor:
        if self.head_type == "flow_matching":
            return outputs.float().pow(2).mean()
        if targets is None:
            raise ValueError("regression loss needs targets")
        return F.mse_loss(outputs.float(), targets.float())


def _resolve_config(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_size": int(params.get("image_size", 96) or 96),
        "patch_size": int(params.get("patch_size", 8) or 8),
        "d_model": int(params.get("d_model", 192) or 192),
        "n_layers": int(params.get("n_layers", 4) or 4),
        "n_heads": int(params.get("n_heads", 6) or 6),
        "expert_layers": int(params.get("expert_layers", 2) or 2),
        "chunk": int(params.get("chunk", 8) or 8),
        "action_dim": int(params.get("action_dim", 2) or 2),
        "max_text_len": int(params.get("max_text_len", 48) or 48),
        "head_type": str(params.get("head_type", "flow_matching")
                         or "flow_matching"),
        "vision_stem": str(params.get("vision_stem", "conv") or "conv"),
        "dropout": float(params.get("dropout", 0.0) or 0.0),
    }


class VLAModelNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "VLAModel"
    CATEGORY = "VLA"
    DESCRIPTION = (
        "A mini vision-language-action policy: vision stem + byte-level "
        "instruction embedding -> transformer trunk -> chunked action "
        "expert. head_type picks the paradigm - flow_matching (pi0/SmolVLA "
        "family: noise the chunk, learn the velocity field, Euler-integrate "
        "at inference) or regression (direct MSE behavior cloning) - with "
        "everything else held fixed, so the two can be compared honestly. "
        "Emits the mode-matched loss_fn itself; wire model+loss_fn to "
        "TrainingLoop and the PushWorldDemos dataset to a DataLoader. "
        "Defaults build ~3.2M params."
    )

    # A MODEL output is a live handle TrainingLoop mutates in place --
    # the #253/#254 rule, same as CausalLMModel.
    cacheable = False

    # Every constructor input, so editing one honestly discards persisted
    # weights. flow_steps / flow_time_dist are deliberately ABSENT: they are
    # runtime behavior refreshed onto the module every run.
    structural_params = (
        "image_size",
        "patch_size",
        "d_model",
        "n_layers",
        "n_heads",
        "expert_layers",
        "chunk",
        "action_dim",
        "max_text_len",
        "head_type",
        "vision_stem",
        "dropout",
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
                    "The policy as an nn.Module: forward((image, tokens, "
                    "actions)) for training, predict_action(image, tokens) "
                    "for rollout. Carries chunk/action_dim/image_size/"
                    "max_text_len/head_type as attributes."
                ),
            ),
            PortDefinition(
                name="loss_fn",
                data_type=DataType.LOSS_FN,
                description=(
                    "The loss matched to head_type - emitted here so a "
                    "mismatched generic loss cannot silently train the "
                    "wrong objective. Wire straight to TrainingLoop."
                ),
            ),
            PortDefinition(
                name="param_count",
                data_type=DataType.SCALAR,
                description="Trainable parameters",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="head_type",
                param_type=ParamType.SELECT,
                default="flow_matching",
                options=HEAD_TYPES,
                description=(
                    "flow_matching: pi0/SmolVLA-style velocity field over "
                    "noised action chunks, Euler sampling at inference. "
                    "regression: direct chunk prediction, MSE. The loss_fn "
                    "output follows this choice automatically."
                ),
            ),
            ParamDefinition(
                name="d_model",
                param_type=ParamType.INT,
                default=192,
                min_value=32,
                max_value=2048,
                description="Width of every token stream (must divide by n_heads)",
            ),
            ParamDefinition(
                name="n_layers",
                param_type=ParamType.INT,
                default=4,
                min_value=1,
                max_value=48,
                description="Trunk depth over [vision; text] tokens",
            ),
            ParamDefinition(
                name="n_heads",
                param_type=ParamType.INT,
                default=6,
                min_value=1,
                max_value=32,
                description="Attention heads, trunk and expert alike",
            ),
            ParamDefinition(
                name="expert_layers",
                param_type=ParamType.INT,
                default=2,
                min_value=1,
                max_value=16,
                description=(
                    "Action-expert depth (self-attention over the chunk "
                    "queries + cross-attention into the trunk)"
                ),
            ),
            ParamDefinition(
                name="chunk",
                param_type=ParamType.INT,
                default=8,
                min_value=1,
                max_value=64,
                description=(
                    "Actions predicted per call (the chunk horizon H) - "
                    "must match PushWorldDemos' chunk"
                ),
            ),
            ParamDefinition(
                name="image_size",
                param_type=ParamType.INT,
                default=96,
                min_value=32,
                max_value=256,
                description="Input frame size - must match PushWorldEnv's image_size",
            ),
            ParamDefinition(
                name="vision_stem",
                param_type=ParamType.SELECT,
                default="conv",
                options=VISION_STEMS,
                description=(
                    "conv: three stride-2 3x3 convs (fine localization; the "
                    "'early convolutions help transformers see' result is "
                    "measurable here). patchify: the classic ViT stem, kept "
                    "for studying exactly that difference."
                ),
                advanced=True,
            ),
            ParamDefinition(
                name="patch_size",
                param_type=ParamType.INT,
                default=8,
                min_value=4,
                max_value=32,
                description="Patchify stem only: square patch edge",
                visible_when={"vision_stem": "patchify"},
                advanced=True,
            ),
            ParamDefinition(
                name="action_dim",
                param_type=ParamType.INT,
                default=2,
                min_value=1,
                max_value=32,
                description="Action vector width (PushWorld is 2: dx, dy)",
                advanced=True,
            ),
            ParamDefinition(
                name="max_text_len",
                param_type=ParamType.INT,
                default=48,
                min_value=8,
                max_value=256,
                description=(
                    "Instruction length in BYTES - must match the dataset's "
                    "encoding (PushWorldDemos uses 48)"
                ),
                advanced=True,
            ),
            ParamDefinition(
                name="flow_steps",
                param_type=ParamType.INT,
                default=10,
                min_value=1,
                max_value=100,
                description=(
                    "flow_matching only: Euler integration steps at "
                    "inference (SmolVLA uses 10). Runtime knob - editing it "
                    "keeps persisted weights."
                ),
                visible_when={"head_type": "flow_matching"},
                advanced=True,
            ),
            ParamDefinition(
                name="flow_time_dist",
                param_type=ParamType.SELECT,
                default="uniform",
                options=FLOW_TIME_DISTS,
                description=(
                    "flow_matching only: how training samples the flow time "
                    "t. beta weights the noisier end of the path "
                    "(pi0-style). Runtime knob - keeps persisted weights."
                ),
                visible_when={"head_type": "flow_matching"},
                advanced=True,
            ),
            ParamDefinition(
                name="dropout",
                param_type=ParamType.FLOAT,
                default=0.0,
                min_value=0.0,
                max_value=0.9,
                description="Dropout through trunk and expert",
                advanced=True,
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description="Weight-init seed",
                advanced=True,
            ),
        ]

    def build_module(self, params: dict[str, Any]) -> nn.Module:
        config = _resolve_config(params)
        if config["d_model"] % config["n_heads"]:
            raise ValueError(
                f"d_model ({config['d_model']}) must divide evenly by "
                f"n_heads ({config['n_heads']})")
        torch.manual_seed(max(0, int(params.get("seed", 0) or 0)))
        return VLAModule(**config)

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        config = _resolve_config(params)
        model = self.get_or_build_module(context, params)
        # Runtime knobs refresh every run precisely BECAUSE they are not
        # structural: a persisted module must still honor today's values.
        model.flow_steps = max(1, int(params.get("flow_steps", 10) or 10))
        model.flow_time_dist = str(
            params.get("flow_time_dist", "uniform") or "uniform")

        param_count = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        note = (
            f"{param_count:,} trainable parameters: {config['n_layers']}-"
            f"layer trunk + {config['expert_layers']}-layer action expert, "
            f"d_model={config['d_model']}, {config['vision_stem']} stem, "
            f"chunk {config['chunk']}, head {config['head_type']}."
        )
        logger.info("VLAModel: %s", note)
        return {
            "model": model,
            "loss_fn": VLABehaviorLoss(config["head_type"]),
            "param_count": param_count,
            "__log__": note,
        }
