"""PositionalEncodingNode — inject position information into token embeddings.

Self-attention is permutation-equivariant: shuffle the inputs and the outputs
shuffle the same way. That means raw token embeddings carry no notion of
*order*, which is fatal for language. The classical fix from Vaswani et al.
(2017) is to add a deterministic positional pattern computed from sines and
cosines at geometrically-spaced frequencies:

    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

Each dimension gets a different "wavelength", so the network can read off
absolute and relative position from inner products of the encoding.

Two modes:

* ``sinusoidal`` — the formula above. Stateless, deterministic, no params.
* ``learnable`` — random init seeded by ``seed``. Returned as a plain tensor
  so a single ``execute`` call is reproducible. (Real training of a
  learnable PE belongs in a future stateful variant; this teaching node
  is stateless on purpose so caches and presets behave predictably.)

The node accepts both ``[seq, D]`` and ``[seq, batch, D]`` inputs and
broadcasts the encoding across the batch dimension when present.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from ...core.step_trace import StepRecorder


class PositionalEncodingNode(BaseNode):
    NODE_NAME = "PositionalEncoding"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Add positional information to token embeddings. `sinusoidal` uses the "
        "Vaswani et al. (2017) formula PE(pos, 2i)=sin(pos/10000^(2i/d)); "
        "`learnable` returns a seeded random pattern (deterministic per seed)."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Input embeddings of shape [seq, D] or [seq, batch, D].",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Input + positional encoding, same shape as input.",
            ),
            PortDefinition(
                name="pe",
                data_type=DataType.TENSOR,
                description="The positional encoding alone, shape [seq, D] — useful for visualisation.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="mode",
                param_type=ParamType.SELECT,
                default="sinusoidal",
                options=["sinusoidal", "learnable"],
                description="sinusoidal=Vaswani formula; learnable=seeded random init.",
            ),
            ParamDefinition(
                name="max_len",
                param_type=ParamType.INT,
                default=512,
                min_value=1,
                description="Maximum sequence length supported. Inputs longer than this raise an error.",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=42,
                description="Seed for `learnable` mode initialisation. Ignored for `sinusoidal`.",
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
        x = inputs.get("tensor")
        if x is None:
            raise ValueError("PositionalEncoding requires a `tensor` input.")
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float()

        if x.ndim == 2:
            seq, d = x.shape
        elif x.ndim == 3:
            seq, _, d = x.shape
        else:
            raise ValueError(
                f"PositionalEncoding expects [seq, D] or [seq, batch, D]; got shape {tuple(x.shape)}"
            )

        mode = str(params.get("mode", "sinusoidal"))
        max_len = int(params.get("max_len", 512))
        seed = int(params.get("seed", 42))

        if seq > max_len:
            raise ValueError(
                f"Input seq_len={seq} exceeds max_len={max_len}; raise max_len or shorten input."
            )

        if mode == "sinusoidal":
            pe = self._sinusoidal_pe(seq, d)
        elif mode == "learnable":
            pe = self._learnable_pe(seq, d, seed)
        else:
            raise ValueError(f"Unknown PositionalEncoding mode: {mode!r}")

        if x.ndim == 2:
            out = x + pe
        else:  # [seq, batch, d]: broadcast over batch dim
            out = x + pe.unsqueeze(1)

        verbose = context is not None and getattr(context, "verbose", False)
        if verbose:
            recorder = StepRecorder()
            recorder.record(
                "input",
                f"Input embeddings of shape {tuple(x.shape)} (seq={seq}, d={d}).",
                tensor=x,
            )
            recorder.record(
                "compute_pe",
                f"Compute {mode} positional encoding of shape [seq, D] = ({seq}, {d}).",
                pe=pe,
            )
            recorder.record(
                "add",
                "Add PE to the input — each token now carries its position alongside its meaning.",
                output=out,
            )
            return {"tensor": out, "pe": pe, "__steps__": recorder.steps}

        return {"tensor": out, "pe": pe}

    @staticmethod
    def _sinusoidal_pe(seq: int, d: int) -> torch.Tensor:
        """Build PE matrix using the Vaswani formula.

        PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
        """
        pe = torch.zeros(seq, d, dtype=torch.float32)
        position = torch.arange(seq, dtype=torch.float32).unsqueeze(1)  # [seq, 1]
        # Frequency exponent for each pair (2i): 2i/d, i = 0 .. d/2
        # div_term = 1 / 10000^(2i/d) = exp(-2i/d * ln(10000))
        div_term = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float32) * -(math.log(10000.0) / d)
        )  # [d/2] (or ceil(d/2) when d is odd)
        # Even dims sin, odd dims cos. Slice carefully when d is odd.
        pe[:, 0::2] = torch.sin(position * div_term[: pe[:, 0::2].shape[1]])
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe

    @staticmethod
    def _learnable_pe(seq: int, d: int, seed: int) -> torch.Tensor:
        """Return a seeded random PE matrix.

        Uses a local generator so it does not perturb the global RNG state
        (which other nodes in the same graph might rely on).
        """
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        # Standard normal init scaled by 1/sqrt(d) — matches typical learnable PE
        # init magnitudes so downstream computation stays well-conditioned.
        return torch.randn(seq, d, generator=gen, dtype=torch.float32) * (1.0 / math.sqrt(d))
