"""EduRNNCellNode — a vanilla RNN cell unrolled over time, every step exposed.

Educational counterpart to the opaque ``nn.RNN`` / ``nn.RNNCell`` node. Where
those run a whole sequence inside a C++ kernel, this one writes out the
textbook recurrence one timestep at a time:

    h_t = activation(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)

so the lesson (I4-1) can watch the hidden state evolve. The two pre-activation
contributions are surfaced separately:

    Wx = W_ih @ x_t  + b_ih      # what the *current input* injects
    Wh = W_hh @ h_{t-1} + b_hh   # what the *memory of the past* carries

and ``h_t = activation(Wx + Wh)``. Seeing Wx and Wh side by side is the whole
point: it makes "the hidden state is a blend of new input and old memory"
concrete rather than a slogan.

The four parameter tensors ``W_ih [hidden, input]``, ``W_hh [hidden, hidden]``,
``b_ih [hidden]``, ``b_hh [hidden]`` are built deterministically from ``seed``
(a seeded ``torch.Generator``, ``randn * 0.1``). They use exactly the layout of
``torch.nn.RNNCell``, so a lesson — or a test — can copy them into a reference
cell and confirm the hand-rolled unroll matches PyTorch bit for bit.

Inputs may be 2-D ``[seq, input_size]`` (a single sequence) or 3-D
``[N, seq, input_size]`` (a batch of sequences, looped one at a time). Output
shape mirrors the input: ``[seq, hidden]`` or ``[N, seq, hidden]``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from app.core.step_trace import StepRecorder

# How many per-timestep steps to record in verbose mode before we start
# sampling. Long sequences would otherwise flood the Teaching Inspector.
_STEP_CAP = 6


class EduRNNCellNode(BaseNode):
    NODE_NAME = "Edu-RNNCell"
    CATEGORY = "RNN"
    DESCRIPTION = (
        "A vanilla RNN unrolled over time. Computes "
        "h_t = activation(W_ih·x_t + b_ih + W_hh·h_{t-1} + b_hh) one timestep at "
        "a time, exposing the input contribution (Wx) and the memory contribution "
        "(Wh) separately so students see the hidden state blend new input with "
        "past state. Weights are deterministic from a seed and share nn.RNNCell's "
        "layout, so the unroll matches PyTorch exactly."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="x_seq",
                data_type=DataType.TENSOR,
                description=(
                    "Input sequence of shape [seq, input_size], or a batch "
                    "[N, seq, input_size] looped one sequence at a time."
                ),
            ),
            PortDefinition(
                name="h0",
                data_type=DataType.TENSOR,
                description=(
                    "Optional initial hidden state [hidden_size] (or "
                    "[N, hidden_size] for a batch). Defaults to zeros."
                ),
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="h_seq",
                data_type=DataType.TENSOR,
                description=(
                    "All hidden states, [seq, hidden] for 2-D input or "
                    "[N, seq, hidden] for 3-D input."
                ),
            ),
            PortDefinition(
                name="h_last",
                data_type=DataType.TENSOR,
                description=(
                    "Final hidden state, [hidden] for 2-D input or [N, hidden] "
                    "for 3-D input."
                ),
            ),
            PortDefinition(
                name="weights",
                data_type=DataType.TENSOR,
                description=(
                    "Input-to-hidden weight W_ih [hidden, input], display-only so "
                    "the lesson can inspect the learned/initialised parameters."
                ),
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="input_size",
                param_type=ParamType.INT,
                default=4,
                min_value=1,
                description="Dimension of each input vector x_t. Must match x_seq's last dim.",
            ),
            ParamDefinition(
                name="hidden_size",
                param_type=ParamType.INT,
                default=8,
                min_value=1,
                description="Dimension of the hidden state h_t.",
            ),
            ParamDefinition(
                name="activation",
                param_type=ParamType.SELECT,
                default="tanh",
                options=["tanh", "relu"],
                description="Nonlinearity applied to the pre-activation sum (matches nn.RNNCell's nonlinearity).",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=0,
                description="Seed for the deterministic W_ih/W_hh/b_ih/b_hh initialisation (randn * 0.1).",
            ),
        ]

    # ------------------------------------------------------------------ #
    # Parameter construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_params(
        input_size: int, hidden_size: int, seed: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deterministically build (W_ih, W_hh, b_ih, b_hh) from a seed.

        Layout matches ``torch.nn.RNNCell``: W_ih is [hidden, input],
        W_hh is [hidden, hidden], biases are [hidden]. Draw order is fixed
        so the same seed always yields the same four tensors.
        """
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        W_ih = torch.randn(hidden_size, input_size, generator=gen) * 0.1
        W_hh = torch.randn(hidden_size, hidden_size, generator=gen) * 0.1
        b_ih = torch.randn(hidden_size, generator=gen) * 0.1
        b_hh = torch.randn(hidden_size, generator=gen) * 0.1
        return W_ih, W_hh, b_ih, b_hh

    @staticmethod
    def _activation_fn(name: str):
        if name == "tanh":
            return torch.tanh
        if name == "relu":
            return F.relu
        raise ValueError(
            f"EduRNNCell: unknown activation '{name}'; expected 'tanh' or 'relu'."
        )

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        x_seq = inputs.get("x_seq")
        if x_seq is None:
            raise ValueError("EduRNNCell requires an `x_seq` input.")
        if not isinstance(x_seq, torch.Tensor):
            x_seq = torch.as_tensor(x_seq, dtype=torch.float32)
        x_seq = x_seq.float()

        input_size = int(params.get("input_size", 4))
        hidden_size = int(params.get("hidden_size", 8))
        activation_name = str(params.get("activation", "tanh"))
        seed = int(params.get("seed", 0))

        if x_seq.ndim not in (2, 3):
            raise ValueError(
                "EduRNNCell expects x_seq of shape [seq, input_size] or "
                f"[N, seq, input_size]; got {tuple(x_seq.shape)}."
            )
        if x_seq.shape[-1] != input_size:
            raise ValueError(
                f"EduRNNCell: x_seq last dim {x_seq.shape[-1]} does not match "
                f"input_size={input_size}."
            )

        activation = self._activation_fn(activation_name)
        W_ih, W_hh, b_ih, b_hh = self._build_params(input_size, hidden_size, seed)

        # Normalise to a batched [N, seq, input] view so one code path handles
        # both 2-D and 3-D inputs.
        batched = x_seq.ndim == 3
        if batched:
            N, seq, _ = x_seq.shape
            x_b = x_seq
        else:
            seq = x_seq.shape[0]
            N = 1
            x_b = x_seq.unsqueeze(0)  # [1, seq, input]

        # Resolve the initial hidden state h0 -> [N, hidden].
        h0 = inputs.get("h0")
        if h0 is None:
            h_prev_b = torch.zeros(N, hidden_size, dtype=x_b.dtype)
        else:
            if not isinstance(h0, torch.Tensor):
                h0 = torch.as_tensor(h0, dtype=torch.float32)
            h0 = h0.float()
            if h0.ndim == 1:
                if h0.shape[0] != hidden_size:
                    raise ValueError(
                        f"EduRNNCell: h0 has shape {tuple(h0.shape)}, expected "
                        f"[{hidden_size}] (or [N, {hidden_size}])."
                    )
                h_prev_b = h0.unsqueeze(0).expand(N, hidden_size).contiguous()
            elif h0.ndim == 2:
                if h0.shape != (N, hidden_size):
                    raise ValueError(
                        f"EduRNNCell: h0 has shape {tuple(h0.shape)}, expected "
                        f"[{N}, {hidden_size}] to match the batch."
                    )
                h_prev_b = h0.clone()
            else:
                raise ValueError(
                    f"EduRNNCell: h0 must be 1-D [hidden] or 2-D [N, hidden]; "
                    f"got {tuple(h0.shape)}."
                )

        verbose = context is not None and getattr(context, "verbose", False)
        recorder = StepRecorder() if verbose else None
        # Which timesteps to record when verbose: all of them if short, else an
        # evenly-spaced sample (always including the first and last) up to the cap.
        record_steps: set[int] = set()
        if recorder is not None:
            if seq <= _STEP_CAP:
                record_steps = set(range(seq))
            else:
                record_steps = {
                    int(round(i * (seq - 1) / (_STEP_CAP - 1)))
                    for i in range(_STEP_CAP)
                }

        # Unroll: for the visible single-sequence view (or the first sequence of
        # a batch) we also record intermediate states for the lesson.
        h_seq_b = torch.empty(N, seq, hidden_size, dtype=x_b.dtype)
        h_prev = h_prev_b
        for t in range(seq):
            x_t = x_b[:, t, :]  # [N, input]
            Wx = x_t @ W_ih.T + b_ih  # [N, hidden] — input contribution
            Wh = h_prev @ W_hh.T + b_hh  # [N, hidden] — memory contribution
            h_t = activation(Wx + Wh)  # [N, hidden]
            h_seq_b[:, t, :] = h_t

            if recorder is not None and t in record_steps:
                # Show the first sequence in the batch (index 0) so the recorded
                # vectors are 1-D [hidden] and read cleanly in the inspector.
                recorder.record(
                    f"t={t}",
                    (
                        f"Timestep {t}: h_{t} = {activation_name}("
                        "W_ih·x_t + b_ih + W_hh·h_{prev} + b_hh). "
                        "Wx is the input's contribution, Wh the memory's."
                    ),
                    x_t=x_t[0],
                    Wx=Wx[0],
                    Wh=Wh[0],
                    h_t=h_t[0],
                )
            h_prev = h_t

        h_last_b = h_prev  # [N, hidden]

        if batched:
            h_seq = h_seq_b
            h_last = h_last_b
        else:
            h_seq = h_seq_b.squeeze(0)  # [seq, hidden]
            h_last = h_last_b.squeeze(0)  # [hidden]

        if recorder is not None:
            recorder.record(
                "h_seq",
                f"Stacked hidden states over all {seq} timesteps, shape {tuple(h_seq.shape)}.",
                scalars={
                    "seq": float(seq),
                    "hidden_size": float(hidden_size),
                },
                h_seq=h_seq,
            )

        result: dict[str, Any] = {
            "h_seq": h_seq,
            "h_last": h_last,
            "weights": W_ih,
        }
        if recorder is not None and recorder.steps:
            result["__steps__"] = recorder.steps
        return result
