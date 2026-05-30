"""EduLSTMCellNode — an LSTM cell unrolled over time, gate by gate.

Educational counterpart to ``torch.nn.LSTMCell`` for lesson **I4-1**. Where the
library cell is a single opaque call, this node writes out the textbook recurrence
and surfaces the four gates plus the cell state at every timestep so a learner can
watch what the network "remembers" and "forgets" step by step:

    f = σ(W_f · [x, h] + b_f)        # forget gate — how much of c_{t-1} to keep
    i = σ(W_i · [x, h] + b_i)        # input gate — how much new info to admit
    g = tanh(W_g · [x, h] + b_g)     # candidate — the new info itself
    o = σ(W_o · [x, h] + b_o)        # output gate — how much of c_t to expose
    c_t = f ⊙ c_{t-1} + i ⊙ g        # new cell state (the "memory")
    h_t = o ⊙ tanh(c_t)              # new hidden state (the output)

To stay validatable against ``torch.nn.LSTMCell`` it uses PyTorch's exact
parameter convention: a single fused ``weight_ih`` ``[4H, input]``, ``weight_hh``
``[4H, hidden]``, ``bias_ih`` ``[4H]``, ``bias_hh`` ``[4H]``, with the four gate
chunks laid out in the order **i, f, g, o**. The fused pre-activations are

    gates = x @ weight_ih.T + bias_ih + h @ weight_hh.T + bias_hh

and ``gates`` is split into its i/f/g/o quarters before the non-linearities.

The four parameter tensors are seeded deterministically from ``seed`` and also
returned as display outputs (``weight_ih``/``weight_hh``/``bias_ih``/``bias_hh``)
so an external check can copy them straight into ``nn.LSTMCell`` and compare.

Inputs may be unbatched ``[seq, input_size]`` or batched ``[N, seq, input_size]``.
``h0``/``c0`` are optional and default to zeros.
"""

from __future__ import annotations

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

# Cap on the number of per-timestep verbose steps. For sequences longer than
# this we sample evenly so the Teaching Inspector never drowns in steps.
_MAX_TIMESTEP_STEPS = 6


class EduLSTMCellNode(BaseNode):
    NODE_NAME = "Edu-LSTMCell"
    CATEGORY = "RNN"
    DESCRIPTION = (
        "An LSTM cell unrolled over time. Computes the forget/input/output gates and "
        "candidate (f, i, g, o), then c_t = f⊙c_{t-1} + i⊙g and h_t = o⊙tanh(c_t) at "
        "every step. Uses PyTorch's fused weight_ih/weight_hh/bias_ih/bias_hh convention "
        "(gate order i,f,g,o) so it matches torch.nn.LSTMCell exactly, and exposes the "
        "four gates and the cell state per timestep in verbose mode."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="x_seq",
                data_type=DataType.TENSOR,
                description="Input sequence, shape [seq, input_size] or [N, seq, input_size].",
            ),
            PortDefinition(
                name="h0",
                data_type=DataType.TENSOR,
                description="Optional initial hidden state, [hidden_size] or [N, hidden_size]. Defaults to zeros.",
                optional=True,
            ),
            PortDefinition(
                name="c0",
                data_type=DataType.TENSOR,
                description="Optional initial cell state, [hidden_size] or [N, hidden_size]. Defaults to zeros.",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="h_seq",
                data_type=DataType.TENSOR,
                description="All hidden states stacked over time — [seq, hidden] or [N, seq, hidden].",
            ),
            PortDefinition(
                name="h_last",
                data_type=DataType.TENSOR,
                description="Hidden state after the final timestep — [hidden] or [N, hidden].",
            ),
            PortDefinition(
                name="c_last",
                data_type=DataType.TENSOR,
                description="Cell state after the final timestep — [hidden] or [N, hidden].",
            ),
            PortDefinition(
                name="weight_ih",
                data_type=DataType.TENSOR,
                description="Fused input→gates weight [4*hidden, input] (gate order i,f,g,o). Sync into nn.LSTMCell.",
            ),
            PortDefinition(
                name="weight_hh",
                data_type=DataType.TENSOR,
                description="Fused hidden→gates weight [4*hidden, hidden] (gate order i,f,g,o). Sync into nn.LSTMCell.",
            ),
            PortDefinition(
                name="bias_ih",
                data_type=DataType.TENSOR,
                description="Input gate bias [4*hidden] (gate order i,f,g,o).",
            ),
            PortDefinition(
                name="bias_hh",
                data_type=DataType.TENSOR,
                description="Hidden gate bias [4*hidden] (gate order i,f,g,o).",
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
                description="Dimension of the hidden / cell state H.",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=0,
                description="Seed for the deterministic small-randn weight init. Same seed → same weights.",
            ),
        ]

    @staticmethod
    def _init_weights(
        input_size: int, hidden_size: int, seed: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deterministic small-randn init in PyTorch's fused [4H, ...] layout."""
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        scale = 0.1
        weight_ih = torch.randn(4 * hidden_size, input_size, generator=gen) * scale
        weight_hh = torch.randn(4 * hidden_size, hidden_size, generator=gen) * scale
        bias_ih = torch.randn(4 * hidden_size, generator=gen) * scale
        bias_hh = torch.randn(4 * hidden_size, generator=gen) * scale
        return weight_ih, weight_hh, bias_ih, bias_hh

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
            raise ValueError("EduLSTMCell requires an `x_seq` input.")
        if not isinstance(x_seq, torch.Tensor):
            x_seq = torch.as_tensor(x_seq, dtype=torch.float32)
        x_seq = x_seq.float()

        input_size = int(params.get("input_size", 4))
        hidden_size = int(params.get("hidden_size", 8))
        seed = int(params.get("seed", 0))

        # ---- Validate the input sequence shape ---------------------------------
        if x_seq.ndim not in (2, 3):
            raise ValueError(
                "EduLSTMCell expects x_seq of shape [seq, input_size] or "
                f"[N, seq, input_size]; got {tuple(x_seq.shape)}."
            )
        if x_seq.shape[-1] != input_size:
            raise ValueError(
                f"EduLSTMCell: x_seq last dim {x_seq.shape[-1]} does not match "
                f"input_size={input_size}."
            )

        # Normalise to batched [N, seq, input_size] so a single code path handles both.
        batched = x_seq.ndim == 3
        if batched:
            n, seq, _ = x_seq.shape
            x_bn = x_seq
        else:
            seq, _ = x_seq.shape
            n = 1
            x_bn = x_seq.unsqueeze(0)  # [1, seq, input_size]
        if seq == 0:
            raise ValueError("EduLSTMCell: x_seq has zero timesteps.")

        # ---- Resolve / validate the initial states ------------------------------
        h = self._init_state(inputs.get("h0"), "h0", n, hidden_size, batched, x_seq.dtype)
        c = self._init_state(inputs.get("c0"), "c0", n, hidden_size, batched, x_seq.dtype)

        # ---- Parameters (PyTorch fused layout, gate order i, f, g, o) -----------
        weight_ih, weight_hh, bias_ih, bias_hh = self._init_weights(
            input_size, hidden_size, seed
        )

        verbose = context is not None and getattr(context, "verbose", False)
        recorder = StepRecorder() if verbose else None
        record_steps = self._steps_to_record(seq) if recorder is not None else set()

        H = hidden_size
        h_states: list[torch.Tensor] = []
        for t in range(seq):
            x_t = x_bn[:, t, :]  # [N, input_size]
            # Fused pre-activations, exactly as nn.LSTMCell computes them.
            gates = (
                F.linear(x_t, weight_ih, bias_ih)
                + F.linear(h, weight_hh, bias_hh)
            )  # [N, 4H]
            i = torch.sigmoid(gates[:, 0:H])       # input gate
            f = torch.sigmoid(gates[:, H:2 * H])   # forget gate
            g = torch.tanh(gates[:, 2 * H:3 * H])  # candidate
            o = torch.sigmoid(gates[:, 3 * H:4 * H])  # output gate
            c = f * c + i * g
            h = o * torch.tanh(c)
            h_states.append(h)

            if recorder is not None and t in record_steps:
                recorder.record(
                    f"timestep_{t}",
                    f"t={t}: gates i,f,g,o → c_t = f⊙c_{{t-1}} + i⊙g, h_t = o⊙tanh(c_t).",
                    scalars={"t": float(t)},
                    x_t=self._unbatch(x_t, batched),
                    i=self._unbatch(i, batched),
                    f=self._unbatch(f, batched),
                    g=self._unbatch(g, batched),
                    o=self._unbatch(o, batched),
                    c_t=self._unbatch(c, batched),
                    h_t=self._unbatch(h, batched),
                )

        h_seq_bn = torch.stack(h_states, dim=1)  # [N, seq, H]

        if batched:
            h_seq = h_seq_bn          # [N, seq, H]
            h_last = h                # [N, H]
            c_last = c                # [N, H]
        else:
            h_seq = h_seq_bn.squeeze(0)  # [seq, H]
            h_last = h.squeeze(0)        # [H]
            c_last = c.squeeze(0)        # [H]

        if recorder is not None:
            recorder.record(
                "h_seq",
                "Stack every hidden state over time into h_seq.",
                scalars={"seq": float(seq), "hidden_size": float(H)},
                h_seq=h_seq,
                h_last=h_last,
                c_last=c_last,
            )

        result: dict[str, Any] = {
            "h_seq": h_seq,
            "h_last": h_last,
            "c_last": c_last,
            "weight_ih": weight_ih,
            "weight_hh": weight_hh,
            "bias_ih": bias_ih,
            "bias_hh": bias_hh,
        }
        if recorder is not None and recorder.steps:
            result["__steps__"] = recorder.steps
        return result

    @staticmethod
    def _init_state(
        raw: Any,
        name: str,
        n: int,
        hidden_size: int,
        batched: bool,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a [N, hidden_size] initial state, validating/broadcasting `raw`."""
        if raw is None:
            return torch.zeros(n, hidden_size, dtype=dtype)
        state = raw if isinstance(raw, torch.Tensor) else torch.as_tensor(raw, dtype=dtype)
        state = state.float()
        if state.ndim == 1:
            if state.shape[0] != hidden_size:
                raise ValueError(
                    f"EduLSTMCell: {name} has length {state.shape[0]}, expected "
                    f"hidden_size={hidden_size}."
                )
            # An unbatched state is broadcast across the batch.
            return state.unsqueeze(0).expand(n, hidden_size).contiguous()
        if state.ndim == 2:
            if state.shape != (n, hidden_size):
                raise ValueError(
                    f"EduLSTMCell: {name} has shape {tuple(state.shape)}, expected "
                    f"[{n}, {hidden_size}] to match x_seq's batch."
                )
            return state.contiguous()
        raise ValueError(
            f"EduLSTMCell: {name} must be 1-D [hidden] or 2-D [N, hidden]; got "
            f"shape {tuple(state.shape)}."
        )

    @staticmethod
    def _steps_to_record(seq: int) -> set[int]:
        """Pick up to _MAX_TIMESTEP_STEPS timestep indices, sampled evenly."""
        if seq <= _MAX_TIMESTEP_STEPS:
            return set(range(seq))
        # Evenly spaced indices including the first and last timestep.
        return {
            round(k * (seq - 1) / (_MAX_TIMESTEP_STEPS - 1))
            for k in range(_MAX_TIMESTEP_STEPS)
        }

    @staticmethod
    def _unbatch(t: torch.Tensor, batched: bool) -> torch.Tensor:
        """Drop the synthetic batch dim for unbatched input so steps read cleanly."""
        return t if batched else t.squeeze(0)
