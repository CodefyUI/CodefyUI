"""Step-trace recorder for verbose / step-by-step node execution.

When verbose mode is enabled (via ExecutionContext.verbose), instrumented
nodes can record intermediate algorithmic tensors via StepRecorder so the
Teaching Inspector can show them as a sequence of named steps.

Convention: an instrumented node returns its normal outputs plus a
reserved ``__steps__`` key in the result dict, e.g.

    recorder = StepRecorder()
    recorder.record("compute_qkv", "Project to Q, K, V", Q=q, K=k, V=v)
    return {"output": out, "__steps__": recorder.steps}

graph_engine.py expands ``__steps__`` into individual entries in
RunOutputStore using port names ``__step__{i}__{tensor_name}`` and
``__step__{i}__meta``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Step:
    """One algorithmic step inside a node's execution."""

    name: str
    description: str
    tensors: dict[str, Any] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)


class StepRecorder:
    """Append-only recorder for algorithmic steps inside a node."""

    def __init__(self) -> None:
        self.steps: list[Step] = []

    def record(
        self,
        name: str,
        description: str = "",
        *,
        scalars: dict[str, float] | None = None,
        **tensors: Any,
    ) -> None:
        """Record a step. Pass tensors as keyword arguments.

        Example:
            recorder.record(
                "scaled_scores",
                "scores = Q @ K.T / sqrt(d_k)",
                scalars={"d_k": 64.0},
                Q=q, K=k, scores=s,
            )
        """
        self.steps.append(
            Step(
                name=name,
                description=description,
                tensors=dict(tensors),
                scalars=dict(scalars) if scalars else {},
            )
        )

    def __len__(self) -> int:
        return len(self.steps)
