"""The deep pack's Edu attention nodes put their masks on the scores' device (#683).

The causal mask used to be built on the CPU and an external mask was applied
as given, so ``causal=true`` or a CPU mask failed as soon as the scores lived
on an accelerator. The meta device stands in for one here: PyTorch refuses a
CPU mask on meta scores the same way it refuses one on MPS scores, so these
tests catch the bug on any machine. The MPS tests then check real numbers
against the CPU run where MPS exists.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import torch

from app.core.execution_context import ExecutionContext
from app.core.graph_engine import execute_graph
from cdui_plugins.deep.nodes.edu_cross_attention_node import EduCrossAttentionNode
from cdui_plugins.deep.nodes.edu_multi_head_attention_node import EduMultiHeadAttentionNode
from cdui_plugins.deep.nodes.edu_self_attention_node import EduSelfAttentionNode

mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
requires_mps = pytest.mark.skipif(not mps_available, reason="needs an MPS device")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MULTI_HEAD_CAUSAL = _REPO_ROOT / "plugins" / "deep" / "examples" / "LLM" / "Multi-Head-Causal" / "graph.json"


def _on_meta(node_cls):
    """The node with its projection module moved to the meta device."""

    class _MetaNode(node_cls):
        def get_or_build_module(self, context, params):
            return super().get_or_build_module(context, params).to("meta")

    return _MetaNode()


_SEQ = 5
_CPU_MASK = torch.tensor(
    [[False, False, False, False, True]] * _SEQ, dtype=torch.bool
)  # every query is blocked from the last key


@pytest.mark.parametrize("node_cls", [EduSelfAttentionNode, EduMultiHeadAttentionNode])
@pytest.mark.parametrize(
    "causal, mask",
    [(True, None), (False, _CPU_MASK), (True, _CPU_MASK)],
    ids=["causal", "cpu-mask", "causal+cpu-mask"],
)
def test_self_attention_masks_follow_the_scores_device(node_cls, causal, mask):
    params = {"embed_dim": 8, "num_heads": 2, "causal": causal, "seed": 0}
    inputs = {"tensor": torch.empty(_SEQ, 8, device="meta")}
    if mask is not None:
        inputs["mask"] = mask

    out = _on_meta(node_cls).execute(inputs, params)

    assert out["output"].device.type == "meta"
    assert out["weights"].device.type == "meta"


def test_cross_attention_cpu_mask_follows_the_scores_device():
    mask = torch.zeros(4, 6, dtype=torch.bool)
    mask[:, -1] = True
    out = _on_meta(EduCrossAttentionNode).execute(
        {
            "query": torch.empty(4, 8, device="meta"),
            "context": torch.empty(6, 8, device="meta"),
            "mask": mask,
        },
        {"embed_dim": 8, "num_heads": 2, "seed": 0},
    )
    assert out["weights"].device.type == "meta"


# ── real accelerator ──────────────────────────────────────────────────────


def _run_on(node, device, inputs, params):
    moved = {k: v.to(device) if k != "mask" else v for k, v in inputs.items()}
    out = node.execute(moved, params, context=ExecutionContext(device=device))
    return {k: v.cpu() for k, v in out.items() if isinstance(v, torch.Tensor)}


@requires_mps
@pytest.mark.parametrize("node_cls", [EduSelfAttentionNode, EduMultiHeadAttentionNode])
def test_causal_self_attention_on_mps_matches_cpu(node_cls):
    x = torch.randn(_SEQ, 8, generator=torch.Generator().manual_seed(1))
    params = {"embed_dim": 8, "num_heads": 2, "causal": True, "seed": 0}
    inputs = {"tensor": x, "mask": _CPU_MASK}

    cpu = _run_on(node_cls(), "cpu", inputs, params)
    mps = _run_on(node_cls(), "mps", inputs, params)

    torch.testing.assert_close(mps["weights"], cpu["weights"], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(mps["output"], cpu["output"], atol=1e-5, rtol=1e-4)
    # The masks really applied: nothing attends above the diagonal or to the
    # last key.
    w = mps["weights"].reshape(-1, _SEQ, _SEQ)
    assert torch.all(w[:, torch.triu(torch.ones(_SEQ, _SEQ, dtype=torch.bool), 1)] == 0)
    assert torch.all(w[..., -1] == 0)


@requires_mps
def test_cross_attention_cpu_mask_on_mps_matches_cpu():
    g = torch.Generator().manual_seed(2)
    mask = torch.zeros(4, 6, dtype=torch.bool)
    mask[:, -1] = True
    inputs = {
        "query": torch.randn(4, 8, generator=g),
        "context": torch.randn(6, 8, generator=g),
        "mask": mask,
    }
    params = {"embed_dim": 8, "num_heads": 2, "seed": 0}

    cpu = _run_on(EduCrossAttentionNode(), "cpu", inputs, params)
    mps = _run_on(EduCrossAttentionNode(), "mps", inputs, params)

    torch.testing.assert_close(mps["weights"], cpu["weights"], atol=1e-5, rtol=1e-4)
    assert torch.all(mps["weights"][..., -1] == 0)


@requires_mps
def test_multi_head_causal_example_runs_on_mps():
    payload = json.loads(_MULTI_HEAD_CAUSAL.read_text(encoding="utf-8"))
    result = asyncio.run(
        execute_graph(
            payload["nodes"],
            payload["edges"],
            context=ExecutionContext(device="mps"),
            error_mode="fail_fast",
        )
    )
    errors = {k: v for k, v in (result.get("errors") or {}).items() if v}
    assert not errors
