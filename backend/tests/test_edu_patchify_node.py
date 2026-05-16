"""Tests for EduPatchifyNode (chapter pack C6)."""

from __future__ import annotations

import pytest
import torch

from cdui_plugins.c6.nodes.edu_patchify_node import EduPatchifyNode


def _run(image, **params):
    base = {"patch_size": 4, "flatten": True}
    base.update(params)
    return EduPatchifyNode().execute({"image": image}, base)


def test_node_metadata():
    assert EduPatchifyNode.NODE_NAME == "EduPatchify"
    assert EduPatchifyNode.CATEGORY == "Transformer"
    out_names = [p.name for p in EduPatchifyNode.define_outputs()]
    assert out_names == ["tokens", "grid"]


def test_token_count_matches_grid():
    # 8×8 image, patch_size=4 → 2×2 = 4 patches
    img = torch.randn(1, 3, 8, 8)
    res = _run(img, patch_size=4)
    assert res["tokens"].shape == (1, 4, 3 * 4 * 4)
    assert res["grid"].tolist() == [2, 2]


def test_flatten_false_keeps_patch_structure():
    img = torch.randn(2, 3, 8, 8)
    res = _run(img, patch_size=4, flatten=False)
    assert res["tokens"].shape == (2, 4, 3, 4, 4)


def test_promotes_3d_input_to_batch_one():
    img = torch.randn(3, 8, 8)
    res = _run(img, patch_size=4)
    assert res["tokens"].shape[0] == 1


def test_rejects_non_divisible_dimensions():
    img = torch.randn(1, 3, 9, 8)  # 9 not divisible by 4
    with pytest.raises(ValueError, match="divisible"):
        _run(img, patch_size=4)


def test_first_patch_is_top_left_corner():
    """flatten=False makes it easy to check patches are read in row-major order."""
    img = torch.zeros(1, 1, 4, 4)
    # Mark top-left patch with 7s.
    img[0, 0, 0:2, 0:2] = 7.0
    # Top-right patch with 9s.
    img[0, 0, 0:2, 2:4] = 9.0
    res = _run(img, patch_size=2, flatten=False)
    tokens = res["tokens"]  # [1, 4, 1, 2, 2]
    assert tokens[0, 0].squeeze().tolist() == [[7.0, 7.0], [7.0, 7.0]]
    assert tokens[0, 1].squeeze().tolist() == [[9.0, 9.0], [9.0, 9.0]]


def test_rejects_4d_with_wrong_dims():
    img = torch.randn(2, 3, 4)  # 3D but wrong shape after squeeze
    # 3D promoted to [1, 2, 3, 4]; W=4 div by 2 works, H=3 not div by 2.
    with pytest.raises(ValueError, match="divisible"):
        _run(img, patch_size=2)


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        EduPatchifyNode().execute({}, {})
