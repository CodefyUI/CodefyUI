"""Tests for ModelSaverNode and ModelLoaderNode."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch
import torch.nn as nn

from app.config import settings
from app.nodes.io.model_loader_node import ModelLoaderNode
from app.nodes.io.model_saver_node import ModelSaverNode


def _model(seed=0):
    torch.manual_seed(seed)
    return nn.Linear(4, 2)


def test_saver_metadata():
    assert ModelSaverNode.NODE_NAME == "ModelSaver"
    assert ModelSaverNode.CATEGORY == "IO"


def test_loader_metadata():
    assert ModelLoaderNode.NODE_NAME == "ModelLoader"
    assert ModelLoaderNode.CATEGORY == "IO"


def test_save_and_load_state_dict_roundtrip():
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = "_roundtrip_test.pt"
    try:
        original = _model(seed=42)
        save_res = ModelSaverNode().execute(
            {"model": original},
            {"path": target_path, "save_mode": "state_dict", "format": "pytorch"},
        )
        assert save_res["model"] is original

        empty_model = nn.Linear(4, 2)
        load_res = ModelLoaderNode().execute(
            {"model": empty_model},
            {"path": target_path, "load_mode": "state_dict", "device": "cpu", "strict": True},
        )
        loaded = load_res["model"]
        x = torch.randn(1, 4)
        assert torch.allclose(original(x), loaded(x))
    finally:
        (settings.MODELS_DIR / target_path).unlink(missing_ok=True)


def test_save_safetensors():
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        model = _model()
        res = ModelSaverNode().execute(
            {"model": model},
            {"path": "_st_test.safetensors", "save_mode": "state_dict", "format": "safetensors"},
        )
        assert res["path"].endswith(".safetensors")
    finally:
        for f in settings.MODELS_DIR.glob("_st_test*"):
            f.unlink(missing_ok=True)


def test_safetensors_with_full_model_raises():
    with pytest.raises(ValueError, match="state_dict"):
        ModelSaverNode().execute(
            {"model": _model()},
            {"path": "_should_fail.safetensors", "save_mode": "full_model", "format": "safetensors"},
        )


def test_load_state_dict_requires_model_input():
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = "_no_model_test.pt"
    try:
        ModelSaverNode().execute(
            {"model": _model()},
            {"path": target_path, "save_mode": "state_dict", "format": "pytorch"},
        )
        with pytest.raises(ValueError, match="state_dict mode"):
            ModelLoaderNode().execute(
                {},
                {"path": target_path, "load_mode": "state_dict", "device": "cpu", "strict": True},
            )
    finally:
        (settings.MODELS_DIR / target_path).unlink(missing_ok=True)


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        ModelLoaderNode().execute(
            {"model": _model()},
            {"path": "_definitely_missing.pt", "load_mode": "state_dict", "device": "cpu", "strict": True},
        )


def test_save_outside_data_dir_raises(tmp_path):
    bad = tmp_path / "bad.pt"
    with pytest.raises(ValueError, match="within the project"):
        ModelSaverNode().execute(
            {"model": _model()},
            {"path": str(bad), "save_mode": "state_dict", "format": "pytorch"},
        )


# ── full_model mode (#222) ───────────────────────────────────────────────
#
# The mode shipped as ``torch.load(..., weights_only=True)`` over a file that
# is by definition a pickled ``nn.Module``, which the restricted unpickler
# refuses -- so it failed for every input it was written to accept, and no
# test covered it, which is why the suite stayed green. These are that test.


#: Set by :func:`_detonate` if the restricted unpickler ever runs the payload
#: in :class:`_Detonator`. A module-level flag rather than a file write or a
#: subprocess, so a regression shows up as a failed assertion instead of as a
#: side effect somebody has to go looking for.
_DETONATED = False


def _detonate() -> None:
    global _DETONATED
    _DETONATED = True


class _Detonator:
    """A pickle that executes on load -- exactly what weights_only stops."""

    def __reduce__(self):
        return (_detonate, ())


class _NotATorchLayer(nn.Module):
    """A real ``nn.Module``, defined outside torch, so outside the allowlist."""

    def forward(self, x):  # pragma: no cover - never reached; load is refused
        return x


@contextmanager
def _saved(name: str):
    """Yield a MODELS_DIR-relative *name*, and delete the file afterwards."""
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        yield name
    finally:
        (settings.MODELS_DIR / name).unlink(missing_ok=True)


def test_full_model_round_trips_through_saver_and_loader():
    """The headline: a saved whole model comes back, and it is the same model.

    Deliberately a multi-layer model rather than the bare ``Linear`` the
    other tests use -- the allowlist is derived by walking subclasses, and a
    single layer type would not notice a walk that stopped early.
    """
    with _saved("_full_model_roundtrip.pt") as path:
        torch.manual_seed(7)
        original = nn.Sequential(
            nn.Conv2d(1, 4, 3), nn.BatchNorm2d(4), nn.ReLU(),
            nn.MaxPool2d(2), nn.Flatten(), nn.Linear(4 * 3 * 3, 2),
        )
        original.eval()
        ModelSaverNode().execute(
            {"model": original},
            {"path": path, "save_mode": "full_model", "format": "pytorch"},
        )

        # No `model` input: rebuilding the architecture is the whole point.
        loaded = ModelLoaderNode().execute(
            {},
            {"path": path, "load_mode": "full_model", "device": "cpu"},
        )["model"]
        loaded.eval()

        assert isinstance(loaded, nn.Sequential)
        x = torch.randn(2, 1, 8, 8)
        with torch.no_grad():
            assert torch.allclose(original(x), loaded(x))


def test_full_model_load_still_refuses_arbitrary_pickle():
    """The guarantee the mode must not buy its way out of.

    ``safe_globals`` widens the restricted unpickler; it does not disable it.
    A payload whose ``__reduce__`` names ``os.system`` is the thing
    ``weights_only=True`` exists to stop, and it has to stay stopped -- if
    this ever passes, the fix for #222 has become the vulnerability #222 was
    careful not to introduce.
    """
    with _saved("_full_model_hostile.pt") as path:
        torch.save({"payload": _Detonator()}, str(settings.MODELS_DIR / path))

        with pytest.raises(ValueError) as excinfo:
            ModelLoaderNode().execute(
                {},
                {"path": path, "load_mode": "full_model", "device": "cpu"},
            )
        assert not _DETONATED, "the restricted unpickler executed the payload"
        assert "could not load" in str(excinfo.value)


def test_full_model_refusal_names_what_it_stopped_on_and_what_to_do():
    """#222's other half: the failure has to be readable.

    A class the allowlist does not cover is refused, and the message has to
    say which class, that a safe default is why, and what to do instead --
    rather than the raw unpickler traceback the mode used to produce.
    """
    with _saved("_full_model_foreign.pt") as path:
        torch.save(_NotATorchLayer(), str(settings.MODELS_DIR / path))

        with pytest.raises(ValueError) as excinfo:
            ModelLoaderNode().execute(
                {},
                {"path": path, "load_mode": "full_model", "device": "cpu"},
            )

        message = str(excinfo.value)
        # What it stopped on, by name.
        assert "_NotATorchLayer" in message
        # Why, in terms the user can act on rather than an unpickler error.
        assert "weights_only=True" in message
        # And the way out, which is the mode that actually works.
        assert "load_mode=state_dict" in message
        assert "ModelSaver" in message


def test_full_model_allowlist_covers_torchs_layers_and_nothing_else():
    """The allowlist is derived, so assert the derivation, not a literal list.

    Two properties: it reaches classes ``torch.nn`` does not re-export at the
    top level (``MultiheadAttention.out_proj`` is one of those, and a walk
    that only read ``vars(torch.nn)`` would miss it), and it admits nothing
    defined outside torch -- a plugin class named ``Linear`` is not torch's.
    """
    from app.nodes.io.model_loader_node import torch_nn_layer_globals

    allowed = torch_nn_layer_globals()
    names = {f"{cls.__module__}.{cls.__qualname__}" for cls in allowed}

    assert nn.Module in allowed
    assert nn.Sequential in allowed
    assert "torch.nn.modules.linear.NonDynamicallyQuantizableLinear" in names
    assert _NotATorchLayer not in allowed
    assert all(name.startswith("torch.nn.") for name in names)
    # Cached: the walk runs once and hands back the same list object.
    assert torch_nn_layer_globals() is allowed
