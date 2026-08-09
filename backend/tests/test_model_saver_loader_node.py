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


# ── full_model saving (#283) ─────────────────────────────────────────────
#
# The other end of the same round trip. ``Reshape``, ``SelectIndex`` and
# ``TransformerEncoderBlock`` -- and four more nobody had noticed -- used to
# be classes defined INSIDE a function, so pickle had no name to write for
# them and ``save_mode="full_model"`` died with
#
#     AttributeError: Can't pickle local object 'Reshape.__new__.<locals>.Mod'
#
# which names an implementation detail and tells the user nothing. #283 asked
# for either a comprehensible refusal or the removal of the impossibility. The
# closure turned out to be incidental -- every one already took its
# configuration through ``__init__`` arguments, and the nesting only kept
# ``import torch`` off the node module's import path -- so both landed: the
# classes moved to module scope, and whatever is still unpicklable is refused
# by name.


#: One config per Sequential-compatible wrapper, keyed by the layer ``type``
#: string the editor writes. All seven, not the three #283 named: they shared
#: one defect and would have shared the next one.
WRAPPER_LAYER_CONFIGS = [
    {"type": "Reshape", "shape": "4,4"},
    {"type": "SelectIndex", "dim": 1, "index": 0},
    {"type": "TransformerEncoder", "d_model": 4, "nhead": 2, "num_layers": 1,
     "dim_feedforward": 8},
    {"type": "TransformerDecoder", "d_model": 4, "nhead": 2, "num_layers": 1,
     "dim_feedforward": 8},
    {"type": "LSTM", "input_size": 4, "hidden_size": 3, "batch_first": True},
    {"type": "GRU", "input_size": 4, "hidden_size": 3, "batch_first": True},
    {"type": "MultiHeadAttention", "embed_dim": 4, "num_heads": 2},
]


def _wrapper_graph_spec() -> dict:
    """A layer-editor graph exercising the three wrappers #283 named."""
    return {
        "version": 2,
        "nodes": [
            {"id": "in", "type": "Input", "ports": [{"id": "p_x", "name": "x"}]},
            {"id": "r", "type": "Reshape", "params": {"shape": "4,4"}},
            {"id": "t", "type": "TransformerEncoder",
             "params": {"d_model": 4, "nhead": 2, "num_layers": 1, "dim_feedforward": 8}},
            {"id": "s", "type": "SelectIndex", "params": {"dim": 1, "index": 0}},
            {"id": "l", "type": "Linear", "params": {"in_features": 4, "out_features": 2}},
            {"id": "out", "type": "Output", "ports": [{"id": "p_y", "name": "y"}]},
        ],
        "edges": [
            {"id": "e1", "source": "in", "sourceHandle": "p_x", "target": "r"},
            {"id": "e2", "source": "r", "target": "t"},
            {"id": "e3", "source": "t", "target": "s"},
            {"id": "e4", "source": "s", "target": "l"},
            {"id": "e5", "source": "l", "target": "out", "targetHandle": "p_y"},
        ],
    }


@pytest.mark.parametrize(
    "cfg", WRAPPER_LAYER_CONFIGS, ids=[c["type"] for c in WRAPPER_LAYER_CONFIGS],
)
def test_every_sequential_wrapper_is_a_class_pickle_can_name(cfg):
    """The property that was missing, asserted the way pickle asks it.

    ``pickle`` stores a class as ``__module__`` + ``__qualname__`` and gets it
    back by importing that module and walking that attribute path. A class
    built inside a function fails the walk -- its qualname carries a literal
    ``<locals>`` segment -- which is the whole of #283. Asserting the walk
    rather than the absence of that string also catches any other way of
    being unreachable.
    """
    import pickle

    from app.nodes.io.model_saver_node import _is_nameable
    from app.nodes.utility.sequential_node import _build_layer

    module = _build_layer(dict(cfg))
    cls = type(module)

    assert "<locals>" not in cls.__qualname__, cls.__qualname__
    assert _is_nameable(cls), f"{cls.__module__}.{cls.__qualname__} is not importable"
    assert pickle.loads(pickle.dumps(module)) is not None


@pytest.mark.parametrize(
    ("cfg", "prefix"),
    [
        ({"type": "TransformerEncoder", "d_model": 4, "nhead": 2}, "encoder."),
        ({"type": "TransformerDecoder", "d_model": 4, "nhead": 2}, "decoder."),
        ({"type": "LSTM", "input_size": 4, "hidden_size": 3}, "lstm."),
        ({"type": "GRU", "input_size": 4, "hidden_size": 3}, "gru."),
        ({"type": "MultiHeadAttention", "embed_dim": 4, "num_heads": 2}, "attn."),
    ],
    ids=["TransformerEncoder", "TransformerDecoder", "LSTM", "GRU", "MultiHeadAttention"],
)
def test_moving_the_wrappers_kept_their_state_dict_keys(cfg, prefix):
    """Attribute names are a file format here, not an implementation detail.

    They are the ``state_dict`` key prefixes, so renaming ``self.encoder``
    while hoisting the class would have silently invalidated every checkpoint
    written before this change -- surfacing as a pile of missing keys under
    ``strict=True``, one node away from the cause.
    """
    from app.nodes.utility.sequential_node import _build_layer

    keys = list(_build_layer(dict(cfg)).state_dict())
    assert keys, cfg["type"]
    assert all(k.startswith(prefix) for k in keys), keys


def test_full_model_saves_a_graph_built_from_the_wrappers():
    """#283's headline: the save that used to raise ``AttributeError``.

    Verified by reading the file back with plain torch rather than through
    ``ModelLoader``, which refuses it by design -- the restricted unpickler
    reconstructs ``torch.nn``'s own classes only (#222), and see the advisory
    test below. What #283 fixes is that a valid file gets written at all.
    """
    import json

    from app.nodes.utility.sequential_node import SequentialModelNode

    with _saved("_full_model_wrappers.pt") as path:
        torch.manual_seed(11)
        model = SequentialModelNode().build_module(
            {"layers": json.dumps(_wrapper_graph_spec())},
        )
        model.eval()

        ModelSaverNode().execute(
            {"model": model},
            {"path": path, "save_mode": "full_model", "format": "pytorch"},
        )

        saved = settings.MODELS_DIR / path
        assert saved.exists() and saved.stat().st_size > 0

        # weights_only=False is safe here and only here: this process wrote
        # the file three lines ago.
        reloaded = torch.load(str(saved), map_location="cpu", weights_only=False)
        reloaded.eval()
        x = torch.randn(2, 16)
        with torch.no_grad():
            assert torch.allclose(model(x=x), reloaded(x=x))


def _function_local_module() -> nn.Module:
    """An ``nn.Module`` whose class is defined inside a function.

    Exactly the shape the wrappers had, kept as a fixture now that they no
    longer have it: a plugin or a custom node can still be written this way,
    and the refusal is what those users will meet.
    """
    class Mod(nn.Module):
        def forward(self, x):  # pragma: no cover - never reached
            return x

    return Mod()


def test_full_model_refuses_a_function_local_class_by_name():
    """The refusal #283 asked for: which class, why, and what to do instead.

    Contrast with what it replaced -- ``Can't pickle local object
    'Reshape.__new__.<locals>.Mod'`` -- which names an internal and offers no
    way forward.
    """
    with _saved("_full_model_local_class.pt") as path:
        model = nn.Sequential(nn.Linear(2, 2), _function_local_module())

        with pytest.raises(ValueError) as excinfo:
            ModelSaverNode().execute(
                {"model": model},
                {"path": path, "save_mode": "full_model", "format": "pytorch"},
            )

        message = str(excinfo.value)
        assert "<locals>" in message                      # which class
        assert "defined inside a function" in message     # why
        assert "save_mode=state_dict" in message          # what to do instead
        # Refused BEFORE the write, so there is no half-written file to clean
        # up and no path handed downstream that points at rubbish.
        assert not (settings.MODELS_DIR / path).exists()


def test_state_dict_mode_is_unaffected_by_an_unpicklable_class():
    """The default path never had this problem and must not acquire one.

    ``state_dict`` stores tensors, not the objects around them, so where a
    layer class was defined is irrelevant to it -- and the #283 guard runs on
    the ``full_model`` branch only.
    """
    with _saved("_state_dict_local_class.pt") as path:
        model = nn.Sequential(nn.Linear(2, 2), _function_local_module())
        res = ModelSaverNode().execute(
            {"model": model},
            {"path": path, "save_mode": "state_dict", "format": "pytorch"},
        )
        assert (settings.MODELS_DIR / path).exists()
        assert "__log__" not in res


def test_the_saver_says_when_the_file_will_not_load_back():
    """A save that succeeds and cannot be undone is a worse trap than one that
    refuses.

    #283 made these models saveable; it did not make them loadable, because
    the loader rebuilds ``torch.nn`` classes only (#222) and neither
    ``GraphModelModule`` nor the wrappers are torch's. The node says so at
    save time rather than letting a user find out one node later.
    """
    import json

    from app.nodes.utility.sequential_node import SequentialModelNode

    with _saved("_full_model_advisory.pt") as path:
        model = SequentialModelNode().build_module(
            {"layers": json.dumps(_wrapper_graph_spec())},
        )
        res = ModelSaverNode().execute(
            {"model": model},
            {"path": path, "save_mode": "full_model", "format": "pytorch"},
        )

        note = res.get("__log__", "")
        assert "cannot be read back by ModelLoader in full_model mode" in note
        assert "GraphModelModule" in note
        assert "save_mode=state_dict" in note
        # Advisory, not a refusal: the file is written, and it is valid.
        assert (settings.MODELS_DIR / path).exists()


def test_a_stock_torch_model_gets_no_advisory():
    """The note has to be absent when the round trip does work.

    ``test_full_model_round_trips_through_saver_and_loader`` above proves this
    shape loads back; a warning on it would be noise, and noise is how a
    warning stops being read.
    """
    with _saved("_full_model_no_advisory.pt") as path:
        model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU(), nn.Flatten())
        res = ModelSaverNode().execute(
            {"model": model},
            {"path": path, "save_mode": "full_model", "format": "pytorch"},
        )
        assert "__log__" not in res
