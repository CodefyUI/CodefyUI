"""Tests for ModelSaverNode and ModelLoaderNode."""

from __future__ import annotations

import sys
import textwrap
from contextlib import contextmanager
from typing import Any

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
    """The torch half is derived, so assert the derivation, not a literal list.

    Two properties: it reaches classes ``torch.nn`` does not re-export at the
    top level (``MultiheadAttention.out_proj`` is one of those, and a walk
    that only read ``vars(torch.nn)`` would miss it), and it admits nothing
    defined outside torch -- a plugin class named ``Linear`` is not torch's.

    The CodefyUI half is a separate function and a separate argument (#288);
    the two must not blur into each other, which is why this still asserts
    every name here starts with ``torch.nn.``.
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

    Read back with PLAIN torch rather than through ``ModelLoader``, which is
    the property #283 bought and #288 does not change: the file is valid
    outside CodefyUI, given ``weights_only=False`` and the classes importable.
    That it now also loads THROUGH ``ModelLoader`` is #288, asserted in
    ``test_a_layer_editor_model_round_trips_through_full_model`` below.
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

    Since #288 the class that triggers this is no longer one of CodefyUI's own
    -- those load back. It is a class from a custom node, a plugin, or a script
    somebody wrote, which is what ``_NotATorchLayer`` stands in for here. The
    node says so at save time rather than letting a user find out one node
    later, and the note is derived from the same allowlist the loader reads, so
    it cannot claim a refusal the loader would not make.
    """
    with _saved("_full_model_advisory.pt") as path:
        model = nn.Sequential(nn.Linear(2, 2), _NotATorchLayer())
        res = ModelSaverNode().execute(
            {"model": model},
            {"path": path, "save_mode": "full_model", "format": "pytorch"},
        )

        note = res.get("__log__", "")
        assert "cannot be read back by ModelLoader in full_model mode" in note
        assert "_NotATorchLayer" in note
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


# ── full_model round trips inside CodefyUI (#288) ────────────────────────
#
# The seam #282 and #287 left open. Each had fixed one half of the round trip
# and neither could close it: the loader admitted ``torch.nn``'s classes and
# the saver could write CodefyUI's, so the product wrote a file it then
# refused -- including every layer-editor model, all of which are a
# ``GraphModelModule``. #288 decided the trust question (intranet deployment,
# a tight curated allowlist of audited attribute-restore-only classes, never a
# wildcard) and these are the tests of that decision. They come in three
# kinds: the round trips now work; the allowlist is still EXACT, so a class it
# does not name is refused even from the same module as one it does; and the
# admission rule is re-derived rather than trusted to a comment.


#: Input shape per wrapper, keyed by the layer ``type`` string. ``Reshape``
#: takes a flat vector and makes it 2-D; every other wrapper is a sequence
#: layer expecting ``(batch, seq, feature)`` with feature = 4 to match
#: ``WRAPPER_LAYER_CONFIGS``.
WRAPPER_INPUT_SHAPES = {"Reshape": (2, 16)}
_DEFAULT_WRAPPER_SHAPE = (2, 3, 4)

#: The two wrappers that #288's allowlist admits and that still do not load,
#: for a reason one level below them: ``nn.TransformerEncoderLayer`` stores its
#: activation as ``torch.nn.functional.relu`` whatever spelling it was
#: constructed with, and #222 left ``torch.nn.functional`` out of the allowlist
#: deliberately -- functions are callables the unpickler may be asked to
#: INVOKE, and that namespace also holds ``handle_torch_function``, which
#: dispatches to an arbitrary object's ``__torch_function__``. #288 records
#: that as a decision still open and does NOT take it, so these two are xfail
#: rather than quietly dropped from the parameter list: the day the functional
#: question is answered, these turn green and say so.
WRAPPERS_BLOCKED_BY_FUNCTIONAL = {"TransformerEncoder", "TransformerDecoder"}


@pytest.mark.parametrize(
    "cfg",
    [
        pytest.param(
            cfg,
            marks=pytest.mark.xfail(
                strict=True,
                reason="stores torch.nn.functional.relu; the functional half "
                       "of the allowlist is still an open decision (#288)",
            ),
        ) if cfg["type"] in WRAPPERS_BLOCKED_BY_FUNCTIONAL else cfg
        for cfg in WRAPPER_LAYER_CONFIGS
    ],
    ids=[c["type"] for c in WRAPPER_LAYER_CONFIGS],
)
def test_every_sequential_wrapper_round_trips_through_full_model(cfg):
    """All seven of #283's hoisted classes, one per case.

    Parameterized over ``WRAPPER_LAYER_CONFIGS`` -- the same list the
    picklability tests above use -- rather than a second hand-written list, so
    a wrapper added to ``sequential_modules`` and wired into ``_build_layer``
    cannot be covered by one and missed by the other.

    ``type(loaded) is type(original)`` is the assertion that matters as much as
    the numbers: the unpickler either reconstructed the real class or refused,
    and a forward comparison alone would also pass on something that merely
    behaved similarly.
    """
    from app.nodes.utility.sequential_node import _build_layer

    with _saved(f"_full_model_wrapper_{cfg['type']}.pt") as path:
        torch.manual_seed(13)
        original = _build_layer(dict(cfg))
        original.eval()

        ModelSaverNode().execute(
            {"model": original},
            {"path": path, "save_mode": "full_model", "format": "pytorch"},
        )
        loaded = ModelLoaderNode().execute(
            {},
            {"path": path, "load_mode": "full_model", "device": "cpu"},
        )["model"]
        loaded.eval()

        assert type(loaded) is type(original)
        x = torch.randn(*WRAPPER_INPUT_SHAPES.get(cfg["type"], _DEFAULT_WRAPPER_SHAPE))
        with torch.no_grad():
            assert torch.allclose(original(x), loaded(x))


def _loadable_wrapper_graph_spec() -> dict:
    """``_wrapper_graph_spec`` minus the transformer block.

    Same three-wrapper shape, with ``TransformerEncoder`` swapped for a plain
    ``Linear``: that block stores ``torch.nn.functional.relu`` one level down
    and so cannot come back yet (see ``WRAPPERS_BLOCKED_BY_FUNCTIONAL``). Kept
    as a separate fixture rather than changing the #283 one, whose whole point
    is the wrappers it names.
    """
    spec = _wrapper_graph_spec()
    for node in spec["nodes"]:
        if node["type"] == "TransformerEncoder":
            node["type"] = "Linear"
            node["params"] = {"in_features": 4, "out_features": 4}
    return spec


def test_a_layer_editor_model_round_trips_through_full_model():
    """#288's headline: the shape that could be saved and not loaded.

    A ``GraphModelModule`` holding the wrappers -- what
    ``test_full_model_saves_a_graph_built_from_the_wrappers`` writes, and what
    could previously only be read back with ``weights_only=False`` outside
    CodefyUI.
    """
    import json

    from app.nodes.utility.graph_model import GraphModelModule
    from app.nodes.utility.sequential_node import SequentialModelNode

    with _saved("_full_model_graph_roundtrip.pt") as path:
        torch.manual_seed(17)
        original = SequentialModelNode().build_module(
            {"layers": json.dumps(_loadable_wrapper_graph_spec())},
        )
        original.eval()

        ModelSaverNode().execute(
            {"model": original},
            {"path": path, "save_mode": "full_model", "format": "pytorch"},
        )
        loaded = ModelLoaderNode().execute(
            {},
            {"path": path, "load_mode": "full_model", "device": "cpu"},
        )["model"]
        loaded.eval()

        assert isinstance(loaded, GraphModelModule)
        x = torch.randn(2, 16)
        with torch.no_grad():
            assert torch.allclose(original(x=x), loaded(x=x))


def test_a_causal_lm_round_trips_through_full_model():
    """The other model family: the LLM wave's ``CausalLMModule`` (#289).

    Listed on the allowlist together with the three classes it is built from
    (``_DecoderBlock``, ``_CausalSelfAttention``, ``_FeedForward``) -- admitting
    only the outer class would have refused the file one level down, which is
    the mistake a list written from the node's public surface would make.

    Tiny on purpose: the point is the class graph, not the parameter count.
    """
    from app.nodes.llm.causal_lm_model_node import CausalLMModelNode, CausalLMModule

    with _saved("_full_model_causal_lm.pt") as path:
        original = CausalLMModelNode().build_module({
            "vocab_size": 16, "d_model": 8, "n_layers": 2, "n_heads": 2,
            "d_ff": 16, "max_seq_len": 8, "seed": 3,
        })
        original.eval()

        ModelSaverNode().execute(
            {"model": original},
            {"path": path, "save_mode": "full_model", "format": "pytorch"},
        )
        loaded = ModelLoaderNode().execute(
            {},
            {"path": path, "load_mode": "full_model", "device": "cpu"},
        )["model"]
        loaded.eval()

        assert isinstance(loaded, CausalLMModule)
        # Public attributes downstream nodes read, so they have to survive the
        # __dict__ restore and not merely the parameters.
        assert loaded.max_seq_len == original.max_seq_len
        assert loaded.vocab_size == original.vocab_size
        ids = torch.randint(0, 16, (2, 5))
        with torch.no_grad():
            assert torch.allclose(original(ids), loaded(ids))


def _diffusion_unet():
    from app.nodes.diffusion.diffusion_unet_node import DiffusionUNetNode

    return DiffusionUNetNode().build_module({
        "in_channels": 1, "base_channels": 8, "channel_mult": "1,2",
        "time_emb_dim": 8, "num_groups": 4, "seed": 0,
    })


def _vla_policy():
    from app.nodes.vla.vla_model_node import VLAModule

    return VLAModule(
        image_size=32, patch_size=8, d_model=16, n_layers=1, n_heads=2,
        expert_layers=1, chunk=2, action_dim=3, max_text_len=4,
        head_type="flow_matching", vision_stem="patchify", dropout=0.0,
    )


def _moe_layer():
    from app.nodes.transformer.moe_layer_node import _MoELayer

    return _MoELayer(num_experts=2, top_k=1, hidden_dim=8, expert_hidden_dim=16)


def _seeded_rnn_cell():
    from app.nodes.rnn.rnn_cell_node import _SeededRNNCell

    return _SeededRNNCell(input_size=4, hidden_size=4, nonlinearity="tanh", seed=0)


def _reward_head():
    from app.nodes.rl.reward_model_node import _RewardHead

    return _RewardHead(input_dim=4, hidden_dim=8)


def _lm_loss():
    from app.nodes.llm.lm_cross_entropy_loss_node import LMCrossEntropyLoss

    return LMCrossEntropyLoss()


def _vla_loss():
    from app.nodes.vla.vla_model_node import VLABehaviorLoss

    return VLABehaviorLoss("flow_matching")


@pytest.mark.parametrize(
    "build",
    [_diffusion_unet, _vla_policy, _moe_layer, _seeded_rnn_cell, _reward_head,
     _lm_loss, _vla_loss],
    ids=["DiffusionUNet", "VLAModule", "MoELayer", "SeededRNNCell",
         "RewardHead", "LMCrossEntropyLoss", "VLABehaviorLoss"],
)
def test_the_remaining_admitted_families_round_trip(build):
    """Being ON the allowlist and actually loading are different claims.

    ``test_the_allowlist_covers_every_codefyui_module_class`` proves the list is
    complete; nothing there proves the entries WORK. The transformer wrappers
    are the cautionary case -- admitted, and still refused for something one
    level down -- so every other family gets exercised rather than assumed,
    including the two loss modules and the composite classes (``VLAModule``
    covers ``_EncoderBlock`` / ``_ExpertBlock``, ``DiffusionUNet`` covers
    ``_ResBlockModule`` / ``_TimestepMLP``, ``_MoELayer`` covers ``_ExpertFFN``).

    Deliberately tiny, and compared by ``state_dict`` rather than a forward
    pass: these take different input signatures (an image and a timestep, an
    image and instruction bytes, logits and targets), and what #288 changed is
    the reconstruction, not the arithmetic.
    """
    with _saved("_full_model_family.pt") as path:
        torch.manual_seed(23)
        original = build()

        ModelSaverNode().execute(
            {"model": original},
            {"path": path, "save_mode": "full_model", "format": "pytorch"},
        )
        loaded = ModelLoaderNode().execute(
            {},
            {"path": path, "load_mode": "full_model", "device": "cpu"},
        )["model"]

        assert type(loaded) is type(original)
        before, after = original.state_dict(), loaded.state_dict()
        assert list(before) == list(after)
        assert all(torch.equal(before[k], after[k]) for k in before)


def test_a_full_model_save_of_our_own_classes_says_what_the_file_needs():
    """The advisory #287 wrote, updated to what #288 made true.

    It used to say the file could not be read back. It can, so saying so would
    be a lie the loader disproves one node later. What is still worth one line
    is that the file stopped being self-contained: an older CodefyUI refuses
    it, and plain torch needs both ``weights_only=False`` and the class
    definitions. Pinned because the whole value of the note is its wording.
    """
    import json

    from app.nodes.utility.sequential_node import SequentialModelNode

    with _saved("_full_model_ours_note.pt") as path:
        model = SequentialModelNode().build_module(
            {"layers": json.dumps(_loadable_wrapper_graph_spec())},
        )
        res = ModelSaverNode().execute(
            {"model": model},
            {"path": path, "save_mode": "full_model", "format": "pytorch"},
        )

        note = res.get("__log__", "")
        # The new truth, not the old warning.
        assert "reads this file back in full_model mode" in note
        assert "cannot be read back" not in note
        # Named, so the user can see which classes carry the condition.
        assert "GraphModelModule" in note
        # And the conditions themselves.
        assert "older than this one refuses it" in note
        assert "weights_only=False" in note
        assert "save_mode=state_dict" in note


def test_the_note_names_a_refused_function_not_only_a_refused_class():
    """The note must not promise a round trip the loader will not honour.

    ``TransformerEncoderBlock`` is ON the allowlist, and a model containing one
    still does not load: ``nn.TransformerEncoderLayer`` stores its activation as
    ``torch.nn.functional.relu``, and #222 kept functions out of the allowlist
    on purpose. A note derived from the class walk alone would have said "reads
    this file back" here -- the exact trap #288 exists to remove, moved one
    category over. So the walk looks at function-valued attributes too, and
    names what it found.
    """
    from app.nodes.utility.sequential_node import _build_layer

    with _saved("_full_model_functional_note.pt") as path:
        model = _build_layer(
            {"type": "TransformerEncoder", "d_model": 4, "nhead": 2,
             "num_layers": 1, "dim_feedforward": 8},
        )
        res = ModelSaverNode().execute(
            {"model": model},
            {"path": path, "save_mode": "full_model", "format": "pytorch"},
        )

        note = res.get("__log__", "")
        assert "cannot be read back by ModelLoader in full_model mode" in note
        assert "torch.nn.functional.relu" in note
        assert "no functions at all" in note

        # And the loader agrees, which is the property that makes the note
        # worth printing rather than a second opinion nobody checked.
        with pytest.raises(ValueError, match="torch.nn.functional.relu"):
            ModelLoaderNode().execute(
                {},
                {"path": path, "load_mode": "full_model", "device": "cpu"},
            )


@pytest.mark.parametrize(
    "payload",
    [{"w": torch.zeros(2)}, 5, [torch.zeros(2)], None],
    ids=["dict", "int", "list", "None"],
)
def test_a_non_module_on_the_model_port_still_gets_a_note(payload):
    """The note path must survive an input that is not an ``nn.Module``.

    ``ModelSaver.model`` is a MODEL port, and ``type_system`` lets an ANY-typed
    output into one -- ``CheckpointLoader.grad_scaler_state`` emits
    dict-or-None, and PythonScript / Switch / Reduce / GraphInput can pass
    anything at all. So a dict, an int or a None genuinely arrives here.

    The #288 function-attribute walk originally reached for ``vars(module)``,
    which raises ``TypeError`` on every one of these -- replacing the note with
    a traceback, before a ``torch.save`` that would have succeeded, where the
    pre-#288 code returned a warning and wrote a valid file. Found in review.
    """
    from app.nodes.io.model_saver_node import _reload_note

    level, text = _reload_note(payload)
    assert level == "warning"
    assert type(payload).__name__ in text


def test_saving_a_non_module_writes_the_file_and_warns():
    """The same regression at the node's edge rather than the helper's.

    A dict of tensors is a legitimate thing to hand ``full_model``: it pickles
    fine, so the node's job is to write it and say it will not come back
    through ``ModelLoader`` -- not to raise.
    """
    with _saved("_full_model_bare_dict.pt") as path:
        res = ModelSaverNode().execute(
            {"model": {"w": torch.zeros(2)}},
            {"path": path, "save_mode": "full_model", "format": "pytorch"},
        )
        assert (settings.MODELS_DIR / path).exists()
        assert "cannot be read back" in res.get("__log__", "")


def test_the_allowlist_is_exact_not_a_module_prefix():
    """A class the list does not name is refused even from a listed module.

    The cheap way to implement #288 would have been ``__module__.startswith
    ("app.")``, and it would have been wrong twice over: ``app.custom_nodes.*``
    is code the user uploaded, and any class a plugin or a script drops into an
    allowlisted module would ride in on the package's reputation. So the
    impostor here is installed INTO ``sequential_modules`` -- pickle can name
    it, ``_is_nameable`` agrees, the save succeeds -- and the load still has to
    refuse it, by name.
    """
    from app.nodes.utility import sequential_modules

    class _ImpostorBlock(nn.Module):
        def forward(self, x):  # pragma: no cover - never reached; load refuses
            return x

    # Make it genuinely reachable at that name, both for the pickler writing
    # the reference and for the unpickler resolving it -- otherwise this would
    # test an import failure, or #283's function-local refusal, instead of the
    # allowlist.
    _ImpostorBlock.__module__ = sequential_modules.__name__
    _ImpostorBlock.__qualname__ = "_ImpostorBlock"
    sequential_modules._ImpostorBlock = _ImpostorBlock
    try:
        with _saved("_full_model_impostor.pt") as path:
            ModelSaverNode().execute(
                {"model": nn.Sequential(nn.Linear(2, 2), _ImpostorBlock())},
                {"path": path, "save_mode": "full_model", "format": "pytorch"},
            )

            with pytest.raises(ValueError) as excinfo:
                ModelLoaderNode().execute(
                    {},
                    {"path": path, "load_mode": "full_model", "device": "cpu"},
                )
            message = str(excinfo.value)
            assert "_ImpostorBlock" in message
            assert "weights_only=True" in message
            assert "load_mode=state_dict" in message
    finally:
        del sequential_modules._ImpostorBlock


#: Bare builtin calls a reachable constructor must not make.
FORBIDDEN_BARE_CALLS = frozenset({
    "open", "eval", "exec", "compile", "__import__", "input", "breakpoint",
})

#: Any call rooted at one of these modules -- ``os.anything``, ``socket.anything``.
FORBIDDEN_CALL_ROOTS = frozenset({
    "os", "subprocess", "shutil", "socket", "requests", "urllib", "pickle",
    "shlex", "webbrowser", "ctypes", "importlib",
})

#: Specific dotted calls. ``torch.manual_seed`` is here and
#: ``<generator>.manual_seed`` deliberately is not: seeding a LOCAL
#: ``torch.Generator`` from file-chosen bytes is the harmless case (it only
#: changes tensors the constructor is about to overwrite), while reseeding the
#: global RNG would reach every other node in the run.
FORBIDDEN_DOTTED_CALLS = frozenset({
    "os.system", "torch.manual_seed", "torch.load", "torch.save",
    "torch.use_deterministic_algorithms", "pathlib.Path", "Path",
})

#: Prefixes for families of dotted calls (``torch.set_default_dtype``, ...).
FORBIDDEN_DOTTED_PREFIXES = ("torch.set_",)


def _dotted_callee(node: Any) -> str | None:
    """``foo.bar.baz`` for an attribute chain rooted at a plain name, else None.

    A call on something that is not a simple name chain -- ``self.layers[i]()``,
    ``build()()`` -- returns None and is not judged here. That is a deliberate
    limit of a static check, and the reason the docstring below still records a
    hand audit.
    """
    import ast

    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _forbidden_calls_reachable_from_constructor(cls: type) -> list[str]:
    """Dangerous calls in *cls*'s source, or in a helper its source reaches.

    Walked as an AST rather than grepped as text, because text is wrong twice
    over: ``_init_module_weights``'s DOCSTRING contains the words
    ``torch.manual_seed`` (explaining why it does not call it), and
    ``module.eval()`` -- torch's train/eval switch -- reads as a call to
    ``eval`` under any word-boundary regex. Both were false positives on the
    first attempt at this test.

    The class body alone is also not the whole constructor: ``CausalLMModule``
    delegates to ``_make_norm`` / ``_sinusoidal_table`` /
    ``_init_module_weights``. And the whole owning MODULE is the other wrong
    answer -- these files also hold the node classes, two of which legitimately
    call ``torch.manual_seed`` where the unpickler cannot reach it. So: the
    class, then every module-level function or class its source names, to a
    fixed point.
    """
    import ast
    import inspect

    module = sys.modules[cls.__module__]
    found: set[str] = set()
    seen: set[str] = set()
    pending: list[Any] = [cls]

    while pending:
        obj = pending.pop()
        name = getattr(obj, "__qualname__", None)
        if name is None or name in seen:
            continue
        seen.add(name)
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
        except (OSError, TypeError, SyntaxError):  # pragma: no cover
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                callee = node.func
                if isinstance(callee, ast.Name) and callee.id in FORBIDDEN_BARE_CALLS:
                    found.add(callee.id)
                    continue
                dotted = _dotted_callee(callee)
                if dotted is None:
                    continue
                if (
                    dotted in FORBIDDEN_DOTTED_CALLS
                    or dotted.split(".")[0] in FORBIDDEN_CALL_ROOTS
                    or dotted.startswith(FORBIDDEN_DOTTED_PREFIXES)
                ):
                    found.add(dotted)
            elif isinstance(node, ast.Name):
                # Follow the module-level helpers this source names.
                candidate = getattr(module, node.id, None)
                if (
                    candidate is not None
                    and getattr(candidate, "__qualname__", None) is not None
                    and getattr(candidate, "__module__", None) == cls.__module__
                ):
                    pending.append(candidate)

    return sorted(found)


def test_every_admitted_class_is_safe_to_reconstruct():
    """The admission rule, re-derived instead of trusted to the comment.

    The rule has two halves and this checks what is mechanically checkable in
    each. Renamed from ``..._is_attribute_restore_only`` in review of #288,
    because that name recorded a rule that was only half right.

    **Half one: no pickle hooks.** A ``__setstate__`` or ``__reduce__`` on an
    admitted class turns a name on the allowlist into a code path, in a file
    nobody reviewing that class would think to open.

    **Half two: the constructor is reachable.** torch's restricted unpickler
    implements REDUCE as ``result = func(*args)`` for user-allowed globals
    (``torch/_weights_only_unpickler.py``, around :415), so a crafted file can
    call an admitted ``cls(...)`` with arguments it chose -- this is NOT only
    ``__new__`` plus a ``__dict__`` update. The same has always been true of
    the torch half (#222): admitting ``nn.Linear`` admits ``nn.Linear(...)``.
    What is checkable is that no admitted constructor, and no module-level
    helper it reaches, calls anything on the filesystem / network / process or
    mutates global state -- see
    :func:`_forbidden_calls_reachable_from_constructor`.

    What NO test can prove is that running these constructors on nonsense
    arguments is merely useless rather than harmful (nor can a static walk see
    through a call on something that is not a plain name chain). That was
    hand-audited on 2026-08-12 across all 24 classes: each builds torch layers
    from numbers, so a hostile argument produces a raised exception or a large
    allocation -- a failed load, not a compromised one.

    ``nn.Module``'s own ``__setstate__`` is excluded from the hook search: it is
    torch's, it back-fills defaults on an old checkpoint, and the #222 half of
    the allowlist already trusts it.
    """
    from app.nodes.io.model_loader_node import codefyui_module_globals

    hooks = ("__reduce__", "__reduce_ex__", "__setstate__", "__getstate__",
             "__getnewargs__", "__getnewargs_ex__", "__new__")

    hook_offenders = {}
    call_offenders = {}
    for cls in codefyui_module_globals():
        name = f"{cls.__module__}.{cls.__qualname__}"
        own = [
            base for base in cls.__mro__
            if (getattr(base, "__module__", "") or "").startswith("app.")
        ]
        found = sorted({h for base in own for h in hooks if h in base.__dict__})
        if found:
            hook_offenders[name] = found

        hits = _forbidden_calls_reachable_from_constructor(cls)
        if hits:
            call_offenders[name] = hits

    assert not hook_offenders, (
        f"These classes are on the full_model allowlist and define pickle "
        f"hooks, so the unpickler would run them on file-controlled state: "
        f"{hook_offenders}. Either remove the hook or remove the class from "
        f"_CODEFYUI_MODULE_CLASSES -- read the comment above it first (#288)."
    )
    assert not call_offenders, (
        f"These classes are on the full_model allowlist and their constructors "
        f"(or a helper those reach) contain a call a hostile file must not be "
        f"able to trigger: {call_offenders}. A crafted checkpoint can call an "
        f"admitted class with arguments of its choosing, so a constructor that "
        f"touches the filesystem, the network, or global state is not "
        f"admissible. Move the call out of the constructor, or remove the "
        f"class from _CODEFYUI_MODULE_CLASSES (#288)."
    )


def test_the_allowlist_covers_every_codefyui_module_class():
    """The list is curated, so make forgetting it fail rather than degrade.

    A curated list's real failure mode is not being too long, it is going
    stale: someone adds an ``nn.Module`` to a node, ``full_model`` silently
    stops round-tripping for models containing it, and the symptom appears as a
    load refusal a release later. This walk is the derived counterpart -- every
    ``nn.Module`` subclass ``app.nodes`` defines has to be either admitted or
    deliberately excluded.

    ``app.custom_nodes`` and ``cdui_plugins`` are outside the walk on purpose:
    that code has not been through review, and admitting it is the thing #288
    decided NOT to do.
    """
    import sys

    from app.nodes.io.model_loader_node import codefyui_module_globals

    # Import every node module, so the subclass walk below sees the classes a
    # session that had only touched ModelLoader would not.
    from app.core.node_registry import NodeRegistry
    from app.config import settings

    NodeRegistry().discover(settings.NODES_DIR, "app.nodes")

    admitted = set(codefyui_module_globals())

    #: Audited and deliberately NOT admitted. Empty today; an entry here needs
    #: a reason, and the reason has to be about the class's deserialization
    #: surface rather than about it being unlikely to be saved.
    excluded: set[type] = set()

    found: set[type] = set()
    pending: list[type] = [nn.Module]
    seen: set[type] = set()
    while pending:
        for subclass in pending.pop().__subclasses__():
            if subclass in seen:
                continue
            seen.add(subclass)
            pending.append(subclass)
            if (getattr(subclass, "__module__", "") or "").startswith("app.nodes."):
                found.add(subclass)

    missing = {
        f"{cls.__module__}.{cls.__qualname__}"
        for cls in found - admitted - excluded
        # A class defined inside a function cannot be saved in the first place,
        # so it has nothing to load back and nothing to audit.
        if "<locals>" not in cls.__qualname__
        # Test-installed impostors and other runtime additions are not the
        # product's classes; only what the module itself defines counts.
        and getattr(sys.modules.get(cls.__module__), cls.__qualname__, None) is cls
    }
    assert not missing, (
        f"These nn.Module classes are CodefyUI's own and are not on the "
        f"full_model allowlist, so a full_model save containing one cannot be "
        f"loaded back: {sorted(missing)}. Audit each against the admission "
        f"rule above _CODEFYUI_MODULE_CLASSES and add it there, or add it to "
        f"this test's `excluded` set with a reason (#288)."
    )

    # And nothing admitted has stopped existing: the resolver raises on a stale
    # entry, so simply calling it is the assertion.
    assert len(admitted) == len(codefyui_module_globals())


def test_state_dict_mode_writes_the_same_bytes_as_before():
    """#288 widened one branch and must not have touched the other.

    ``state_dict`` is the default and the mode every shipped example uses, so
    the widening has to be invisible to it. Compared as BYTES against a plain
    ``torch.save`` of the same ``state_dict`` -- a comparison that fails if the
    node ever starts writing anything extra alongside the tensors, which is the
    only way this branch could have changed.

    Written to the SAME path both times, not two paths: ``torch.save`` names
    the zip archive after the file stem, so two names would differ in their
    bytes for a reason that has nothing to do with the contents.
    """
    with _saved("_state_dict_bytes.pt") as path:
        target = settings.MODELS_DIR / path
        model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
        res = ModelSaverNode().execute(
            {"model": model},
            {"path": path, "save_mode": "state_dict", "format": "pytorch"},
        )
        written = target.read_bytes()
        # No note on this branch either: the #288 note is a full_model thing.
        assert "__log__" not in res

        target.unlink()
        torch.save(model.state_dict(), str(target))
        assert written == target.read_bytes()
