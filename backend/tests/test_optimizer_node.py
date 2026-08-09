"""Tests for OptimizerNode (creating optimizers from a model)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from app.nodes.training import optimizer_node
from app.nodes.training.optimizer_node import OptimizerNode


def _model():
    return nn.Linear(4, 2)


def test_node_metadata():
    assert OptimizerNode.NODE_NAME == "Optimizer"
    assert OptimizerNode.CATEGORY == "Training"


def test_default_creates_adam():
    res = OptimizerNode().execute({"model": _model()}, {"type": "Adam", "lr": 0.001, "weight_decay": 0.0})
    assert res["optimizer"].__class__.__name__ == "Adam"


def test_sgd_optimizer():
    res = OptimizerNode().execute({"model": _model()}, {"type": "SGD", "lr": 0.01})
    assert res["optimizer"].__class__.__name__ == "SGD"
    assert res["optimizer"].param_groups[0]["lr"] == 0.01


def test_adamw_includes_weight_decay():
    res = OptimizerNode().execute({"model": _model()}, {"type": "AdamW", "lr": 0.001, "weight_decay": 0.05})
    assert res["optimizer"].param_groups[0]["weight_decay"] == 0.05


def test_rprop_drops_default_weight_decay_zero():
    # Rprop doesn't accept weight_decay, but zero should silently drop
    res = OptimizerNode().execute({"model": _model()}, {"type": "Rprop", "lr": 0.01, "weight_decay": 0.0})
    assert res["optimizer"].__class__.__name__ == "Rprop"


def test_rprop_rejects_nonzero_weight_decay():
    with pytest.raises(ValueError, match="weight_decay"):
        OptimizerNode().execute(
            {"model": _model()},
            {"type": "Rprop", "lr": 0.01, "weight_decay": 0.01},
        )


def test_unsupported_optimizer_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        OptimizerNode().execute({"model": _model()}, {"type": "Bogus", "lr": 0.001})


def test_lr_is_set_correctly():
    res = OptimizerNode().execute({"model": _model()}, {"type": "Adam", "lr": 0.5})
    assert res["optimizer"].param_groups[0]["lr"] == 0.5


@pytest.mark.parametrize("opt_type", ["Adam", "SGD", "AdamW", "RMSprop", "Adagrad", "RAdam", "NAdam", "ASGD"])
def test_all_supported_optimizers_create(opt_type):
    """Each listed optimizer should construct without error."""
    res = OptimizerNode().execute({"model": _model()}, {"type": opt_type, "lr": 0.001, "weight_decay": 0.0})
    assert res["optimizer"] is not None


# ── full parameterization (core#134) ──────────────────────────────────────
#
# Every test below compares the CONSTRUCTED optimizer against a raw torch
# reference. A schema that accepts a value and then drops it would pass a
# "the param is in define_params()" test and fail every one of these.


def _param_group(params: dict) -> dict:
    return OptimizerNode().execute(
        {"model": _model()}, params)["optimizer"].param_groups[0]


def test_betas_are_parsed_and_applied():
    group = _param_group({"type": "Adam", "betas": "0.85, 0.995"})
    assert group["betas"] == (0.85, 0.995)


@pytest.mark.parametrize("spelling", ["0.85, 0.995", "(0.85, 0.995)",
                                      "0.85 0.995", "[0.85; 0.995]"])
def test_betas_accept_the_spellings_people_actually_paste(spelling):
    assert _param_group({"type": "Adam", "betas": spelling})["betas"] == (0.85, 0.995)


def test_betas_reject_the_wrong_arity():
    with pytest.raises(ValueError, match="exactly 2 values"):
        _param_group({"type": "Adam", "betas": "0.9"})


def test_betas_reject_nonsense():
    with pytest.raises(ValueError, match="not a number"):
        _param_group({"type": "Adam", "betas": "fast, slow"})


def test_betas_change_the_update_versus_a_raw_torch_reference():
    """The value reaches the maths, not just ``param_groups``.

    Two steps with DIFFERENT gradients, deliberately: Adam's bias correction
    makes the first step ``lr``-sized whatever beta1 is, and a constant
    gradient keeps it that way forever. Only a changing gradient lets the
    momentum term diverge, which is the thing beta1 controls.
    """
    torch.manual_seed(0)
    model = _model()
    state = {k: v.clone() for k, v in model.state_dict().items()}

    def _two_steps(node_params: dict | None, reference_betas=None):
        model.load_state_dict(state)
        if node_params is not None:
            optimizer = OptimizerNode().execute({"model": model}, node_params)["optimizer"]
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1,
                                         betas=reference_betas)
        for scale in (1.0, 4.0):
            model.weight.grad = torch.full_like(model.weight, scale)
            model.bias.grad = torch.full_like(model.bias, scale)
            optimizer.step()
        return model.weight.detach().clone()

    node_tuned = _two_steps({"type": "Adam", "lr": 0.1, "betas": "0.5, 0.999"})
    torch_tuned = _two_steps(None, reference_betas=(0.5, 0.999))
    torch_default = _two_steps(None, reference_betas=(0.9, 0.999))

    assert torch.equal(node_tuned, torch_tuned)
    assert not torch.equal(node_tuned, torch_default)


def test_eps_is_applied_where_torch_shares_our_default():
    assert _param_group({"type": "Adam", "eps": 1e-3})["eps"] == 1e-3
    assert _param_group({"type": "RMSprop", "eps": 1e-3})["eps"] == 1e-3


def test_eps_leaves_adagrads_own_default_alone():
    """Adagrad defaults eps to 1e-10, not 1e-8.

    Forwarding this node's default would silently retune every existing
    Adagrad graph on upgrade, so Adagrad is excluded from the eps set and
    keeps torch's value.
    """
    reference = torch.optim.Adagrad(_model().parameters(), lr=0.01)
    assert _param_group({"type": "Adagrad", "lr": 0.01})["eps"] == \
        reference.param_groups[0]["eps"]


def test_momentum_is_applied_to_sgd_and_rmsprop():
    assert _param_group({"type": "SGD", "momentum": 0.9})["momentum"] == 0.9
    assert _param_group({"type": "RMSprop", "momentum": 0.5})["momentum"] == 0.5


def test_momentum_actually_accelerates_versus_plain_sgd():
    """Two identical steps: with momentum the second one moves further."""
    def _two_steps(momentum: float) -> float:
        torch.manual_seed(0)
        model = _model()
        optimizer = OptimizerNode().execute(
            {"model": model},
            {"type": "SGD", "lr": 0.1, "momentum": momentum})["optimizer"]
        start = model.weight.detach().clone()
        for _ in range(2):
            model.weight.grad = torch.ones_like(model.weight)
            model.bias.grad = torch.ones_like(model.bias)
            optimizer.step()
        return (start - model.weight.detach()).abs().sum().item()

    assert _two_steps(0.9) > _two_steps(0.0)


def test_amsgrad_is_applied_and_recorded_in_state():
    group = _param_group({"type": "Adam", "amsgrad": True})
    assert group["amsgrad"] is True

    torch.manual_seed(0)
    model = _model()
    optimizer = OptimizerNode().execute(
        {"model": model}, {"type": "Adam", "amsgrad": True})["optimizer"]
    model.weight.grad = torch.ones_like(model.weight)
    model.bias.grad = torch.ones_like(model.bias)
    optimizer.step()
    # The AMSGrad variant is the only one that keeps a running maximum.
    assert any("max_exp_avg_sq" in s for s in optimizer.state.values())


def test_nesterov_and_dampening_reach_sgd():
    group = _param_group({"type": "SGD", "momentum": 0.9, "nesterov": True})
    assert group["nesterov"] is True
    assert _param_group({"type": "SGD", "momentum": 0.9,
                         "dampening": 0.5})["dampening"] == 0.5


def test_dampening_changes_the_second_step():
    def _second_step_delta(dampening: float) -> float:
        torch.manual_seed(0)
        model = _model()
        optimizer = OptimizerNode().execute(
            {"model": model},
            {"type": "SGD", "lr": 0.1, "momentum": 0.9,
             "dampening": dampening})["optimizer"]
        for _ in range(1):
            model.weight.grad = torch.ones_like(model.weight)
            model.bias.grad = torch.ones_like(model.bias)
            optimizer.step()
        before = model.weight.detach().clone()
        model.weight.grad = torch.ones_like(model.weight)
        model.bias.grad = torch.ones_like(model.bias)
        optimizer.step()
        return (before - model.weight.detach()).abs().sum().item()

    assert _second_step_delta(0.0) > _second_step_delta(0.5)


def test_an_inapplicable_visible_param_is_an_error_not_a_shrug():
    """``weight_decay`` is always on the form, so Rprop must complain."""
    with pytest.raises(ValueError, match="does not accept weight_decay"):
        _param_group({"type": "Rprop", "weight_decay": 0.1})


@pytest.mark.parametrize("param,value,other_type", [
    ("momentum", 0.9, "Adam"),
    ("nesterov", True, "Adam"),
    ("dampening", 0.5, "Adam"),
    ("amsgrad", True, "SGD"),
    ("betas", "0.5, 0.9", "SGD"),
    ("eps", 1e-07, "Adagrad"),
])
def test_a_param_the_type_hides_is_ignored_rather_than_fatal(
    param, value, other_type,
):
    """Tune it on Adam, switch to Adagrad, and the run must still work.

    The canvas writes every default onto a node and never clears one when
    the sibling that hides it changes, so the leftover value is invisible:
    an error naming it points at a field that is not on the form and cannot
    be reset without hand-editing the graph JSON. Hidden means "not set".

    Regression for the #188 review's I5 — every one of these raised before.
    """
    group = _param_group({"type": other_type, param: value})

    assert group["lr"] == 0.001
    # And the leftover really is inert, not quietly applied.
    assert param not in group or group[param] != value


def test_a_hidden_leftover_comes_back_when_it_applies_again():
    """The value is preserved, not destroyed — same as the editor does."""
    tuned = {"type": "Adam", "eps": 1e-07}
    _param_group({**tuned, "type": "Adagrad"})       # ignored, no raise
    assert _param_group(tuned)["eps"] == 1e-07       # honoured again


@pytest.mark.parametrize("opt_type", ["Adam", "SGD", "AdamW", "RMSprop",
                                      "Adagrad", "RAdam", "NAdam", "Rprop",
                                      "ASGD"])
def test_defaults_reproduce_the_pre_change_optimizer_exactly(opt_type):
    """The new params are no-ops until someone edits them.

    The reference is what THIS NODE built before core#134 -- ``lr`` plus
    ``weight_decay`` where accepted -- not torch's own defaults, because the
    node has always overridden AdamW's 0.01 decay with 0.0. An existing saved
    graph carries none of the new keys, so it must produce a byte-identical
    optimizer.
    """
    import inspect

    cls = getattr(torch.optim, opt_type)
    legacy_kwargs = {"lr": 0.01}
    if "weight_decay" in inspect.signature(cls.__init__).parameters:
        legacy_kwargs["weight_decay"] = 0.0

    built = _param_group({"type": opt_type, "lr": 0.01})
    reference = cls(_model().parameters(), **legacy_kwargs).param_groups[0]
    for key, expected in reference.items():
        if key == "params":
            continue
        assert built[key] == expected, f"{opt_type}.{key} drifted"


# ── the invariant that keeps the rejection honest (#188 re-review, D5) ────


def test_every_conditional_param_hides_exactly_where_it_does_not_apply():
    """Visibility and applicability are the SAME set, per conditional param.

    So an inapplicable value is a hidden value, and hidden means "not set" —
    which is why ``_reject_inapplicable`` fires for exactly one param today:
    ``weight_decay``, the only inapplicable one with no rule to hide it.
    That is a property of these definitions, not a coincidence, and this
    test is what makes it one: add a param that is visible where it does not
    apply and it fails, which is the moment to decide what should happen.
    """
    definitions = {p.name: p for p in OptimizerNode.define_params()}

    for param, applies_to in {
        "momentum": optimizer_node._MOMENTUM_TYPES,
        "betas": optimizer_node._BETAS_TYPES,
        "eps": optimizer_node._EPS_TYPES,
        "amsgrad": optimizer_node._AMSGRAD_TYPES,
        "nesterov": optimizer_node._SGD_ONLY,
        "dampening": optimizer_node._SGD_ONLY,
    }.items():
        assert definitions[param].visible_when == {"type": sorted(applies_to)}, (
            f"{param} is shown for algorithms that cannot take it")

    always_visible = {name for name, d in definitions.items()
                      if not d.visible_when}
    assert always_visible == {"type", "lr", "weight_decay"}, (
        "a new always-visible param needs an explicit decision: every "
        "algorithm must accept it, or execute() must reject it where it "
        "does not (see weight_decay)")


# ── the tables against the torch that is actually installed (#189) ────────


def _accepting(param, node_default):
    """Algorithms whose signature takes *param* AND defaults it to ours."""
    import inspect

    import torch.optim as optim

    found = set()
    for name in optimizer_node.OPTIMIZER_TYPES:
        signature = inspect.signature(getattr(optim, name).__init__).parameters
        if param in signature and signature[param].default == node_default:
            found.add(name)
    return found


def test_the_applicability_tables_match_the_installed_torch_signatures():
    """The declared sets ARE "accepts it, and agrees with our default".

    #134 declares applicability instead of inferring it, and that stays the
    right call: inferring it would forward ``eps`` to Adagrad, whose torch
    default is 1e-10 against this node's 1e-8, silently retuning every
    existing Adagrad graph on upgrade. What declaring costs is that the
    table can quietly stop describing torch — a release that adds
    ``momentum`` to another algorithm, or that changes a default, leaves the
    set behind with nothing to say so.

    This closes the "with nothing to say so" part, and only that: it asserts
    the declaration, it does not derive it. A failure here is a decision to
    make by hand, not a line to delete.
    """
    from app.core.param_values import parse_float_sequence

    betas_default = parse_float_sequence(
        optimizer_node.DEFAULT_BETAS, name="betas", length=2)

    for param, node_default, declared in (
        ("momentum", 0.0, optimizer_node._MOMENTUM_TYPES),
        ("betas", betas_default, optimizer_node._BETAS_TYPES),
        ("eps", optimizer_node.DEFAULT_EPS, optimizer_node._EPS_TYPES),
        ("amsgrad", False, optimizer_node._AMSGRAD_TYPES),
        ("nesterov", False, optimizer_node._SGD_ONLY),
        ("dampening", 0.0, optimizer_node._SGD_ONLY),
    ):
        found = _accepting(param, node_default)
        assert found == set(declared), (
            f"the {param} table no longer matches this torch: it accepts-"
            f"and-agrees for {sorted(found)}, the node declares "
            f"{sorted(declared)}. Decide per algorithm — forwarding is only "
            f"a no-op where torch's default equals ours (see eps/Adagrad "
            f"for why that matters)")


def test_adagrad_is_excluded_from_eps_because_its_default_differs():
    """The one exclusion that is a JUDGEMENT rather than an absence.

    Every other algorithm missing from a table is missing because torch has
    no such knob. Adagrad has ``eps`` and is left out anyway, so a reader
    checking the table against the signature would find a discrepancy and
    "fix" it. Pinned here so the reason is executable rather than a comment.
    """
    import inspect

    import torch.optim as optim

    signature = inspect.signature(optim.Adagrad.__init__).parameters
    assert "eps" in signature, "Adagrad stopped taking eps; revisit the table"
    assert signature["eps"].default != optimizer_node.DEFAULT_EPS, (
        "Adagrad's eps default now matches ours, so the reason it is "
        "excluded from _EPS_TYPES has gone away — including it would now be "
        "the no-op it was not before")
    assert "Adagrad" not in optimizer_node._EPS_TYPES


#: Every keyword each optimizer took when this was written (torch 2.13),
#: minus ``self``/``params``. A snapshot, and deliberately a whole one --
#: see the test below for why it is not narrowed to the knobs we expose.
_TORCH_OPTIMIZER_KWARGS = {
    "Adam": {"amsgrad", "betas", "capturable", "decoupled_weight_decay",
             "differentiable", "eps", "foreach", "fused", "lr", "maximize",
             "weight_decay"},
    "SGD": {"dampening", "differentiable", "foreach", "fused", "lr",
            "maximize", "momentum", "nesterov", "weight_decay"},
    "AdamW": {"amsgrad", "betas", "capturable", "differentiable", "eps",
              "foreach", "fused", "lr", "maximize", "weight_decay"},
    "RMSprop": {"alpha", "capturable", "centered", "differentiable", "eps",
                "foreach", "lr", "maximize", "momentum", "weight_decay"},
    "Adagrad": {"differentiable", "eps", "foreach", "fused",
                "initial_accumulator_value", "lr", "lr_decay", "maximize",
                "weight_decay"},
    "RAdam": {"betas", "capturable", "decoupled_weight_decay",
              "differentiable", "eps", "foreach", "lr", "maximize",
              "weight_decay"},
    "NAdam": {"betas", "capturable", "decoupled_weight_decay",
              "differentiable", "eps", "foreach", "lr", "maximize",
              "momentum_decay", "weight_decay"},
    "Rprop": {"capturable", "differentiable", "etas", "foreach", "lr",
              "maximize", "step_sizes"},
    "ASGD": {"alpha", "capturable", "differentiable", "foreach", "lambd",
             "lr", "maximize", "t0", "weight_decay"},
}


def test_no_new_torch_optimizer_knob_arrives_unnoticed():
    """A new hyperparameter must break something. This is the something.

    Deliberately a snapshot of the WHOLE signature rather than of the knobs
    the node exposes: the gap #189 names is a torch release ADDING one, and
    a test that only looks at the params already declared cannot see a param
    that is not declared yet.

    It will therefore also fire for plumbing nobody wants on a node
    (``fused``, ``capturable``). That is the intended cost — the fix is one
    line in the dict above, written by someone who looked at what changed,
    which is the entire point. A table that drifts for years in silence is
    the alternative being rejected here.
    """
    import inspect

    import torch.optim as optim

    current = {}
    for name in optimizer_node.OPTIMIZER_TYPES:
        signature = inspect.signature(getattr(optim, name).__init__).parameters
        current[name] = {k for k in signature if k not in ("self", "params")}

    assert current == _TORCH_OPTIMIZER_KWARGS, (
        "this torch's optimizer signatures differ from the recorded ones. "
        "For each added keyword: expose it (a ParamDefinition plus an "
        "applicability set) or record it above as reviewed-and-declined. Do "
        "NOT start forwarding automatically — see the eps/Adagrad note at "
        "the top of optimizer_node.py.")
