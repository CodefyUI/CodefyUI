"""Tests for LRSchedulerNode."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from app.nodes.training.lr_scheduler_node import LRSchedulerNode


def _opt():
    model = nn.Linear(4, 2)
    return torch.optim.Adam(model.parameters(), lr=0.1)


def test_node_metadata():
    assert LRSchedulerNode.NODE_NAME == "LRScheduler"


def test_step_lr_decays_lr():
    opt = _opt()
    res = LRSchedulerNode().execute(
        {"optimizer": opt},
        {"type": "StepLR", "step_size": 1, "gamma": 0.5},
    )
    sched = res["scheduler"]
    # Take an optimizer step (required before scheduler step in newer torch)
    opt.step()
    sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.05)


def test_cosine_annealing_lr():
    opt = _opt()
    res = LRSchedulerNode().execute({"optimizer": opt}, {"type": "CosineAnnealingLR", "T_max": 10})
    assert res["scheduler"].__class__.__name__ == "CosineAnnealingLR"


def test_exponential_lr():
    opt = _opt()
    res = LRSchedulerNode().execute({"optimizer": opt}, {"type": "ExponentialLR", "gamma": 0.9})
    sched = res["scheduler"]
    opt.step()
    sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.09)


def test_reduce_lr_on_plateau():
    opt = _opt()
    res = LRSchedulerNode().execute({"optimizer": opt}, {"type": "ReduceLROnPlateau", "gamma": 0.5})
    assert res["scheduler"].__class__.__name__ == "ReduceLROnPlateau"


def test_one_cycle_lr():
    opt = _opt()
    res = LRSchedulerNode().execute(
        {"optimizer": opt},
        {"type": "OneCycleLR", "max_lr": 0.1, "total_steps": 100},
    )
    assert res["scheduler"].__class__.__name__ == "OneCycleLR"


def test_unsupported_scheduler_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        LRSchedulerNode().execute({"optimizer": _opt()}, {"type": "Bogus"})


# ── #700: the knobs researchers tune ──────────────────────────────────────


def _lrs(opt, sched, steps, metric=None):
    """The LR after each of *steps* scheduler steps."""
    out = []
    for _ in range(steps):
        opt.step()
        if metric is None:
            sched.step()
        else:
            sched.step(metric)
        out.append(opt.param_groups[0]["lr"])
    return out


def _reference(factory, steps, metric=None):
    """The same sequence from a scheduler built directly with torch."""
    opt = _opt()
    return _lrs(opt, factory(opt), steps, metric)


def test_plateau_patience_reaches_torch():
    # The issue's repro: gamma=0.5, a flat metric for 8 epochs. At torch's
    # default patience of 10 the LR is still 0.1; at patience=2 it halves.
    opt = _opt()
    sched = LRSchedulerNode().execute(
        {"optimizer": opt},
        {"type": "ReduceLROnPlateau", "gamma": 0.5, "patience": 2},
    )["scheduler"]
    got = _lrs(opt, sched, 8, metric=1.0)
    assert got[-1] < 0.1
    assert got == _reference(
        lambda o: torch.optim.lr_scheduler.ReduceLROnPlateau(o, factor=0.5, patience=2),
        8, metric=1.0)


def test_plateau_min_lr_and_threshold_reach_torch():
    opt = _opt()
    sched = LRSchedulerNode().execute(
        {"optimizer": opt},
        {"type": "ReduceLROnPlateau", "gamma": 0.1, "patience": 0,
         "min_lr": 0.005, "threshold": 0.5},
    )["scheduler"]
    # A metric improving by 10% each epoch is not "better" at threshold=0.5,
    # so the LR keeps dropping, and stops at min_lr.
    got = []
    for m in [1.0, 0.9, 0.81, 0.73, 0.66]:
        opt.step()
        sched.step(m)
        got.append(opt.param_groups[0]["lr"])
    assert got[-1] == pytest.approx(0.005)
    assert min(got) >= 0.005 - 1e-12


def test_multistep_milestones_reach_torch():
    opt = _opt()
    sched = LRSchedulerNode().execute(
        {"optimizer": opt},
        {"type": "MultiStepLR", "milestones": "3, 5", "gamma": 0.1, "step_size": 30},
    )["scheduler"]
    got = _lrs(opt, sched, 6)
    assert got == _reference(
        lambda o: torch.optim.lr_scheduler.MultiStepLR(o, milestones=[3, 5], gamma=0.1), 6)
    assert got[2] == pytest.approx(0.01)


def test_multistep_rejects_a_milestone_that_is_not_a_number():
    with pytest.raises(ValueError, match="milestones"):
        LRSchedulerNode().execute(
            {"optimizer": _opt()}, {"type": "MultiStepLR", "milestones": "30, eighty"})


def test_cosine_eta_min_reaches_torch():
    opt = _opt()
    sched = LRSchedulerNode().execute(
        {"optimizer": opt}, {"type": "CosineAnnealingLR", "T_max": 4, "eta_min": 0.01},
    )["scheduler"]
    got = _lrs(opt, sched, 4)
    assert got[-1] == pytest.approx(0.01)
    assert got == pytest.approx(_reference(
        lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=4, eta_min=0.01), 4))


def test_warm_restarts_t_mult_and_eta_min_reach_torch():
    opt = _opt()
    sched = LRSchedulerNode().execute(
        {"optimizer": opt},
        {"type": "CosineAnnealingWarmRestarts", "T_max": 2, "T_mult": 2, "eta_min": 0.02},
    )["scheduler"]
    got = _lrs(opt, sched, 8)
    assert got == pytest.approx(_reference(
        lambda o: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            o, T_0=2, T_mult=2, eta_min=0.02), 8))
    # The second cycle is 4 steps long, so step 4 (index 3) is not a restart.
    assert got[3] < 0.1


def test_one_cycle_pct_start_reaches_torch():
    opt = _opt()
    sched = LRSchedulerNode().execute(
        {"optimizer": opt},
        {"type": "OneCycleLR", "max_lr": 0.1, "total_steps": 20, "pct_start": 0.1},
    )["scheduler"]
    got = _lrs(opt, sched, 19)
    assert got == pytest.approx(_reference(
        lambda o: torch.optim.lr_scheduler.OneCycleLR(
            o, max_lr=0.1, total_steps=20, pct_start=0.1), 19))
    # The peak comes after 10% of the 20 steps, at the first step; the
    # default 30% would put it at the sixth.
    assert max(range(19), key=got.__getitem__) == 0


@pytest.mark.parametrize("params, factory, metric", [
    ({"type": "ReduceLROnPlateau", "gamma": 0.5},
     lambda o: torch.optim.lr_scheduler.ReduceLROnPlateau(o, factor=0.5), 1.0),
    ({"type": "MultiStepLR", "step_size": 2, "gamma": 0.5},
     lambda o: torch.optim.lr_scheduler.MultiStepLR(o, milestones=[2, 4, 6, 8], gamma=0.5), None),
    ({"type": "CosineAnnealingLR", "T_max": 5},
     lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=5), None),
    ({"type": "CosineAnnealingWarmRestarts", "T_max": 3},
     lambda o: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(o, T_0=3), None),
    ({"type": "OneCycleLR", "max_lr": 0.1, "total_steps": 20},
     lambda o: torch.optim.lr_scheduler.OneCycleLR(o, max_lr=0.1, total_steps=20), None),
])
def test_a_saved_graph_without_the_new_params_keeps_its_lr_sequence(params, factory, metric):
    steps = 12 if metric is not None else 10
    opt = _opt()
    sched = LRSchedulerNode().execute({"optimizer": opt}, params)["scheduler"]
    assert _lrs(opt, sched, steps, metric) == pytest.approx(
        _reference(factory, steps, metric))


def test_declared_defaults_keep_the_old_lr_sequence():
    # The canvas fills every declared default in, so those defaults have to
    # be the values the node used before the params existed.
    defaults = {p.name: p.default for p in LRSchedulerNode.define_params()}
    for sched_type, factory in [
        ("ReduceLROnPlateau",
         lambda o: torch.optim.lr_scheduler.ReduceLROnPlateau(o, factor=0.1)),
        ("MultiStepLR",
         lambda o: torch.optim.lr_scheduler.MultiStepLR(o, milestones=[10, 20, 30, 40])),
        ("CosineAnnealingLR",
         lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=50)),
        ("CosineAnnealingWarmRestarts",
         lambda o: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(o, T_0=50)),
    ]:
        opt = _opt()
        sched = LRSchedulerNode().execute(
            {"optimizer": opt}, {**defaults, "type": sched_type})["scheduler"]
        metric = 1.0 if sched_type == "ReduceLROnPlateau" else None
        assert _lrs(opt, sched, 45, metric) == pytest.approx(
            _reference(factory, 45, metric)), sched_type


def test_exported_python_carries_the_new_params():
    """Export Python hands the node the same params, so the knobs still work."""
    import ast

    from app.core.codegen import generate_python

    script = generate_python(
        [{"id": "s1", "type": "Start", "data": {"params": {}}},
         {"id": "model", "type": "SequentialModel", "data": {"params": {}}},
         {"id": "opt", "type": "Optimizer", "data": {"params": {}}},
         {"id": "sched", "type": "LRScheduler",
          "data": {"params": {"type": "ReduceLROnPlateau", "gamma": 0.5,
                              "patience": 2, "min_lr": 0.01}}}],
        [{"id": "t1", "source": "s1", "target": "model", "sourceHandle": "trigger",
          "targetHandle": "", "type": "trigger"},
         {"id": "d1", "source": "model", "target": "opt", "sourceHandle": "model",
          "targetHandle": "model", "type": "data"},
         {"id": "d2", "source": "opt", "target": "sched", "sourceHandle": "optimizer",
          "targetHandle": "optimizer", "type": "data"}],
        name="knobs")
    literal = next(
        line.split("=", 1)[1].strip()
        for line in script.splitlines()
        if line.strip().startswith("params = {") and "patience" in line)
    exported = ast.literal_eval(literal)
    opt = _opt()
    sched = LRSchedulerNode().execute({"optimizer": opt}, exported)["scheduler"]
    got = _lrs(opt, sched, 8, metric=1.0)
    assert got == _reference(
        lambda o: torch.optim.lr_scheduler.ReduceLROnPlateau(
            o, factor=0.5, patience=2, min_lr=0.01), 8, metric=1.0)
