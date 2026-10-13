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



# ── #700: ReduceLROnPlateau's mode ─────────────────────────────────────────


def _lrs_on(opt, sched, metrics):
    """The LR after one ``sched.step(m)`` per metric (``None`` -> no metric)."""
    out = []
    for m in metrics:
        opt.step()
        if m is None:
            sched.step()
        else:
            sched.step(m)
        out.append(opt.param_groups[0]["lr"])
    return out


def test_plateau_mode_reaches_torch():
    # A metric that rises every epoch: improvement under max, a plateau under
    # min. patience=1, so min halves the LR every second epoch.
    rising = [0.5, 0.6, 0.7, 0.8, 0.9]
    by_mode = {}
    for mode in ("min", "max"):
        opt = _opt()
        sched = LRSchedulerNode().execute(
            {"optimizer": opt},
            {"type": "ReduceLROnPlateau", "gamma": 0.5, "patience": 1, "mode": mode},
        )["scheduler"]
        assert sched.mode == mode
        by_mode[mode] = _lrs_on(opt, sched, rising)
        ref_opt = _opt()
        assert by_mode[mode] == _lrs_on(
            ref_opt,
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                ref_opt, mode=mode, factor=0.5, patience=1),
            rising)
    assert by_mode["max"] == [0.1] * 5
    assert by_mode["min"][-1] < 0.1


def test_plateau_mode_defaults_to_min_and_is_shown_only_for_plateau():
    param = next(p for p in LRSchedulerNode.define_params() if p.name == "mode")
    assert param.default == "min"
    assert param.options == ["min", "max"]
    assert param.visible_when == {"type": ["ReduceLROnPlateau"]}
    sched = LRSchedulerNode().execute(
        {"optimizer": _opt()}, {"type": "ReduceLROnPlateau"})["scheduler"]
    assert sched.mode == "min"


def test_plateau_mode_rejects_anything_but_min_or_max():
    with pytest.raises(ValueError, match="mode"):
        LRSchedulerNode().execute(
            {"optimizer": _opt()}, {"type": "ReduceLROnPlateau", "mode": "maximize"})


def test_every_per_type_knob_is_shown_only_for_the_types_that_read_it():
    visible = {p.name: p.visible_when for p in LRSchedulerNode.define_params()}
    assert visible["milestones"] == {"type": ["MultiStepLR"]}
    assert visible["eta_min"] == {
        "type": ["CosineAnnealingLR", "CosineAnnealingWarmRestarts"]}
    assert visible["T_mult"] == {"type": ["CosineAnnealingWarmRestarts"]}
    for name in ("mode", "patience", "threshold", "min_lr"):
        assert visible[name] == {"type": ["ReduceLROnPlateau"]}, name
    assert visible["pct_start"] == {"type": ["OneCycleLR"]}


def test_declared_defaults_keep_the_old_lr_sequence_for_the_other_types():
    """The types #700 added no knob to, plus OneCycleLR's pct_start."""
    defaults = {p.name: p.default for p in LRSchedulerNode.define_params()}
    for sched_type, factory in [
        ("StepLR", lambda o: torch.optim.lr_scheduler.StepLR(o, step_size=10)),
        ("ExponentialLR", lambda o: torch.optim.lr_scheduler.ExponentialLR(o, gamma=0.1)),
        ("OneCycleLR",
         lambda o: torch.optim.lr_scheduler.OneCycleLR(o, max_lr=0.01, total_steps=1000)),
    ]:
        opt = _opt()
        sched = LRSchedulerNode().execute(
            {"optimizer": opt}, {**defaults, "type": sched_type})["scheduler"]
        assert _lrs(opt, sched, 45) == pytest.approx(
            _reference(factory, 45)), sched_type


_EXPORT_CASES = [
    ({"type": "ReduceLROnPlateau", "gamma": 0.5, "mode": "max", "patience": 1,
      "threshold": 0.01, "min_lr": 0.0002},
     lambda o: torch.optim.lr_scheduler.ReduceLROnPlateau(
         o, mode="max", factor=0.5, patience=1, threshold=0.01, min_lr=0.0002),
     [0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]),
    ({"type": "MultiStepLR", "milestones": "2, 5", "gamma": 0.5},
     lambda o: torch.optim.lr_scheduler.MultiStepLR(o, milestones=[2, 5], gamma=0.5),
     [None] * 10),
    ({"type": "CosineAnnealingLR", "T_max": 6, "eta_min": 0.0001},
     lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=6, eta_min=0.0001),
     [None] * 10),
    ({"type": "CosineAnnealingWarmRestarts", "T_max": 2, "T_mult": 2, "eta_min": 0.0001},
     lambda o: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
         o, T_0=2, T_mult=2, eta_min=0.0001),
     [None] * 10),
    ({"type": "OneCycleLR", "max_lr": 0.01, "total_steps": 12, "pct_start": 0.2},
     lambda o: torch.optim.lr_scheduler.OneCycleLR(
         o, max_lr=0.01, total_steps=12, pct_start=0.2),
     [None] * 10),
]


@pytest.mark.parametrize(
    "params, factory, metrics", _EXPORT_CASES,
    ids=[case[0]["type"] for case in _EXPORT_CASES])
def test_an_exported_script_builds_the_same_scheduler(params, factory, metrics):
    """Export Python emits every knob, and the script's scheduler follows the
    same LR sequence as one built with torch directly (#700 acceptance 3)."""
    import ast

    from app.core.codegen import generate_python
    from app.core.execution_context import ExecutionContext

    script = generate_python(
        [{"id": "s1", "type": "Start", "data": {"params": {}}},
         {"id": "model", "type": "SequentialModel", "data": {"params": {}}},
         {"id": "opt", "type": "Optimizer", "data": {"params": {"lr": 0.001}}},
         {"id": "sched", "type": "LRScheduler", "data": {"params": params}}],
        [{"id": "t1", "source": "s1", "target": "model", "sourceHandle": "trigger",
          "targetHandle": "", "type": "trigger"},
         {"id": "d1", "source": "model", "target": "opt", "sourceHandle": "model",
          "targetHandle": "model", "type": "data"},
         {"id": "d2", "source": "opt", "target": "sched", "sourceHandle": "optimizer",
          "targetHandle": "optimizer", "type": "data"}],
        name="knobs")

    emitted = [
        ast.literal_eval(line.split("=", 1)[1].strip())
        for line in script.splitlines()
        if line.strip().startswith("params = {")]
    assert params in emitted

    module: dict = {"__name__": "lr_scheduler_export"}
    exec(compile(script, "<exported graph>", "exec"), module)  # noqa: S102
    module["_RT"] = module["_load_runtime"](None)
    results = module["run_graph"](ExecutionContext(seed=module.get("GRAPH_SEED")), {})
    sched = results["sched"]["scheduler"]
    exported_opt = sched.optimizer

    ref_opt = torch.optim.Adam(nn.Linear(4, 2).parameters(), lr=0.001)
    assert _lrs_on(exported_opt, sched, metrics) == pytest.approx(
        _lrs_on(ref_opt, factory(ref_opt), metrics))


# ── #700: TrainingLoop feeds a mode="max" plateau schedule val_accuracy ────


class _RecordingPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen: list[float] = []

    def step(self, metrics, *args, **kwargs):  # type: ignore[override]
        self.seen.append(float(metrics))
        return super().step(metrics, *args, **kwargs)


def _loader(n: int = 12):
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(n, 4, generator=gen)
    y = torch.randint(0, 2, (n,), generator=gen)
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=4, shuffle=False)


def _train_with(sched, opt, model, *, val=True, loss_fn=None):
    from app.nodes.training.training_loop_node import TrainingLoopNode

    inputs = {"model": model, "dataloader": _loader(), "optimizer": opt,
              "loss_fn": loss_fn or nn.CrossEntropyLoss(), "lr_scheduler": sched}
    if val:
        inputs["val_dataloader"] = _loader()
    return TrainingLoopNode().execute(inputs, {"device": "cpu", "epochs": 3})


@pytest.mark.parametrize("mode, series", [
    ("max", "final_val_accuracy"), ("min", "final_val_loss")])
def test_training_loop_feeds_the_metric_the_mode_expects(mode, series):
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = _RecordingPlateau(opt, mode=mode, patience=0)
    result = _train_with(sched, opt, model)
    assert len(sched.seen) == 3
    assert sched.seen[-1] == pytest.approx(result["metrics"][series])
    if mode == "max":
        # Accuracies over 12 validation samples.
        assert all(abs(v * 12 - round(v * 12)) < 1e-9 for v in sched.seen)


@pytest.mark.parametrize("val, loss_fn, reason", [
    (False, None, "no val_dataloader"),
    (True, nn.MSELoss(), "not a classification loss"),
])
def test_training_loop_refuses_mode_max_without_an_accuracy(val, loss_fn, reason):
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    before = model.weight.detach().clone()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max")
    with pytest.raises(ValueError, match=reason):
        _train_with(sched, opt, model, val=val, loss_fn=loss_fn)
    # Refused before the first batch: the weights are untouched.
    assert torch.equal(model.weight.detach(), before)


def test_training_loop_skips_a_mode_max_step_when_an_epoch_has_no_accuracy(caplog):
    import logging

    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = _RecordingPlateau(opt, mode="max", patience=0)
    empty = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)),
        batch_size=4)
    from app.nodes.training.training_loop_node import TrainingLoopNode

    with caplog.at_level(logging.WARNING):
        TrainingLoopNode().execute(
            {"model": model, "dataloader": _loader(), "optimizer": opt,
             "loss_fn": nn.CrossEntropyLoss(), "lr_scheduler": sched,
             "val_dataloader": empty},
            {"device": "cpu", "epochs": 3})
    assert sched.seen == []
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1)
    assert sum("mode='max'" in r.getMessage() for r in caplog.records) == 1

# ── 2026-10-13 audit: lengths below 1 are refused by validation ─────────
# step_size=0 or T_max=0 passed validate_graph and then raised
# ZeroDivisionError after a whole epoch of training.


@pytest.mark.parametrize("name,minimum", [
    ("step_size", 1), ("T_max", 1), ("total_steps", 1),
    ("gamma", 0.0), ("max_lr", 0.0),
])
def test_lengths_and_factors_declare_a_lower_bound(name, minimum):
    params = {p.name: p for p in LRSchedulerNode.define_params()}
    assert params[name].min_value == minimum


@pytest.mark.parametrize("params", [
    {"type": "StepLR", "step_size": 0},
    {"type": "CosineAnnealingLR", "T_max": 0},
    {"type": "OneCycleLR", "total_steps": 0},
])
def test_validation_refuses_a_zero_length(params):
    from app.core.graph_engine import validate_graph

    errors = validate_graph(
        [{"id": "n", "type": "LRScheduler", "data": {"params": params}}], [])
    name = next(k for k in params if k != "type")
    assert any(f"Parameter '{name}'" in e and "below minimum 1" in e
               for e in errors), errors
