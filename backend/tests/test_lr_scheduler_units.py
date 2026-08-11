"""LRScheduler's units must be stated where the user is standing.

At the default `TrainingLoop.scheduler_step = epoch`, `TrainingLoop` steps
the scheduler once per EPOCH (training_loop_node.py, inside the epoch loop,
after validation). Three of this node's parameters are lengths in that unit,
and two of them are named after PyTorch concepts that mean something else:

- `T_max` is a cosine cycle length that should normally equal
  `TrainingLoop.epochs`. Getting it wrong costs a few points of accuracy and
  looks like an architecture problem (#205).
- `total_steps` is OneCycleLR's cycle length. Upstream, a "step" is an
  optimizer step (a batch); in epoch mode it is an epoch. The default of 1000
  is far larger than any usual epoch count, so leaving it alone traverses only
  the start of the cycle.

**Both units, not one (#308).** #303 added `scheduler_step=optimizer_step`,
in which all three lengths ARE optimizer-step counts -- and these
descriptions went on asserting the epoch rule as unconditional, flatly
contradicting `TrainingLoop.scheduler_step`'s own description one node over.
A user following one of the two got wrong advice whichever mode they were
in. The mode guards below exist so that cannot come back: stating one unit
as though it were the only one is the regression.

These are documentation guards, not behaviour tests -- the wording IS the
fix, so it is what regresses. Each asserts the load-bearing fact rather than
an exact sentence, so the prose can still be edited.
"""

from __future__ import annotations

import pytest

from app.nodes.training.lr_scheduler_node import LRSchedulerNode


def _param(name: str):
    for p in LRSchedulerNode.define_params():
        if p.name == name:
            return p
    raise AssertionError(f"LRScheduler has no param {name!r}")


def _desc(name: str) -> str:
    return _param(name).description.lower()


@pytest.mark.parametrize("name", ["step_size", "T_max", "total_steps"])
def test_epoch_unit_parameters_say_epoch(name):
    """The unit is the trap, so the unit has to be in the description."""
    assert "epoch" in _desc(name), f"{name} does not state its unit"


@pytest.mark.parametrize("name", ["step_size", "T_max", "total_steps"])
def test_every_length_parameter_states_both_modes(name):
    """#308: the unit depends on TrainingLoop.scheduler_step, so saying only
    one of the two is the same trap the epoch-only wording was."""
    desc = _desc(name)
    assert "optimizer_step" in desc, f"{name} never mentions the other mode"
    assert "scheduler_step" in desc, (
        f"{name} does not name the parameter that decides its unit")


def test_the_warmup_families_say_what_a_scheduler_step_is_in_epoch_mode():
    """warmup_steps is denominated in scheduler steps, and in the DEFAULT
    mode a scheduler step is an epoch -- which is why the node's default of
    100 never finishes ramping on a normal epoch budget (#308)."""
    desc = _desc("warmup_steps")
    assert "optimizer_step" in desc
    assert "epoch" in desc


def test_t_max_names_the_coupling_to_the_training_loop():
    desc = _desc("T_max")
    assert "trainingloop.epochs" in desc.replace(" ", "").replace("`", "")


def test_t_max_does_not_present_the_coupling_as_a_rule():
    """A truncated schedule is legitimate; warm restarts require inequality."""
    desc = _desc("T_max")
    assert "not enforced" in desc or "deliberately not" in desc


def test_t_max_documents_its_second_meaning():
    """The same value is passed as CosineAnnealingWarmRestarts' T_0."""
    assert "warmrestarts" in _desc("T_max").replace(" ", "")
    assert "t_0" in _desc("T_max")


def test_total_steps_warns_that_it_is_not_batches():
    desc = _desc("total_steps")
    assert "batch" in desc, "does not warn against the upstream meaning"


def test_gamma_lists_every_scheduler_that_reads_it():
    """It is also ReduceLROnPlateau's `factor`, which the old text omitted."""
    desc = _desc("gamma")
    for sched in ("steplr", "multisteplr", "exponentiallr", "reducelronplateau"):
        assert sched in desc.replace(" ", ""), f"gamma omits {sched}"


def test_step_size_documents_the_multistep_reuse():
    assert "multisteplr" in _desc("step_size").replace(" ", "")


def test_every_documented_scheduler_type_is_actually_supported():
    """Guards the descriptions against naming a type the node cannot build."""
    options = {o.lower() for o in _param("type").options}
    named = {"steplr", "multisteplr", "exponentiallr", "reducelronplateau",
             "cosineannealinglr", "cosineannealingwarmrestarts", "onecyclelr"}
    assert named <= options
