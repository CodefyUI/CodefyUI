"""Tests for CheckpointSaverNode and CheckpointLoaderNode."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from app.config import settings
from app.nodes.io.checkpoint_node import CheckpointLoaderNode, CheckpointSaverNode
from app.nodes.training.lr_scheduler_node import LRSchedulerNode


def _model_and_opt():
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    return model, opt


def test_saver_metadata():
    assert CheckpointSaverNode.NODE_NAME == "CheckpointSaver"
    assert CheckpointSaverNode.CATEGORY == "IO"


def test_loader_metadata():
    assert CheckpointLoaderNode.NODE_NAME == "CheckpointLoader"


def test_save_and_load_checkpoint_roundtrip():
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = "_ckpt_test.pt"
    try:
        model, opt = _model_and_opt()
        losses = torch.tensor([0.5, 0.3, 0.1])
        save_res = CheckpointSaverNode().execute(
            {"model": model, "optimizer": opt, "losses": losses},
            {"path": target, "epoch": 5},
        )
        assert save_res["model"] is model

        new_model, new_opt = _model_and_opt()
        load_res = CheckpointLoaderNode().execute(
            {"model": new_model, "optimizer": new_opt},
            {"path": target, "device": "cpu"},
        )
        assert load_res["epoch"] == 5
        assert torch.equal(load_res["losses"], losses)
        # Weights should match
        x = torch.randn(1, 4)
        assert torch.allclose(model(x), new_model(x))
    finally:
        (settings.MODELS_DIR / target).unlink(missing_ok=True)


def test_save_without_losses():
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = "_ckpt_no_loss.pt"
    try:
        model, opt = _model_and_opt()
        res = CheckpointSaverNode().execute(
            {"model": model, "optimizer": opt},
            {"path": target, "epoch": 0},
        )
        assert res["path"].endswith(".pt")
    finally:
        (settings.MODELS_DIR / target).unlink(missing_ok=True)


def test_load_missing_checkpoint_raises():
    with pytest.raises(FileNotFoundError):
        model, opt = _model_and_opt()
        CheckpointLoaderNode().execute(
            {"model": model, "optimizer": opt},
            {"path": "_does_not_exist.pt", "device": "cpu"},
        )


# --- scheduler state in the payload (#118) --------------------------------


# Every scheduler LRSchedulerNode can build, read off the node itself so a
# newly supported type is covered the moment it is added to the SELECT.
SCHEDULER_TYPES = next(
    p.options for p in LRSchedulerNode.define_params() if p.name == "type"
)

# Enough to satisfy every branch of LRSchedulerNode.execute() at once.
SCHEDULER_PARAMS = {
    "step_size": 2, "gamma": 0.5, "T_max": 10, "max_lr": 0.1, "total_steps": 20,
}


def _build_scheduler(sched_type, opt):
    return LRSchedulerNode().execute(
        {"optimizer": opt}, {"type": sched_type, **SCHEDULER_PARAMS}
    )["scheduler"]


def _advance(sched, opt, steps=3):
    """Walk ``steps`` epochs so the stored state is not the initial one."""
    plateau = isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)
    for i in range(steps):
        opt.step()
        # ReduceLROnPlateau is metric-driven; feed a worsening metric so its
        # bad-epoch counter moves and the state is genuinely non-trivial.
        sched.step(1.0 + i) if plateau else sched.step()
    return sched


def _advanced_scheduler(opt, steps=3):
    """A StepLR that has already walked ``steps`` epochs of its schedule."""
    return _advance(_build_scheduler("StepLR", opt), opt, steps)


@pytest.mark.parametrize("sched_type", SCHEDULER_TYPES)
def test_scheduler_state_roundtrip(sched_type):
    """Every supported scheduler must survive save + ``weights_only=True`` load.

    This is not only about the schedule. ``CheckpointLoader`` reads the whole
    file with ``weights_only=True``, so a scheduler state dict holding a type
    outside torch's unpickler allowlist would raise ``UnpicklingError`` and
    take the **model weights** down with it. The saver therefore has to store
    something the loader can always read back, for every type the
    ``LRScheduler`` node is able to produce -- and torch has changed what those
    state dicts contain between releases (this repo pins only ``torch>=2.0``,
    so CI resolves whatever is newest).

    The type to watch is ``MultiStepLR.milestones``, which is a
    ``collections.Counter`` rather than a plain dict. torch allowlists it on
    2.11 and 2.12; a release that stopped doing so, or a new scheduler holding
    something more exotic, would fail here rather than in a user's resume.
    """
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = f"_ckpt_sched_{sched_type}.pt"
    try:
        model, opt = _model_and_opt()
        sched = _advance(_build_scheduler(sched_type, opt), opt)

        CheckpointSaverNode().execute(
            {"model": model, "optimizer": opt, "lr_scheduler": sched},
            {"path": target, "epoch": 3},
        )

        # The whole payload, not just the scheduler key: an unreadable
        # scheduler state loses the model too.
        raw = torch.load(settings.MODELS_DIR / target, map_location="cpu", weights_only=True)
        assert raw["scheduler_class"] == type(sched).__name__
        assert set(raw["scheduler_state_dict"]) == set(sched.state_dict())
        assert raw["model_state_dict"].keys() == model.state_dict().keys()

        new_model, new_opt = _model_and_opt()
        new_sched = _build_scheduler(sched_type, new_opt)
        res = CheckpointLoaderNode().execute(
            {"model": new_model, "optimizer": new_opt, "lr_scheduler": new_sched},
            {"path": target, "device": "cpu"},
        )

        assert res["lr_scheduler"] is new_sched
        assert new_sched.state_dict() == sched.state_dict()
        assert new_sched.last_epoch == sched.last_epoch
        if hasattr(sched, "_step_count"):
            assert new_sched._step_count == sched._step_count
        if hasattr(sched, "get_last_lr"):
            assert new_sched.get_last_lr() == sched.get_last_lr()
    finally:
        (settings.MODELS_DIR / target).unlink(missing_ok=True)


def test_checkpoint_omits_scheduler_key_when_none_wired():
    """The key is additive: an unwired scheduler leaves the payload as it was."""
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = "_ckpt_no_sched.pt"
    try:
        model, opt = _model_and_opt()
        CheckpointSaverNode().execute(
            {"model": model, "optimizer": opt},
            {"path": target, "epoch": 1},
        )
        raw = torch.load(settings.MODELS_DIR / target, map_location="cpu", weights_only=True)
        assert set(raw) == {"epoch", "model_state_dict", "optimizer_state_dict"}
    finally:
        (settings.MODELS_DIR / target).unlink(missing_ok=True)


def test_loader_refuses_scheduler_state_from_a_different_class():
    """StepLR state spliced into a CosineAnnealingLR would corrupt it."""
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = "_ckpt_sched_mismatch.pt"
    try:
        model, opt = _model_and_opt()
        CheckpointSaverNode().execute(
            {"model": model, "optimizer": opt, "lr_scheduler": _advanced_scheduler(opt)},
            {"path": target, "epoch": 3},
        )

        new_model, new_opt = _model_and_opt()
        other = torch.optim.lr_scheduler.CosineAnnealingLR(new_opt, T_max=10)
        CheckpointLoaderNode().execute(
            {"model": new_model, "optimizer": new_opt, "lr_scheduler": other},
            {"path": target, "device": "cpu"},
        )

        assert other.last_epoch == 0
        assert not hasattr(other, "step_size")
    finally:
        (settings.MODELS_DIR / target).unlink(missing_ok=True)


def test_saver_rejects_a_non_scheduler_on_the_lr_scheduler_port():
    """The port is ANY, so a mis-wire must fail with a readable message."""
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = "_ckpt_bad_sched.pt"
    try:
        model, opt = _model_and_opt()
        with pytest.raises(ValueError, match="no state_dict"):
            CheckpointSaverNode().execute(
                {"model": model, "optimizer": opt, "lr_scheduler": torch.tensor([1.0])},
                {"path": target, "epoch": 1},
            )
        assert not (settings.MODELS_DIR / target).exists(), (
            "the failure must happen before anything is written"
        )
    finally:
        (settings.MODELS_DIR / target).unlink(missing_ok=True)


def test_loader_reports_no_scheduler_when_none_wired():
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = "_ckpt_sched_unused.pt"
    try:
        model, opt = _model_and_opt()
        CheckpointSaverNode().execute(
            {"model": model, "optimizer": opt, "lr_scheduler": _advanced_scheduler(opt)},
            {"path": target, "epoch": 3},
        )
        new_model, new_opt = _model_and_opt()
        res = CheckpointLoaderNode().execute(
            {"model": new_model, "optimizer": new_opt},
            {"path": target, "device": "cpu"},
        )
        assert res["lr_scheduler"] is None
        assert res["epoch"] == 3
    finally:
        (settings.MODELS_DIR / target).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Atomic write (#122)
# ---------------------------------------------------------------------------


def test_a_failed_save_leaves_the_previous_checkpoint_intact(tmp_path, monkeypatch):
    """A crash mid-``torch.save`` must not truncate the file already there.

    A checkpoint is only ever read after something went wrong, so "the save
    that died left an unloadable file" is the worst possible failure mode.
    ``write_checkpoint`` stages to a sibling temp file and ``os.replace``s
    it, which is atomic on the same volume.
    """
    from app.core import checkpoints

    models = tmp_path / "data" / "models"
    models.mkdir(parents=True)
    monkeypatch.setattr(settings, "MODELS_DIR", models)

    model, opt = _model_and_opt()
    target = checkpoints.write_checkpoint("run.pt", model, opt, epoch=1)
    good = target.read_bytes()

    real_save = torch.save

    def exploding_save(obj, path, *args, **kwargs):
        # Write a partial file first, exactly as a disk-full or a killed
        # process would, then fail.
        with open(path, "wb") as fh:
            fh.write(b"truncated")
        raise RuntimeError("disk full")

    monkeypatch.setattr(torch, "save", exploding_save)
    with pytest.raises(RuntimeError, match="disk full"):
        checkpoints.write_checkpoint("run.pt", model, opt, epoch=2)

    monkeypatch.setattr(torch, "save", real_save)
    assert target.read_bytes() == good, "the good checkpoint was clobbered"
    assert list(models.glob("*.tmp")) == [], "staging debris was left behind"
    # ...and it still loads.
    restored = torch.load(str(target), map_location="cpu", weights_only=True)
    assert restored["epoch"] == 1
