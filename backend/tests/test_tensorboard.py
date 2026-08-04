"""TensorBoard event files (core#136), read back by TensorBoard itself.

``core.tensorboard`` hand-encodes TFRecord and two protobuf messages so the
feature costs an install nothing. The whole point of that trade is that the
format is verified against the real reader rather than against a reader from
the same file -- so every assertion about what was written goes through
``tensorboard.backend.event_processing``, which is in the ``dev`` extra and
is a TEST dependency only.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from app.core.execution_context import ArtifactSignal, ExecutionContext
from app.core.tensorboard import (
    ARTIFACT_KIND,
    EventFileWriter,
    close_and_register,
    crc32c,
    event_filename,
    open_run_writer,
    run_logdir,
)
from app.nodes.training.training_loop_node import TrainingLoopNode

event_accumulator = pytest.importorskip(
    "tensorboard.backend.event_processing.event_accumulator",
    reason="the `dev` extra installs tensorboard; see backend/pyproject.toml",
)


def _read(logdir: Path) -> dict[str, list[tuple[int, float]]]:
    """Every scalar series under *logdir*, through TensorBoard's own reader."""
    accumulator = event_accumulator.EventAccumulator(str(logdir))
    accumulator.Reload()
    return {
        tag: [(point.step, point.value)
              for point in accumulator.Scalars(tag)]
        for tag in accumulator.Tags()["scalars"]
    }


# ── the framing ──────────────────────────────────────────────────────────


def test_crc32c_matches_the_published_check_value():
    """Pins the POLYNOMIAL, not just the round trip.

    ``zlib.crc32`` is a different polynomial and would produce a file every
    reader rejects, while a writer-and-reader pair from this repo would
    happily agree on it. 0xE3069283 for "123456789" is CRC-32C's standard
    check value.
    """
    assert crc32c(b"123456789") == 0xE3069283


def test_the_filename_is_the_one_tensorboards_scan_looks_for(tmp_path):
    name = event_filename(now=1700000000)
    assert name.startswith("events.out.tfevents.1700000000.")


def test_run_logdir_sits_under_the_data_root(monkeypatch, tmp_path):
    from app.config import settings

    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "data" / "models")
    assert run_logdir("abc123", "loop1") == (
        tmp_path / "data" / "runs" / "abc123" / "tb" / "loop1")


def test_run_logdir_sanitises_ids_that_are_not_filename_safe(
        monkeypatch, tmp_path):
    from app.config import settings

    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "data" / "models")
    logdir = run_logdir("../escape", "a/b")
    assert logdir.is_relative_to(tmp_path / "data" / "runs")


# ── what the reader sees ─────────────────────────────────────────────────


def test_scalars_round_trip_through_the_real_reader(tmp_path):
    with EventFileWriter(tmp_path) as writer:
        for step in range(1, 4):
            writer.add_scalar("train_loss", 1.0 / step, step)
            writer.add_scalar("lr", 0.01, step)

    series = _read(tmp_path)
    assert set(series) == {"train_loss", "lr"}
    assert [step for step, _ in series["train_loss"]] == [1, 2, 3]
    values = [value for _, value in series["train_loss"]]
    assert values == pytest.approx([1.0, 0.5, 1 / 3], rel=1e-6)


def test_a_file_with_no_scalars_is_still_a_valid_event_file(tmp_path):
    """The version record alone has to parse, or an empty run looks corrupt."""
    EventFileWriter(tmp_path).close()
    assert _read(tmp_path) == {}


def test_a_tag_with_a_slash_keeps_its_grouping(tmp_path):
    """TensorBoard groups by the part before the slash; it must survive."""
    with EventFileWriter(tmp_path) as writer:
        writer.add_scalar("loss/train", 1.0, 1)
        writer.add_scalar("loss/val", 2.0, 1)
    assert set(_read(tmp_path)) == {"loss/train", "loss/val"}


def test_non_finite_values_are_dropped_rather_than_written(tmp_path):
    with EventFileWriter(tmp_path) as writer:
        writer.add_scalar("loss", 1.0, 1)
        writer.add_scalar("loss", float("nan"), 2)
        writer.add_scalar("loss", float("inf"), 3)
        writer.add_scalar("loss", 4.0, 4)
    assert [step for step, _ in _read(tmp_path)["loss"]] == [1, 4]


def test_a_non_numeric_value_is_ignored_instead_of_raising(tmp_path):
    with EventFileWriter(tmp_path) as writer:
        writer.add_scalar("loss", "not a number", 1)
        writer.add_scalar("loss", 2.0, 2)
    assert [step for step, _ in _read(tmp_path)["loss"]] == [2]


def test_writing_after_close_is_a_no_op(tmp_path):
    writer = EventFileWriter(tmp_path)
    writer.add_scalar("loss", 1.0, 1)
    writer.close()
    writer.add_scalar("loss", 2.0, 2)
    writer.close()  # idempotent
    assert [step for step, _ in _read(tmp_path)["loss"]] == [1]


def test_a_writer_that_is_never_closed_still_flushes(tmp_path):
    """The exception path: a node that raises never reaches its close."""
    writer = EventFileWriter(tmp_path)
    writer.add_scalar("loss", 1.0, 1)
    del writer
    assert [step for step, _ in _read(tmp_path)["loss"]] == [1]


# ── the run-facing helpers ───────────────────────────────────────────────


def _recording_context(tmp_path, monkeypatch) -> ExecutionContext:
    from app.config import settings

    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "data" / "models")
    context = ExecutionContext(signals_recorded=True)
    context.current_node_id = "loop1"
    return context


def test_no_writer_when_the_run_cannot_record_artifacts(tmp_path, monkeypatch):
    """No row, no file -- the rule ``core.checkpoints`` states in full."""
    from app.config import settings

    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "data" / "models")
    context = ExecutionContext(signals_recorded=False)
    assert open_run_writer(context) is None
    assert not (tmp_path / "data" / "runs").exists()


def test_no_writer_without_a_context():
    assert open_run_writer(None) is None


def test_close_and_register_queues_the_directory_not_the_file(
        tmp_path, monkeypatch):
    context = _recording_context(tmp_path, monkeypatch)
    writer = open_run_writer(context)
    assert writer is not None
    writer.add_scalar("loss", 1.0, 1)
    logdir = close_and_register(writer, context)

    items, _dropped = context.outbox.drain()
    artifacts = [item for item in items if isinstance(item, ArtifactSignal)]
    assert len(artifacts) == 1
    assert artifacts[0].kind == ARTIFACT_KIND
    assert artifacts[0].path == logdir
    assert Path(logdir).is_dir()
    assert artifacts[0].meta["events"].startswith("events.out.tfevents.")


def test_close_and_register_tolerates_no_writer():
    assert close_and_register(None, ExecutionContext()) is None


# ── through TrainingLoop ─────────────────────────────────────────────────


def _tiny_training(params, context):
    torch.manual_seed(0)
    features = torch.randn(8, 4)
    targets = torch.randint(0, 2, (8,))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, targets), batch_size=4)
    model = nn.Linear(4, 2)
    return TrainingLoopNode().execute(
        {"model": model, "dataloader": loader,
         "optimizer": torch.optim.SGD(model.parameters(), lr=0.01),
         "loss_fn": nn.CrossEntropyLoss()},
        {"epochs": 2, "device": "cpu", **params},
        context=context,
    )


def test_training_loop_writes_the_same_series_it_records(tmp_path, monkeypatch):
    context = _recording_context(tmp_path, monkeypatch)
    result = _tiny_training({"tensorboard": True}, context)

    logdir = result["metrics"]["tensorboard_logdir"]
    series = _read(Path(logdir))
    assert set(series) == {"train_loss", "lr"}
    assert [step for step, _ in series["train_loss"]] == [1, 2]

    # The same numbers the run's own metric store got, to the float32 the
    # event file stores them in.
    from app.core.execution_context import MetricSignal

    items, _dropped = context.outbox.drain()
    recorded = [item for item in items
                if isinstance(item, MetricSignal) and item.name == "train_loss"]
    assert [point.step for point in recorded] == [1, 2]
    assert [value for _, value in series["train_loss"]] == pytest.approx(
        [point.value for point in recorded], rel=1e-6)


def test_training_loop_registers_the_logdir_as_an_artifact(
        tmp_path, monkeypatch):
    context = _recording_context(tmp_path, monkeypatch)
    result = _tiny_training({"tensorboard": True}, context)

    items, _dropped = context.outbox.drain()
    artifacts = [item for item in items if isinstance(item, ArtifactSignal)]
    assert [item.kind for item in artifacts] == [ARTIFACT_KIND]
    assert artifacts[0].path == result["metrics"]["tensorboard_logdir"]
    assert artifacts[0].node_id == "loop1"


def test_training_loop_writes_nothing_when_the_param_is_off(
        tmp_path, monkeypatch):
    context = _recording_context(tmp_path, monkeypatch)
    result = _tiny_training({}, context)
    assert "tensorboard_logdir" not in result["metrics"]
    assert not (tmp_path / "data" / "runs").exists()


def test_training_loop_skips_it_when_the_run_records_no_artifacts(
        tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "data" / "models")
    context = ExecutionContext(signals_recorded=False)
    context.current_node_id = "loop1"
    result = _tiny_training({"tensorboard": True}, context)
    assert "tensorboard_logdir" not in result["metrics"]
    assert not (tmp_path / "data" / "runs").exists()


def test_batch_metrics_reach_tensorboard_too(tmp_path, monkeypatch):
    context = _recording_context(tmp_path, monkeypatch)
    result = _tiny_training(
        {"tensorboard": True, "batch_metrics": True}, context)
    series = _read(Path(result["metrics"]["tensorboard_logdir"]))
    # 8 samples, batch_size 4, 2 epochs -> global batches 1..4
    assert [step for step, _ in series["train_loss_batch"]] == [1, 2, 3, 4]


def test_two_training_nodes_of_one_run_get_separate_logdirs(
        tmp_path, monkeypatch):
    """TensorBoard colours each sub-directory of --logdir as its own run."""
    context = _recording_context(tmp_path, monkeypatch)
    first = _tiny_training({"tensorboard": True}, context)
    context.current_node_id = "loop2"
    second = _tiny_training({"tensorboard": True}, context)
    assert (first["metrics"]["tensorboard_logdir"]
            != second["metrics"]["tensorboard_logdir"])
    assert Path(first["metrics"]["tensorboard_logdir"]).parent == \
        Path(second["metrics"]["tensorboard_logdir"]).parent


def test_the_param_is_advanced_and_off_by_default():
    param = [p for p in TrainingLoopNode.define_params()
             if p.name == "tensorboard"][0]
    assert param.default is False
    assert param.advanced is True
