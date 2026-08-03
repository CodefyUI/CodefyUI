"""Mixed precision: what each device honours, and what it buys (#135).

Everything down to ``requires_cuda`` runs on a CPU, because CPU autocast is
real autocast -- the dtype promotion rules, the ``GradScaler`` and the
checkpoint round-trip are all exercisable without a graphics card. The two
assertions that genuinely need one (does bf16 converge like fp32, and does
it use less memory) are marked and skipped elsewhere.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from app.core.amp import PRECISIONS, AmpPolicy, normalize_precision, supported_precisions
from app.core.checkpoints import build_checkpoint
from app.nodes.training.training_loop_node import TrainingLoopNode

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ── resolution ────────────────────────────────────────────────────────────


def test_the_three_precisions_are_the_declared_vocabulary():
    assert PRECISIONS == ("fp32", "bf16", "fp16")
    param = next(p for p in TrainingLoopNode.define_params()
                 if p.name == "precision")
    assert param.options == list(PRECISIONS)
    assert param.default == "fp32"
    # Advanced, not basic: a classroom's first training loop should not open
    # with a dtype question.
    assert param.advanced is True


def test_an_unknown_precision_degrades_to_fp32_instead_of_raising():
    assert normalize_precision("float8") == "fp32"
    assert normalize_precision(None) == "fp32"
    assert normalize_precision(" BF16 ") == "bf16"


def test_cpu_honours_every_precision():
    assert supported_precisions("cpu") == PRECISIONS


def test_mps_declines_everything_but_fp32():
    assert supported_precisions("mps") == ("fp32",)
    policy = AmpPolicy.for_device("bf16", "mps")
    assert policy.precision == "fp32"
    assert policy.requested == "bf16"
    assert policy.fell_back is True


def test_fp32_is_a_pure_no_op():
    policy = AmpPolicy.for_device("fp32", "cpu")
    assert policy.enabled is False
    assert policy.uses_scaler is False
    assert policy.state_dict() is None
    loss = torch.tensor(2.0)
    assert policy.scale(loss) is loss
    with policy.autocast():
        # No promotion, no demotion: float64 stays float64, which autocast
        # would not allow.
        assert (torch.ones(2, dtype=torch.float64)
                @ torch.ones(2, dtype=torch.float64)).dtype == torch.float64


def test_bf16_autocasts_without_a_scaler():
    policy = AmpPolicy.for_device("bf16", "cpu")
    assert policy.precision == "bf16"
    assert policy.uses_scaler is False
    with policy.autocast():
        assert (torch.randn(4, 4) @ torch.randn(4, 4)).dtype == torch.bfloat16


def test_fp16_autocasts_and_brings_a_scaler():
    policy = AmpPolicy.for_device("fp16", "cpu")
    assert policy.precision == "fp16"
    assert policy.uses_scaler is True
    with policy.autocast():
        assert (torch.randn(4, 4) @ torch.randn(4, 4)).dtype == torch.float16
    loss = torch.tensor(1.0)
    assert policy.scale(loss).item() > loss.item()


def test_a_device_string_with_an_index_resolves_by_type():
    policy = AmpPolicy.for_device("fp32", "cuda:3")
    assert policy.device_type == "cuda"


# ── scaler state across a checkpoint ──────────────────────────────────────


def test_the_scaler_state_round_trips_through_a_checkpoint_payload():
    policy = AmpPolicy.for_device("fp16", "cpu")
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    payload = build_checkpoint(model, optimizer,
                               scaler_state=policy.state_dict())
    assert payload["scaler_state_dict"]["scale"] == policy.scaler.get_scale()
    # Plain scalars only -- the format requires weights_only=True unpickling.
    assert all(isinstance(v, (int, float))
               for v in payload["scaler_state_dict"].values())

    # Restored into a FRESH policy, from a scale deliberately unlike the
    # default one a fresh GradScaler starts at -- otherwise the assertion
    # would pass just as well against a load that did nothing.
    saved = dict(payload["scaler_state_dict"], scale=1024.0)
    restored = AmpPolicy.for_device("fp16", "cpu")
    assert restored.scaler.get_scale() != 1024.0
    assert restored.load_state_dict(saved) is True
    assert restored.scaler.get_scale() == 1024.0


def test_a_live_scaler_is_accepted_where_a_state_dict_is():
    policy = AmpPolicy.for_device("fp16", "cpu")
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    payload = build_checkpoint(model, optimizer, scaler_state=policy.scaler)
    assert payload["scaler_state_dict"]["scale"] == policy.scaler.get_scale()


def test_a_checkpoint_without_a_scaler_has_no_scaler_key():
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    assert "scaler_state_dict" not in build_checkpoint(model, optimizer)
    assert "scaler_state_dict" not in build_checkpoint(
        model, optimizer, scaler_state=None)


def test_nonsense_wired_into_the_scaler_port_does_not_lose_the_checkpoint():
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    payload = build_checkpoint(model, optimizer, scaler_state="not a scaler")
    assert "scaler_state_dict" not in payload
    assert "model_state_dict" in payload


def test_restoring_into_a_run_with_no_scaler_declines_quietly():
    policy = AmpPolicy.for_device("bf16", "cpu")
    assert policy.load_state_dict({"scale": 512.0}) is False
    assert policy.load_state_dict(None) is False


# ── the node ──────────────────────────────────────────────────────────────


def _dataset(n: int = 16) -> torch.utils.data.Dataset:
    x = torch.linspace(-1.0, 1.0, n * 4).reshape(n, 4)
    y = torch.linspace(1.0, -1.0, n * 2).reshape(n, 2)
    return torch.utils.data.TensorDataset(x, y)


def _train(precision: str, **params):
    torch.manual_seed(7)
    model = nn.Linear(4, 2)
    loader = torch.utils.data.DataLoader(_dataset(), batch_size=4)
    return TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": loader,
            "optimizer": torch.optim.SGD(model.parameters(), lr=0.05),
            "loss_fn": nn.MSELoss(),
            **params.pop("inputs", {}),
        },
        {"epochs": 2, "device": "cpu", "precision": precision, **params},
    )


def test_the_node_reports_the_precision_it_actually_ran_in():
    result = _train("bf16")
    assert result["metrics"]["precision"] == "bf16"
    # No fallback happened on a CPU, so the request is not repeated.
    assert "precision_requested" not in result["metrics"]
    assert result["grad_scaler_state"] is None


def test_an_fp16_run_hands_its_loss_scale_out_for_a_checkpoint():
    result = _train("fp16")
    state = result["grad_scaler_state"]
    assert isinstance(state, dict) and "scale" in state
    assert result["metrics"]["skipped_steps"] >= 0


def test_an_fp16_run_resumes_from_the_scale_it_was_handed():
    handed = {"scale": 1024.0, "growth_factor": 2.0, "backoff_factor": 0.5,
              "growth_interval": 2000, "_growth_tracker": 0}
    result = _train("fp16", inputs={"grad_scaler_state": handed}, epochs=1)
    # Four clean steps at growth_interval 2000 cannot grow the scale, so it
    # is still the value that came in -- which is the point: a fresh scaler
    # would report 65536.0 here.
    assert result["grad_scaler_state"]["scale"] == 1024.0


def test_fp32_training_is_unchanged_by_the_precision_plumbing():
    """The default path must be bit-identical to the pre-#135 loop."""
    explicit = _train("fp32")
    torch.manual_seed(7)
    model = nn.Linear(4, 2)
    loader = torch.utils.data.DataLoader(_dataset(), batch_size=4)
    omitted = TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": loader,
            "optimizer": torch.optim.SGD(model.parameters(), lr=0.05),
            "loss_fn": nn.MSELoss(),
        },
        {"epochs": 2, "device": "cpu"},
    )
    assert torch.equal(explicit["losses"], omitted["losses"])


def test_bf16_still_learns_on_a_cpu():
    """Not a convergence claim -- that one needs a GPU and is marked below.

    This is the weaker statement that the autocast path is wired up at all:
    a loop whose backward never reached the optimizer would produce a flat
    loss curve rather than a falling one.
    """
    result = _train("bf16", epochs=8)
    losses = result["losses"]
    assert losses[-1].item() < losses[0].item()


# ── the two assertions that need real hardware ────────────────────────────


def _conv_net() -> nn.Module:
    """A small conv net over CIFAR-shaped inputs (3 x 32 x 32).

    Deliberately activation-heavy rather than parameter-heavy: autocast
    leaves parameters in float32 and halves ACTIVATIONS, so a model whose
    memory is dominated by its weights would show no saving at all and the
    measurement below would be measuring nothing.
    """
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10),
    )


def _cifar_shaped_fixture(n: int = 512) -> torch.utils.data.Dataset:
    """Synthetic CIFAR-shaped data (3 x 32 x 32, ten classes) that is learnable.

    There is no CIFAR download in this repository and a test must not fetch
    one, so the fixture is CIFAR's SHAPE with a signal a small conv net can
    actually find: each class is a distinct (colour channel, amplitude)
    pair over Gaussian noise. Ten classes from three channels and four
    amplitudes, all distinct.

    The point is not realism -- it is that "did bf16 converge like fp32"
    needs a run where both of them converge, inside a test's runtime.
    """
    generator = torch.Generator().manual_seed(20260801)
    labels = torch.randint(0, 10, (n,), generator=generator)
    images = torch.randn(n, 3, 32, 32, generator=generator) * 0.2
    for index in range(10):
        images[labels == index, index % 3] += 1.0 + (index // 3)
    return torch.utils.data.TensorDataset(images, labels)


def _train_conv(precision: str, *, epochs: int = 8, batch_size: int = 128):
    torch.manual_seed(4242)
    model = _conv_net()
    loader = torch.utils.data.DataLoader(
        _cifar_shaped_fixture(), batch_size=batch_size, shuffle=False)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    result = TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": loader,
            "optimizer": torch.optim.SGD(model.parameters(), lr=0.1,
                                         momentum=0.9),
            "loss_fn": nn.CrossEntropyLoss(),
        },
        {"epochs": epochs, "device": "cuda", "precision": precision},
    )
    peak = torch.cuda.max_memory_allocated()
    del result["model"]
    torch.cuda.empty_cache()
    return result, peak


@requires_cuda
def test_bf16_converges_like_fp32_and_costs_less_memory():
    """The acceptance criterion, on the card that is actually present.

    Two claims in one test because they are one experiment: the same graph,
    the same seed, the same data, one knob changed.
    """
    fp32, fp32_peak = _train_conv("fp32")
    bf16, bf16_peak = _train_conv("bf16")

    assert bf16["metrics"]["precision"] == "bf16", (
        "this card reports no bfloat16 support; the comparison below would "
        "be fp32 against fp32")

    fp32_losses = [v.item() for v in fp32["losses"]]
    bf16_losses = [v.item() for v in bf16["losses"]]
    # Both learned: the last epoch is well below the first.
    assert fp32_losses[-1] < fp32_losses[0] * 0.75
    assert bf16_losses[-1] < bf16_losses[0] * 0.75
    # And they learned the SAME thing, within bfloat16's much coarser
    # mantissa. 5% of the fp32 loss is far tighter than the gap between a
    # converging run and a diverging one, and far looser than bf16 rounding.
    assert bf16_losses[-1] == pytest.approx(fp32_losses[-1], rel=0.05,
                                            abs=0.05)

    # A margin, not merely "less": a one-byte difference would satisfy a
    # bare `<` while proving nothing about activation memory.
    assert bf16_peak < fp32_peak * 0.9, (
        f"bf16 peak {bf16_peak} was not meaningfully below fp32 peak "
        f"{fp32_peak}")


@requires_cuda
def test_an_out_of_range_cuda_index_still_trains():
    """``cuda:9`` on a machine with fewer cards lands on a real device.

    The degradation is deliberate (see ``device_utils.resolve_device``): a
    user who asked for a GPU gets a GPU, not a silent forty-minute CPU run.
    """
    torch.manual_seed(3)
    model = nn.Linear(4, 2)
    loader = torch.utils.data.DataLoader(_dataset(), batch_size=4)
    result = TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": loader,
            "optimizer": torch.optim.SGD(model.parameters(), lr=0.05),
            "loss_fn": nn.MSELoss(),
        },
        {"epochs": 1, "device": "cuda:9"},
    )
    assert next(result["model"].parameters()).device.type == "cuda"


@requires_cuda
def test_accumulation_and_bf16_compose():
    """Both memory levers at once, on the device they were written for."""
    result, _ = _train_conv("bf16", epochs=1)
    assert result["metrics"]["total_steps"] == 4  # 512 samples / batch 128

    torch.manual_seed(4242)
    model = _conv_net()
    loader = torch.utils.data.DataLoader(
        _cifar_shaped_fixture(), batch_size=32, shuffle=False)
    accumulated = TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": loader,
            "optimizer": torch.optim.SGD(model.parameters(), lr=0.05,
                                         momentum=0.9),
            "loss_fn": nn.CrossEntropyLoss(),
        },
        {"epochs": 1, "device": "cuda", "precision": "bf16",
         "accumulate_steps": 4},
    )
    assert accumulated["metrics"]["precision"] == "bf16"
    assert accumulated["metrics"]["total_batches"] == 16
    assert accumulated["metrics"]["total_steps"] == 4
    del accumulated["model"]
    torch.cuda.empty_cache()
