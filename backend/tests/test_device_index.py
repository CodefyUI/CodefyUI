"""Addressing a specific GPU by index: ``cuda:N`` end to end (#135).

Everything here runs without a graphics card. ``torch.cuda.device_count``
and ``torch.cuda.is_available`` are monkeypatched to describe a machine with
four cards, which is the only way to test multi-GPU behaviour on a box that
has one -- and the behaviour under test is string handling and option
enumeration, not anything the driver does.
"""

from __future__ import annotations

import pytest
import torch

from app.core import device_utils
from app.core.device_utils import (
    cuda_device_count,
    describe_accelerator,
    device_options,
    get_available_devices,
    resolve_device,
    split_device,
)
from app.core.run_service import DEVICE_PATTERN, canonical_queue_key


@pytest.fixture
def four_cards(monkeypatch):
    """Pretend this machine has four CUDA devices, numbered 0 to 3."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name",
                        lambda index=0: f"Fake GPU {index}")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    get_available_devices.cache_clear()
    yield
    get_available_devices.cache_clear()


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    get_available_devices.cache_clear()
    yield
    get_available_devices.cache_clear()


# ── parsing ───────────────────────────────────────────────────────────────


def test_split_device_separates_the_kind_from_the_index():
    assert split_device("cuda:1") == ("cuda", 1)
    assert split_device("cuda") == ("cuda", None)
    assert split_device("cpu") == ("cpu", None)
    assert split_device("mps:0") == ("mps", 0)


def test_an_unparseable_index_reads_as_no_index_rather_than_raising():
    """``None`` and ``0`` mean different things and must stay distinct."""
    assert split_device("cuda:x") == ("cuda", None)
    assert split_device("") == ("", None)


# ── resolution ────────────────────────────────────────────────────────────


def test_a_valid_index_survives(four_cards):
    for index in range(4):
        assert resolve_device(f"cuda:{index}") == f"cuda:{index}"


def test_an_out_of_range_index_degrades_to_the_current_card(four_cards, caplog):
    """Not to the CPU. The user asked for a GPU; they get one, loudly.

    Answering "train on GPU 7" with a forty-minute CPU run is a worse
    surprise than answering it with the card that exists -- and the warning
    names the count so the substitution is visible.
    """
    with caplog.at_level("WARNING"):
        assert resolve_device("cuda:7") == "cuda:0"
    assert "cuda:0" in caplog.text and "4" in caplog.text


def test_a_negative_index_is_out_of_range_too(four_cards):
    assert resolve_device("cuda:-1") == "cuda:0"


def test_an_index_on_a_machine_with_no_cuda_still_lands_on_the_cpu(no_cuda):
    assert resolve_device("cuda:2") == "cpu"
    assert resolve_device("cuda") == "cpu"


def test_mps_has_no_index_vocabulary_beyond_zero(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("mps:0") == "mps:0"
    assert resolve_device("mps:1") == "mps"


def test_nonsense_still_degrades_to_the_cpu(four_cards):
    assert resolve_device("cudda:0") == "cpu"
    assert resolve_device("auto") == "cpu"


# ── enumeration ───────────────────────────────────────────────────────────


def test_multiple_cards_are_listed_individually(four_cards):
    assert get_available_devices() == [
        "cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    assert cuda_device_count() == 4


def test_a_single_card_is_not_listed_twice(monkeypatch):
    """``cuda`` and ``cuda:0`` are the same hardware; offering both is a
    choice between two spellings of one answer."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    get_available_devices.cache_clear()
    try:
        assert "cuda" in get_available_devices()
        assert "cuda:0" not in get_available_devices()
    finally:
        get_available_devices.cache_clear()


def test_node_device_options_expand_to_every_card(four_cards):
    assert device_options("device", ["auto", "cpu", "cuda", "mps"]) == [
        "auto", "cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]


def test_node_device_options_still_drop_absent_backends(no_cuda):
    assert device_options("device", ["auto", "cpu", "cuda", "mps"]) == [
        "auto", "cpu"]


def test_a_param_that_is_not_a_device_is_untouched(four_cards):
    assert device_options("mode", ["a", "b"]) == ["a", "b"]
    assert device_options("device", []) == []


def test_the_index_ordering_is_numeric_not_lexical(monkeypatch):
    """``cuda:10`` must not sort before ``cuda:2``."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 11)
    get_available_devices.cache_clear()
    try:
        options = device_options("device", ["cpu", "cuda"])
        assert options[-3:] == ["cuda:8", "cuda:9", "cuda:10"]
    finally:
        get_available_devices.cache_clear()


def test_the_global_selector_labels_each_card(four_cards):
    described = describe_accelerator()
    values = [d["value"] for d in described["devices"]]
    assert values == ["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    assert described["default"] == "cuda"
    per_card = [d for d in described["devices"] if d["value"] == "cuda:2"][0]
    assert per_card["label"].endswith("#2")
    assert per_card["detail"] == "Fake GPU 2"


def test_the_global_selector_keeps_one_entry_for_one_card(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=0: "One")
    get_available_devices.cache_clear()
    try:
        values = [d["value"] for d in describe_accelerator()["devices"]]
        assert values == ["cpu", "cuda"]
    finally:
        get_available_devices.cache_clear()


# ── the run queue ─────────────────────────────────────────────────────────


def test_the_submit_pattern_accepts_an_index():
    for device in ("cpu", "auto", "cuda", "cuda:0", "cuda:3", "mps", "mps:0"):
        assert DEVICE_PATTERN.match(device), device
    for device in ("cudda", "cuda:", "cuda:x", "gpu0", "cuda:1:2"):
        assert DEVICE_PATTERN.match(device) is None, device


def test_each_card_gets_its_own_queue(four_cards):
    """One FIFO per physical device, so a job on card 1 never waits on card 0."""
    assert canonical_queue_key("cuda:1") == "cuda:1"
    assert canonical_queue_key("cuda:3") == "cuda:3"
    # A bare ``cuda`` collapses onto the index it actually means, so
    # ``--device cuda`` and ``--device cuda:0`` share one queue instead of
    # running concurrently over one card's VRAM.
    assert canonical_queue_key("cuda") == "cuda:0"
    assert canonical_queue_key("cpu") == "cpu"


def test_a_bad_index_cannot_open_a_queue_for_a_card_that_is_not_there(
        four_cards):
    """The queue key derives from the RESOLVED device, so the substitution
    that ``resolve_device`` makes is the one the queue sees."""
    assert canonical_queue_key(resolve_device("cuda:9")) == "cuda:0"


# ── nodes ─────────────────────────────────────────────────────────────────


def test_every_device_param_offers_the_indexed_forms(four_cards):
    """The node API is the only place a saved graph's device can come from."""
    from app.api.routes_nodes import _node_to_definition
    from app.core.node_registry import registry

    checked = 0
    for name, cls in registry.nodes.items():
        options = {p.name: p.options for p in cls.define_params()}
        if "device" not in options or "cuda" not in options["device"]:
            continue
        definition = _node_to_definition(name, cls)
        device_param = next(p for p in definition.params if p.name == "device")
        assert "cuda:2" in device_param.options, name
        checked += 1
    assert checked >= 4, "expected several device-aware nodes"
