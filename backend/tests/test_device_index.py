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


def test_a_negative_index_is_rejected(four_cards):
    """Caught by the syntax check (``\\d+``), so it lands where a malformed
    string does -- which is the same place an out-of-range one does."""
    assert resolve_device("cuda:-1") == "cuda:0"


# ── malformed strings, at every entry point they can arrive through ───────


@pytest.mark.parametrize("bad", [
    "cuda:abc",     # a typo, or an unexpanded ${CUDA_INDEX}
    "cuda:",        # a trailing separator
    "cuda:0:1",     # two indices
    "cuda:1e3",     # scientific notation; int() rejects it, humans do not
    "cuda: 0",      # a space -- int(" 0") is 0, so only a SYNTAX check sees it
    "cuda : 0",     # ... and the same mistake one character earlier
    "cuda:+1",      # a sign
    "cuda:\u0660",       # a non-ASCII digit; int() accepts it, torch does not
])
def test_a_malformed_index_never_reaches_torch(four_cards, bad):
    """Every one of these used to be returned VERBATIM.

    torch then answered ``RuntimeError: Invalid device string``, naming
    neither the graph nor the parameter the string came from. They arrive
    through the exported script's free-form ``--device`` and through a
    hand-edited SELECT param, neither of which a run submission validates.
    """
    resolved = resolve_device(bad)
    assert resolved == "cuda:0", bad
    # The real proof: torch accepts what came back.
    assert torch.device(resolved).type == "cuda"


@pytest.mark.parametrize("bad", ["cuda:abc", "cuda:", "cuda:0:1", "cuda: 0"])
def test_a_malformed_index_lands_on_the_cpu_when_there_is_no_cuda(no_cuda, bad):
    assert resolve_device(bad) == "cpu"


def test_a_malformed_mps_index_is_rejected_too(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("mps:abc") == "mps"
    assert resolve_device("mps:") == "mps"


def test_an_unrecognised_kind_still_goes_to_the_cpu(four_cards):
    assert resolve_device("gpu:0") == "cpu"
    assert resolve_device("cudda:0") == "cpu"


def test_the_resolved_string_is_rebuilt_rather_than_echoed(four_cards):
    """``cuda: 0`` is why. The space survives ``int()``, so a
    parse-and-check would accept it and hand torch a string it rejects."""
    assert resolve_device("  CUDA:2  ") == "cuda:2"
    assert resolve_device("CUDA") == "cuda"


def test_split_device_is_a_parser_and_not_a_validator():
    """It cannot tell "no index" from "broken index" -- both are None.

    Pinned because that is exactly the gap the malformed-index bug lived
    in: a check written against ``split_device`` looks like validation and
    is not.
    """
    assert split_device("cuda") == ("cuda", None)
    assert split_device("cuda:abc") == ("cuda", None)


def test_the_exported_script_documents_the_indexed_form():
    """The one device entry point with no `choices=` to constrain it."""
    from app.core import codegen

    source = __import__("inspect").getsource(codegen)
    assert "cuda:N" in source


def test_an_index_on_a_machine_with_no_cuda_still_lands_on_the_cpu(no_cuda):
    assert resolve_device("cuda:2") == "cpu"
    assert resolve_device("cuda") == "cpu"


def test_mps_has_no_index_vocabulary_beyond_zero(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    # "mps" and "mps:0" name the same hardware, so both canonicalise to
    # the bare form -- REBUILT, not echoed (#194; see the dedicated test
    # below for why the echo was reachable and unsafe).
    assert resolve_device("mps:0") == "mps"
    assert resolve_device("mps:1") == "mps"


def test_mps_index_zero_is_rebuilt_rather_than_echoed(monkeypatch):
    """#194: the mps branch used to ``return device`` verbatim whenever the
    index was exactly 0, instead of rebuilding like every other branch
    does (see test_the_resolved_string_is_rebuilt_rather_than_echoed for
    the cuda side, closed by #135).

    ``DEVICE_SYNTAX``'s ``\\d`` is Unicode-aware and ``int()`` parses
    Unicode decimal digits too, so ``mps:\u0660`` (Arabic-Indic digit
    zero) and ``mps:00`` both satisfy "index == 0" without being spelled
    "0" -- and the old code handed the untouched original string to
    torch, which rejected it with a RuntimeError naming neither the
    graph nor the parameter it came from.
    """
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("mps:00") == "mps"
    assert resolve_device("mps:\u0660") == "mps"
    # The real proof, same style as the cuda malformed-index test: torch
    # accepts what came back.
    assert torch.device(resolve_device("mps:00")).type == "mps"


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
