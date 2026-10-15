"""Device inventory with memory facts (#767): ``core/resources.py``.

The probes talk to NVML, torch and psutil, none of which this suite can
steer on real hardware (CI has no GPU), so each test installs fake modules
in ``sys.modules`` -- the probes import lazily, so a fake there is what they
get. The torch-free guarantee is checked for real in a fresh interpreter.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core import device_utils, resources
from app.core.resources import DeviceFact, probe_devices, with_memory

GIB = 1024 ** 3
BACKEND = Path(__file__).resolve().parent.parent


# ── fakes ────────────────────────────────────────────────────────────

class FakeNVML:
    """A ``pynvml`` stand-in. ``cards`` is (name, total, free) per NVML index."""

    def __init__(self, cards, *, init_error=None, count_error=None,
                 broken=(), bytes_names=False):
        self.cards = cards
        self.init_error = init_error
        self.count_error = count_error
        self.broken = set(broken)
        self.bytes_names = bytes_names
        self.shutdowns = 0

    def nvmlInit(self):
        if self.init_error:
            raise self.init_error

    def nvmlShutdown(self):
        self.shutdowns += 1

    def nvmlDeviceGetCount(self):
        if self.count_error:
            raise self.count_error
        return len(self.cards)

    def nvmlDeviceGetHandleByIndex(self, index):
        if index in self.broken:
            raise RuntimeError("NVML_ERROR_GPU_IS_LOST")
        return index

    def nvmlDeviceGetName(self, handle):
        name = self.cards[handle][0]
        return name.encode() if self.bytes_names else name

    def nvmlDeviceGetMemoryInfo(self, handle):
        _, total, free = self.cards[handle]
        return SimpleNamespace(total=total, free=free, used=total - free)


def fake_torch(*, cuda=(), hip=None, mps=None, mem_error=False,
               name_error=False, cuda_error=False):
    """A torch namespace. ``cuda`` is (name, total, free) per card; ``mps``
    is (recommended_max, driver_allocated) or None for no MPS."""

    def is_available():
        if cuda_error:
            raise RuntimeError("driver too old")
        return bool(cuda)

    def mem_get_info(index):
        if mem_error:
            raise RuntimeError("CUDA error: out of memory")
        _, total, free = cuda[index]
        return free, total

    def get_device_name(index):
        if name_error:
            raise RuntimeError("no name")
        return cuda[index][0]

    def recommended_max_memory():
        if mps is None or mps[0] is None:
            raise RuntimeError("not supported")
        return mps[0]

    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=is_available,
                             device_count=lambda: len(cuda),
                             mem_get_info=mem_get_info,
                             get_device_name=get_device_name),
        version=SimpleNamespace(hip=hip),
        backends=SimpleNamespace(mps=SimpleNamespace(
            is_available=lambda: mps is not None)),
        mps=SimpleNamespace(recommended_max_memory=recommended_max_memory,
                            driver_allocated_memory=lambda: mps[1]),
    )


@pytest.fixture
def no_nvml(monkeypatch):
    # None in sys.modules makes `import pynvml` raise ImportError.
    monkeypatch.setitem(sys.modules, "pynvml", None)


@pytest.fixture
def cpu_only_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch())


@pytest.fixture(autouse=True)
def _no_visible_devices_env(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


# ── CPU ──────────────────────────────────────────────────────────────

def test_cpu_comes_first_from_psutil(no_nvml, cpu_only_torch, monkeypatch):
    vm = SimpleNamespace(total=16 * GIB, available=5 * GIB)
    monkeypatch.setitem(sys.modules, "psutil",
                        SimpleNamespace(virtual_memory=lambda: vm))

    facts = probe_devices()

    assert len(facts) == 1
    cpu = facts[0]
    assert (cpu.value, cpu.kind, cpu.index) == ("cpu", "cpu", None)
    assert (cpu.total_bytes, cpu.free_bytes, cpu.source) == (16 * GIB, 5 * GIB, "psutil")
    assert cpu.memory() == {"total_bytes": 16 * GIB, "free_bytes": 5 * GIB,
                            "source": "psutil"}


def test_real_psutil_answers_on_this_machine(no_nvml, cpu_only_torch):
    """CI (no GPU): CPU only, from psutil, with plausible numbers."""
    facts = probe_devices()
    assert [f.value for f in facts] == ["cpu"]
    assert facts[0].source == "psutil"
    assert 0 < facts[0].free_bytes <= facts[0].total_bytes


def test_broken_psutil_degrades_to_null_memory(no_nvml, cpu_only_torch, monkeypatch):
    def boom():
        raise OSError("no /proc")
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(virtual_memory=boom))

    cpu = probe_devices()[0]

    assert (cpu.value, cpu.total_bytes, cpu.free_bytes, cpu.source) == ("cpu", None, None, "none")
    assert cpu.memory() is None


def test_cpu_name_falls_back_when_platform_is_silent(no_nvml, cpu_only_torch, monkeypatch):
    monkeypatch.setattr(resources.platform, "processor", lambda: "")
    monkeypatch.setattr(resources.platform, "machine", lambda: "")
    assert probe_devices()[0].name == "CPU"


# ── CUDA via NVML ────────────────────────────────────────────────────

def test_nvml_reports_every_card_torch_free(monkeypatch):
    nvml = FakeNVML([("RTX 4090", 24 * GIB, 20 * GIB),
                     ("RTX 3060", 12 * GIB, 11 * GIB)], bytes_names=True)
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    # A torch that would explode if touched: the torch-free path must not.
    monkeypatch.setitem(sys.modules, "torch", None)

    facts = probe_devices(allow_torch=False)

    gpus = facts[1:]
    assert [(f.value, f.kind, f.index, f.name, f.total_bytes, f.free_bytes, f.source)
            for f in gpus] == [
        ("cuda:0", "cuda", 0, "RTX 4090", 24 * GIB, 20 * GIB, "nvml"),
        ("cuda:1", "cuda", 1, "RTX 3060", 12 * GIB, 11 * GIB, "nvml"),
    ]
    assert nvml.shutdowns == 1


def test_nvml_wins_over_torch_when_both_answer(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml",
                        FakeNVML([("A100", 80 * GIB, 70 * GIB)]))
    monkeypatch.setitem(sys.modules, "torch",
                        fake_torch(cuda=[("A100", 80 * GIB, 1 * GIB)]))

    gpu = probe_devices()[1]

    assert (gpu.free_bytes, gpu.source) == (70 * GIB, "nvml")


def test_visible_devices_maps_cuda_index_to_nvml_index(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2, 0")
    monkeypatch.setitem(sys.modules, "pynvml", FakeNVML([
        ("card-a", 8 * GIB, 1 * GIB),
        ("card-b", 8 * GIB, 2 * GIB),
        ("card-c", 8 * GIB, 3 * GIB),
    ]))

    gpus = probe_devices(allow_torch=False)[1:]

    assert [(f.value, f.name) for f in gpus] == [("cuda:0", "card-c"),
                                                 ("cuda:1", "card-a")]


def test_visible_devices_stops_at_an_invalid_entry(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,,7,0")
    monkeypatch.setitem(sys.modules, "pynvml", FakeNVML([
        ("card-a", 8 * GIB, 1 * GIB), ("card-b", 8 * GIB, 2 * GIB)]))

    gpus = probe_devices(allow_torch=False)[1:]

    assert [(f.value, f.name) for f in gpus] == [("cuda:0", "card-b")]


def test_visible_devices_by_uuid_is_left_to_torch(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-8f3a")
    monkeypatch.setitem(sys.modules, "pynvml",
                        FakeNVML([("card-a", 8 * GIB, 1 * GIB)]))

    assert probe_devices(allow_torch=False)[1:] == []

    monkeypatch.setitem(sys.modules, "torch",
                        fake_torch(cuda=[("card-a", 8 * GIB, 4 * GIB)]))
    gpu = probe_devices()[1]
    assert (gpu.value, gpu.free_bytes, gpu.source) == ("cuda:0", 4 * GIB, "torch.cuda")


# ── broken NVML degrades: torch, then null ───────────────────────────

@pytest.mark.parametrize("nvml", [
    pytest.param(None, id="not-installed"),
    pytest.param(FakeNVML([], init_error=RuntimeError("NVML_ERROR_DRIVER_NOT_LOADED")),
                 id="init-raises"),
    pytest.param(FakeNVML([("x", 1, 1)], count_error=RuntimeError("boom")),
                 id="count-raises"),
])
def test_broken_nvml_falls_back_to_torch(monkeypatch, nvml):
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setitem(sys.modules, "torch",
                        fake_torch(cuda=[("RTX 4090", 24 * GIB, 9 * GIB)]))

    gpu = probe_devices()[1]

    assert (gpu.value, gpu.kind, gpu.name, gpu.total_bytes, gpu.free_bytes, gpu.source) == (
        "cuda:0", "cuda", "RTX 4090", 24 * GIB, 9 * GIB, "torch.cuda")


def test_one_broken_card_falls_back_alone(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", FakeNVML(
        [("good", 8 * GIB, 7 * GIB), ("lost", 8 * GIB, 6 * GIB)], broken={1}))
    monkeypatch.setitem(sys.modules, "torch", fake_torch(
        cuda=[("good", 8 * GIB, 1 * GIB), ("lost", 8 * GIB, 2 * GIB)]))

    gpus = probe_devices()[1:]

    assert [(f.value, f.free_bytes, f.source) for f in gpus] == [
        ("cuda:0", 7 * GIB, "nvml"), ("cuda:1", 2 * GIB, "torch.cuda")]


def test_a_failing_shutdown_is_swallowed(monkeypatch):
    nvml = FakeNVML([("T4", 16 * GIB, 15 * GIB)])

    def bad_shutdown():
        raise RuntimeError("NVML_ERROR_UNINITIALIZED")
    nvml.nvmlShutdown = bad_shutdown
    monkeypatch.setitem(sys.modules, "pynvml", nvml)

    assert probe_devices(allow_torch=False)[1].source == "nvml"


def test_broken_card_without_torch_has_null_memory(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml",
                        FakeNVML([("lost", 8 * GIB, 6 * GIB)], broken={0}))

    gpu = probe_devices(allow_torch=False)[1]

    assert (gpu.value, gpu.total_bytes, gpu.free_bytes, gpu.source) == ("cuda:0", None, None, "none")
    assert gpu.memory() is None


def test_broken_nvml_and_broken_torch_yield_null(no_nvml, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(
        cuda=[("RTX 4090", 24 * GIB, 9 * GIB)], mem_error=True, name_error=True))

    gpu = probe_devices()[1]

    assert (gpu.value, gpu.name, gpu.total_bytes, gpu.free_bytes, gpu.source) == (
        "cuda:0", "", None, None, "none")


def test_rocm_is_seen_through_torch(no_nvml, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(
        cuda=[("AMD Instinct MI300X", 192 * GIB, 190 * GIB)], hip="6.2"))

    gpu = probe_devices()[1]

    assert (gpu.kind, gpu.source) == ("rocm", "torch.cuda")


def test_torch_without_cuda_keeps_nvml_cards(monkeypatch):
    """A CPU-only torch wheel on a GPU box: NVML still sees the card."""
    monkeypatch.setitem(sys.modules, "pynvml", FakeNVML([("T4", 16 * GIB, 15 * GIB)]))
    monkeypatch.setitem(sys.modules, "torch", fake_torch())

    assert [f.value for f in probe_devices()] == ["cpu", "cuda:0"]


@pytest.mark.parametrize("torch_module", [
    pytest.param(None, id="torch-missing"),
    pytest.param(fake_torch(cuda_error=True), id="cuda-probe-raises"),
])
def test_unusable_torch_degrades(no_nvml, monkeypatch, torch_module):
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    assert [f.value for f in probe_devices()] == ["cpu"]


# ── MPS ──────────────────────────────────────────────────────────────

def test_mps_total_and_free_from_torch_mps(no_nvml, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch",
                        fake_torch(mps=(16 * GIB, 3 * GIB)))

    facts = probe_devices()

    mps = facts[-1]
    assert (mps.value, mps.kind, mps.index, mps.name) == ("mps", "mps", None, "Apple MPS")
    assert (mps.total_bytes, mps.free_bytes, mps.source) == (16 * GIB, 13 * GIB, "torch.mps")


def test_mps_free_never_goes_negative(no_nvml, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch",
                        fake_torch(mps=(4 * GIB, 5 * GIB)))
    assert probe_devices()[-1].free_bytes == 0


def test_mps_memory_probe_failure_is_null(no_nvml, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(mps=(None, 0)))

    mps = probe_devices()[-1]

    assert (mps.value, mps.source, mps.memory()) == ("mps", "none", None)


def test_mps_is_not_reported_without_torch(no_nvml, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(mps=(16 * GIB, 0)))
    assert [f.value for f in probe_devices(allow_torch=False)] == ["cpu"]


def test_mps_probe_tolerates_a_broken_torch(no_nvml, monkeypatch):
    broken = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                             backends=SimpleNamespace(
                                 mps=SimpleNamespace(is_available=_raise)))
    monkeypatch.setitem(sys.modules, "torch", broken)
    assert [f.value for f in probe_devices()] == ["cpu"]


def _raise():
    raise RuntimeError("Metal unavailable")


# ── torch-free mode in a fresh interpreter ───────────────────────────

def test_torch_free_mode_never_imports_torch():
    """The agent's mode (#771): after the call, torch is not in sys.modules."""
    script = (
        "import json, sys\n"
        "from app.core.resources import probe_devices\n"
        "facts = probe_devices(allow_torch=False)\n"
        "print(json.dumps({'torch': 'torch' in sys.modules,\n"
        "                  'values': [f.value for f in facts]}))\n"
    )
    out = subprocess.run([sys.executable, "-c", script], cwd=BACKEND,
                         capture_output=True, text=True, timeout=120, check=True)

    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["torch"] is False
    assert result["values"][0] == "cpu"


# ── merging into describe_accelerator() ──────────────────────────────

def _fact(value, total, free, source="nvml"):
    return DeviceFact(value, value.split(":")[0], None, "", total, free, source)


def test_with_memory_adds_memory_per_device(monkeypatch):
    monkeypatch.setattr(device_utils, "_current_cuda_index", lambda: 1)
    described = {
        "default": "cuda",
        "devices": [
            {"value": "cpu", "label": "CPU", "detail": "", "available": True},
            {"value": "cuda", "label": "NVIDIA CUDA", "detail": "B", "available": True},
            {"value": "cuda:0", "label": "NVIDIA CUDA #0", "detail": "A", "available": True},
            {"value": "cuda:1", "label": "NVIDIA CUDA #1", "detail": "B", "available": True},
            {"value": "mps", "label": "Apple MPS", "detail": "", "available": True},
        ],
    }
    facts = [_fact("cpu", 64, 32, "psutil"), _fact("cuda:0", 24, 20),
             _fact("cuda:1", 12, 11), _fact("mps", None, None, "none")]

    merged = with_memory(described, facts)

    memory = {d["value"]: d["memory"] for d in merged["devices"]}
    assert memory == {
        "cpu": {"total_bytes": 64, "free_bytes": 32, "source": "psutil"},
        # bare cuda means the current card, here #1
        "cuda": {"total_bytes": 12, "free_bytes": 11, "source": "nvml"},
        "cuda:0": {"total_bytes": 24, "free_bytes": 20, "source": "nvml"},
        "cuda:1": {"total_bytes": 12, "free_bytes": 11, "source": "nvml"},
        "mps": None,
    }
    assert merged["default"] == "cuda"
    assert merged["devices"][1]["label"] == "NVIDIA CUDA"
    assert "memory" not in described["devices"][0]  # input untouched


def test_with_memory_unknown_device_is_null():
    merged = with_memory({"default": "cpu", "devices": [{"value": "cpu"}]}, [])
    assert merged["devices"] == [{"value": "cpu", "memory": None}]
