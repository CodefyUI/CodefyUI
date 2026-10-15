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

UUID_A = "aaaaaaaa-1111-2222-3333-444444444444"
UUID_B = "bbbbbbbb-1111-2222-3333-444444444444"
UUID_C = "cccccccc-1111-2222-3333-444444444444"


# ── fakes ────────────────────────────────────────────────────────────

def card(name, total, free, uuid, *, mig=False, broken=False, mig_error=False,
         uuid_error=False, name_error=False, memory_error=False):
    return dict(name=name, total=total, free=free, uuid=uuid, mig=mig,
                broken=broken, mig_error=mig_error, uuid_error=uuid_error,
                name_error=name_error, memory_error=memory_error)


class FakeNVML:
    """A ``pynvml`` stand-in over a list of :func:`card` dicts (PCI order)."""

    def __init__(self, cards, *, init_error=None, count_error=None,
                 bytes_strings=False):
        self.cards = cards
        self.init_error = init_error
        self.count_error = count_error
        self.bytes_strings = bytes_strings
        self.shutdowns = 0

    def _s(self, text):
        return text.encode() if self.bytes_strings else text

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
        if self.cards[index]["broken"]:
            raise RuntimeError("NVML_ERROR_GPU_IS_LOST")
        return self.cards[index]

    def nvmlDeviceGetUUID(self, handle):
        if handle["uuid_error"]:
            raise RuntimeError("uuid")
        return self._s(f"GPU-{handle['uuid']}")

    def nvmlDeviceGetName(self, handle):
        if handle["name_error"]:
            raise RuntimeError("name")
        return self._s(handle["name"])

    def nvmlDeviceGetMigMode(self, handle):
        if handle["mig_error"]:
            raise RuntimeError("NVML_ERROR_NOT_SUPPORTED")
        return (1 if handle["mig"] else 0, 0)

    def nvmlDeviceGetMemoryInfo(self, handle):
        if handle["memory_error"]:
            raise RuntimeError("memory")
        return SimpleNamespace(total=handle["total"], free=handle["free"],
                               used=handle["total"] - handle["free"])


def fake_torch(*, cuda=(), hip=None, mps=None, initialized=False,
               mem_error=False, props_error=False, uuids=True):
    """A torch namespace. ``cuda`` is a list of :func:`card` dicts in torch's
    order; ``mps`` is (recommended_max, driver_allocated) or None."""

    def mem_get_info(index):
        if mem_error:
            raise RuntimeError("CUDA error")
        return cuda[index]["free"], cuda[index]["total"]

    def get_device_properties(index):
        if props_error:
            raise RuntimeError("props")
        c = cuda[index]
        props = SimpleNamespace(total_memory=c["total"], name=c["name"])
        if uuids:
            props.uuid = c["uuid"]
        return props

    def recommended_max_memory():
        if mps[0] is None:
            raise RuntimeError("not supported")
        return mps[0]

    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: bool(cuda),
                             device_count=lambda: len(cuda),
                             is_initialized=lambda: initialized,
                             mem_get_info=mem_get_info,
                             get_device_properties=get_device_properties,
                             get_device_name=lambda i: cuda[i]["name"]),
        version=SimpleNamespace(hip=hip),
        backends=SimpleNamespace(mps=SimpleNamespace(
            is_available=lambda: mps is not None)),
        mps=SimpleNamespace(recommended_max_memory=recommended_max_memory,
                            driver_allocated_memory=lambda: mps[1]),
    )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    # Default: no NVML, a CPU-only torch. Tests override.
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch())


def use(monkeypatch, *, nvml=None, torch=None):
    if nvml is not None:
        monkeypatch.setitem(sys.modules, "pynvml", nvml)
    if torch is not None:
        monkeypatch.setitem(sys.modules, "torch", torch)


def summary(facts):
    return [(f.value, f.kind, f.name, f.total_bytes, f.free_bytes, f.source)
            for f in facts]


# ── CPU ──────────────────────────────────────────────────────────────

def test_cpu_comes_first_from_psutil(monkeypatch):
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


def test_real_psutil_answers_on_this_machine():
    """CI (no GPU): CPU only, from psutil, with plausible numbers."""
    facts = probe_devices()
    assert [f.value for f in facts] == ["cpu"]
    assert facts[0].source == "psutil"
    assert 0 < facts[0].free_bytes <= facts[0].total_bytes


def test_broken_psutil_degrades_to_null_memory(monkeypatch):
    def boom():
        raise OSError("no /proc")
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(virtual_memory=boom))

    cpu = probe_devices()[0]

    assert (cpu.value, cpu.total_bytes, cpu.free_bytes, cpu.source) == ("cpu", None, None, "none")
    assert cpu.memory() is None


def test_cpu_name_falls_back_when_platform_is_silent(monkeypatch):
    monkeypatch.setattr(resources.platform, "processor", lambda: "")
    monkeypatch.setattr(resources.platform, "machine", lambda: "")
    assert probe_devices()[0].name == "CPU"


# ── torch-free: NVML alone ───────────────────────────────────────────

def test_torch_free_lists_every_nvml_card(monkeypatch):
    nvml = FakeNVML([card("RTX 4090", 24 * GIB, 20 * GIB, UUID_A),
                     card("RTX 3060", 12 * GIB, 11 * GIB, UUID_B)],
                    bytes_strings=True)
    use(monkeypatch, nvml=nvml)

    facts = probe_devices(allow_torch=False)

    assert summary(facts[1:]) == [
        ("cuda:0", "cuda", "RTX 4090", 24 * GIB, 20 * GIB, "nvml"),
        ("cuda:1", "cuda", "RTX 3060", 12 * GIB, 11 * GIB, "nvml"),
    ]
    assert [f.index for f in facts[1:]] == [0, 1]
    assert nvml.shutdowns == 1


def test_torch_free_without_nvml_lists_no_cards():
    assert [f.value for f in probe_devices(allow_torch=False)] == ["cpu"]


def test_torch_free_mig_parent_and_broken_cards_have_null_memory(monkeypatch):
    use(monkeypatch, nvml=FakeNVML([
        card("A100", 80 * GIB, 70 * GIB, UUID_A, mig=True),
        card("lost", 8 * GIB, 6 * GIB, UUID_B, broken=True),
        card("no-mem", 8 * GIB, 6 * GIB, UUID_C, memory_error=True,
             name_error=True, uuid_error=True, mig_error=True),
    ]))

    facts = probe_devices(allow_torch=False)[1:]

    assert summary(facts) == [
        ("cuda:0", "cuda", "A100", None, None, "none"),
        ("cuda:1", "cuda", "", None, None, "none"),
        ("cuda:2", "cuda", "", None, None, "none"),
    ]
    assert all(f.memory() is None for f in facts)


def test_a_failing_shutdown_is_swallowed(monkeypatch):
    nvml = FakeNVML([card("T4", 16 * GIB, 15 * GIB, UUID_A)])

    def bad_shutdown():
        raise RuntimeError("NVML_ERROR_UNINITIALIZED")
    nvml.nvmlShutdown = bad_shutdown
    use(monkeypatch, nvml=nvml)

    assert probe_devices(allow_torch=False)[1].source == "nvml"


@pytest.mark.parametrize("nvml", [
    pytest.param(FakeNVML([], init_error=RuntimeError("NVML_ERROR_DRIVER_NOT_LOADED")),
                 id="init-raises"),
    pytest.param(FakeNVML([card("x", 1, 1, UUID_A)], count_error=RuntimeError("boom")),
                 id="count-raises"),
])
def test_torch_free_broken_nvml_lists_no_cards(monkeypatch, nvml):
    use(monkeypatch, nvml=nvml)
    assert [f.value for f in probe_devices(allow_torch=False)] == ["cpu"]


# ── CUDA_VISIBLE_DEVICES, CUDA's rules ───────────────────────────────

THREE = [card("card-a", 8 * GIB, 1 * GIB, UUID_A),
         card("card-b", 8 * GIB, 2 * GIB, UUID_B),
         card("card-c", 8 * GIB, 3 * GIB, UUID_C)]


@pytest.mark.parametrize("env, names", [
    pytest.param("2, 0", ["card-c", "card-a"], id="indices-reordered"),
    pytest.param("1,,0", ["card-b"], id="stops-at-empty"),
    pytest.param("1,7,0", ["card-b"], id="stops-at-out-of-range"),
    pytest.param("1,-1,0", ["card-b"], id="stops-at-negative"),
    pytest.param("1,banana,0", ["card-b"], id="stops-at-garbage"),
    pytest.param("0,2,0", [], id="duplicate-hides-everything"),
    pytest.param("GPU-cccc,GPU-AAAAAAAA-1111", ["card-c", "card-a"], id="uuid-prefix"),
    pytest.param("0,GPU-aaaa", [], id="uuid-duplicates-index"),
    pytest.param("GPU-,1", [], id="ambiguous-uuid-prefix-stops"),
    pytest.param("GPU-dddd,1", [], id="unknown-uuid-stops"),
    pytest.param("", [], id="empty-hides-everything"),
])
def test_visible_devices(monkeypatch, env, names):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", env)
    use(monkeypatch, nvml=FakeNVML(THREE))

    facts = probe_devices(allow_torch=False)[1:]

    assert [f.name for f in facts] == names
    assert [f.value for f in facts] == [f"cuda:{i}" for i in range(len(names))]


def test_visible_mig_instance_is_listed_without_memory(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-1234abcd,1,mig-1234ABCD")
    use(monkeypatch, nvml=FakeNVML(THREE))

    facts = probe_devices(allow_torch=False)[1:]

    # The second spelling of the same MIG instance is a duplicate.
    assert facts == []

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-1234abcd,1")
    facts = probe_devices(allow_torch=False)[1:]
    assert summary(facts) == [
        ("cuda:0", "cuda", "", None, None, "none"),
        ("cuda:1", "cuda", "card-b", 8 * GIB, 2 * GIB, "nvml"),
    ]


# ── with torch: NVML matched by UUID ─────────────────────────────────

def test_nvml_is_matched_to_torch_by_uuid_not_index(monkeypatch):
    """torch orders fastest-first, NVML by PCI bus: the UUID decides."""
    nvml = FakeNVML([card("slow", 8 * GIB, 7 * GIB, UUID_A),
                     card("fast", 24 * GIB, 20 * GIB, UUID_B)])
    torch_cards = [card("fast", 24 * GIB, 0, UUID_B), card("slow", 8 * GIB, 0, UUID_A)]
    use(monkeypatch, nvml=nvml, torch=fake_torch(cuda=torch_cards))

    assert summary(probe_devices()[1:]) == [
        ("cuda:0", "cuda", "fast", 24 * GIB, 20 * GIB, "nvml"),
        ("cuda:1", "cuda", "slow", 8 * GIB, 7 * GIB, "nvml"),
    ]


def test_torch_uuid_with_gpu_prefix_still_matches(monkeypatch):
    use(monkeypatch, nvml=FakeNVML([card("A100", 80 * GIB, 70 * GIB, UUID_A)]),
        torch=fake_torch(cuda=[card("A100", 80 * GIB, 0, f"GPU-{UUID_A.upper()}")]))
    assert probe_devices()[1].source == "nvml"


def test_nvml_name_fills_a_missing_torch_name(monkeypatch):
    use(monkeypatch, nvml=FakeNVML([card("A100", 80 * GIB, 70 * GIB, UUID_A)]),
        torch=fake_torch(cuda=[card("", 80 * GIB, 0, UUID_A)]))
    assert probe_devices()[1].name == "A100"


def test_unmatched_or_mig_card_falls_back_to_torch(monkeypatch):
    use(monkeypatch,
        nvml=FakeNVML([card("A100", 80 * GIB, 70 * GIB, UUID_A, mig=True),
                       card("lost", 8 * GIB, 6 * GIB, UUID_B, memory_error=True)]),
        torch=fake_torch(cuda=[card("A100", 80 * GIB, 0, UUID_A),
                               card("lost", 8 * GIB, 0, UUID_B),
                               card("other", 4 * GIB, 0, UUID_C)]))

    assert summary(probe_devices()[1:]) == [
        ("cuda:0", "cuda", "A100", 80 * GIB, None, "torch.cuda"),
        ("cuda:1", "cuda", "lost", 8 * GIB, None, "torch.cuda"),
        ("cuda:2", "cuda", "other", 4 * GIB, None, "torch.cuda"),
    ]


def test_torch_without_uuid_falls_back_to_torch(monkeypatch):
    use(monkeypatch, nvml=FakeNVML([card("A100", 80 * GIB, 70 * GIB, UUID_A)]),
        torch=fake_torch(cuda=[card("A100", 80 * GIB, 0, UUID_A)], uuids=False))
    assert probe_devices()[1].source == "torch.cuda"


# ── torch fallback: no new CUDA context ──────────────────────────────

@pytest.mark.parametrize("nvml", [
    pytest.param(None, id="not-installed"),
    pytest.param(FakeNVML([], init_error=RuntimeError("NVML_ERROR_LIB_RM_VERSION_MISMATCH")),
                 id="init-raises"),
    pytest.param(FakeNVML([card("x", 1, 1, UUID_A)], count_error=RuntimeError("boom")),
                 id="count-raises"),
])
def test_broken_nvml_falls_back_to_torch_total_only(monkeypatch, nvml):
    if nvml is not None:
        use(monkeypatch, nvml=nvml)
    use(monkeypatch, torch=fake_torch(cuda=[card("RTX 4090", 24 * GIB, 9 * GIB, UUID_A)]))

    assert summary(probe_devices()[1:]) == [
        ("cuda:0", "cuda", "RTX 4090", 24 * GIB, None, "torch.cuda")]


def test_initialised_cuda_reads_free_through_mem_get_info(monkeypatch):
    use(monkeypatch, torch=fake_torch(
        cuda=[card("RTX 4090", 24 * GIB, 9 * GIB, UUID_A)], initialized=True))

    gpu = probe_devices()[1]

    assert (gpu.total_bytes, gpu.free_bytes, gpu.source) == (24 * GIB, 9 * GIB, "torch.cuda")


def test_failing_mem_get_info_keeps_the_total(monkeypatch):
    use(monkeypatch, torch=fake_torch(
        cuda=[card("RTX 4090", 24 * GIB, 9 * GIB, UUID_A)], initialized=True,
        mem_error=True))
    gpu = probe_devices()[1]
    assert (gpu.total_bytes, gpu.free_bytes, gpu.source) == (24 * GIB, None, "torch.cuda")


def test_failing_properties_yield_null(monkeypatch):
    use(monkeypatch, torch=fake_torch(
        cuda=[card("RTX 4090", 24 * GIB, 9 * GIB, UUID_A)], props_error=True))

    gpu = probe_devices()[1]

    assert (gpu.value, gpu.name, gpu.source, gpu.memory()) == ("cuda:0", "RTX 4090", "none", None)


def test_properties_without_total_yield_null(monkeypatch):
    torch = fake_torch(cuda=[card("RTX 4090", 24 * GIB, 9 * GIB, UUID_A)])
    torch.cuda.get_device_properties = lambda i: SimpleNamespace(uuid=UUID_A)
    use(monkeypatch, torch=torch)
    assert probe_devices()[1].source == "none"


def test_torch_import_failing_after_count_lists_nothing(monkeypatch):
    monkeypatch.setattr(device_utils, "cuda_device_count", lambda: 1)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert [f.value for f in probe_devices()] == ["cpu"]


# ── ROCm: torch only ─────────────────────────────────────────────────

def test_rocm_never_uses_nvml(monkeypatch):
    use(monkeypatch,
        nvml=FakeNVML([card("not this", 99 * GIB, 99 * GIB, UUID_A)]),
        torch=fake_torch(cuda=[card("AMD Instinct MI300X", 192 * GIB, 190 * GIB, UUID_A)],
                         hip="6.2"))

    assert summary(probe_devices()[1:]) == [
        ("cuda:0", "rocm", "AMD Instinct MI300X", 192 * GIB, None, "torch.cuda")]


# ── NVML cards with a torch that cannot use CUDA ─────────────────────

def test_cpu_only_torch_still_lists_nvml_cards(monkeypatch):
    """Inventory, not runnability: a CPU wheel on a GPU box."""
    use(monkeypatch, nvml=FakeNVML([card("T4", 16 * GIB, 15 * GIB, UUID_A)]))

    assert summary(probe_devices()[1:]) == [
        ("cuda:0", "cuda", "T4", 16 * GIB, 15 * GIB, "nvml")]


def test_missing_torch_degrades():
    sys.modules["torch"] = None  # the autouse fixture restores it
    assert [f.value for f in probe_devices()] == ["cpu"]


# ── MPS ──────────────────────────────────────────────────────────────

def _psutil(available):
    vm = SimpleNamespace(total=64 * GIB, available=available)
    return SimpleNamespace(virtual_memory=lambda: vm)


def test_mps_total_and_free_from_torch_mps(monkeypatch):
    use(monkeypatch, torch=fake_torch(mps=(16 * GIB, 3 * GIB)))
    monkeypatch.setitem(sys.modules, "psutil", _psutil(40 * GIB))

    mps = probe_devices()[-1]

    assert (mps.value, mps.kind, mps.index, mps.name) == ("mps", "mps", None, "Apple MPS")
    assert (mps.total_bytes, mps.free_bytes, mps.source) == (16 * GIB, 13 * GIB, "torch.mps")


def test_mps_free_is_capped_by_available_ram(monkeypatch):
    """Unified memory: the GPU cannot have more free than the system."""
    use(monkeypatch, torch=fake_torch(mps=(16 * GIB, 3 * GIB)))
    monkeypatch.setitem(sys.modules, "psutil", _psutil(5 * GIB))

    assert probe_devices()[-1].free_bytes == 5 * GIB


def test_mps_free_without_psutil_is_uncapped(monkeypatch):
    def boom():
        raise OSError("no psutil")
    use(monkeypatch, torch=fake_torch(mps=(16 * GIB, 3 * GIB)))
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(virtual_memory=boom))

    assert probe_devices()[-1].free_bytes == 13 * GIB


def test_mps_free_never_goes_negative(monkeypatch):
    use(monkeypatch, torch=fake_torch(mps=(4 * GIB, 5 * GIB)))
    assert probe_devices()[-1].free_bytes == 0


def test_mps_memory_probe_failure_is_null(monkeypatch):
    use(monkeypatch, torch=fake_torch(mps=(None, 0)))

    mps = probe_devices()[-1]

    assert (mps.value, mps.source, mps.memory()) == ("mps", "none", None)


def test_mps_is_not_reported_without_torch(monkeypatch):
    use(monkeypatch, torch=fake_torch(mps=(16 * GIB, 0)))
    assert [f.value for f in probe_devices(allow_torch=False)] == ["cpu"]


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
             _fact("cuda:1", 12, None, "torch.cuda"), _fact("mps", None, None, "none")]

    merged = with_memory(described, facts)

    memory = {d["value"]: d["memory"] for d in merged["devices"]}
    assert memory == {
        "cpu": {"total_bytes": 64, "free_bytes": 32, "source": "psutil"},
        # bare cuda means the current card, here #1
        "cuda": {"total_bytes": 12, "free_bytes": None, "source": "torch.cuda"},
        "cuda:0": {"total_bytes": 24, "free_bytes": 20, "source": "nvml"},
        "cuda:1": {"total_bytes": 12, "free_bytes": None, "source": "torch.cuda"},
        "mps": None,
    }
    assert merged["default"] == "cuda"
    assert merged["devices"][1]["label"] == "NVIDIA CUDA"
    assert "memory" not in described["devices"][0]  # input untouched


def test_with_memory_unknown_device_is_null():
    merged = with_memory({"default": "cpu", "devices": [{"value": "cpu"}]}, [])
    assert merged["devices"] == [{"value": "cpu", "memory": None}]
