"""Device inventory with memory facts (#767).

:func:`probe_devices` lists the compute devices this machine has, each with
its total and free memory where something can say. It is display-only: no
admission decision reads it yet (that is #788), and its memory figures are
not cached, so every call is a fresh reading and the caller decides how
often to pay for one. Only the CPU's name (:func:`cpu_name`) is kept: a
brand found once, and a miss for :data:`CPU_BRAND_RETRY_S`.

Sources, per kind of device:

- **NVIDIA CUDA** -- NVML (``pynvml``, from the ``nvidia-ml-py``
  dependency), because it answers without creating a CUDA context. With
  torch allowed, NVML cards are matched to torch devices by UUID, never by
  index. A card NVML cannot describe falls back to torch (below).
- **ROCm** -- torch only; NVML facts are never used on a ROCm build (the
  same rule ``env_fingerprint`` follows for the driver version).
- **torch fallback** -- ``get_device_properties(i).total_memory`` for the
  total. The free figure needs ``torch.cuda.mem_get_info``, which creates a
  CUDA context on the card, so it is read only when this process has
  already initialised CUDA; otherwise ``free_bytes`` is ``None``.
- **MPS** -- see :func:`_mps_fact`.
- **CPU** -- ``psutil.virtual_memory()`` (``total`` / ``available``).

Card numbering in torch-free mode
---------------------------------
``allow_torch=False`` is the worker agent's mode (#771): the agent process
cannot import torch, because only the run subprocess may initialise CUDA,
and it must still see its cards. NVML numbers cards in PCI bus order; CUDA
numbers them fastest-first unless ``CUDA_DEVICE_ORDER=PCI_BUS_ID``. Ranking
cards "the way CUDA does" from NVML is not reliable, so torch-free indices
ASSUME ``CUDA_DEVICE_ORDER=PCI_BUS_ID``: under that setting ``cuda:N`` here
is the same card as ``cuda:N`` in a child process.

The worker agent (#771) sets ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` in its own
environment and in every run subprocess it starts
(``app.agent.runner.child_environment``), so the indices it reports from
this module are the indices its runs use.

Every probe degrades instead of raising: a missing library, a broken driver
or an odd answer yields ``None`` fields (or no entry) and a debug log line.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceFact:
    """One device and what is known about its memory.

    ``value`` is the device string a run would use (``cpu``, ``mps``,
    ``cuda:1``). This is an inventory, not a promise the device runs
    anything: a card NVML sees is listed even when the installed torch
    cannot use CUDA (a CPU-only wheel on a GPU machine).
    ``kind`` is ``cpu`` / ``cuda`` / ``rocm`` / ``mps``. ``index`` is the
    card index within its kind (``None`` for CPU and MPS). ``source`` names
    where the memory figures came from (``nvml``, ``torch.cuda``,
    ``torch.mps``, ``psutil``) or ``none`` when no probe answered, in which
    case both byte counts are ``None``. ``free_bytes`` alone may be ``None``
    (``torch.cuda`` without an initialised CUDA context).
    ``uuid`` is a CUDA card's physical identity (the NVML / torch UUID,
    lower case, without the ``GPU-`` prefix), or None when nothing reported
    one; it is how a same-host worker's card is matched to this process's
    (#773).
    """

    value: str
    kind: str
    index: int | None
    name: str
    total_bytes: int | None
    free_bytes: int | None
    source: str
    uuid: str | None = None

    def memory(self) -> dict[str, Any] | None:
        """The ``memory`` object ``GET /api/system/devices`` serves, or None."""
        if self.source == "none":
            return None
        return {
            "total_bytes": self.total_bytes,
            "free_bytes": self.free_bytes,
            "source": self.source,
        }


def probe_devices(*, allow_torch: bool = True) -> list[DeviceFact]:
    """Every device this machine offers, CPU first, with memory facts.

    With ``allow_torch=False`` nothing imports torch: CUDA cards come from
    NVML alone (numbered as described in the module docstring) and MPS is
    not reported.
    """
    facts = [_cpu_fact()]
    if allow_torch:
        facts.extend(_cuda_facts_with_torch())
        mps = _mps_fact()
        if mps is not None:
            facts.append(mps)
    else:
        facts.extend(_cuda_facts_from_nvml())
    return facts


# ── CPU ──────────────────────────────────────────────────────────────

def _available_ram() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception as exc:  # noqa: BLE001
        logger.debug("psutil memory probe failed: %s", exc)
        return None


def _cpu_brand_darwin() -> str | None:
    """``machdep.cpu.brand_string`` through ``sysctlbyname``: an in-process
    call, so the probe never waits on a subprocess (it may run on the event
    loop, #840)."""
    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.dylib",
                           use_errno=True)
        key = b"machdep.cpu.brand_string"
        size = ctypes.c_size_t(0)
        if libc.sysctlbyname(key, None, ctypes.byref(size), None, 0) != 0 \
                or size.value == 0:
            return None
        buf = ctypes.create_string_buffer(size.value)
        if libc.sysctlbyname(key, buf, ctypes.byref(size), None, 0) != 0:
            return None
    except (OSError, AttributeError):
        return None
    return buf.value.decode("utf-8", "replace").strip() or None


#: /proc/cpuinfo keys that name the processor, best first: x86 and most ARM
#: kernels say ``model name``, POWER says ``cpu``, older ARM kernels
#: ``Hardware`` (the SoC) or ``Processor``. ``Model`` is never used: on a
#: Raspberry Pi it names the board.
_CPUINFO_KEYS = ("model name", "cpu", "hardware", "processor")


def _cpu_brand_linux() -> str | None:
    found: dict[str, str] = {}
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                key, _, value = line.partition(":")
                key, value = key.strip().lower(), value.strip()
                # "processor : 0" is the core index on x86 and ARM.
                if key in _CPUINFO_KEYS and value and not value.isdigit():
                    found.setdefault(key, value)
    except OSError:
        return None
    return next((found[key] for key in _CPUINFO_KEYS if key in found), None)


def _cpu_brand_windows() -> str | None:
    try:
        import winreg  # type: ignore[import-not-found]

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor"
                            r"\0") as key:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
    except (ImportError, OSError):
        return None
    return str(value).strip() or None


#: A probe that found no brand is not repeated for this long (#840): an
#: aarch64 host whose /proc/cpuinfo has no name line would otherwise be
#: parsed again on every device probe, while a passing failure still heals.
CPU_BRAND_RETRY_S = 600.0
#: The brand string, once a probe found one.
_cpu_brand: str | None = None
#: When (monotonic) a probe last found none, or None.
_cpu_brand_missed_at: float | None = None


def cpu_name() -> str:
    """The CPU's brand string (``Apple M3``, ``AMD EPYC 7763 64-Core
    Processor``), for the device lists (#840); the architecture
    (``platform.processor()``, ``platform.machine()``) when the system does
    not say, which is all ``arm`` / ``x86_64`` told before. Every probe is
    in-process (no subprocess), so a call on the event loop is cheap; a
    found brand is kept, a miss for :data:`CPU_BRAND_RETRY_S`."""
    global _cpu_brand, _cpu_brand_missed_at
    if _cpu_brand is not None:
        return _cpu_brand
    missed = _cpu_brand_missed_at
    if missed is None or time.monotonic() - missed >= CPU_BRAND_RETRY_S:
        probe = {"darwin": _cpu_brand_darwin,
                 "win32": _cpu_brand_windows}.get(sys.platform,
                                                  _cpu_brand_linux)
        brand = probe()
        if brand:
            _cpu_brand = brand
            return brand
        _cpu_brand_missed_at = time.monotonic()
    return platform.processor() or platform.machine() or "CPU"


def _cpu_fact() -> DeviceFact:
    name = cpu_name()
    try:
        import psutil

        vm = psutil.virtual_memory()
        return DeviceFact("cpu", "cpu", None, name,
                          int(vm.total), int(vm.available), "psutil")
    except Exception as exc:  # noqa: BLE001 -- display only, never fatal
        logger.debug("psutil memory probe failed: %s", exc)
        return DeviceFact("cpu", "cpu", None, name, None, None, "none")


# ── NVML ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _NvmlCard:
    """One physical card as NVML reports it, in NVML (PCI bus) order."""

    uuid: str | None         # normalised: no GPU- prefix, lower case
    name: str
    total_bytes: int | None
    free_bytes: int | None
    mig_enabled: bool

    @property
    def usable(self) -> bool:
        """Whether this card's NVML memory figures describe one CUDA device.

        A MIG-enabled parent's memory is split between instances that CUDA
        enumerates separately, so its figures describe none of them.
        """
        return not self.mig_enabled and self.total_bytes is not None


def _normalise_uuid(raw: Any) -> str | None:
    if raw is None:
        return None
    text = raw.decode() if isinstance(raw, bytes) else str(raw)
    text = text.strip().lower()
    if text.startswith("gpu-"):
        text = text[4:]
    return text or None


def _nvml_cards() -> list[_NvmlCard] | None:
    """Every card NVML sees, or None when NVML cannot answer at all."""
    try:
        import pynvml  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 -- not installed / broken
        logger.debug("pynvml unavailable: %s", exc)
        return None
    try:
        pynvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001 -- no driver, no GPU
        logger.debug("nvmlInit failed: %s", exc)
        return None
    try:
        cards = []
        for index in range(int(pynvml.nvmlDeviceGetCount())):
            cards.append(_nvml_card(pynvml, index))
        return cards
    except Exception as exc:  # noqa: BLE001
        logger.debug("NVML enumeration failed: %s", exc)
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


def _nvml_card(pynvml: Any, index: int) -> _NvmlCard:
    uuid = None
    name = ""
    total = free = None
    mig = False
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    except Exception as exc:  # noqa: BLE001 -- one card, not all
        logger.debug("NVML handle for card %d failed: %s", index, exc)
        return _NvmlCard(None, "", None, None, False)
    try:
        uuid = _normalise_uuid(pynvml.nvmlDeviceGetUUID(handle))
    except Exception as exc:  # noqa: BLE001
        logger.debug("NVML UUID of card %d failed: %s", index, exc)
    try:
        raw = pynvml.nvmlDeviceGetName(handle)
        name = raw.decode() if isinstance(raw, bytes) else str(raw)
    except Exception as exc:  # noqa: BLE001
        logger.debug("NVML name of card %d failed: %s", index, exc)
    try:
        current, _pending = pynvml.nvmlDeviceGetMigMode(handle)
        mig = bool(current)
    except Exception:  # noqa: BLE001 -- not MIG-capable: NOT_SUPPORTED
        mig = False
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total, free = int(info.total), int(info.free)
    except Exception as exc:  # noqa: BLE001
        logger.debug("NVML memory of card %d failed: %s", index, exc)
    return _NvmlCard(uuid, name, total, free, mig)


def _visible_cards(cards: list[_NvmlCard]) -> list[_NvmlCard | None]:
    """The cards this process sees as ``cuda:0``, ``cuda:1``..., by CUDA's rules.

    Applies ``CUDA_VISIBLE_DEVICES`` the way the CUDA runtime does: entries
    are indices or (prefixes of) ``GPU-`` UUIDs; parsing stops at the first
    empty or invalid entry (out of range, no unique UUID match); a card
    named twice hides every device. A ``MIG-`` entry is a MIG instance NVML
    is not asked about here, so it stays in the list as ``None`` (no memory
    facts). Integer entries assume ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` (see the
    module docstring).
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return list(cards)
    chosen: list[int | str] = []
    for part in raw.split(","):
        entry = part.strip()
        if not entry:
            break
        if entry.lower().startswith("mig-"):
            key: int | str = entry.lower()
        elif entry.isdigit():
            index = int(entry)
            if index >= len(cards):
                break
            key = index
        elif entry.lower().startswith("gpu-"):
            prefix = _normalise_uuid(entry)
            matches = [i for i, card in enumerate(cards)
                       if card.uuid and prefix and card.uuid.startswith(prefix)]
            if len(matches) != 1:
                break
            key = matches[0]
        else:
            break
        if key in chosen:
            return []
        chosen.append(key)
    return [cards[key] if isinstance(key, int) else None for key in chosen]


def _nvml_fact(position: int, card: _NvmlCard | None) -> DeviceFact:
    value = f"cuda:{position}"
    if card is None:
        return DeviceFact(value, "cuda", position, "", None, None, "none")
    if not card.usable:
        return DeviceFact(value, "cuda", position, card.name, None, None,
                          "none", card.uuid)
    return DeviceFact(value, "cuda", position, card.name,
                      card.total_bytes, card.free_bytes, "nvml", card.uuid)


def _cuda_facts_from_nvml(cards: list[_NvmlCard] | None = None) -> list[DeviceFact]:
    """Torch-free: the visible NVML cards, numbered by CUDA_VISIBLE_DEVICES."""
    if cards is None:
        cards = _nvml_cards()
    if not cards:
        return []
    return [_nvml_fact(position, card)
            for position, card in enumerate(_visible_cards(cards))]


# ── CUDA / ROCm through torch ────────────────────────────────────────

def _torch_cuda_fact(torch: Any, index: int, kind: str, name: str,
                     props: Any) -> DeviceFact:
    value = f"cuda:{index}"
    try:
        total = int(props.total_memory)
    except Exception as exc:  # noqa: BLE001
        logger.debug("torch total memory of cuda:%d failed: %s", index, exc)
        return DeviceFact(value, kind, index, name, None, None, "none")
    free = None
    try:
        # mem_get_info creates a CUDA context on the card; only pay for
        # that where this process has already initialised CUDA.
        if torch.cuda.is_initialized():
            free = int(torch.cuda.mem_get_info(index)[0])
    except Exception as exc:  # noqa: BLE001
        logger.debug("torch.cuda.mem_get_info(%d) failed: %s", index, exc)
    return DeviceFact(value, kind, index, name, total, free, "torch.cuda")


def _cuda_facts_with_torch() -> list[DeviceFact]:
    from .device_utils import cuda_device_count, cuda_device_name, cuda_is_rocm

    count = cuda_device_count()
    if count == 0:
        # No usable torch CUDA. ROCm cannot be the reason NVML sees cards,
        # so list them: inventory, not runnability (see DeviceFact).
        return _cuda_facts_from_nvml()
    rocm = cuda_is_rocm()
    kind = "rocm" if rocm else "cuda"
    try:
        import torch
    except Exception:  # noqa: BLE001 -- count > 0 means torch imported
        return []

    by_uuid: dict[str, _NvmlCard] = {}
    if not rocm:
        for card in _nvml_cards() or []:
            if card.uuid and card.usable:
                by_uuid[card.uuid] = card

    facts: list[DeviceFact] = []
    for index in range(count):
        name = cuda_device_name(index)
        try:
            props = torch.cuda.get_device_properties(index)
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_device_properties(%d) failed: %s", index, exc)
            facts.append(DeviceFact(f"cuda:{index}", kind, index, name,
                                    None, None, "none"))
            continue
        uuid = _normalise_uuid(getattr(props, "uuid", None))
        card = by_uuid.get(uuid) if uuid else None
        if card is not None:
            facts.append(DeviceFact(f"cuda:{index}", kind, index,
                                    name or card.name, card.total_bytes,
                                    card.free_bytes, "nvml", uuid))
        else:
            fact = _torch_cuda_fact(torch, index, kind, name, props)
            facts.append(dataclasses.replace(fact, uuid=uuid)
                         if uuid and kind == "cuda" else fact)
    return facts


# ── MPS ──────────────────────────────────────────────────────────────

def _mps_fact() -> DeviceFact | None:
    """Apple MPS: total and free from ``torch.mps``, capped by system RAM.

    Total is ``torch.mps.recommended_max_memory()``; free is that minus
    ``torch.mps.driver_allocated_memory()``. Apple silicon has unified
    memory, so the GPU's headroom is also bounded by what the whole system
    has left: free is capped at ``psutil.virtual_memory().available``.
    """
    from .device_utils import _mps_available

    if not _mps_available():
        return None
    name = "Apple MPS"
    try:
        import torch

        total = int(torch.mps.recommended_max_memory())
        allocated = int(torch.mps.driver_allocated_memory())
    except Exception as exc:  # noqa: BLE001
        logger.debug("torch.mps memory probe failed: %s", exc)
        return DeviceFact("mps", "mps", None, name, None, None, "none")
    free = max(total - allocated, 0)
    ram = _available_ram()
    if ram is not None:
        free = min(free, ram)
    return DeviceFact("mps", "mps", None, name, total, free, "torch.mps")


# ── merging into describe_accelerator() ──────────────────────────────

def with_memory(described: dict[str, Any],
                facts: list[DeviceFact]) -> dict[str, Any]:
    """``describe_accelerator()``'s answer with a ``memory`` key per device.

    A bare ``cuda`` entry takes the facts of the card a bare ``cuda`` means
    in this process (``torch.cuda.current_device()``). A device the probe
    did not see gets ``"memory": null``. The input is not modified.
    """
    by_value = {fact.value: fact for fact in facts}
    devices: list[dict[str, Any]] = []
    for entry in described.get("devices", []):
        value = entry.get("value")
        if value == "cuda":
            from .device_utils import _current_cuda_index

            value = f"cuda:{_current_cuda_index()}"
        fact = by_value.get(value)
        devices.append({**entry, "memory": fact.memory() if fact else None})
    return {**described, "devices": devices}

