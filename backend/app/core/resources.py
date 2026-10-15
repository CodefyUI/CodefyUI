"""Device inventory with memory facts (#767).

:func:`probe_devices` lists the compute devices this machine has, each with
its total and free memory where something can say. It is display-only: no
admission decision reads it yet (that is #788), and nothing here is cached,
so every call is a fresh reading and the caller decides how often to pay
for one.

Sources, in order, per kind of device:

- **CUDA** -- ``pynvml`` (the optional ``nvidia-ml-py`` dependency) first,
  because it answers without creating a CUDA context; then
  ``torch.cuda.mem_get_info`` when torch is allowed. ROCm cards have no NVML
  and are only seen through the torch path.
- **MPS** -- ``torch.mps.recommended_max_memory()`` as the total and that
  minus ``torch.mps.driver_allocated_memory()`` as the free figure. Needs
  torch, so the torch-free mode reports no MPS device.
- **CPU** -- ``psutil.virtual_memory()`` (``total`` / ``available``).

``allow_torch=False`` is the worker agent's mode (#771): the agent process
must never import torch, because only the run subprocess may initialise
CUDA. Nothing on that path imports torch, and a test checks it in a fresh
interpreter.

Every probe degrades instead of raising: a missing library, a broken driver
or an odd answer yields ``None`` fields (or no entry) and a debug log line.
"""

from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceFact:
    """One device and what is known about its memory.

    ``value`` is the device string a run would use (``cpu``, ``mps``,
    ``cuda:1``). ``kind`` is ``cpu`` / ``cuda`` / ``rocm`` / ``mps``.
    ``index`` is the card index within its kind (``None`` for CPU and MPS).
    ``source`` names where the memory figures came from (``nvml``,
    ``torch.cuda``, ``torch.mps``, ``psutil``) or ``none`` when no probe
    answered, in which case both byte counts are ``None``.
    """

    value: str
    kind: str
    index: int | None
    name: str
    total_bytes: int | None
    free_bytes: int | None
    source: str

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
    NVML alone and MPS is not reported.
    """
    facts = [_cpu_fact()]
    facts.extend(_cuda_facts(allow_torch=allow_torch))
    if allow_torch:
        mps = _mps_fact()
        if mps is not None:
            facts.append(mps)
    return facts


# ── CPU ──────────────────────────────────────────────────────────────

def _cpu_fact() -> DeviceFact:
    name = platform.processor() or platform.machine() or "CPU"
    try:
        import psutil

        vm = psutil.virtual_memory()
        return DeviceFact("cpu", "cpu", None, name,
                          int(vm.total), int(vm.available), "psutil")
    except Exception as exc:  # noqa: BLE001 -- display only, never fatal
        logger.debug("psutil memory probe failed: %s", exc)
        return DeviceFact("cpu", "cpu", None, name, None, None, "none")


# ── CUDA / ROCm ──────────────────────────────────────────────────────

def _visible_nvml_indices(count: int) -> list[int] | None:
    """NVML indices of the cards this process sees as ``cuda:0``, ``cuda:1``...

    NVML numbers every card on the machine; CUDA numbers only the cards
    ``CUDA_VISIBLE_DEVICES`` lets through. An integer list is mapped
    entry by entry. A UUID / MIG spelling is not something this probe
    resolves, so it answers ``None`` and the caller falls back to torch.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return list(range(count))
    indices: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            index = int(part)
        except ValueError:
            return None
        # CUDA stops at the first invalid entry; so does this.
        if index < 0 or index >= count:
            break
        indices.append(index)
    return indices


def _nvml_facts() -> list[DeviceFact] | None:
    """Per-card facts from NVML, or None when NVML cannot answer at all."""
    try:
        import pynvml  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 -- optional dependency
        logger.debug("pynvml unavailable: %s", exc)
        return None
    try:
        pynvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001 -- no driver, no GPU
        logger.debug("nvmlInit failed: %s", exc)
        return None
    try:
        visible = _visible_nvml_indices(int(pynvml.nvmlDeviceGetCount()))
        if visible is None:
            return None
        facts: list[DeviceFact] = []
        for cuda_index, nvml_index in enumerate(visible):
            name = ""
            total = free = None
            source = "none"
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
                raw_name = pynvml.nvmlDeviceGetName(handle)
                name = (raw_name.decode() if isinstance(raw_name, bytes)
                        else str(raw_name))
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total, free, source = int(info.total), int(info.free), "nvml"
            except Exception as exc:  # noqa: BLE001 -- one card, not all
                logger.debug("NVML probe of card %d failed: %s", nvml_index, exc)
            facts.append(DeviceFact(f"cuda:{cuda_index}", "cuda", cuda_index,
                                    name, total, free, source))
        return facts
    except Exception as exc:  # noqa: BLE001
        logger.debug("NVML enumeration failed: %s", exc)
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


def _torch_cuda_fact(torch: Any, index: int, kind: str) -> DeviceFact:
    try:
        name = str(torch.cuda.get_device_name(index))
    except Exception:  # noqa: BLE001 -- name is best-effort
        name = ""
    try:
        free, total = torch.cuda.mem_get_info(index)
        return DeviceFact(f"cuda:{index}", kind, index, name,
                          int(total), int(free), "torch.cuda")
    except Exception as exc:  # noqa: BLE001
        logger.debug("torch.cuda.mem_get_info(%d) failed: %s", index, exc)
        return DeviceFact(f"cuda:{index}", kind, index, name, None, None, "none")


def _cuda_facts(*, allow_torch: bool) -> list[DeviceFact]:
    nvml = _nvml_facts()
    if not allow_torch:
        return nvml or []

    try:
        import torch

        if not torch.cuda.is_available():
            return nvml or []
        count = int(torch.cuda.device_count())
        kind = "rocm" if getattr(torch.version, "hip", None) is not None else "cuda"
    except Exception as exc:  # noqa: BLE001 -- torch missing or broken
        logger.debug("torch CUDA probe failed: %s", exc)
        return nvml or []

    by_value = {fact.value: fact for fact in nvml or []}
    facts: list[DeviceFact] = []
    for index in range(count):
        fact = by_value.get(f"cuda:{index}")
        if fact is None or fact.source == "none":
            fact = _torch_cuda_fact(torch, index, kind)
        facts.append(fact)
    return facts


# ── MPS ──────────────────────────────────────────────────────────────

def _mps_fact() -> DeviceFact | None:
    try:
        import torch

        if not (hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()):
            return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("torch MPS probe failed: %s", exc)
        return None
    name = "Apple MPS"
    try:
        total = int(torch.mps.recommended_max_memory())
        allocated = int(torch.mps.driver_allocated_memory())
        return DeviceFact("mps", "mps", None, name,
                          total, max(total - allocated, 0), "torch.mps")
    except Exception as exc:  # noqa: BLE001
        logger.debug("torch.mps memory probe failed: %s", exc)
        return DeviceFact("mps", "mps", None, name, None, None, "none")


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
