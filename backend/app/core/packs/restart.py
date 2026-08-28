"""What the server knows about a restart-mode install, and about this GPU.

Two halves, and they are here together because the Package Center asks them
in one breath ("can I switch PyTorch, and to which wheel?"):

* the RESTART half is a stub in this release. ``restart_available()`` is
  ``False``, so a ``mode="restart"`` install is refused with the exact command
  to type instead. A later PR replaces the body with the real supervisor
  check; nothing else in the feature has to change when it does, because
  every caller already goes through :func:`install_command_for` for the
  command and through the refusal for the decision.
* the GPU half MIRRORS ``scripts/dev.py``. The installer CLI and this server
  have to agree about which CUDA wheel a driver can load -- a panel that
  recommends ``cu128`` while ``cdui install --gpu auto`` picks ``cu124`` is
  worse than no panel. dev.py is a standalone script (it must run before the
  backend is installed at all, from a Python with nothing on it), so it
  cannot be imported from here; the mapping is copied instead and
  ``test_gpu_info_never_raises_and_mirrors_dev_py`` fails the day the two
  drift apart.

:func:`gpu_info` never raises. It is read by ``GET /api/packs``, which is the
route that draws the whole panel -- a machine whose ``nvidia-smi`` hangs, or
whose torch is half-installed, must still get a Package Center it can use.
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import subprocess

from . import runner, state
from .catalog import Pack
from .paths import last_restart_file

log = logging.getLogger(__name__)

#: The pack whose install is a wheel swap rather than a download.
_GPU_TORCH_PACK_ID = "gpu-torch"

#: Mirror of ``scripts/dev.py``'s ``TORCH_INDEX_URLS`` (see the module
#: docstring for why it is a copy). ``None`` means "let PyPI resolve it";
#: ``"__skip__"`` means "do not touch torch at all".
TORCH_INDEX_URLS: dict[str, str | None] = {
    "auto":    None,                                            # resolved at runtime
    "cpu":     "https://download.pytorch.org/whl/cpu",
    "cu118":   "https://download.pytorch.org/whl/cu118",
    "cu121":   "https://download.pytorch.org/whl/cu121",
    "cu124":   "https://download.pytorch.org/whl/cu124",
    "cu126":   "https://download.pytorch.org/whl/cu126",
    "cu128":   "https://download.pytorch.org/whl/cu128",
    "rocm6.1": "https://download.pytorch.org/whl/rocm6.1",
    "rocm6.2": "https://download.pytorch.org/whl/rocm6.2",
    "mps":     None,                                            # default PyPI on Apple Silicon
    "skip":    "__skip__",                                      # leave torch untouched
}

#: The wheel variants a user may actually pick. ``auto`` is a request to
#: decide, not a variant, and ``skip`` is a request to decide nothing -- so
#: neither belongs in a dropdown of builds.
VARIANTS: tuple[str, ...] = tuple(
    key for key in TORCH_INDEX_URLS if key not in {"auto", "skip"})

#: How long ``nvidia-smi`` gets before we call it a failure. Same value as
#: dev.py: a driver that cannot answer in five seconds is not one we want to
#: hold an HTTP request open for.
_SMI_TIMEOUT_S = 5

#: Memoised :func:`detect_gpu` answer. Detection shells out, and
#: ``GET /api/packs`` is POLLED while an install runs -- running nvidia-smi
#: on every poll would block the event loop several times a second for a
#: fact that cannot change while the process is alive (a GPU is not
#: hot-plugged, and a driver upgrade requires a reboot). Tests that patch
#: detection reset this to None.
_detected: tuple[str | None, str] | None = None


def restart_available() -> bool:
    """Can this server restart itself to finish an install?

    Always False in this release: nothing yet writes the pending-restart
    file or re-launches the process, so promising a restart would strand the
    user with a server that went away and never came back. The refusal
    carries the CLI command instead, which works today.
    """
    return False


def recommended_cu_for_driver(driver_version: str) -> str:
    """Map an NVIDIA driver version to the newest wheel it can load.

    Mirrors ``dev._recommended_cu_for_driver`` exactly, floors included: the
    two must recommend the same build for the same driver or the panel and
    the CLI disagree in front of the user. Conservative on purpose -- an
    older wheel runs slower, a too-new one does not load at all.
    """
    try:
        major = int(driver_version.split(".")[0])
    except (ValueError, IndexError):
        return "cu121"
    if major >= 560:
        return "cu128"
    if major >= 555:
        return "cu126"
    if major >= 545:
        return "cu124"
    if major >= 530:
        return "cu121"
    if major >= 520:
        return "cu118"
    return "cpu"


def detect_gpu() -> tuple[str, str]:
    """Best-effort ``(display_label, recommended_variant)``.

    Mirrors ``dev.detect_gpu``: Apple Silicon first (its acceleration ships
    in the default wheel), then nvidia-smi, then rocm-smi on Linux, then
    "CPU only". Detection failures collapse to ``("CPU only", "cpu")``
    rather than raising -- the caller is drawing a panel, not installing.

    The ANSWERS mirror dev.py; how the probe is started does not, and must
    not. dev.py runs in a console the user is looking at, while this runs
    inside a server that ``cdui start`` detaches -- so nvidia-smi is given
    the same treatment as ``flows.verify_imports``: no console window on
    Windows (otherwise one flashes over the editor the first time somebody
    opens the panel), and no stdin to block on.
    """
    if platform.system() == "Darwin":
        if platform.machine() in ("arm64", "aarch64"):
            return ("Apple Silicon (MPS)", "mps")
        return ("macOS x86_64", "cpu")

    if shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=_SMI_TIMEOUT_S,
                check=True, stdin=subprocess.DEVNULL,
                creationflags=runner.creation_flags(),
            )
            first = (proc.stdout or "").strip().splitlines()[0] if proc.stdout else ""
            if first:
                name, _, driver = first.partition(",")
                name, driver = name.strip(), driver.strip()
                return (f"{name} (driver {driver})",
                        recommended_cu_for_driver(driver))
        except (subprocess.SubprocessError, OSError, ValueError, IndexError):
            pass

    if platform.system() == "Linux" and shutil.which("rocm-smi"):
        return ("AMD GPU (ROCm)", "rocm6.2")

    return ("CPU only", "cpu")


def gpu_info() -> dict:
    """Everything the Package Center needs to offer the GPU PyTorch pack.

    ``detected_label`` is what this machine has, ``recommended_variant`` is
    what it should install, ``installed_variant`` is what it already has (or
    None when that cannot be told -- see ``state.torch_variant``), and
    ``install_command`` is the line to type.

    Never raises. Every field has an honest fallback, because the alternative
    is a 500 on the one route that draws the panel.
    """
    global _detected

    if _detected is None:
        try:
            _detected = detect_gpu()
        except Exception:
            log.warning("GPU detection failed; reporting CPU", exc_info=True)
            _detected = (None, "cpu")
    label, recommended = _detected

    try:
        installed = state.torch_variant()
    except Exception:
        log.warning("could not read the installed torch variant", exc_info=True)
        installed = None

    return {
        "detected_label": label,
        "recommended_variant": recommended,
        "installed_variant": installed,
        "variants": list(VARIANTS),
        "install_command": f"cdui install --gpu {recommended}",
    }


def install_command_for(pack: Pack, variant: str | None = None) -> str:
    """The terminal command that installs *pack* outside the server.

    Shown wherever the in-app install is refused, so it has to be the whole
    line: "a restart is required" without saying what to type leaves the user
    guessing at a CLI they have never run.

    The GPU pack is not installed by ``cdui packs`` at all -- it is a wheel
    swap the installer owns -- so it gets ``cdui install --gpu <variant>``,
    defaulting to whatever this machine should have.

    *variant* is checked against :data:`VARIANTS` even though every caller
    today validates it first. This function's whole output is a line the user
    is invited to paste into a shell, so an unchecked value would be text of
    somebody else's choosing wearing a command's clothes -- and the day a CLI
    flag or a restart record reaches here unvalidated, this refuses instead of
    printing it.
    """
    if variant is not None and variant not in VARIANTS:
        raise ValueError(
            f"unknown torch variant {variant!r}; expected one of "
            f"{', '.join(VARIANTS)}")
    if pack.pack_id == _GPU_TORCH_PACK_ID:
        chosen = variant or gpu_info()["recommended_variant"]
        return f"cdui install --gpu {chosen}"
    return f"cdui packs install {pack.pack_id}"


def read_last_restart() -> dict | None:
    """The record of the most recent restart-mode job, or None.

    None when the file is absent (the usual case), unreadable, or not a JSON
    object. A corrupt record is treated as no record: this exists so the UI
    can report what happened while it was not running, and a half-written
    file is not something to report.
    """
    path = last_restart_file()
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None
