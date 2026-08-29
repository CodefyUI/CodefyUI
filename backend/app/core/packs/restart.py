"""What the server knows about a restart-mode install, and about this GPU.

Two halves, and they are here together because the Package Center asks them
in one breath ("can I switch PyTorch, and to which wheel?"):

* the RESTART half answers "can this server finish an install that would
  have to replace something it has already imported?" -- by going away and
  coming back. It is offered only to a process ``cdui start`` launched
  (:func:`restart_available`), and it runs as a handshake across the gap
  where this process does not exist: write down what to install
  (:func:`write_pending`), start a helper that outlives us
  (:func:`spawn_helper`), then ask uvicorn to shut down gracefully
  (:func:`schedule_self_shutdown`). The helper waits for this pid to go,
  reinstalls, records the outcome where :func:`read_last_restart` will find
  it, and relaunches the server. Where a restart is NOT available the
  install is refused with the exact command to type, through
  :func:`install_command_for`, which works today.
* the GPU half MIRRORS ``scripts/dev.py``. The installer CLI and this server
  have to agree about which CUDA wheel a driver can load -- a panel that
  recommends ``cu128`` while ``cdui install --gpu auto`` picks ``cu124`` is
  worse than no panel. dev.py is a standalone script (it must run before the
  backend is installed at all, from a Python with nothing on it), so it
  cannot be imported from here; the mapping is copied instead and
  ``test_gpu_info_never_raises_and_mirrors_dev_py`` fails the day the two
  drift apart.

Two files carry the restart half across the gap, and both are read by
``scripts/dev.py``'s helper, which runs from an interpreter that has none of
this installed and so cannot import a line of it. Their SCHEMAS are the
contract, written down here (:class:`PendingRestart` and
:func:`write_last_restart`) because this is where they are designed; dev.py
duplicates the twenty lines that read and write them rather than importing
the app, which is the price of a helper that can run while the app's
packages are being replaced under it.

:func:`gpu_info` never raises. It is read by ``GET /api/packs``, which is the
route that draws the whole panel -- a machine whose ``nvidia-smi`` hangs, or
whose torch is half-installed, must still get a Package Center it can use.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import runner, state
from .catalog import Pack
from .errors import PackInstallError, PendingExists
from .paths import job_log_dir, last_restart_file, pending_restart_file

log = logging.getLogger(__name__)

#: The pack whose install is a wheel swap rather than a download.
_GPU_TORCH_PACK_ID = "gpu-torch"

#: Kill switch. ``"0"`` refuses restart-mode installs even under
#: ``cdui start``; anything else (including unset) leaves them on. A kill
#: switch rather than an opt-in because the thing it guards against -- a
#: server that goes away and does not come back on some particular machine
#: -- is discovered by an operator who then needs to turn it off for a whole
#: classroom without downgrading, and because a feature nobody can reach by
#: default is a feature nobody tests.
ENABLE_ENV = "CODEFYUI_ENABLE_RESTART_INSTALL"

#: JSON list: the OUTER interpreter and ``scripts/dev.py``, exported by
#: ``cdui start``. NOT the ``cdui`` shim -- the shim's whole job is to FIND
#: that interpreter, and asking a detached child to find it again is a
#: second chance to find a different one (a venv that moved, a PATH that
#: changed). The first element is checked for existence before a restart is
#: offered, because a checkout that has been deleted cannot bring anything
#: back.
LAUNCHER_ENV = "CODEFYUI_LAUNCHER"

#: JSON list: the arguments ``cdui start`` was given (host, port, project,
#: extras). The helper relaunches with exactly these, so the server comes
#: back on the address the browser is still pointing at.
RELAUNCH_ARGV_ENV = "CODEFYUI_RELAUNCH_ARGV"

#: Version of the pending-restart file. dev.py's helper reads that file
#: without importing this module, so the number is the handshake: a helper
#: from an older install refuses a file it does not understand instead of
#: guessing at it.
PENDING_SCHEMA = 1

#: Version of the outcome record -- see :func:`write_last_restart`.
OUTCOME_SCHEMA = 1

#: How long a pending file may sit before it is treated as abandoned. A
#: restart is a few seconds of shutdown plus however long uv takes; fifteen
#: minutes is longer than a torch reinstall over a slow link and shorter
#: than anybody's patience, so a claim left behind by a machine that lost
#: power does not block the next attempt forever.
STALE_PENDING_S = 15 * 60

#: The ``dev.py`` subcommand that finishes the install once this process has
#: exited.
HELPER_COMMAND = "packs-run-pending"

#: No console at all, which is what a process outliving its parent needs.
#: Not exposed by ``subprocess`` off Windows, so it is spelled out here --
#: ``dev.py``'s ``start`` does the same for the server it daemonises.
DETACHED_PROCESS = 0x00000008

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


def _env_argv(name: str) -> tuple[str, ...]:
    """A JSON list of strings from the environment, or ``()``.

    Never raises. Another program writes these variables, and every caller's
    honest answer to "that was not a list of strings" is the same as its
    answer to "it was not set": there is no launcher here.
    """
    raw = os.environ.get(name)
    if not raw:
        return ()
    try:
        value = json.loads(raw)
    except ValueError:
        log.warning("%s is not JSON; ignoring it", name)
        return ()
    if not isinstance(value, list) or not all(
            isinstance(part, str) for part in value):
        log.warning("%s is not a list of strings; ignoring it", name)
        return ()
    return tuple(value)


def restart_available() -> bool:
    """Can this server restart itself to finish an install?

    Three facts, all of which have to hold, and none of which this process
    can talk itself into:

    * ``CODEFYUI_MANAGED == "start"`` -- something launched this process that
      knows how to launch it again. ``cdui dev`` reloads in place and does
      not relaunch; a bare ``uvicorn`` has nobody at all, and promising
      either of them a restart strands the user with a server that went away
      and never came back.
    * :data:`LAUNCHER_ENV` names a file that is STILL THERE. The launcher is
      what brings the server back, and a checkout that has been moved or
      deleted since ``cdui start`` ran cannot.
    * the kill switch (:data:`ENABLE_ENV`) is not thrown.

    Never raises, for the same reason :func:`gpu_info` does not: the caller
    is deciding what to draw in a panel, and the honest failure is "no".
    """
    try:
        if os.environ.get("CODEFYUI_MANAGED") != "start":
            return False
        if os.environ.get(ENABLE_ENV, "1") == "0":
            return False
        launcher = _env_argv(LAUNCHER_ENV)
        return bool(launcher) and Path(launcher[0]).is_file()
    except Exception:  # pragma: no cover - is_file() eats its own OSErrors
        log.warning("the restart availability check failed", exc_info=True)
        return False


def runs_active(app) -> bool:
    """Is a graph run in flight -- running, or waiting for a device?

    A restart-mode install ends this process, and a run that dies with it is
    minutes or hours of somebody's training thrown away with no output and
    no error anyone asked for. So this is a veto, not a hint.

    QUEUED runs count as much as running ones. The queue lives in this
    process's memory, so a restart does not postpone a queued run, it LOSES
    it: ``RunService.recover_interrupted`` retires the rows a dead process
    left behind at the next startup.

    Answers True when it cannot tell. A restart refused because the service
    could not be read costs a retry; a restart allowed on a wrong "no" costs
    the run.
    """
    service = getattr(getattr(app, "state", None), "run_service", None)
    if service is None:
        # Before the lifespan built one, or after it tore it down. Nothing
        # can be running, because nothing can have been started.
        return False
    try:
        if service.active_run_ids():
            return True
        return any(service.queue_snapshot().values())
    except Exception:
        log.warning("could not read the run service; assuming it is busy",
                    exc_info=True)
        return True


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


def resolve_gpu_torch(variant: str | None) -> tuple[str, str]:
    """``(variant, index_url)`` for a GPU PyTorch install.

    ``None`` and ``"auto"`` are the same request -- "decide for me" -- and
    are answered by :func:`gpu_info`, so the wheel a restart installs is the
    wheel the panel offered.

    Two refusals, both deliberate:

    * ``"mps"`` has nothing to switch to. Apple Silicon's acceleration ships
      in the default PyPI wheel, so there is no index to reinstall from --
      and a pending file carrying ``index_url: null`` would reach the
      helper's ``--index-url`` as the four letters ``None``. This fires for
      an explicit ``"mps"`` and for an ``auto`` that resolves to it.
    * anything else unknown -- ``"skip"`` included, which is a request to
      change nothing -- is refused BY NAME, listing what would have been
      accepted. The value ends up in a subprocess argument list, so the day
      one arrives from a request body unvalidated, the check is already
      here.
    """
    chosen = (gpu_info()["recommended_variant"]
              if variant in (None, "auto") else variant)

    if chosen == "mps":
        raise ValueError(
            "torch variant 'mps' has no wheel index to switch to; Apple "
            "Silicon acceleration ships in the default PyPI build")
    if chosen not in VARIANTS:
        raise ValueError(
            f"unknown torch variant {chosen!r}; expected one of "
            f"{', '.join(VARIANTS)}")

    index_url = TORCH_INDEX_URLS[chosen]
    if not index_url or index_url == "__skip__":  # pragma: no cover
        # Unreachable while every VARIANTS entry but mps has a real URL;
        # here so that adding one without an index fails loudly rather than
        # writing a pending file the helper cannot install from.
        raise ValueError(f"torch variant {chosen!r} has no index URL")
    return chosen, index_url


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


def write_last_restart(record: dict) -> None:
    """Record how a restart-mode job ended, for the UI that comes back.

    The schema, which ``dev.py``'s helper writes with its own two-line copy
    of this function (it cannot import ``app`` -- it runs while ``app``'s
    packages are being replaced)::

        {"schema": 1, "job_id": str, "pack_id": str, "kind": "torch"|"pip",
         "status": "ok"|"failed", "returncode": int|None, "message": str,
         "log_tail": str, "finished_at": iso8601}

    ``status`` and ``message`` are the CONTRACT: the SPA reads exactly those
    two to tell the user what happened while it was not running (packStore's
    ``checkInProgress``). Everything else is for whoever opens the file.

    Atomic, and deliberately NOT validated. The writer that matters most has
    no access to this module, so a check here would describe only one of the
    two writers -- :func:`read_last_restart` is where a bad record is
    handled, on the reading side that both of them share.
    """
    text = json.dumps(record, indent=2)
    path = last_restart_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


# -- the pending file ------------------------------------------------------

def _require_text(data: dict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"pending restart file: {key!r} must be a non-empty string")
    return value


def _require_optional_text(data: dict, key: str) -> str | None:
    value = data.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(
            f"pending restart file: {key!r} must be a string or null")
    return value


def _require_text_tuple(data: dict, key: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not all(
            isinstance(part, str) for part in value):
        raise ValueError(
            f"pending restart file: {key!r} must be a list of strings")
    return tuple(value)


def _require_pid(data: dict, key: str) -> int:
    value = data.get(key)
    # ``isinstance(True, int)`` is True, and ``True == 1``: without the bool
    # check a JSON ``true`` would arrive here as the pid 1, which on POSIX
    # is init and never exits -- a pending file nothing could ever clear.
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"pending restart file: {key!r} must be a positive integer")
    return value


@dataclass(frozen=True)
class PendingRestart:
    """What a restart-mode install asked for: the file the helper reads.

    Written by the server just before it shuts down, read by ``dev.py
    packs-run-pending`` from an interpreter that cannot import this class.
    The JSON is therefore the interface, and this dataclass is one of its two
    implementations -- which is why :meth:`from_json` validates rather than
    trusts, and why the field list is flat and dull (no nesting, no enums,
    nothing that needs a library to read).

    The fields, and why each one has to be in the file rather than worked
    out again later:

    * ``kind`` -- ``"torch"`` swaps the wheel with ``--index-url``,
      ``"pip"`` installs ``specs`` with no constraints file. They take
      different command lines, so the helper must not have to guess.
    * ``index_url`` / ``packages`` / ``specs`` -- decided HERE, where the
      GPU probe and the catalog are available. The helper has neither.
    * ``venv_python`` -- the interpreter to install INTO. The helper's own
      is the outer one, which is a different environment entirely.
    * ``server_pid`` -- who to wait for. Replacing packages while this
      process still has them imported is the failure the restart exists to
      avoid, and it is also how :func:`write_pending` tells a live claim
      from an abandoned one.
    * ``launcher`` / ``relaunch_argv`` -- how to bring the server back, on
      the same address the browser is still pointing at.
    * ``created_at`` -- how :data:`STALE_PENDING_S` is measured.

    Nothing here is trusted by the helper on the strength of being in the
    file: it validates ``venv_python`` against ``backend/.venv`` and
    ``launcher`` against the repo root before it runs anything (R3). This
    file is written by the server into the user's own data directory, and
    that is exactly the assumption worth double-checking.
    """

    schema: int
    job_id: str
    pack_id: str
    kind: str                       # "torch" | "pip"
    index_url: str | None
    packages: tuple[str, ...]       # torch kind: ("torch", "torchvision")
    specs: tuple[str, ...]          # pip kind: the pack's PEP 508 specs
    venv_python: str
    server_pid: int
    launcher: tuple[str, ...]
    relaunch_argv: tuple[str, ...]
    created_at: str

    def to_json(self) -> str:
        """The file's exact contents. Indented: a person debugging a restart
        that did not come back reads this file with an editor."""
        return json.dumps({
            "schema": self.schema,
            "job_id": self.job_id,
            "pack_id": self.pack_id,
            "kind": self.kind,
            "index_url": self.index_url,
            "packages": list(self.packages),
            "specs": list(self.specs),
            "venv_python": self.venv_python,
            "server_pid": self.server_pid,
            "launcher": list(self.launcher),
            "relaunch_argv": list(self.relaunch_argv),
            "created_at": self.created_at,
        }, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "PendingRestart":
        """Parse and CHECK a pending file. ``ValueError`` on any bad shape.

        One exception type for every rejection -- a bad parse, a schema from
        the future, a missing key, a pid that is a string -- because every
        caller does the same thing with all of them: treat the file as if it
        were not there. ``json.JSONDecodeError`` is already a ``ValueError``,
        so the promise costs nothing to keep.
        """
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("pending restart file is not a JSON object")

        schema = data.get("schema")
        if isinstance(schema, bool) or schema != PENDING_SCHEMA:
            raise ValueError(
                f"pending restart file: schema {schema!r} is not "
                f"{PENDING_SCHEMA}")

        kind = data.get("kind")
        if kind not in ("torch", "pip"):
            raise ValueError(
                f"pending restart file: kind {kind!r} is not 'torch' or 'pip'")

        return cls(
            schema=PENDING_SCHEMA,
            job_id=_require_text(data, "job_id"),
            pack_id=_require_text(data, "pack_id"),
            kind=kind,
            index_url=_require_optional_text(data, "index_url"),
            packages=_require_text_tuple(data, "packages"),
            specs=_require_text_tuple(data, "specs"),
            venv_python=_require_text(data, "venv_python"),
            server_pid=_require_pid(data, "server_pid"),
            launcher=_require_text_tuple(data, "launcher"),
            relaunch_argv=_require_text_tuple(data, "relaunch_argv"),
            created_at=_require_text(data, "created_at"),
        )


def build_pending(pack: Pack, *, job_id: str, kind: str,
                  variant: str | None = None) -> PendingRestart:
    """The claim this server is about to write down.

    ``venv_python`` is THIS interpreter, not "the venv next to the helper's
    working directory" -- ``runner`` pins ``--python`` for the same reason,
    and a server started from somewhere else would otherwise have its
    packages installed into a different environment than the one that asked.

    ``launcher`` and ``relaunch_argv`` are read from the environment
    ``cdui start`` exported. An empty launcher is not refused here: this
    function builds a record, and :func:`spawn_helper` is where a record
    that cannot be acted on is caught -- before the shutdown is scheduled.
    """
    if kind == "torch":
        _, index_url = resolve_gpu_torch(variant)
        packages: tuple[str, ...] = ("torch", "torchvision")
        specs: tuple[str, ...] = ()
    elif kind == "pip":
        index_url = None
        packages, specs = (), tuple(pack.pip)
    else:
        raise ValueError(
            f"unknown restart install kind {kind!r}; expected 'torch' or 'pip'")

    return PendingRestart(
        schema=PENDING_SCHEMA,
        job_id=job_id,
        pack_id=pack.pack_id,
        kind=kind,
        index_url=index_url,
        packages=packages,
        specs=specs,
        venv_python=sys.executable,
        server_pid=os.getpid(),
        launcher=_env_argv(LAUNCHER_ENV),
        relaunch_argv=_env_argv(RELAUNCH_ARGV_ENV),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


#: Windows' "this process has not exited yet" exit code.
_STILL_ACTIVE = 259
#: ``PROCESS_QUERY_LIMITED_INFORMATION`` -- the smallest access right that
#: still answers "is it running?", and the one that works across users.
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
#: ``ERROR_ACCESS_DENIED``: the process exists, it is simply not ours.
_ERROR_ACCESS_DENIED = 5


def _pid_alive_windows(pid: int) -> bool:
    """The Windows half of :func:`_pid_alive`, via ``ctypes``.

    ``tasklist`` would also answer (``dev.py`` uses it), but this runs
    inside the server: shelling out costs a process and, without
    ``CREATE_NO_WINDOW``, flashes a console over the editor.

    ``OpenProcess`` alone is not the answer. A handle can still be opened
    for a process that has exited but whose object has not been released --
    the pid would read as alive forever, and the pending file it wrote could
    never be replaced. ``GetExitCodeProcess`` is what distinguishes the two.
    Its one blind spot is a process that genuinely exited with code 259; no
    Python interpreter does, and the cost of being wrong is one refused
    install rather than a corrupted environment.
    """
    import ctypes  # Windows-only, and only on this path.

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # Declared rather than left to ctypes' defaults: a HANDLE is
        # pointer-sized and the default ``c_int`` return would truncate it
        # on 64-bit Windows.
        kernel32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int,
                                         ctypes.c_uint32)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.GetExitCodeProcess.argtypes = (ctypes.c_void_p,
                                                ctypes.POINTER(ctypes.c_ulong))
        kernel32.GetExitCodeProcess.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel32.CloseHandle.restype = ctypes.c_int

        handle = kernel32.OpenProcess(
            _PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
        try:
            code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == _STILL_ACTIVE
            return True
        finally:
            kernel32.CloseHandle(handle)
    except (OSError, AttributeError, ValueError):
        log.warning("could not ask Windows about pid %s; assuming it is alive",
                    pid, exc_info=True)
        return True


def _pid_alive(pid: int) -> bool:
    """Is a process with this id running right now?

    ``psutil`` is not a dependency of this project and will not become one
    for a single predicate, so this is the stdlib version of it: a null
    signal on POSIX, a process handle on Windows.

    Every unknown answers TRUE. The only caller that acts on a False is the
    one that DELETES another server's pending file, and being wrong in that
    direction means two helpers installing into one site-packages.
    """
    if pid <= 0:
        # 0 is "every process in my group" to ``os.kill`` and would signal
        # this server; negatives are process groups. Neither is a pid we
        # could have written.
        return False
    if sys.platform == "win32":
        return _pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True     # it exists; it belongs to somebody else
    except OSError:
        log.warning("could not signal pid %s; assuming it is alive", pid,
                    exc_info=True)
        return True
    return True


def _read_pending(path: Path) -> "PendingRestart | None":
    """The claim on disk, or None when there is not a readable one."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return PendingRestart.from_json(raw)
    except ValueError:
        log.warning("the pending restart file is unreadable; "
                    "treating it as absent", exc_info=True)
        return None


def _age_seconds(pending: PendingRestart) -> "float | None":
    """How long ago the claim was made, or None when it does not say."""
    try:
        created = datetime.fromisoformat(pending.created_at)
    except ValueError:
        return None
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - created).total_seconds()


def _is_stale(pending: PendingRestart) -> bool:
    """Has this claim been abandoned -- dead writer, or too old to be real?

    Either fact is enough. The pid catches the ordinary case (a server that
    crashed between writing the file and exiting); the age catches the one
    the pid cannot -- a machine that rebooted and handed the same number to
    something else, which would otherwise read as alive forever.
    """
    if not _pid_alive(pending.server_pid):
        return True
    age = _age_seconds(pending)
    return age is not None and age > STALE_PENDING_S


def write_pending(pending: PendingRestart) -> Path:
    """Write the claim, atomically, refusing to trample a live one.

    Atomic (temp file plus ``os.replace``) because the reader is another
    process that may look at any moment: a half-written file would be a
    restart the helper refuses, on a server that has already gone away.

    :raises PendingExists: a claim is already there, made by a process that
        is still alive, and recent enough to still be under way. Two claims
        means two ``uv`` runs over one site-packages. A STALE claim -- dead
        writer, or older than :data:`STALE_PENDING_S` -- is overwritten
        without ceremony, and so is one that cannot be parsed.
    """
    path = pending_restart_file()
    existing = _read_pending(path)
    if existing is not None and not _is_stale(existing):
        raise PendingExists(
            f"a restart-mode install is already pending for "
            f"{existing.pack_id} (job {existing.job_id}), waiting for "
            f"process {existing.server_pid} to exit",
            hint="wait for that restart to finish before starting another")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(pending.to_json(), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
    return path


def clear_stale_pending() -> bool:
    """Delete an abandoned claim. True when one was deleted.

    For startup, on both sides: a server that comes back to find a pending
    file from a process that no longer exists is looking at the wreckage of
    a restart that did not happen, and leaving it there would refuse every
    future install with "one is already pending".

    An unparseable file is deleted too. Writes here are atomic, so a file
    nobody can read is not a half-written claim -- it is not a claim.
    """
    path = pending_restart_file()
    if not path.exists():
        return False
    pending = _read_pending(path)
    if pending is not None and not _is_stale(pending):
        return False
    try:
        path.unlink()
    except OSError:
        log.warning("could not delete the stale pending restart file",
                    exc_info=True)
        return False
    log.info("cleared a stale pending restart file")
    return True


# -- the helper, and going away --------------------------------------------

def _log_file_name(job_id: str) -> str:
    """The helper's log file for *job_id*, as a NAME and never a path.

    The id is read back out of a file on disk and then concatenated into a
    path, which is the shape of every directory-traversal bug ever written.
    Nothing puts anything but a uuid4 hex in there today, so the
    substitution should never fire -- and it is a substitution rather than a
    rejection for that reason: an odd job id should cost an ugly log name,
    not a restart that refuses to start.
    """
    return f"restart-{re.sub(r'[^A-Za-z0-9._-]', '_', job_id)}.log"


def spawn_helper(pending_path: Path) -> int:
    """Start the process that finishes the install, and outlives this one.

    Returns its pid, which is what the caller logs; nothing waits for it,
    because this server is about to stop existing.

    Detached in whichever way the OS understands. POSIX gets
    ``start_new_session``: its own session and process group, so neither the
    terminal's Ctrl-C nor the SIGHUP of a closing shell reaches it. Windows
    gets ``DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP``: no console to
    inherit, and no console to be Ctrl-C'd through -- the same pair
    ``cdui start`` uses to daemonise the server this will replace.

    Output goes to a FILE, never a pipe. The parent is seconds from exiting,
    and a pipe with nobody left to read it fills up and blocks the helper
    mid-install. That log is the only record of an install nobody watched.

    The pending file is read here rather than passed in as an object, and
    that is the point: it proves the handshake works while there is still a
    server to report a failure. A file that cannot be parsed, or that names
    no launcher, raises HERE -- before :func:`schedule_self_shutdown` is
    called and the chance to say anything is gone.
    """
    pending_path = Path(pending_path)
    pending = PendingRestart.from_json(
        pending_path.read_text(encoding="utf-8"))
    if not pending.launcher:
        raise PackInstallError(
            "cannot start the restart helper: this server was launched "
            f"without {LAUNCHER_ENV}")

    argv = [*pending.launcher, HELPER_COMMAND, str(pending_path)]
    log_dir = job_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / _log_file_name(pending.job_id)

    detach: dict = {}
    if sys.platform == "win32":
        detach["creationflags"] = (DETACHED_PROCESS
                                   | runner.CREATE_NEW_PROCESS_GROUP)
    else:
        detach["start_new_session"] = True

    log.info("starting the restart helper: %s (log: %s)",
             " ".join(argv), log_path)
    with open(log_path, "ab") as log_file:
        # An argv list and no shell, like every other subprocess here: the
        # parts come from the environment and from a file this server wrote,
        # and neither is a reason to hand the box's shell a string.
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=runner.pip_env(),
            **detach,
        )
    return proc.pid


def schedule_self_shutdown(loop, delay: float = 0.5) -> None:
    """Ask uvicorn to stop, shortly after the response has gone out.

    DELAYED, because the client is still waiting for the 202 that tells it a
    restart is under way. Shutting down inside the request handler answers
    that request with a closed socket, and the SPA cannot tell "the restart
    started" from "the server crashed".

    SIGINT rather than either alternative, and both were considered:

    * ``Server.should_exit = True`` needs the ``uvicorn.Server`` object, and
      nothing hands one to an application -- it is not on ``app.state``, and
      digging it out of the loop's tasks is a guess about uvicorn's
      internals that a version bump gets to break.
    * ``os.kill(os.getpid(), SIGINT)`` is not the same call on Windows,
      where ``os.kill`` means ``TerminateProcess`` for everything except
      ``CTRL_C_EVENT``/``CTRL_BREAK_EVENT`` -- and those go to a process
      GROUP, which under ``cdui start`` includes the console the user is
      typing in. Terminating also skips the lifespan shutdown entirely: the
      database never closes and in-flight runs are never retired.

    ``signal.raise_signal`` raises the signal in THIS process, on every
    platform, so uvicorn's own handler runs: it sets ``should_exit``, the
    server stops accepting, and the lifespan's shutdown half runs to the
    end. It is what Ctrl-C does, without a console.
    """
    loop.call_later(delay, signal.raise_signal, signal.SIGINT)
