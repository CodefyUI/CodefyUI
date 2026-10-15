"""Storage mount labels: one name for one shared directory on every host (#782).

The lab's NAS is mounted at a different path on each machine. A label
names it the same way everywhere: a worker agent declares
``--mount data=/nas/lab/codefyui/data`` (``hello.mounts``), and the control
plane declares that its own data directory is ``data`` too
(``CODEFYUI_MOUNT_LABEL``, default ``data``). Two hosts holding the same
label see the same files, so

- a run that needs files (:mod:`core.run_needs`) is only placed on a worker
  that holds the label (``scheduling.WorkerScheduler``);
- a file such a run produces is recorded as ``(backend="mount",
  mount_label, path relative to the mount)`` and translated to this
  control plane's own path when it is read back (:func:`translate`).

A **same-host** worker (its machine identity is this server's, #773) holds
the control plane's label automatically: it is on this machine's
filesystem. When it declares no directory for the label itself, the
assignment LENDS it this server's data directories exactly as this server
has them configured (:func:`lends_mount`, :func:`lent_dirs`,
``worker_protocol.Storage.dirs``). Either way its files are on this
machine, so they are recorded as ``local`` rows with their absolute path,
never as ``mount`` rows (``Storage.relative_artifacts`` is off).

The mount root and the data directories under it
------------------------------------------------
A mount is a directory laid out like the control plane's data directory,
so one path derives every data setting (:func:`data_root_env`), mirroring
``config.py``:

- a plain data directory (``backend/data`` in a default install):
  ``files/`` (``DATA_FILES_DIR``), ``images/``, ``models/``, ``media/``,
  ``graphs/``, ``presets/``;
- a project directory (it holds ``codefyui.project.toml``):
  ``CODEFYUI_PROJECT_DIR=<root>``, and ``config.py``'s project derivation
  does the rest (``assets/images``, ``assets/models``, ``assets/media``,
  ``graphs``).

On the control plane the root of its label is the project directory in
project mode and the data root (``MODELS_DIR.parent``) otherwise
(:func:`control_plane_root`), so a relative path means the same file on
both sides.

This module imports nothing heavy and reads ``settings`` only inside the
control-plane helpers: the agent imports it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath
from typing import Any

#: ``exec_run_artifacts.backend`` values. ``local``: an absolute path on this
#: machine (every row before #782). ``mount``: ``path`` is relative to the
#: mount ``mount_label`` names. ``transfer`` (#785) and ``s3`` (#791) are
#: reserved for the backends that move files.
BACKEND_LOCAL = "local"
BACKEND_MOUNT = "mount"
BACKEND_TRANSFER = "transfer"
BACKEND_S3 = "s3"
BACKENDS = (BACKEND_LOCAL, BACKEND_MOUNT, BACKEND_TRANSFER, BACKEND_S3)

#: The label a control plane gives its data directory unless
#: ``CODEFYUI_MOUNT_LABEL`` says otherwise.
DEFAULT_LABEL = "data"

#: Labels, matched with ``fullmatch``: the worker-name alphabet.
LABEL_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,62}")

#: The file that makes a directory a CodefyUI project.
PROJECT_MANIFEST = "codefyui.project.toml"

#: The data settings a plain data directory derives, as ``config.py`` lays
#: them out under ``backend/data``: env var -> subdirectory.
DATA_DIR_LAYOUT: dict[str, str] = {
    "CODEFYUI_DATA_FILES_DIR": "files",
    "CODEFYUI_IMAGES_DIR": "images",
    "CODEFYUI_MODELS_DIR": "models",
    "CODEFYUI_MEDIA_DIR": "media",
    "CODEFYUI_GRAPHS_DIR": "graphs",
    "CODEFYUI_USER_PRESETS_DIR": "presets",
}


def parse_mount(raw: str) -> tuple[str, str]:
    """``"data=/nas/x"`` -> ``("data", "/nas/x")``; ``ValueError`` when malformed.

    The path may be empty (``data=``): that removes the label from a saved
    agent configuration.
    """
    label, sep, path = raw.partition("=")
    label = label.strip().lower()
    if not sep:
        raise ValueError(f"--mount {raw!r}: expected LABEL=PATH")
    if LABEL_PATTERN.fullmatch(label) is None:
        raise ValueError(f"--mount {raw!r}: the label must be lower-case "
                         "letters, digits, '-' or '_' (at most 63)")
    return label, path.strip()


def is_project_dir(root: Path) -> bool:
    return (root / PROJECT_MANIFEST).is_file()


def data_root_env(root: str | Path) -> tuple[dict[str, str], list[str]]:
    """``(set, unset)``: the environment that points CodefyUI's data settings
    at the mount *root*.

    A project directory sets ``CODEFYUI_PROJECT_DIR`` and unsets the
    per-directory variables, which would otherwise win over the project
    derivation as explicitly set roots. A plain data directory sets each
    of :data:`DATA_DIR_LAYOUT` and unsets ``CODEFYUI_PROJECT_DIR``.
    """
    base = Path(root)
    if is_project_dir(base):
        return ({"CODEFYUI_PROJECT_DIR": str(base)}, list(DATA_DIR_LAYOUT))
    return ({name: str(base / sub) for name, sub in DATA_DIR_LAYOUT.items()},
            ["CODEFYUI_PROJECT_DIR"])


def relative_to_mount(path: str | os.PathLike[str], root: str | Path) -> str | None:
    """*path* relative to the mount *root*, POSIX-spelled, or None when it
    is not inside it. Both sides are resolved first (a symlinked mount
    point, ``..`` segments)."""
    try:
        resolved = Path(path).resolve()
        base = Path(root).resolve()
    except (OSError, ValueError):
        return None
    if not resolved.is_relative_to(base):
        return None
    return resolved.relative_to(base).as_posix()


# ── the control plane's side ─────────────────────────────────────────────


def control_plane_label() -> str:
    """This server's label for its data directory (``CODEFYUI_MOUNT_LABEL``)."""
    from ..config import settings

    label = str(settings.MOUNT_LABEL or "").strip().lower()
    return label or DEFAULT_LABEL


def control_plane_root() -> Path:
    """The directory this server's label names: the project directory in
    project mode, else the data root every node write stays under."""
    from ..config import settings
    from .data_paths import data_root

    if settings.PROJECT_DIR is not None:
        return Path(settings.PROJECT_DIR).resolve()
    return data_root()


def control_plane_mounts() -> dict[str, str]:
    """``{label: path}`` for this server: one entry in v1."""
    return {control_plane_label(): str(control_plane_root())}


#: The data settings a same-host worker borrows (:func:`lent_dirs`).
LENT_SETTINGS = ("DATA_FILES_DIR", "IMAGES_DIR", "MODELS_DIR", "MEDIA_DIR")


def lends_mount(worker: Any, label: str) -> bool:
    """The lending rule, in one place: this server lends *worker* its own
    data directories for *label* when the worker runs on this machine,
    *label* is this server's label, and the worker declared no directory
    of its own for it. ``assignment_storage``, the Workers list and
    :func:`holds_mount` all ask this."""
    mounts = getattr(worker, "mounts", None) or {}
    return (bool(getattr(worker, "same_host", False))
            and label == control_plane_label() and label not in mounts)


def lent_dirs() -> dict[str, str]:
    """``CODEFYUI_<NAME>`` -> path for each of :data:`LENT_SETTINGS`, as
    this server has them configured (an explicit ``CODEFYUI_MODELS_DIR``,
    project mode, the defaults alike)."""
    from ..config import settings

    return {f"CODEFYUI_{name}": str(Path(getattr(settings, name)).resolve())
            for name in LENT_SETTINGS}


def holds_mount(worker: Any, label: str) -> bool:
    """Whether *worker* can reach the files under *label*: it declared the
    label, or this server lends it (:func:`lends_mount`)."""
    mounts = getattr(worker, "mounts", None) or {}
    return label in mounts or lends_mount(worker, label)


def translate(backend: str | None, mount_label: str | None,
              path: str) -> str | None:
    """The path a stored artifact row names on this control plane, or
    None when this server cannot name one.

    A ``local`` row (and a row from before #782 with no backend) comes back
    as stored. A ``mount`` row is joined onto this server's root when its
    label is this server's; for any other label, or a stored path that is
    absolute or climbs out of the mount, there is no local path: None, and
    the caller keeps the relative path and the label instead. A relative
    string is never handed out as if it were a path on this machine.
    """
    if backend != BACKEND_MOUNT:
        return path
    if not mount_label or mount_label != control_plane_label():
        return None
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    return str(control_plane_root().joinpath(*relative.parts))
