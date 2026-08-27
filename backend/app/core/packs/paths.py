"""Where Package Center downloads and control files live on disk.

Two roots, for two lifetimes:

* the asset CACHE holds the bytes -- model snapshots and downloaded files.
  Deleting it costs a re-download and nothing else.
* the user DATA root holds the control files -- what a restart-mode install
  asked for, and the log of the job that ran it. Those have to survive the
  server exiting, which is the entire point of a restart-mode install.

Both come from the existing helpers rather than from a fresh
``os.environ`` read, so ``CODEFYUI_USER_DATA_DIR`` -- the switch that keeps
a dev clone out of the OS-wide cache -- keeps working without this module
knowing the variable exists.

These functions only COMPUTE paths. Creating the directory is the caller's
job (``mkdir(parents=True, exist_ok=True)``), so merely asking where
something would go never leaves an empty directory behind.

Nothing here reads or sets ``HF_HOME``: that variable is the whole
machine's Hugging Face cache, shared with every other tool the user runs,
and it belongs to its owner.
"""

from __future__ import annotations

from pathlib import Path

from ...config import _user_data_root
from ..asset_cache import cache_dir


def hf_cache_dir() -> Path:
    """Root for Hugging Face snapshots downloaded by the Package Center."""
    return cache_dir() / "hf"


def asset_dir() -> Path:
    """Root for single-file assets (the GloVe table and friends)."""
    return cache_dir()


def sentinel_dir() -> Path:
    """Root for the small JSON files recording "this item finished downloading".

    A sentinel, not a directory listing: a half-finished snapshot looks
    exactly like a complete one on disk.
    """
    return cache_dir() / "packs" / "state"


def sentinel_path(pack_id: str, item_id: str) -> Path:
    """The sentinel for one item of one pack."""
    return sentinel_dir() / f"{pack_id}__{item_id}.json"


def control_dir() -> Path:
    """Root for state that has to outlive the server process."""
    return _user_data_root() / "packs"


def pending_restart_file() -> Path:
    """What a restart-mode install asked for, read back after the restart."""
    return control_dir() / "pending_restart.json"


def last_restart_file() -> Path:
    """The outcome of the most recent restart-mode job, so the UI can report
    what happened while it was not running."""
    return control_dir() / "last_restart_job.json"


def job_log_dir() -> Path:
    """Per-job install logs -- the only record of a pip run nobody watched."""
    return control_dir() / "logs"
