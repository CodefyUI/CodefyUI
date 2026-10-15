"""The agent's saved configuration: ``<user-data>/agent.json`` (#782).

What an operator sets once and the agent keeps across restarts. Today that
is the storage mounts (``--mount LABEL=PATH``): ``{"mounts": {"data":
"/nas/lab/codefyui/data"}}``. The file is read and rewritten as a whole
object, keeping keys this module does not know, so later settings can live
beside the mounts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..core.mounts import parse_mount

CONFIG_NAME = "agent.json"


def config_path(user_data_dir: Path) -> Path:
    return user_data_dir / CONFIG_NAME


def load(user_data_dir: Path) -> dict[str, Any]:
    """The saved configuration, ``{}`` when there is none or it is unreadable."""
    try:
        data = json.loads(config_path(user_data_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save(user_data_dir: Path, config: dict[str, Any]) -> None:
    """Write *config* atomically (a temp file, then ``os.replace``)."""
    path = config_path(user_data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def saved_mounts(config: dict[str, Any]) -> dict[str, str]:
    raw = config.get("mounts")
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if isinstance(v, str) and v}


def merge_mounts(saved: dict[str, str], flags: list[str] | None
                 ) -> dict[str, str]:
    """*saved* updated by ``--mount`` *flags*, in order. ``LABEL=PATH`` sets
    a label; ``LABEL=`` (no path) removes it. A path is made absolute
    (``~`` expanded). ``ValueError`` for a malformed flag."""
    mounts = dict(saved)
    for raw in flags or ():
        label, path = parse_mount(raw)
        if path:
            mounts[label] = str(Path(path).expanduser().absolute())
        else:
            mounts.pop(label, None)
    return mounts


def missing_dirs(mounts: dict[str, str]) -> list[str]:
    """``LABEL=PATH`` for every mount whose path is not a directory."""
    return [f"{label}={path}" for label, path in sorted(mounts.items())
            if not Path(path).is_dir()]
