"""Which machine this process runs on, as an opaque digest (#773).

A worker agent sends :func:`machine_identity` in its ``hello``; the control
plane compares it with its own to decide whether the worker runs on the
same machine, and therefore on the same physical devices. A peer address
cannot answer that: an ``ssh -R`` tunnel or a local proxy makes a remote
agent look like loopback.

The source is the operating system's own machine id:

- Linux: ``/etc/machine-id`` (or ``/var/lib/dbus/machine-id``);
- macOS: ``IOPlatformUUID`` from ``ioreg -rd1 -c IOPlatformExpertDevice``;
- Windows: ``MachineGuid`` under
  ``HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Cryptography``.

The raw id is never sent: ``machine-id(5)`` asks that it be treated as
confidential, so what travels is a SHA-256 of it under a CodefyUI-specific
prefix, which is still equal on both ends of one machine. None when no
source answers; None never matches anything, so an unknown machine is
always treated as another machine.

Torch-free (the agent imports it).
"""

from __future__ import annotations

import functools
import hashlib
import logging
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_LINUX_PATHS = (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id"))
_IOREG_UUID = re.compile(r'"IOPlatformUUID"\s*=\s*"([^"]+)"')
_DIGEST_PREFIX = b"codefyui-machine-identity:"


def _raw_linux() -> str | None:
    for path in _LINUX_PATHS:
        try:
            text = path.read_text(encoding="ascii").strip()
        except (OSError, UnicodeDecodeError):
            continue
        if text:
            return text
    return None


def _raw_macos() -> str | None:
    try:
        output = subprocess.run(
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            capture_output=True, text=True, timeout=5, check=False).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("ioreg failed: %s", exc)
        return None
    match = _IOREG_UUID.search(output or "")
    return match.group(1) if match else None


def _raw_windows() -> str | None:
    try:
        import winreg  # type: ignore[import-not-found]

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SOFTWARE\Microsoft\Cryptography",
                            0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
            value, _kind = winreg.QueryValueEx(key, "MachineGuid")
        return str(value)
    except Exception as exc:  # noqa: BLE001 -- no key, no access
        logger.debug("MachineGuid lookup failed: %s", exc)
        return None


def raw_machine_id(platform: str | None = None) -> str | None:
    """The OS machine id for *platform* (``sys.platform``), or None."""
    platform = sys.platform if platform is None else platform
    if platform == "darwin":
        raw = _raw_macos()
    elif platform.startswith("win"):
        raw = _raw_windows()
    else:
        raw = _raw_linux()
    raw = (raw or "").strip().lower()
    return raw or None


def digest(raw: str | None) -> str | None:
    """The identity sent over the wire for one raw machine id."""
    if not raw:
        return None
    return hashlib.sha256(_DIGEST_PREFIX + raw.encode()).hexdigest()


@functools.lru_cache(maxsize=1)
def machine_identity() -> str | None:
    """This machine's identity digest, or None when it cannot be read."""
    identity = digest(raw_machine_id())
    if identity is None:
        logger.info("no machine id found; workers will not be treated as "
                    "running on this machine")
    return identity


def same_machine(a: str | None, b: str | None) -> bool:
    """Equal and known: None never matches, not even None."""
    return a is not None and a == b
