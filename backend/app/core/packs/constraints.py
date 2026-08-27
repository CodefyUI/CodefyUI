"""The constraints file that makes a live pack install ADD-ONLY.

Installing a pack into the interpreter that is currently serving the app is
only safe under one condition: pip must not be allowed to REPLACE anything
already imported. On Windows the loader holds every loaded ``.pyd`` open and
the replacement fails halfway through, leaving a half-uninstalled package on
disk; everywhere else it "succeeds" and the running process keeps the old code
in memory until something segfaults on a mismatched C extension.

So every install runs under a constraints file that pins EVERY distribution
in this interpreter to the version it already has. uv is then free to add the
packages a pack needs, and free to do nothing else: any resolution that would
upgrade, downgrade or replace an installed package is unsatisfiable and fails
before a single byte is written.

Two details carry the weight:

* the pin keeps the LOCAL version tag (``2.11.0+cu128``, not ``2.11.0``).
  Without it the pin says "keep torch 2.11.0" and a resolver is free to honour
  that by installing the CPU wheel over the CUDA one -- a silent, invisible
  downgrade of the whole machine.
* the project itself is never pinned. CodefyUI is installed editable and
  exists on no index; a pin on it makes every resolve unsatisfiable.

The file is written per job into a caller-owned temporary directory, never
cached: it describes this interpreter at this moment, and the moment ends as
soon as an install succeeds.
"""

from __future__ import annotations

import importlib.metadata
import json
import re
from pathlib import Path

#: PEP 503 normalisation: runs of ``-``, ``_`` and ``.`` are one separator,
#: and case never matters. ``huggingface_hub`` and ``huggingface-hub`` are one
#: distribution and must not become two conflicting pins.
_SEPARATORS = re.compile(r"[-_.]+")

#: What may appear on the left of ``==``. Deliberately stricter than PEP 508:
#: a constraints file is options-and-requirements text, so a name carrying a
#: newline would become a second LINE, and uv reads lines starting with ``-``
#: as flags. Nothing on a healthy machine fails this; it is here so that the
#: one that does gets dropped rather than turned into an argument.
#:
#: Matched with ``fullmatch``, and it has to stay that way: ``$`` also matches
#: BEFORE a trailing newline, so ``re.match(r"^...$", "evil\n")`` succeeds --
#: waving through the exact character this guard exists to reject.
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

CONSTRAINTS_FILENAME = "constraints.txt"


def _canonical(name: str) -> str:
    """The PEP 503 canonical form of a distribution name."""
    return _SEPARATORS.sub("-", name).lower()


def _is_editable(dist) -> bool:
    """Was this distribution installed with ``pip install -e``?

    PEP 610 records that in ``direct_url.json`` as ``dir_info.editable``. The
    flat ``{"editable": true}`` spelling is accepted too: it costs one ``or``
    and means a tool that writes the shorter form cannot smuggle the project
    into the constraints file.
    """
    try:
        raw = dist.read_text("direct_url.json")
    except Exception:
        return False
    if not raw:
        return False
    try:
        direct_url = json.loads(raw)
    except ValueError:
        return False
    if not isinstance(direct_url, dict):
        return False
    dir_info = direct_url.get("dir_info")
    nested = dir_info.get("editable") if isinstance(dir_info, dict) else None
    return bool(nested or direct_url.get("editable"))


def installed_distributions() -> dict[str, str]:
    """``{canonical_name: version}`` for every distribution in THIS interpreter.

    Editable installs are excluded BY NAME rather than per record, and this
    venv is the reason: the project appears twice, once as the editable
    ``.dist-info`` that declares itself and once as a ``codefyui_backend.egg-info``
    sitting next to ``pyproject.toml``, which carries no ``direct_url.json``
    at all and so looks like an ordinary installed package. Dropping only the
    record that admits to being editable leaves the other one to be pinned,
    which is exactly the pin that must never exist.

    Where two records share a canonical name and neither is editable, the
    first wins -- that is the one earlier on ``sys.path``, and therefore the
    one this interpreter actually imported.

    A record that cannot be read at all (an unreadable or truncated
    ``.dist-info``) costs its own pin and no other: an install must not be
    blocked by an unrelated broken package.
    """
    versions: dict[str, str] = {}
    editable: set[str] = set()

    for dist in importlib.metadata.distributions():
        try:
            metadata = dist.metadata
            name = (metadata["Name"] if metadata else None) or ""
            version = dist.version or ""
        except Exception:
            continue
        if not name or not version:
            continue

        canonical = _canonical(str(name))
        if _is_editable(dist):
            editable.add(canonical)
        elif canonical not in versions:
            versions[canonical] = str(version)

    for canonical in editable:
        versions.pop(canonical, None)
    return versions


def constraints_text(dists: dict[str, str] | None = None) -> str:
    """Render *dists* (default: this interpreter) as constraints-file text.

    One ``name==version`` line per distribution, sorted by name, LF endings,
    trailing newline. Entries whose name or version could change how uv reads
    the file are dropped rather than escaped -- see ``_SAFE_NAME``.
    """
    if dists is None:
        dists = installed_distributions()

    lines = []
    for name, version in sorted(dists.items()):
        if not _SAFE_NAME.fullmatch(name):
            continue
        if not version or any(character.isspace() for character in version):
            continue
        lines.append(f"{name}=={version}\n")
    return "".join(lines)


def write_constraints_file(
    directory: Path, dists: dict[str, str] | None = None,
) -> Path:
    """Write ``constraints.txt`` into *directory* and return its path.

    The caller owns *directory* (a per-job temporary directory) and its
    lifetime, so this does not create or clean up anything around the file.
    Written as UTF-8 with explicit LF newlines: on Windows the default would
    translate every line ending to CRLF.
    """
    path = Path(directory) / CONSTRAINTS_FILENAME
    path.write_text(constraints_text(dists), encoding="utf-8", newline="\n")
    return path
