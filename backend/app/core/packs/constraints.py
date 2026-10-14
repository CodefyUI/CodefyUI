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
from dataclasses import dataclass
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement

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


#: Pinned even when an install may replace packages: the PyTorch build is
#: chosen per machine (``cdui install --gpu``), and an index only has to
#: carry the same version under another build for a resolver to swap it.
_MACHINE_CHOSEN = frozenset({"torch", "torchvision", "torchaudio"})


def _machine_chosen(name: str, version: str) -> bool:
    """Was this package picked for this machine's hardware?

    The PyTorch family, and anything carrying a local version tag
    (``+cu128``, ``+rocm6.4``). Replacing one is the silent CPU-wheel
    downgrade the module docstring describes, so it stays pinned even when
    an install may replace packages.
    """
    return name in _MACHINE_CHOSEN or "+" in version


@dataclass(frozen=True)
class Replacement:
    """How an install may replace packages when no server is running.

    ``names`` are handed to uv beside the plugin's own specs, so it resolves
    them as one set: every installed package's requirements are honoured,
    and an installed version is kept wherever it still fits. Without them uv
    resolves the plugin's specs alone and does not check the packages it
    leaves in place -- 3.0.1 upgraded ``huggingface-hub`` to 2.x under a
    ``datasets`` that requires ``<2.0``.

    ``constraints`` holds what must not move: the packages picked for this
    machine's hardware, every package installed from a URL or a path
    (nothing on the index can stand in for it), and the requirements those
    and the editable project declare -- they are not re-resolved, so their
    requirements would otherwise go unchecked too.
    """

    names: tuple[str, ...]
    constraints: str


def _declared(dist) -> list[str]:
    """*dist*'s requirements as constraint lines, extras and URLs left out.

    An extra's requirements apply only if the extra was asked for, which
    the installed record does not say; a URL requirement cannot be written
    as a constraint. Neither is needed to keep the rest consistent.
    """
    lines = []
    for raw in dist.requires or ():
        try:
            requirement = Requirement(raw)
        except InvalidRequirement:
            continue
        name = _canonical(requirement.name)
        marker = str(requirement.marker) if requirement.marker else ""
        if (requirement.url or not requirement.specifier
                or "extra" in marker or not _SAFE_NAME.fullmatch(name)):
            continue
        line = f"{name}{requirement.specifier}"
        if marker:
            line += f"; {marker}"
        if "\n" not in line and "\r" not in line:
            lines.append(line)
    return lines


def replacement_plan() -> Replacement:
    """The :class:`Replacement` for this interpreter as it is now."""
    names: dict[str, None] = {}
    pinned: dict[str, str] = {}
    declared: list[str] = []
    editable: set[str] = set()

    for dist in importlib.metadata.distributions():
        try:
            metadata = dist.metadata
            name = _canonical(str((metadata["Name"] if metadata else None) or ""))
            version = str(dist.version or "")
            is_editable = _is_editable(dist)
            from_url = not is_editable and bool(dist.read_text("direct_url.json"))
        except Exception:
            continue
        if not name or not version or not _SAFE_NAME.fullmatch(name):
            continue
        if is_editable:
            editable.add(name)
            declared.extend(_declared(dist))
        elif from_url or _machine_chosen(name, version):
            pinned.setdefault(name, version)
            declared.extend(_declared(dist))
        else:
            names.setdefault(name, None)

    for name in editable:
        names.pop(name, None)
        pinned.pop(name, None)
    for name in pinned:
        names.pop(name, None)
    text = constraints_text(pinned) + "".join(
        f"{line}\n" for line in dict.fromkeys(declared))
    return Replacement(names=tuple(names), constraints=text)


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
    *, text: str | None = None,
) -> Path:
    """Write ``constraints.txt`` into *directory* and return its path.

    The caller owns *directory* (a per-job temporary directory) and its
    lifetime, so this does not create or clean up anything around the file.
    Written as UTF-8 with explicit LF newlines: on Windows the default would
    translate every line ending to CRLF. *text*, when given, is written as
    it is instead of pins rendered from *dists* (a :class:`Replacement`'s).
    """
    path = Path(directory) / CONSTRAINTS_FILENAME
    if text is None:
        text = constraints_text(dists)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path
