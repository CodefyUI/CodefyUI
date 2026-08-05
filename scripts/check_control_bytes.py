#!/usr/bin/env python3
"""CI guard (#209): fail when a tracked source file holds a raw C0 control byte.

Five times so far in this repo's history, a composite map key was built as a
template literal with a delimiter written as a *raw* byte -- on the reasoning
that "no id can contain a NUL, so it's a safe separator" -- instead of the
escaped form (``\\0``, ``\\x01``, ...). One instance reached `main`. The
correct escaped form is four ordinary ASCII characters (backslash, x, 0, 0);
a raw control byte is a single invisible byte sitting in the file's own byte
stream.

This has to be a byte count, not a grep. The failure is silent everywhere
else: it's runtime-identical so nothing type-checks or lints differently,
editors render the byte invisibly, `git diff --stat` misreports the line
count, and -- the sharpest one -- ripgrep and `git grep` both classify a file
containing these bytes as binary and skip it, reporting "no match" rather
than "not searched". See issue #209 for the full incident list.

Usage (from anywhere; resolves the repo root from this file's own path):

    python scripts/check_control_bytes.py

Exit 0: no raw C0 control bytes in any tracked file (besides tab/LF/CR).
Exit 1: at least one was found; every hit is printed as `path:line  0xHH`.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Force UTF-8 on Windows so this prints cleanly regardless of the console
# code page. Mirrors the same guard in scripts/device_smoke.py.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

_REPO_ROOT = Path(__file__).resolve().parents[1]

# C0 controls are U+0000-U+001F. Tab/LF/CR are legitimate in text source
# (CR shows up in CRLF-checked-out files) and are excluded. Everything else
# in the block -- 0x00, and the same construct's other observed victims
# 0x01 and 0x02, plus the rest of the range -- is flagged. DEL (0x7F) is
# outside the formal C0 block and outside every incident on record, so it
# is deliberately not included.
_ALLOWED_C0 = frozenset({0x09, 0x0A, 0x0D})
_FLAGGED_C0 = frozenset(b for b in range(0x20) if b not in _ALLOWED_C0)

# Vendored binary content that is not source and must not be scanned, even
# though it is tracked. Ordinary binary assets (PNG, ico, the .gz dumps)
# already fail the UTF-8-validity check below and need no listing here.
# This one directory is the sole exception found while validating this
# script against the real tree: `backend/data/MNIST/raw/*-labels-idx1-ubyte`
# is the *uncompressed* IDX label dump, one byte per label (0-9) plus a
# small header -- every byte in it happens to be < 0x80, which makes it
# accidentally valid UTF-8 and would otherwise register ~9000 false hits.
_EXCLUDED_PREFIXES = (
    "backend/data/MNIST/",
)


def _tracked_files() -> list[Path]:
    """Every git-tracked file, as absolute paths. Never touches untracked
    files (build output, .venv, node_modules, ...) because it asks git
    directly instead of walking the filesystem."""
    result = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "ls-files", "-z"],
        capture_output=True, check=True,
    )
    rels = [raw.decode("utf-8") for raw in result.stdout.split(b"\x00") if raw]
    return [
        _REPO_ROOT / rel for rel in rels
        if not rel.startswith(_EXCLUDED_PREFIXES)
    ]


def _scan(data: bytes) -> list[tuple[int, int]]:
    """[(line_number, byte_value), ...] for every flagged C0 byte in data.

    Binary files are skipped by attempting a UTF-8 decode first: arbitrary
    binary data (images, fonts, the vendored MNIST dumps) essentially never
    decodes as valid UTF-8, while every hazard instance on record is a raw
    control byte sitting inside an otherwise-normal UTF-8 source file (the
    byte itself is legal UTF-8 on its own -- it is wrong *semantically*,
    not structurally). This needs no hand-maintained extension allowlist or
    denylist, so it can't silently miss a new source language later the way
    one would.
    """
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return []
    hits = []
    line = 1
    for byte in data:
        if byte == 0x0A:
            line += 1
        elif byte in _FLAGGED_C0:
            hits.append((line, byte))
    return hits


def main() -> int:
    files = _tracked_files()
    hits_by_file: list[tuple[Path, list[tuple[int, int]]]] = []
    for path in files:
        if not path.is_file():
            continue
        found = _scan(path.read_bytes())
        if found:
            hits_by_file.append((path, found))

    if not hits_by_file:
        print(f"OK: no raw C0 control bytes in {len(files)} tracked files")
        return 0

    print("Raw C0 control byte(s) found. Use the escaped form (e.g. \\x00, "
          "\\x01) instead of a literal byte in the source:\n")
    for path, found in hits_by_file:
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for line, byte in found:
            print(f"  {rel}:{line}  0x{byte:02x}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
