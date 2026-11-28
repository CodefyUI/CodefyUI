"""Project files an in-editor agent may read and write (round 6, H5).

The routes are ``GET``/``PUT /api/agent/files`` (``api.routes_agent_files``);
every rule is here, as plain functions, so it is written once and tested
without HTTP. The design, threat model and review are in the program's
``reports/round6-files-design.md`` and ``reviews/round6-files-design.md``.

Who calls this: a plugin's JavaScript on behalf of an LLM. The plugin shows
the user an approval card for every write; the host's job is that a write
which got past the card still cannot reach anything outside the allowed set,
and still cannot run code without the host's own trust prompt.

The rules, in short:

- A path is project-relative, ``/``-separated, with no ``..``, ``.``, empty
  or absolute part, no ``:`` or ``\\``, and each part a name a file can have
  on every OS (``data_paths.check_file_name``). No existing part may be a
  symbolic link or another reparse point.
- Read: everything in the project except ``.git/``, dotenv files, any other
  dot-path outside a short list (``.ssh/``, ``.netrc`` ...) and names shaped
  like secrets (``*.pem``, ``id_rsa*``, ``credentials*`` ...). What is read
  goes to whatever LLM provider the user picked.
- Write: text files under a short allowlist of folders, each with its own
  extensions. The one place an agent can put Python is ``mods/nodes/``, and
  a write there marks the project so its mods ask for the user's trust before
  they load (``mods.mark_agent_write``). Nothing is imported or run here.
- A write is compare-and-swap on the file's sha256 and atomic (a temporary
  file in the same folder, then a rename, which never writes through a hard
  link).

Known gap: on Windows the folder walk cannot hold directory handles
(``dir_fd``), so a folder swapped for a link between the check and the
rename by another process of the same user is not caught. On POSIX it is.
"""

from __future__ import annotations

import codecs
import fnmatch
import hashlib
import json
import os
import secrets
import stat
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .data_paths import UnstorableName, check_file_name
from .fs_paths import canonical_path, is_within
from .git.paths import is_env_secret_path

#: Longest path string accepted.
MAX_PATH_CHARS = 1024
#: Most of a text file a read returns.
READ_CONTENT_BYTES = 64 * 1024
#: Bytes inspected for a NUL to call a file binary.
BINARY_SNIFF_BYTES = 8 * 1024
#: Files larger than this are not hashed on request (a checkpoint).
HASH_LIMIT_BYTES = 16 * 1024 * 1024
#: Most entries one directory listing returns.
LIST_LIMIT = 500
#: Largest file a write may store, after encoding.
WRITE_LIMIT_BYTES = 256 * 1024

#: Dot-names a read may still open: the repository's own text files and
#: CI folder. Every other path part starting with ``.`` is refused.
READABLE_DOT_NAMES = frozenset({".gitignore", ".gitattributes", ".env.example", ".github"})

#: File names (compared lower-cased) that look like credentials. Listed in a
#: directory -- a name is not a secret -- but never read.
SECRET_NAME_PATTERNS = (
    "*.pem", "*.key", "*.p12", "*.pfx", "*.keystore",
    "id_rsa*", "id_ed25519*", "credentials*", "secret*", "*token*.json",
)

#: Extensions a write may use under each folder (compared lower-cased).
DATA_EXTENSIONS = frozenset({".csv", ".tsv", ".json", ".jsonl", ".txt", ".md", ".yaml", ".yml"})
DOC_EXTENSIONS = frozenset({".md", ".txt", ".json", ".yaml", ".yml", ".csv"})

#: The file an agent's standing instructions are read from
#: (``routes_agent_context``); a write to it is flagged.
AGENT_INSTRUCTIONS_FILE = "agents.md"

#: What the allowlist says, for the refusal message.
WRITABLE_SUMMARY = (
    "mods/nodes/<name>.py, mods/presets/<name>.json, "
    "docs/** and notes/** (" + ", ".join(sorted(DOC_EXTENSIONS)) + "), "
    "data/** and assets/data/** (" + ", ".join(sorted(DATA_EXTENSIONS))
    + "; not data/samples/), and <name>.md at the project root"
)

_FILE_ATTRIBUTE_REPARSE_POINT = 0x400

_write_lock = threading.Lock()


class AgentFileError(Exception):
    """A refusal, answered as ``{"detail": {"code", "message", ...}}``."""

    def __init__(self, status: int, code: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.extra = extra

    def detail(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, **self.extra}


def _fold(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


# ── path syntax ──────────────────────────────────────────────────────────

def parse_path(raw: Any, *, allow_root: bool = False) -> list[str]:
    """The parts of a project-relative path, or :class:`AgentFileError`
    (400 ``invalid_path``). Nothing on disk is looked at."""
    def bad(why: str) -> AgentFileError:
        return AgentFileError(400, "invalid_path", f"Invalid path {raw!r}: {why}.")

    if not isinstance(raw, str):
        raise bad("not a string")
    if raw == "" and allow_root:
        return []
    if not raw:
        raise bad("empty")
    if len(raw) > MAX_PATH_CHARS:
        raise bad(f"longer than {MAX_PATH_CHARS} characters")
    if "\x00" in raw:
        raise bad("it contains a NUL character")
    if "\\" in raw:
        raise bad("use / to separate folders, not \\")
    if raw.startswith("/"):
        raise bad("it must be relative to the project, not absolute")
    parts = raw.split("/")
    for part in parts:
        if part in ("", ".", ".."):
            raise bad("empty, '.' and '..' parts are not allowed")
        if ":" in part:
            raise bad("':' is not allowed (a drive or a stream on Windows)")
        try:
            check_file_name(part)
        except UnstorableName as exc:
            raise bad(str(exc).rstrip(".")) from None
    return parts


# ── locating it on disk ──────────────────────────────────────────────────

def _is_link(info: os.stat_result) -> bool:
    if stat.S_ISLNK(info.st_mode):
        return True
    return bool(getattr(info, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT)


@dataclass
class Located:
    project: Path
    #: The parts as the disk spells them (the given spelling past the first
    #: part that does not exist).
    parts: list[str]
    path: Path
    #: ``os.lstat`` of the target, None when it does not exist.
    info: os.stat_result | None = None
    created_dirs: list[str] = field(default_factory=list)

    @property
    def rel(self) -> str:
        return "/".join(self.parts)


def locate(project: Path, parts: list[str], *, refuse_code: str) -> Located:
    """Walk *parts* under *project* without following anything.

    Each existing part is re-spelled as its folder lists it (a
    case-insensitive disk) and must not be a link or reparse point; the
    result must be inside the project. *refuse_code* is the code a link is
    refused with (``path_not_readable`` / ``path_not_writable``).
    """
    root = canonical_path(project)
    current = root
    spelled: list[str] = []
    info: os.stat_result | None = None
    for index, part in enumerate(parts):
        candidate = current / part
        if not os.path.lexists(candidate):
            spelled.extend(parts[index:])
            current = current.joinpath(*parts[index:])
            info = None
            break
        try:
            names = os.listdir(current)
        except OSError:
            names = [part]
        if part not in names:
            matches = [n for n in names if _fold(n) == _fold(part)]
            if len(matches) == 1:
                part = matches[0]
                candidate = current / part
        info = os.lstat(candidate)
        if _is_link(info):
            raise AgentFileError(403, refuse_code,
                                 f"'{'/'.join([*spelled, part])}' is a symbolic link; "
                                 "links are never followed.")
        spelled.append(part)
        current = candidate
    if not is_within(current, root):
        raise AgentFileError(403, refuse_code, "The path leaves the project.")
    return Located(project=root, parts=spelled, path=current, info=info)


# ── read ─────────────────────────────────────────────────────────────────

def read_refusal(parts: list[str]) -> str | None:
    """Why *parts* may not be read, or None."""
    for part in parts:
        folded = _fold(part)
        if folded == ".git":
            return "'.git/' is never read."
        if part.startswith(".") and folded not in READABLE_DOT_NAMES:
            if is_env_secret_path(part):
                return f"'{part}' is a dotenv file, which holds secrets; it is never read."
            return (f"'{part}' is a hidden path, never read (it may hold credentials, "
                    "such as .ssh/ or .netrc).")
    if parts:
        name = parts[-1]
        if is_env_secret_path(name):
            return f"'{name}' is a dotenv file, which holds secrets; it is never read."
        lowered = name.lower()
        for pattern in SECRET_NAME_PATTERNS:
            if fnmatch.fnmatchcase(lowered, pattern):
                return (f"'{name}' looks like a credential ({pattern}); it is never read, "
                        "because what is read is sent to the model provider.")
    return None


def _listed(name: str) -> bool:
    """Whether a directory listing shows *name*: everything except ``.git``
    and dotenv files (other secret-shaped names are shown, not readable)."""
    return _fold(name) != ".git" and not is_env_secret_path(name)


def list_directory(located: Located) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    omitted = 0
    truncated = False
    prefix = located.rel + "/" if located.parts else ""
    with os.scandir(located.path) as it:
        items = sorted(it, key=lambda e: e.name)
    for entry in items:
        if not _listed(entry.name):
            continue
        try:
            info = entry.stat(follow_symlinks=False)
        except OSError:
            omitted += 1
            continue
        if _is_link(info):
            omitted += 1
            continue
        if stat.S_ISDIR(info.st_mode):
            kind, size = "dir", None
        elif stat.S_ISREG(info.st_mode):
            kind, size = "file", info.st_size
        else:
            omitted += 1
            continue
        if len(entries) >= LIST_LIMIT:
            truncated = True
            break
        entries.append({"path": prefix + entry.name, "kind": kind, "size": size})
    return {"path": located.rel, "entries": entries, "truncated": truncated, "omitted": omitted}


def read_file(located: Located) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(located.path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise AgentFileError(403, "path_not_readable", "Not a regular file.")
        size = info.st_size
        answer: dict[str, Any] = {"path": located.rel, "size": size, "binary": False,
                                  "content": None, "sha256": None, "truncated": False}
        if size > HASH_LIMIT_BYTES:
            # Not hashed on request, and too large to be text worth reading.
            answer["truncated"] = True
            return answer
        digest = hashlib.sha256()
        head = bytearray()
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            if len(head) <= READ_CONTENT_BYTES:
                head.extend(chunk[: READ_CONTENT_BYTES + 1 - len(head)])
        answer["sha256"] = digest.hexdigest()
    finally:
        os.close(fd)
    if b"\x00" in head[:BINARY_SNIFF_BYTES]:
        answer["binary"] = True
        return answer
    truncated = len(head) > READ_CONTENT_BYTES
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        # final=not truncated: a character cut at the 64 KB boundary is held
        # back rather than called invalid.
        text = decoder.decode(bytes(head[:READ_CONTENT_BYTES]), final=not truncated)
    except UnicodeDecodeError:
        answer["binary"] = True
        return answer
    answer["content"] = text
    answer["truncated"] = truncated
    return answer


# ── write ────────────────────────────────────────────────────────────────

@dataclass
class WriteRule:
    kind: str                  # "mods_node" | "mods_preset" | "docs" | "data" | "root_md"
    agent_instructions: bool = False

    @property
    def mods(self) -> bool:
        return self.kind.startswith("mods_")


def _not_writable(message: str) -> AgentFileError:
    return AgentFileError(403, "path_not_writable",
                          f"{message} Only these can be written: {WRITABLE_SUMMARY}.")


#: Category folders whose spelling must be exact (lower case), by position.
_CATEGORY_NAMES = {0: {"mods", "docs", "notes", "data", "assets"}}


def write_rule(parts: list[str]) -> WriteRule:
    """Which allowlist row *parts* (disk spelling) falls under, or
    :class:`AgentFileError` 403 ``path_not_writable``."""
    def exact(index: int, expected: set[str]) -> None:
        if len(parts) > index and _fold(parts[index]) in expected and parts[index] not in expected:
            raise _not_writable(
                f"'{parts[index]}' must be spelled '{_fold(parts[index])}' exactly.")

    exact(0, {"mods", "docs", "notes", "data", "assets"})
    if parts and parts[0] == "mods":
        exact(1, {"nodes", "presets"})
    if parts and parts[0] == "assets":
        exact(1, {"data"})
    name = parts[-1]
    suffix = Path(name).suffix
    lowered = suffix.lower()

    if len(parts) == 1:
        if lowered == ".md":
            return WriteRule("root_md", agent_instructions=_fold(name) == AGENT_INSTRUCTIONS_FILE)
        raise _not_writable(f"'{name}' is at the project root and is not a .md file.")

    head = parts[0]
    if head == "mods":
        if len(parts) == 3 and parts[1] == "nodes" and suffix == ".py" and name != "__init__.py":
            return WriteRule("mods_node")
        if len(parts) == 3 and parts[1] == "presets" and suffix == ".json":
            return WriteRule("mods_preset")
        raise _not_writable(
            "Under mods/ only a .py directly in mods/nodes/ (not __init__.py) or a .json "
            "directly in mods/presets/ is loaded, so only those can be written.")
    if head in ("docs", "notes"):
        if lowered in DOC_EXTENSIONS:
            return WriteRule("docs")
        raise _not_writable(
            f"'{name}' has an extension {head}/ does not take "
            f"(allowed: {', '.join(sorted(DOC_EXTENSIONS))}).")
    data_tail: list[str] | None = None
    if head == "data":
        data_tail = parts[1:]
    elif head == "assets" and len(parts) >= 3 and parts[1] == "data":
        data_tail = parts[2:]
    if data_tail is not None:
        if len(data_tail) > 1 and _fold(data_tail[0]) == "samples":
            raise _not_writable(
                "data/samples/ is kept for the names of the bundled samples: a file there "
                "would replace a sample in every example that reads it.")
        if lowered in DATA_EXTENSIONS:
            return WriteRule("data")
        raise _not_writable(
            f"'{name}' has an extension a data folder does not take "
            f"(allowed: {', '.join(sorted(DATA_EXTENSIONS))}).")
    raise _not_writable(f"'{'/'.join(parts)}' is not in a writable folder.")


def prepare_content(content: Any, rule: WriteRule) -> bytes:
    """The bytes to store: LF line endings, UTF-8, checked."""
    if not isinstance(content, str):
        raise AgentFileError(400, "invalid_content", "content must be a string.")
    text = content.replace("\r\n", "\n").replace("\r", "\n")
    for index, char in enumerate(text):
        code = ord(char)
        if (code < 32 and char not in "\t\n") or code == 127:
            line = text.count("\n", 0, index) + 1
            raise AgentFileError(400, "invalid_content",
                                 f"content holds the control character U+{code:04X} "
                                 f"on line {line}; only text can be written.")
    try:
        data = text.encode("utf-8")
    except UnicodeEncodeError:
        raise AgentFileError(400, "invalid_content",
                             "content is not valid Unicode (a lone surrogate).") from None
    if len(data) > WRITE_LIMIT_BYTES:
        raise AgentFileError(413, "file_too_large",
                             f"The file would be {len(data)} bytes; the limit is "
                             f"{WRITE_LIMIT_BYTES} bytes.", limit=WRITE_LIMIT_BYTES)
    if rule.kind == "mods_preset":
        try:
            json.loads(text)
        except ValueError as exc:
            raise AgentFileError(400, "invalid_content",
                                 f"A preset must be valid JSON: {exc}.") from None
    return data


_HAS_DIR_FD = (os.open in os.supports_dir_fd and os.mkdir in os.supports_dir_fd
               and os.rename in os.supports_dir_fd and os.stat in os.supports_dir_fd
               and os.unlink in os.supports_dir_fd)


def _current_sha(fd_or_path: Any, name: str | None) -> str | None:
    """The sha256 of the existing target, or None when there is none."""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = (os.open(name, flags, dir_fd=fd_or_path) if name is not None
              else os.open(fd_or_path, flags))
    except FileNotFoundError:
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise AgentFileError(403, "path_not_writable", "The target is not a regular file.")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(fd)


def _check_cas(current: str | None, expected: str | None) -> None:
    if expected is None and current is not None:
        raise AgentFileError(409, "file_changed",
                             "The file already exists; read it and pass its sha256 as "
                             "expected_sha256 to replace it.", current_sha256=current)
    if expected is not None and current != expected:
        message = ("The file no longer exists." if current is None
                   else "The file changed since it was read; read it again.")
        raise AgentFileError(409, "file_changed", message, current_sha256=current)


def _temp_name(name: str) -> str:
    return f".{name}.tmp-{secrets.token_hex(6)}"


def _write_posix(project: Path, parts: list[str], data: bytes, expected: str | None,
                 before_replace: Any) -> bool:
    """Walk the folders by handle (``O_NOFOLLOW``), so a folder swapped for a
    link mid-way fails instead of being followed. Returns ``created``."""
    dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(project, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        for part in parts[:-1]:
            try:
                os.mkdir(part, 0o777, dir_fd=fd)
            except FileExistsError:
                pass
            try:
                child = os.open(part, dir_flags, dir_fd=fd)
            except OSError:
                raise AgentFileError(403, "path_not_writable",
                                     f"'{part}' is not a folder (or is a link).") from None
            os.close(fd)
            fd = child
        name = parts[-1]
        current = _current_sha(fd, name)
        _check_cas(current, expected)
        try:
            mode = stat.S_IMODE(os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode)
        except FileNotFoundError:
            mode = 0o666
        temp = _temp_name(name)
        tfd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                      mode, dir_fd=fd)
        try:
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(tfd, view):]
                os.fsync(tfd)
            finally:
                os.close(tfd)
            before_replace()
            os.rename(temp, name, src_dir_fd=fd, dst_dir_fd=fd)
        except BaseException:
            try:
                os.unlink(temp, dir_fd=fd)
            except OSError:
                pass
            raise
        return current is None
    finally:
        os.close(fd)


def _write_portable(project: Path, parts: list[str], data: bytes, expected: str | None,
                    before_replace: Any) -> bool:
    """Windows: no directory handles, so every folder is re-checked with
    ``lstat`` immediately before the rename (see the module docstring)."""
    folder = project
    for part in parts[:-1]:
        folder = folder / part
        if not os.path.lexists(folder):
            folder.mkdir()
        info = os.lstat(folder)
        if _is_link(info) or not stat.S_ISDIR(info.st_mode):
            raise AgentFileError(403, "path_not_writable", f"'{part}' is not a folder (or is a link).")
    target = folder / parts[-1]
    if os.path.lexists(target) and _is_link(os.lstat(target)):
        raise AgentFileError(403, "path_not_writable", "The target is a link.")
    current = _current_sha(target, None)
    _check_cas(current, expected)
    temp = folder / _temp_name(parts[-1])
    try:
        with open(temp, "xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        before_replace()
        locate(project, parts, refuse_code="path_not_writable")
        os.replace(temp, target)
    except BaseException:
        try:
            temp.unlink()
        except OSError:
            pass
        raise
    return current is None


def write_file(project: Path, raw_path: Any, content: Any, expected_sha256: Any) -> dict[str, Any]:
    """Store *content* at *raw_path* in *project*. Returns the PUT answer."""
    from . import mods

    if expected_sha256 is not None and not (
            isinstance(expected_sha256, str) and len(expected_sha256) == 64
            and all(c in "0123456789abcdef" for c in expected_sha256)):
        raise AgentFileError(400, "invalid_content",
                             "expected_sha256 must be null or 64 lower-case hex digits.")
    parts = parse_path(raw_path)
    located = locate(project, parts, refuse_code="path_not_writable")
    rule = write_rule(located.parts)
    if located.info is not None and not stat.S_ISREG(located.info.st_mode):
        raise AgentFileError(403, "path_not_writable", "The target is not a regular file.")
    data = prepare_content(content, rule)
    sha = hashlib.sha256(data).hexdigest()

    def before_replace() -> None:
        # The marker goes in before the file does: a crash between the two
        # leaves the stricter state.
        if rule.mods:
            mods.mark_agent_write(located.project, "/".join(located.parts[1:]), sha)

    with _write_lock:
        writer = _write_posix if _HAS_DIR_FD else _write_portable
        created = writer(located.project, located.parts, data, expected_sha256, before_replace)
    return {"path": located.rel, "sha256": sha, "created": created,
            "mods_consent_required": rule.mods,
            "agent_instructions": rule.agent_instructions}


def read_path(project: Path, raw_path: Any) -> dict[str, Any]:
    """The GET answer for *raw_path*: a listing or a file."""
    parts = parse_path("" if raw_path is None else raw_path, allow_root=True)
    located = locate(project, parts, refuse_code="path_not_readable")
    refusal = read_refusal(located.parts)
    if refusal:
        raise AgentFileError(403, "path_not_readable", refusal)
    if located.parts and located.info is None:
        raise AgentFileError(404, "path_not_found", f"Nothing at '{located.rel}'.")
    if not located.parts or stat.S_ISDIR(located.info.st_mode):
        return list_directory(located)
    if not stat.S_ISREG(located.info.st_mode):
        raise AgentFileError(403, "path_not_readable", "Not a regular file or folder.")
    return read_file(located)


# ── data files (GET /api/agent/data-files) ───────────────────────────────

#: Most entries per group.
DATA_FILES_LIMIT = 200
#: CSVs at least this large report their header only (no row count).
CSV_COUNT_LIMIT = 2 * 1024 * 1024
_TEXT_SUFFIXES = {".txt", ".md", ".jsonl", ".json", ".yaml", ".yml", ".tsv"}


def _csv_facts(path: Path) -> dict[str, Any]:
    import csv
    import io

    facts: dict[str, Any] = {}
    try:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            first = handle.readline(64 * 1024)
            header = next(csv.reader(io.StringIO(first.decode("utf-8-sig", errors="replace"))), [])
            facts["columns"] = [c.strip() for c in header]
            if size < CSV_COUNT_LIMIT:
                facts["rows"] = sum(1 for line in handle if line.strip())
    except (OSError, StopIteration):
        pass
    return facts


def _entry(path: Path, spelled: str) -> dict[str, Any]:
    if path.is_dir():
        return {"path": spelled, "size": None, "kind": "dir"}
    suffix = path.suffix.lower()
    entry: dict[str, Any] = {"path": spelled, "size": path.stat().st_size,
                             "kind": "csv" if suffix == ".csv"
                             else "text" if suffix in _TEXT_SUFFIXES else "other"}
    if suffix == ".csv":
        entry.update(_csv_facts(path))
    return entry


def _walk_files(root: Path):
    """Regular files under *root*, sorted, without following links or
    entering dot-folders."""
    if not root.is_dir() or root.is_symlink():
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith(".") and not (base / d).is_symlink())
        for name in sorted(filenames):
            path = base / name
            if name.startswith(".") or path.is_symlink() or not path.is_file():
                continue
            yield path


def data_files(project: Path | None) -> dict[str, Any]:
    """Every data file a node parameter can name, spelled the way it takes
    them: the bundled samples, the project's ``data/`` and ``assets/data/``
    (project-relative), and the uploads (a bare name, as the dropdown)."""
    from ..config import settings
    from .data_paths import bundled_sample_names

    groups: dict[str, list[dict[str, Any]]] = {"bundled": [], "project": [], "uploads": []}
    truncated = {name: False for name in groups}

    def add(group: str, path: Path, spelled: str) -> bool:
        if len(groups[group]) >= DATA_FILES_LIMIT:
            truncated[group] = True
            return False
        try:
            groups[group].append(_entry(path, spelled))
        except OSError:
            pass
        return True

    backend = Path(__file__).resolve().parents[2]
    for name in bundled_sample_names():
        if not add("bundled", backend / name, name):
            break
    if project is not None:
        root = canonical_path(project)
        for folder in ("data", "assets/data"):
            stop = False
            for path in _walk_files(root / folder):
                rel = path.relative_to(root).as_posix()
                if read_refusal(rel.split("/")):
                    continue
                if not add("project", path, rel):
                    stop = True
                    break
            if stop:
                break
    uploads = Path(settings.DATA_FILES_DIR)
    if uploads.is_dir():
        for path in sorted(uploads.iterdir()):
            if path.name.startswith(".") or path.is_symlink() or not path.is_file():
                continue
            if not add("uploads", path, path.name):
                break
    return {**groups, "truncated": truncated}
