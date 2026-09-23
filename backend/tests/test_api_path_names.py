"""A client-supplied name that cannot be a file is a 400, never a 500 (#483).

Eight helpers across ``app/api`` each spelled out "resolve this name inside
that directory" for their own routes, and the ones in six route modules let
``Path.resolve`` raise straight through. An embedded NUL makes it raise
``ValueError``, and the custom-node toggle also raised ``TypeError`` for a
``filename`` that is not a string at all, so every request in ``_REFUSED``
below answered 500 with a traceback in the server's log. Five of them are
open GETs that take no session token.

The rule is now written once, as ``data_paths.resolve_under``, and each route
keeps its own answer for a name it refuses. The unit tests pin the rule; the
route table pins that every route really goes through it.

The second half of this file is #520: which names a stored file may have. On
a Windows server ``what?.csv`` answered 500, ``a:b.csv`` and ``x\\b.csv`` were
stored as ``b.csv`` over whatever had that name, and ``run 12:30.csv``
vanished into an NTFS stream. ``data_paths.check_file_name`` refuses them at
upload with a 400 that names the character, on every OS, so those tests fail
on Linux CI too without it. A lookup (download, delete, toggle) refuses only
what no file on the server itself can have (``data_paths.check_lookup_name``):
the Windows characters on a Windows server only, which the lookup tests fake
on any host, so a file a Linux server already holds under such a name stays
reachable. The media and examples routes above apply neither, and keep
"Invalid filename" / "Invalid path".

Two traps for whoever extends this file
---------------------------------------
- The upload bodies are built by hand. httpx ``files=`` percent-encodes a NUL
  in a multipart filename, and so does a browser: the upload then succeeds
  and stores a file literally named ``a%00b.csv``, so a test written that way
  passes on the code it exists to catch.
  ``test_the_hand_built_upload_body_is_one_the_server_accepts`` is the control
  that proves these bodies are read at all.
- A NUL is written as the escape ``\\x00``, or ``%00`` in a URL, never as the
  byte itself: ``byte-scan.yml`` fails the build on a raw C0 byte in a source
  file.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import routes_custom_nodes
from app.config import settings
from app.core import data_paths
from app.core.auth import TOKEN_HEADER, session_token
from app.main import app

#: Each setting a route below reads its directory from, and the key the
#: ``dirs`` fixture hands that directory back under.
_DIRECTORIES = {
    "DATA_FILES_DIR": "files",
    "IMAGES_DIR": "images",
    "MODELS_DIR": "models",
    "CUSTOM_NODES_DIR": "custom_nodes",
    "MEDIA_DIR": "media",
    "EXAMPLES_DIR": "examples",
}

_BOUNDARY = "cdui-483-boundary"

_NUL = "\x00"

#: U+0001, the control character #520 was reported with.
_SOH = "\x01"

#: What the routes that apply the #520 name rule answer for a NUL: it is a
#: control character, refused by name before anything is resolved.
_NUL_DETAIL = ("File names cannot contain the control character U+0000. "
               "Rename the file and try again.")


@pytest.fixture
def dirs(tmp_path, monkeypatch) -> dict[str, Path]:
    """Every directory these routes read, as an empty temp dir.

    Each route reads its setting per request, so patching the attribute is
    enough, and a regression that writes something writes it here.
    """
    made: dict[str, Path] = {}
    for setting, key in _DIRECTORIES.items():
        directory = tmp_path / key
        directory.mkdir()
        monkeypatch.setattr(settings, setting, directory)
        made[key] = directory
    return made


@pytest.fixture
async def client():
    """The app, with ``raise_app_exceptions=False``.

    The default re-raises an unhandled exception inside the test, so a
    regression would read as a ``ValueError`` traceback. This way it reads as
    what a client gets: ``500 == 400`` failing.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url=f"http://127.0.0.1:{settings.PORT}",
        headers={TOKEN_HEADER: session_token()},
    ) as http:
        yield http


def _raw_upload(filename: str, payload: bytes = b"x") -> dict[str, Any]:
    """Request arguments for a multipart upload whose filename reaches the
    server exactly as written. Not httpx ``files=``: see the module
    docstring."""
    head = (f"--{_BOUNDARY}\r\n"
            'Content-Disposition: form-data; name="file"; '
            f'filename="{filename}"\r\n'
            "Content-Type: application/octet-stream\r\n"
            "\r\n")
    tail = f"\r\n--{_BOUNDARY}--\r\n"
    return {
        "content": head.encode("utf-8") + payload + tail.encode("ascii"),
        "headers": {
            "Content-Type": f"multipart/form-data; boundary={_BOUNDARY}"},
    }


# ── the rule: data_paths.resolve_under ───────────────────────────────────


@pytest.mark.parametrize("direct_child", [False, True])
def test_a_nul_is_none(tmp_path, direct_child):
    assert data_paths.resolve_under(
        tmp_path, f"a{_NUL}b.csv", direct_child=direct_child) is None


def test_a_nul_is_none_even_where_resolve_accepts_it(tmp_path, monkeypatch):
    """On Windows from Python 3.13, ``Path.resolve`` does not raise on a NUL.

    ``ntpath.realpath`` hands the path back unchanged (gh-106242) and the NUL
    fails only at the write: every upload route answered 500 there even with
    the ``ValueError`` caught. Faked with a ``resolve`` that only makes the
    path absolute, which is all it does on that platform.
    """
    monkeypatch.setattr(type(tmp_path), "resolve",
                        lambda self, strict=False: self.absolute())
    assert data_paths.resolve_under(tmp_path, f"a{_NUL}b.csv") is None


def test_a_name_the_os_will_not_look_up_is_none(tmp_path, monkeypatch):
    """The ``OSError`` half of the catch: ``resolve`` refusing the name."""
    real_resolve = type(tmp_path).resolve

    def _refusing(self, strict=False):
        if self.name == "refused.csv":
            raise OSError(22, "Invalid argument")
        return real_resolve(self, strict=strict)

    monkeypatch.setattr(type(tmp_path), "resolve", _refusing)
    assert data_paths.resolve_under(tmp_path, "refused.csv") is None


def test_a_name_resolve_cannot_encode_is_none(tmp_path, monkeypatch):
    """The ``ValueError`` half of the catch.

    With the NUL refused up front, what is left for it is a name the
    filesystem encoding cannot represent. On POSIX a lone surrogate (JSON
    ``"\\ud800.py"`` on the toggle) makes ``resolve`` raise
    ``UnicodeEncodeError``, a ``ValueError``, and uncaught that is a 500.
    Faked, because Windows resolves the same name without complaint and CI
    runs both.
    """
    real_resolve = type(tmp_path).resolve

    def _unencodable(self, strict=False):
        if self.name == "unencodable.py":
            raise UnicodeEncodeError("utf-8", "\ud800", 0, 1,
                                     "surrogates not allowed")
        return real_resolve(self, strict=strict)

    monkeypatch.setattr(type(tmp_path), "resolve", _unencodable)
    assert data_paths.resolve_under(tmp_path, "unencodable.py") is None


@pytest.mark.parametrize("name", [5, None, ["a.csv"]])
def test_a_name_that_is_not_a_string_is_none(tmp_path, name):
    """A JSON body can hand a route anything, and ``directory / 5`` raises
    ``TypeError``."""
    assert data_paths.resolve_under(tmp_path, name) is None


@pytest.mark.parametrize("direct_child", [False, True])
def test_a_parent_segment_is_none(tmp_path, direct_child):
    assert data_paths.resolve_under(
        tmp_path, "../x", direct_child=direct_child) is None


def test_a_direct_child_is_returned_in_both_modes(tmp_path):
    """The answer that has to keep working, or refusing everything would
    pass every test above."""
    expected = tmp_path.resolve() / "x.csv"
    assert data_paths.resolve_under(tmp_path, "x.csv") == expected
    assert data_paths.resolve_under(
        tmp_path, "x.csv", direct_child=True) == expected


def test_a_nested_name_is_kept_unless_a_direct_child_is_required(tmp_path):
    """Media and model downloads serve nested names by design; a preset file
    never is one."""
    assert data_paths.resolve_under(tmp_path, "sub/x.csv") == (
        tmp_path.resolve() / "sub" / "x.csv")
    assert data_paths.resolve_under(
        tmp_path, "sub/x.csv", direct_child=True) is None


def test_an_empty_name_is_the_directory_unless_a_direct_child_is_required(
        tmp_path):
    """The default mode answers with the directory itself, as the routes'
    own checks always did, so they keep answering "Not a file" for it."""
    assert data_paths.resolve_under(tmp_path, "") == tmp_path.resolve()
    assert data_paths.resolve_under(tmp_path, "", direct_child=True) is None


# ── every route that takes a name from a client ──────────────────────────


#: (method, url, request arguments, the route's own detail). Every row
#: answered 500 before #483. The NUL rows of the routes #520 covers answer
#: with the name rule's detail, which names the character.
_REFUSED = [
    pytest.param("POST", "/api/files/upload", _raw_upload(f"a{_NUL}b.csv"),
                 _NUL_DETAIL, id="files-upload"),
    pytest.param("POST", "/api/images/upload", _raw_upload(f"a{_NUL}b.png"),
                 _NUL_DETAIL, id="images-upload"),
    pytest.param("POST", "/api/models/upload", _raw_upload(f"a{_NUL}b.pt"),
                 _NUL_DETAIL, id="models-upload"),
    pytest.param("POST", "/api/custom-nodes/upload",
                 _raw_upload(f"a{_NUL}b.py"), _NUL_DETAIL,
                 id="custom-nodes-upload"),
    pytest.param("GET", "/api/files/download/a%00b.csv", {},
                 _NUL_DETAIL, id="files-download"),
    pytest.param("GET", "/api/images/download/a%00b.png", {},
                 _NUL_DETAIL, id="images-download"),
    pytest.param("GET", "/api/models/download/a%00b.pt", {},
                 _NUL_DETAIL, id="models-download"),
    pytest.param("DELETE", "/api/files/a%00b.csv", {},
                 _NUL_DETAIL, id="files-delete"),
    pytest.param("DELETE", "/api/images/a%00b.png", {},
                 _NUL_DETAIL, id="images-delete"),
    pytest.param("DELETE", "/api/models/a%00b.pt", {},
                 _NUL_DETAIL, id="models-delete"),
    pytest.param("DELETE", "/api/custom-nodes/a%00b.py", {},
                 _NUL_DETAIL, id="custom-nodes-delete"),
    pytest.param("POST", "/api/custom-nodes/toggle",
                 {"json": {"filename": f"a{_NUL}b.py"}},
                 _NUL_DETAIL, id="toggle-nul"),
    pytest.param("POST", "/api/custom-nodes/toggle",
                 {"json": {"filename": 5}},
                 "Invalid filename", id="toggle-number"),
    pytest.param("POST", "/api/custom-nodes/toggle",
                 {"json": {"filename": None}},
                 "Invalid filename", id="toggle-null"),
    pytest.param("GET", "/api/media/a%00b.png", {},
                 "Invalid filename", id="media"),
    pytest.param("GET", "/api/examples/load?path=a%00b", {},
                 "Invalid path", id="examples-load"),
]


@pytest.mark.parametrize("method, url, arguments, detail", _REFUSED)
async def test_a_name_that_cannot_be_a_file_is_a_400_not_a_500(
        client, dirs, method, url, arguments, detail):
    response = await client.request(method, url, **arguments)
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == detail
    # Refused before anything reached the disk, under any name.
    assert [entry for directory in dirs.values()
            for entry in directory.iterdir()] == []


async def test_the_hand_built_upload_body_is_one_the_server_accepts(
        client, dirs):
    """The control for the upload rows above: the same body with an ordinary
    name uploads, so a refusal there is the name being refused and not the
    body going unread."""
    response = await client.post("/api/files/upload",
                                  **_raw_upload("ok.csv", b"a,b\n1,2\n"))
    assert response.status_code == 200, response.text
    assert response.json() == {"filename": "ok.csv", "size": 8}
    assert (dirs["files"] / "ok.csv").read_bytes() == b"a,b\n1,2\n"


# ── #520, the rule: data_paths.check_file_name ───────────────────────────


@pytest.mark.parametrize("name, code, fields", [
    ("what?.csv", "name_reserved_character", {"character": "?"}),
    ("loss|acc.csv", "name_reserved_character", {"character": "|"}),
    ("a*b.csv", "name_reserved_character", {"character": "*"}),
    ("a<b.csv", "name_reserved_character", {"character": "<"}),
    ("a>b.csv", "name_reserved_character", {"character": ">"}),
    ('a"b.csv', "name_reserved_character", {"character": '"'}),
    ("a:b.csv", "name_separator", {"character": ":"}),
    ("run 12:30.csv", "name_separator", {"character": ":"}),
    ("x\\b.csv", "name_separator", {"character": "\\"}),
    ("a/b.csv", "name_separator", {"character": "/"}),
    (f"a{_NUL}b.csv", "name_control_character", {"codepoint": 0}),
    (f"a{_SOH}b.csv", "name_control_character", {"codepoint": 1}),
    ("a\x1fb.csv", "name_control_character", {"codepoint": 31}),
    ("a\x7fb.csv", "name_control_character", {"codepoint": 127}),
    ("", "name_empty", {}),
    ("   ", "name_empty", {}),
    ("..", "name_dot_segment", {}),
    ("...", "name_dot_segment", {}),
    ("con.csv", "name_reserved_device", {"reserved": "con"}),
    ("NUL.txt", "name_reserved_device", {"reserved": "nul"}),
    ("Com1.pt", "name_reserved_device", {"reserved": "com1"}),
    ("lpt9", "name_reserved_device", {"reserved": "lpt9"}),
    ("aux.tar.gz", "name_reserved_device", {"reserved": "aux"}),
    # The spellings older Windows versions also resolve to a device, as
    # ntpath.isreserved lists them: spaces before the dot, the console's two
    # halves, and superscript digits.
    ("con .csv", "name_reserved_device", {"reserved": "con"}),
    ("NUL  .txt", "name_reserved_device", {"reserved": "nul"}),
    ("CONIN$.txt", "name_reserved_device", {"reserved": "conin$"}),
    ("conout$", "name_reserved_device", {"reserved": "conout$"}),
    ("COM\N{SUPERSCRIPT ONE}.pt", "name_reserved_device",
     {"reserved": "com\N{SUPERSCRIPT ONE}"}),
    ("lpt\N{SUPERSCRIPT THREE}", "name_reserved_device",
     {"reserved": "lpt\N{SUPERSCRIPT THREE}"}),
    # Longer than any file system CodefyUI runs on takes: 256 bytes, and 90
    # Chinese characters, which NTFS stores and ext4 does not (270 bytes).
    ("a" * 252 + ".csv", "name_too_long", {"limit": 255}),
    ("\N{CJK UNIFIED IDEOGRAPH-4E2D}" * 90 + ".csv", "name_too_long",
     {"limit": 255}),
])
def test_a_name_windows_cannot_store_is_refused(name, code, fields):
    """The same answer on every OS: a name that works on Linux and not on
    Windows makes a project that cannot move between them."""
    with pytest.raises(data_paths.UnstorableName) as refused:
        data_paths.check_file_name(name)
    assert refused.value.code == code
    assert refused.value.fields == fields


@pytest.mark.parametrize("name", [
    "train.csv", "REPORT.CSV", "b .csv", "run 12-30.csv", "block-2_v3.pt",
    "視覺 分類器.png", "console.csv", "com10.pt", "con_x.csv", "a..b.csv",
    ".hidden.csv", "x.py.disabled", "com0.pt", "conin.txt",
    # Exactly at the limit, in both units a name can be long in.
    "a" * 251 + ".csv", "\N{CJK UNIFIED IDEOGRAPH-4E2D}" * 85,
])
def test_an_ordinary_name_passes(name):
    """The counterweight: refusing everything would pass the test above."""
    data_paths.check_file_name(name)


def test_the_name_typed_and_the_file_written_are_checked_for_different_things():
    """``stored_as``, which presets use: a title becomes
    ``<title, lowercased, spaces as _>.json``. The characters are read in
    what was typed; the length and the device name in what is written."""
    title = "a" * 251
    with pytest.raises(data_paths.UnstorableName) as refused:
        data_paths.check_file_name(title, stored_as=title + ".json")
    assert refused.value.code == "name_too_long"
    # "CON " is stored as con_.json, which is no device on any Windows.
    data_paths.check_file_name("CON ", stored_as="con_.json")


@pytest.mark.parametrize("name, named", [
    ("what?.csv", "'?'"),
    ("x\\b.csv", "'\\'"),
    ("run 12:30.csv", "':'"),
    (f"a{_SOH}b.csv", "U+0001"),
    ("con.csv", "'con'"),
    ("a" * 300 + ".csv", "at most 255 bytes"),
])
def test_the_refusal_says_what_is_wrong(name, named):
    """``str()`` is the upload routes' ``detail``, the sentence the editor
    shows; it has to name what to take out of the name."""
    with pytest.raises(data_paths.UnstorableName) as refused:
        data_paths.check_file_name(name)
    assert named in str(refused.value)


@pytest.mark.parametrize("sent, stored", [
    ("train.csv", "train.csv"),
    ("../../evil.csv", "evil.csv"),
    ("subdir/nested.csv", "nested.csv"),
])
def test_an_upload_is_stored_under_the_part_after_the_last_slash(sent, stored):
    """The reduction ``test_api_data_files`` pins, kept: a browser never
    sends a ``/``, and one that arrives is a path, not a name."""
    assert data_paths.upload_file_name(sent) == stored


@pytest.mark.parametrize("sent, character", [
    ("x\\b.csv", "\\"),
    ("a:b.csv", ":"),
    ("C:b.csv", ":"),
    ("dir/a:b.csv", ":"),
])
def test_an_upload_reads_a_backslash_or_colon_as_a_character(sent, character):
    """``WindowsPath(...).name`` read both as separators and stored the
    upload as ``b.csv``; ``PosixPath`` kept them. Refused on both, nothing is
    stored under a name the user did not pick."""
    with pytest.raises(data_paths.UnstorableName) as refused:
        data_paths.upload_file_name(sent)
    assert refused.value.fields == {"character": character}


# ── #520, lookups: data_paths.check_lookup_name ──────────────────────────


@pytest.fixture
def windows_lookups(monkeypatch):
    """Lookups as a Windows server makes them, on any host.

    Safe on Linux: the check runs before any call reaches the filesystem, so
    nothing here depends on the host being able to hold the name.
    """
    monkeypatch.setattr(data_paths, "_LOOKUP_REFUSES_WINDOWS_CHARACTERS", True)


@pytest.fixture
def posix_lookups(monkeypatch):
    """Lookups as a Linux or macOS server makes them, on any host."""
    monkeypatch.setattr(data_paths, "_LOOKUP_REFUSES_WINDOWS_CHARACTERS", False)


#: Names a Linux or macOS server can hold and a Windows server cannot.
_WINDOWS_ONLY_REFUSALS = [
    "runs\\exp1\\model.pt", "runs/2026-09-23T12:30/model.pt", "what?.csv",
    "Screenshot from 2019-05-12 11:16:07.png",
]


@pytest.mark.parametrize("name", _WINDOWS_ONLY_REFUSALS)
def test_a_windows_server_refuses_a_lookup_by_a_name_it_cannot_hold(
        windows_lookups, name):
    with pytest.raises(data_paths.UnstorableName):
        data_paths.check_lookup_name(name)


@pytest.mark.parametrize("name", _WINDOWS_ONLY_REFUSALS)
def test_a_linux_or_macos_server_looks_such_a_name_up(posix_lookups, name):
    """A file there may have it: uploaded before #520, written by a node such
    as ModelSaver, or put there by hand. ``resolve_under`` keeps the lookup
    inside its directory either way."""
    data_paths.check_lookup_name(name)


@pytest.mark.parametrize("host", ["windows_lookups", "posix_lookups"])
def test_every_server_refuses_a_control_character_in_a_lookup(request, host):
    request.getfixturevalue(host)
    with pytest.raises(data_paths.UnstorableName) as refused:
        data_paths.check_lookup_name(f"a{_SOH}b.csv")
    assert refused.value.fields == {"codepoint": 1}


@pytest.mark.parametrize("host", ["windows_lookups", "posix_lookups"])
def test_a_lookup_reads_a_slash_as_the_separator_of_a_nested_name(
        request, host):
    """Model downloads serve nested names by design."""
    request.getfixturevalue(host)
    data_paths.check_lookup_name("runs/exp1/model.pt")


@pytest.mark.parametrize("host", ["windows_lookups", "posix_lookups"])
@pytest.mark.parametrize("name", [
    "a" * 300 + ".csv",
    # 274 bytes and 94 characters: stored by NTFS, HFS+ and APFS, which count
    # characters, and refused by ext4, which counts bytes.
    "\N{CJK UNIFIED IDEOGRAPH-4E2D}" * 90 + ".csv",
])
def test_a_lookup_refuses_no_length(request, host, name):
    """File systems disagree on how long a name is, so a limit here would
    refuse a file some volume really holds. A name the volume in use cannot
    hold is answered 404 by ``lookup_exists``."""
    request.getfixturevalue(host)
    data_paths.check_lookup_name(name)


@pytest.mark.parametrize("name", [5, None, ["a.csv"]])
def test_a_lookup_leaves_a_name_that_is_not_a_string_to_resolve_under(name):
    """``resolve_under`` answers those, as "Invalid filename" (#483)."""
    data_paths.check_lookup_name(name)


# ── #520, the routes ─────────────────────────────────────────────────────


@pytest.fixture
def reloads(monkeypatch) -> list[None]:
    """Each rediscovery a custom-node route asks for, recorded instead of run.

    A real one rebuilds the process-wide registry from the temp directory
    ``dirs`` points ``CUSTOM_NODES_DIR`` at. A refused request must not ask
    for one at all.
    """
    calls: list[None] = []
    monkeypatch.setattr(routes_custom_nodes, "_reload_all",
                        lambda: calls.append(None))
    return calls


#: Every upload route: its URL, the ``dirs`` key it writes to, and an
#: extension it takes.
_UPLOAD_ROUTES = [
    pytest.param("/api/files/upload", "files", ".csv", id="files"),
    pytest.param("/api/images/upload", "images", ".png", id="images"),
    pytest.param("/api/models/upload", "models", ".pt", id="models"),
    pytest.param("/api/custom-nodes/upload", "custom_nodes", ".py",
                 id="custom-nodes"),
]

#: Names no file can be stored under on Windows, as the multipart header
#: spells them (``{ext}`` is the route's extension), and what the refusal
#: must name. Every one of them answered 500 or 200 before #520.
_UNSTORABLE_UPLOADS = [
    pytest.param("what?{ext}", "'?'", id="question-mark"),
    pytest.param("loss|acc{ext}", "'|'", id="pipe"),
    pytest.param("a*b{ext}", "'*'", id="asterisk"),
    pytest.param("a<b{ext}", "'<'", id="less-than"),
    pytest.param("a>b{ext}", "'>'", id="greater-than"),
    # A browser sends a quote as %22; a client that escapes it the way the
    # header syntax allows gets it through as the character itself.
    pytest.param('a\\"b{ext}', "'\"'", id="quote"),
    # Windows read "a:" as a drive and stored the upload as b<ext>.
    pytest.param("a:b{ext}", "':'", id="drive-letter"),
    # A longer prefix: the bytes went into an NTFS stream of a zero-byte file
    # "run 12" that no list shows.
    pytest.param("run 12:30{ext}", "':'", id="timestamp"),
    pytest.param("b{ext}:hidden{ext}", "':'", id="stream-on-a-stored-file"),
    pytest.param("x\\b{ext}", "'\\'", id="backslash"),
    pytest.param(f"a{_SOH}b{{ext}}", "U+0001", id="control-character"),
    pytest.param("con{ext}", "'con'", id="device-name"),
    pytest.param("con {ext}", "'con'", id="device-name-with-a-space"),
    pytest.param("CONIN${ext}", "'conin$'", id="console-input"),
    # 300 + the extension: the write failed and the route answered 500.
    pytest.param("a" * 300 + "{ext}", "at most 255 bytes", id="too-long"),
]


@pytest.mark.parametrize("route, key, ext", _UPLOAD_ROUTES)
@pytest.mark.parametrize("spelled, named", _UNSTORABLE_UPLOADS)
async def test_an_upload_name_windows_cannot_store_is_a_400_on_every_os(
        client, dirs, reloads, route, key, ext, spelled, named):
    stored = dirs[key] / f"b{ext}"
    stored.write_bytes(b"seed")

    response = await client.post(
        route, **_raw_upload(spelled.format(ext=ext), b"x = 1\n"))

    assert response.status_code == 400, response.text
    assert named in response.json()["detail"]
    # Nothing new in the directory -- not even the zero-byte file an NTFS
    # stream hangs off -- and the stored file is untouched.
    assert sorted(p.name for p in dirs[key].iterdir()) == [f"b{ext}"]
    assert stored.read_bytes() == b"seed"
    assert reloads == []


@pytest.mark.parametrize("route, key, ext", _UPLOAD_ROUTES)
async def test_an_ordinary_name_still_uploads_on_every_route(
        client, dirs, reloads, route, key, ext):
    """The control for the table above, per route."""
    response = await client.post(
        route, **_raw_upload(f"run 12-30{ext}", b"x = 1\n"))

    assert response.status_code == 200, response.text
    assert response.json()["filename"] == f"run 12-30{ext}"
    assert (dirs[key] / f"run 12-30{ext}").read_bytes() == b"x = 1\n"


#: Every route that looks a stored file up by a name the client sends:
#: method, URL (``{}`` takes the percent-encoded name; None is the toggle,
#: which takes it in a JSON body), ``dirs`` key and extension.
_LOOKUPS = [
    pytest.param("GET", "/api/files/download/{}", "files", ".csv",
                 id="files-download"),
    pytest.param("DELETE", "/api/files/{}", "files", ".csv",
                 id="files-delete"),
    pytest.param("GET", "/api/images/download/{}", "images", ".png",
                 id="images-download"),
    pytest.param("DELETE", "/api/images/{}", "images", ".png",
                 id="images-delete"),
    pytest.param("GET", "/api/models/download/{}", "models", ".pt",
                 id="models-download"),
    pytest.param("DELETE", "/api/models/{}", "models", ".pt",
                 id="models-delete"),
    pytest.param("DELETE", "/api/custom-nodes/{}", "custom_nodes", ".py",
                 id="custom-nodes-delete"),
    pytest.param("POST", None, "custom_nodes", ".py",
                 id="custom-nodes-toggle"),
]

#: Names a Windows server cannot hold, as a lookup receives them. There
#: ``a:b.csv`` resolved outside the directory and ``run 12:30.csv`` read or
#: deleted a stream of ``run 12``; the rest answered 404.
_WINDOWS_UNSTORABLE_LOOKUPS = [
    pytest.param("what?{ext}", "'?'", id="question-mark"),
    pytest.param("loss|acc{ext}", "'|'", id="pipe"),
    pytest.param("a*b{ext}", "'*'", id="asterisk"),
    pytest.param("a<b{ext}", "'<'", id="less-than"),
    pytest.param("a>b{ext}", "'>'", id="greater-than"),
    pytest.param('a"b{ext}', "'\"'", id="quote"),
    pytest.param("a:b{ext}", "':'", id="drive-letter"),
    pytest.param("run 12:30{ext}", "':'", id="timestamp"),
    pytest.param("b{ext}:hidden{ext}", "':'", id="stream-on-a-stored-file"),
    pytest.param("x\\b{ext}", "'\\'", id="backslash"),
]

async def _look_up(client, method, url, name):
    if url is None:
        return await client.post("/api/custom-nodes/toggle",
                                 json={"filename": name})
    return await client.request(method, url.format(quote(name, safe="")))


@pytest.mark.parametrize("method, url, key, ext", _LOOKUPS)
@pytest.mark.parametrize("spelled, named", _WINDOWS_UNSTORABLE_LOOKUPS)
async def test_a_windows_server_refuses_a_lookup_by_such_a_name(
        client, dirs, reloads, windows_lookups,
        method, url, key, ext, spelled, named):
    stored = dirs[key] / f"b{ext}"
    stored.write_bytes(b"seed")

    response = await _look_up(client, method, url, spelled.format(ext=ext))

    assert response.status_code == 400, response.text
    assert named in response.json()["detail"]
    assert sorted(p.name for p in dirs[key].iterdir()) == [f"b{ext}"]
    assert stored.read_bytes() == b"seed"
    assert reloads == []


@pytest.mark.parametrize("host", ["windows_lookups", "posix_lookups"])
@pytest.mark.parametrize("method, url, key, ext", _LOOKUPS)
async def test_every_server_refuses_a_lookup_by_a_control_character(
        request, client, dirs, reloads, host, method, url, key, ext):
    request.getfixturevalue(host)
    stored = dirs[key] / f"b{ext}"
    stored.write_bytes(b"seed")

    response = await _look_up(client, method, url, f"a{_SOH}b{ext}")

    assert response.status_code == 400, response.text
    assert "U+0001" in response.json()["detail"]
    assert sorted(p.name for p in dirs[key].iterdir()) == [f"b{ext}"]
    assert reloads == []


@pytest.mark.parametrize("method, url, key, ext", _LOOKUPS)
async def test_a_lookup_by_a_part_too_long_for_the_volume_is_not_found(
        client, dirs, reloads, method, url, key, ext):
    """No length rule on a lookup: such a part names a file that is not
    there. On Linux ``exists`` raises ENAMETOOLONG for it, which answered
    500; on Windows it answers False."""
    stored = dirs[key] / f"b{ext}"
    stored.write_bytes(b"seed")

    response = await _look_up(client, method, url, "a" * 300 + ext)

    assert response.status_code == 404, response.text
    assert sorted(p.name for p in dirs[key].iterdir()) == [f"b{ext}"]
    assert reloads == []


@pytest.mark.parametrize("method, url, key, ext", _LOOKUPS)
async def test_an_os_error_from_a_lookup_is_not_found(
        client, dirs, reloads, monkeypatch, method, url, key, ext):
    """``Path.exists`` as Python 3.10-3.12 runs it on Linux for a part longer
    than the volume holds -- raising instead of answering False -- faked so
    every host takes that path."""
    path_type = type(dirs[key])
    real_exists = path_type.exists

    def _exists(self, *args, **kwargs):
        if len(self.name) > 255:
            raise OSError(errno.ENAMETOOLONG, "File name too long", str(self))
        return real_exists(self, *args, **kwargs)

    monkeypatch.setattr(path_type, "exists", _exists)

    response = await _look_up(client, method, url, "a" * 300 + ext)

    assert response.status_code == 404, response.text
    assert reloads == []


@pytest.mark.skipif(os.name == "nt", reason="Windows cannot store these names")
async def test_a_file_stored_under_such_a_name_stays_reachable_off_windows(
        client, dirs):
    """What the host-dependent lookup is for. A Linux or macOS server can
    already hold these: an upload from before #520, a path typed into a
    ModelSaver, a GNOME screenshot copied in by hand. The upload rule keeps
    new ones out; it must not lock the old ones in."""
    (dirs["files"] / "run 12:30.csv").write_bytes(b"a,b\n")
    run = dirs["models"] / "runs" / "2026-09-23T12:30"
    run.mkdir(parents=True)
    (run / "model.pt").write_bytes(b"weights")
    screenshot = "Screenshot from 2019-05-12 11:16:07.png"
    (dirs["images"] / screenshot).write_bytes(b"png")

    response = await client.get("/api/files/download/run%2012%3A30.csv")
    assert response.status_code == 200, response.text
    assert response.content == b"a,b\n"
    response = await client.get(
        "/api/models/download/runs/2026-09-23T12%3A30/model.pt")
    assert response.status_code == 200, response.text
    assert response.content == b"weights"
    response = await client.get(
        f"/api/images/download/{quote(screenshot, safe='')}")
    assert response.status_code == 200, response.text

    response = await client.delete("/api/files/run%2012%3A30.csv")
    assert response.status_code == 200, response.text
    assert not (dirs["files"] / "run 12:30.csv").exists()
