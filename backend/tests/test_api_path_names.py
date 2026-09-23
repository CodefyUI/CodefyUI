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

from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

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
#: answered 500 before #483.
_REFUSED = [
    pytest.param("POST", "/api/files/upload", _raw_upload(f"a{_NUL}b.csv"),
                 "Invalid filename", id="files-upload"),
    pytest.param("POST", "/api/images/upload", _raw_upload(f"a{_NUL}b.png"),
                 "Invalid filename", id="images-upload"),
    pytest.param("POST", "/api/models/upload", _raw_upload(f"a{_NUL}b.pt"),
                 "Invalid filename", id="models-upload"),
    pytest.param("POST", "/api/custom-nodes/upload",
                 _raw_upload(f"a{_NUL}b.py"), "Invalid filename",
                 id="custom-nodes-upload"),
    pytest.param("GET", "/api/files/download/a%00b.csv", {},
                 "Invalid filename", id="files-download"),
    pytest.param("GET", "/api/images/download/a%00b.png", {},
                 "Invalid filename", id="images-download"),
    pytest.param("GET", "/api/models/download/a%00b.pt", {},
                 "Invalid filename", id="models-download"),
    pytest.param("DELETE", "/api/files/a%00b.csv", {},
                 "Invalid filename", id="files-delete"),
    pytest.param("DELETE", "/api/images/a%00b.png", {},
                 "Invalid filename", id="images-delete"),
    pytest.param("DELETE", "/api/models/a%00b.pt", {},
                 "Invalid filename", id="models-delete"),
    pytest.param("DELETE", "/api/custom-nodes/a%00b.py", {},
                 "Invalid filename", id="custom-nodes-delete"),
    pytest.param("POST", "/api/custom-nodes/toggle",
                 {"json": {"filename": f"a{_NUL}b.py"}},
                 "Invalid filename", id="toggle-nul"),
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
