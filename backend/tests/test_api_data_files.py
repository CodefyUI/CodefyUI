"""Security properties of the data-files API (core#242 item 1).

``routes_data_files.py`` takes a filename straight from the user on three of
its four routes, and those routes were verified BY HAND at review time — which
is exactly the state that lets a guard regress silently. This file pins the
behaviour that review observed, so a change to ``_safe_path``, to the
extension whitelist or to a route's path converter has to break a test rather
than a deployment.

Reading the traversal tests: the exact string the ROUTE sees
------------------------------------------------------------
httpx removes ``..`` segments from a URL before it sends it, the same way a
browser does — ``/api/files/download/../../etc/passwd`` leaves this process as
``/api/etc/passwd`` and answers 404 from the router without ever reaching the
handler. A test written that way passes whether or not the guard exists.

So the tests below percent-encode the dots (``%2e%2e/``). httpx forwards that
untouched, and the ASGI layer decodes it back into ``scope["path"]`` — leaving
the handler with the literal ``../../etc/passwd``, which is precisely what an
attacker using ``curl --path-as-is``, a raw socket or a proxy that does not
normalise produces against a real uvicorn. The encoding is a transport
detail; the server-side input is the real one.

Where an outcome differs between Windows and POSIX it is branched explicitly
rather than smoothed over — backslash is a path separator on one and an
ordinary filename character on the other, and CI runs both.
"""

from __future__ import annotations

import os

import pytest

from app.api.routes_data_files import ALLOWED_EXTENSIONS

# No module-level asyncio mark: pyproject sets asyncio_mode = "auto", and a
# blanket mark would also land on the synchronous unit test below.

# Traversal payloads from the core#242 table that mean the same thing on both
# platforms. The backslash entry is handled separately, below, because it does
# not.
TRAVERSAL_INPUTS = [
    "../../etc/passwd",
    "subdir/../../../secret.csv",
    "/etc/passwd",
    "a/../../../../../../tmp/x.csv",
]

BACKSLASH_INPUT = "..\\..\\windows\\system32\\config\\sam"


def _verbatim(filename: str) -> str:
    """Encode *filename* so the route receives it character for character.

    Only the dots need encoding: ``.`` is what triggers a client's dot-segment
    removal, and ``%2e`` is decoded back to ``.`` before the handler sees it.
    Slashes are left alone so the URL keeps the same shape an attacker's would.
    """
    return filename.replace(".", "%2e")


@pytest.fixture
def data_files_dir(tmp_path, monkeypatch):
    """Redirect settings.DATA_FILES_DIR at a temp dir for each test.

    Read at request time by every route, so patching the attribute is enough —
    and it means no test can touch the real ``backend/data/data_files``.
    """
    d = tmp_path / "data_files"
    d.mkdir()
    monkeypatch.setattr("app.config.settings.DATA_FILES_DIR", d)
    return d


# ── path traversal: download ─────────────────────────────────────────────


@pytest.mark.parametrize("filename", TRAVERSAL_INPUTS)
async def test_download_traversal_is_refused(test_client, data_files_dir, filename):
    """Every traversal payload core#242 recorded as refused, still refused.

    400 rather than 404 is the assertion that matters: it proves the request
    reached the handler and ``_safe_path`` turned it away, instead of being
    lost to URL normalisation on the way in (which would make this test pass
    with the guard deleted).
    """
    resp = await test_client.get(f"/api/files/download/{_verbatim(filename)}")
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "Invalid filename"


async def test_download_backslash_traversal_never_escapes(test_client, data_files_dir):
    """``..\\..\\windows\\...`` — one payload, two correct answers.

    On Windows a backslash separates path components, so this escapes the data
    dir and ``_safe_path`` refuses it (400). On POSIX the same bytes are a
    single, legal — if bizarre — filename that resolves INSIDE the data dir,
    so the honest answer is "no such file" (404). Both platforms run this
    suite in CI; neither serves anything from outside, which is the property
    being pinned.
    """
    resp = await test_client.get(f"/api/files/download/{_verbatim(BACKSLASH_INPUT)}")
    if os.name == "nt":
        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"] == "Invalid filename"
    else:
        assert resp.status_code == 404, resp.text
        assert resp.json()["detail"].startswith("File not found:")


async def test_download_never_serves_a_real_file_above_the_data_dir(
    test_client, data_files_dir, tmp_path,
):
    """The guard against a file that actually exists, one level up.

    The parametrised cases above point at paths that are absent on the test
    machine, so a broken guard could still answer 404 there and look fine.
    Here the target is real, readable and has an allowed extension — the only
    thing standing between it and the response body is ``_safe_path``.
    """
    secret = tmp_path / "secret.csv"
    secret.write_bytes(b"api_key,value\nprod,hunter2\n")

    resp = await test_client.get("/api/files/download/%2e%2e/secret%2ecsv")
    assert resp.status_code == 400, resp.text
    assert b"hunter2" not in resp.content
    assert secret.exists()


async def test_download_of_dot_quad_resolves_inside_the_data_dir(
    test_client, data_files_dir,
):
    """``....//....//x.csv`` is NOT a traversal, and core#242 says so.

    ``....`` is an ordinary directory name — only ``.`` and ``..`` are dot
    segments — so this resolves to ``<data dir>/..../x.csv``, stays inside the
    sandbox, and is simply not there. Pinned because the distinction is easy
    to lose: the answer is 404 from the HANDLER (note the echoed filename in
    the detail, which a router-level 404 does not carry), never 400, and a
    future "fix" that starts refusing it would be refusing a legal name.
    """
    resp = await test_client.get(f"/api/files/download/{_verbatim('....//....//x.csv')}")
    assert resp.status_code == 404, resp.text
    assert resp.json()["detail"] == "File not found: ....//....//x.csv"


async def test_download_of_percent_encoded_traversal_is_refused(
    test_client, data_files_dir, tmp_path,
):
    """``%2e%2e%2fsecret.csv``, sent with its encoding intact, is refused.

    core#242 filed this one alongside ``....//....//x.csv`` as "resolved
    inside the base dir", which is what a BROWSER produces: it decodes the
    escapes itself and collapses the result before the request leaves, so the
    server is handed ``/api/files/secret.csv`` and the traversal is gone. A
    client that forwards the escapes — this one, curl --path-as-is — gets them
    decoded by the ASGI layer instead, the handler sees ``../secret.csv``, and
    the guard answers 400. Both paths are safe; only the second one exercises
    the guard, so that is the one worth a test.
    """
    (tmp_path / "secret.csv").write_bytes(b"leak")

    resp = await test_client.get("/api/files/download/%2e%2e%2fsecret.csv")
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "Invalid filename"
    assert b"leak" not in resp.content


# ── path traversal: delete ───────────────────────────────────────────────


@pytest.mark.parametrize("filename", TRAVERSAL_INPUTS)
async def test_delete_traversal_never_reaches_the_handler(
    test_client, data_files_dir, tmp_path, filename,
):
    """Delete refuses these one layer earlier than download does.

    ``@router.delete("/{filename}")`` has no ``:path`` converter, so a name
    containing a separator matches no route at all and the router answers 404
    before any handler runs — which is why these are 404 here and 400 on the
    download route. The distinction is worth pinning: widening that converter
    to ``{filename:path}`` would silently move the entire burden onto
    ``_safe_path``, and this test is what would notice.
    """
    decoy = tmp_path / "secret.csv"
    decoy.write_bytes(b"still here")

    resp = await test_client.delete(f"/api/files/{_verbatim(filename)}")
    assert resp.status_code == 404, resp.text
    assert decoy.exists()


async def test_delete_backslash_traversal_never_escapes(
    test_client, data_files_dir, tmp_path,
):
    """The backslash payload is the one that DOES reach delete's handler.

    It has no forward slash, so it matches ``{filename}`` on both platforms.
    From there Windows resolves it out of the data dir and ``_safe_path``
    refuses it (400), while on POSIX it is one odd-looking filename that is
    simply absent (404). Nothing outside the data dir is unlinked either way.
    """
    decoy = tmp_path / "secret.csv"
    decoy.write_bytes(b"still here")

    resp = await test_client.delete(f"/api/files/{_verbatim(BACKSLASH_INPUT)}")
    if os.name == "nt":
        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"] == "Invalid filename"
    else:
        assert resp.status_code == 404, resp.text
        assert resp.json()["detail"].startswith("File not found:")
    assert decoy.exists()


# ── the extension whitelist ──────────────────────────────────────────────


def test_allowed_extensions_are_exactly_the_documented_four():
    """A whitelist is only a whitelist while it is short.

    Pinned as a set so that adding an entry is a deliberate act with a test
    change attached — the failure mode this guards against is someone
    accepting ``.py`` or ``.zip`` for one node's convenience and handing the
    upload route an arbitrary-file-write surface.
    """
    assert ALLOWED_EXTENSIONS == {".csv", ".tsv", ".txt", ".json"}


@pytest.mark.parametrize("filename", ["evil.exe", "keys.pem", "bundle.zip", "noext"])
async def test_upload_rejects_a_disallowed_extension(
    test_client, data_files_dir, filename,
):
    """Upload is where a bad extension costs the most — it is the only route
    that writes. ``noext`` is in the list because a suffix-less name yields
    ``""``, and an empty string must fail the membership test like any other
    value rather than slipping through some falsy shortcut.
    """
    resp = await test_client.post(
        "/api/files/upload",
        files={"file": (filename, b"payload", "application/octet-stream")},
    )
    assert resp.status_code == 400, resp.text
    assert "Unsupported file type" in resp.json()["detail"]
    assert list(data_files_dir.iterdir()) == []


async def test_upload_extension_check_is_case_insensitive(test_client, data_files_dir):
    """``.CSV`` is a CSV. The check lowercases before comparing, and it has to
    stay that way — a case-sensitive whitelist would reject a file a student
    exported from Excel on Windows.
    """
    resp = await test_client.post(
        "/api/files/upload",
        files={"file": ("REPORT.CSV", b"a,b\n1,2\n", "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    assert (data_files_dir / "REPORT.CSV").exists()


async def test_download_refuses_a_file_with_a_disallowed_extension(
    test_client, data_files_dir,
):
    """A file inside the data dir is still not downloadable unless its
    extension is on the list.

    This is the second half of the whitelist and the half that is easy to
    forget: something else may have written into that directory (a node's
    output, a stray key file a user dropped in), and the upload guard says
    nothing about those.
    """
    (data_files_dir / "id_rsa.pem").write_bytes(b"-----BEGIN PRIVATE KEY-----")

    resp = await test_client.get("/api/files/download/id_rsa.pem")
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "Not a data file"
    assert b"PRIVATE KEY" not in resp.content


async def test_delete_refuses_a_file_with_a_disallowed_extension(
    test_client, data_files_dir,
):
    """The same whitelist on delete, and here it is a write guard: without it
    this route deletes any file in the directory, not just data files.
    """
    victim = data_files_dir / "id_rsa.pem"
    victim.write_bytes(b"-----BEGIN PRIVATE KEY-----")

    resp = await test_client.delete("/api/files/id_rsa.pem")
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "Not a data file"
    assert victim.exists()


async def test_download_and_delete_refuse_a_directory(test_client, data_files_dir):
    """A directory name passes ``_safe_path`` — it is inside the sandbox — so
    the ``is_file`` check is what stops FileResponse being handed a directory
    and ``unlink`` being called on one.
    """
    (data_files_dir / "nested").mkdir()

    resp = await test_client.get("/api/files/download/nested")
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "Not a file"

    resp = await test_client.delete("/api/files/nested")
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "Not a file"
    assert (data_files_dir / "nested").is_dir()


# ── upload filename reduction ────────────────────────────────────────────


async def test_upload_strips_directory_components_from_the_filename(
    test_client, data_files_dir, tmp_path,
):
    """``Path(filename).name`` is upload's guard, and it runs BEFORE
    ``_safe_path``.

    A multipart filename never passes through URL routing, so nothing
    normalises it on the way in: the handler gets whatever the client typed.
    The reduction is what turns ``../../evil.csv`` into ``evil.csv``, and the
    assertion that matters is not just where the file landed but that the two
    levels it aimed at are untouched.
    """
    resp = await test_client.post(
        "/api/files/upload",
        files={"file": ("../../evil.csv", b"x,y\n1,2\n", "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["filename"] == "evil.csv"
    assert (data_files_dir / "evil.csv").read_bytes() == b"x,y\n1,2\n"
    # Nothing appeared beside or above the data dir.
    assert [p.name for p in tmp_path.iterdir()] == ["data_files"]
    assert not (tmp_path.parent / "evil.csv").exists()


async def test_upload_flattens_a_nested_filename(test_client, data_files_dir):
    """The same reduction with no traversal in it: ``subdir/nested.csv`` lands
    flat. Pinned so the guard cannot be "fixed" into creating subdirectories,
    which ``list`` does not show and ``delete``'s route cannot address.
    """
    resp = await test_client.post(
        "/api/files/upload",
        files={"file": ("subdir/nested.csv", b"a\n", "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["filename"] == "nested.csv"
    assert (data_files_dir / "nested.csv").exists()
    assert not (data_files_dir / "subdir").exists()


# ── the size ceiling ─────────────────────────────────────────────────────


async def test_upload_over_the_size_limit_is_refused_with_413(
    test_client, data_files_dir, monkeypatch,
):
    """The route's own ceiling, in the route's own words.

    Deliberately only 1 KB over: ``core.body_limit`` caps the multipart BODY
    at MAX_UPLOAD_SIZE plus an envelope allowance, so a payload this size
    sails past the middleware and is refused by the handler's own
    ``len(content)`` check — which is the one that enforces the documented
    number exactly. Nothing is written.
    """
    monkeypatch.setattr("app.config.settings.MAX_UPLOAD_SIZE", 1024)

    resp = await test_client.post(
        "/api/files/upload",
        files={"file": ("big.csv", b"c" * 2048, "text/csv")},
    )
    assert resp.status_code == 413, resp.text
    assert resp.json()["detail"] == "File too large"
    assert not (data_files_dir / "big.csv").exists()


async def test_upload_at_the_size_limit_still_succeeds(
    test_client, data_files_dir, monkeypatch,
):
    """The counterweight to the test above: a file of exactly the limit is
    accepted. Without this, refusing every upload would pass the 413 test.
    """
    monkeypatch.setattr("app.config.settings.MAX_UPLOAD_SIZE", 1024)

    resp = await test_client.post(
        "/api/files/upload",
        files={"file": ("exact.csv", b"c" * 1024, "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["size"] == 1024
    assert (data_files_dir / "exact.csv").stat().st_size == 1024


# ── listing ──────────────────────────────────────────────────────────────


async def test_list_skips_subdirectories_and_other_extensions(
    test_client, data_files_dir,
):
    """``list`` feeds the DATA_FILE dropdown, so anything it returns is
    something a learner can pick and a node will try to open.

    A subdirectory would be un-openable, and a file outside the whitelist
    would be un-downloadable — both are entries that can only produce a
    confusing error later. ``is_file()`` and the suffix test keep them out.
    """
    (data_files_dir / "train.csv").write_bytes(b"a,b\n1,2\n")
    (data_files_dir / "notes.txt").write_bytes(b"hello")
    (data_files_dir / "id_rsa.pem").write_bytes(b"secret")
    nested = data_files_dir / "nested"
    nested.mkdir()
    (nested / "inner.csv").write_bytes(b"a\n")

    resp = await test_client.get("/api/files")
    assert resp.status_code == 200, resp.text
    listed = {item["filename"]: item["size"] for item in resp.json()}
    assert set(listed) == {"train.csv", "notes.txt"}
    assert listed["train.csv"] == 8


async def test_list_creates_the_data_dir_when_it_is_missing(
    test_client, tmp_path, monkeypatch,
):
    """First run on a fresh install: the directory does not exist yet, and
    listing must answer an empty list rather than raising.
    """
    missing = tmp_path / "not-created-yet"
    monkeypatch.setattr("app.config.settings.DATA_FILES_DIR", missing)

    resp = await test_client.get("/api/files")
    assert resp.status_code == 200, resp.text
    assert resp.json() == []
    assert missing.is_dir()


# ── the ordinary path ────────────────────────────────────────────────────


async def test_upload_list_download_delete_roundtrip(test_client, data_files_dir):
    """The whole lifecycle a learner actually walks, in one test.

    Its job is to keep the guards above honest: every other test in this file
    asserts that something is refused, and all of them would still pass if the
    router refused everything.
    """
    payload = b"feature,label\n1.0,0\n2.0,1\n"

    resp = await test_client.post(
        "/api/files/upload",
        files={"file": ("train.csv", payload, "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"filename": "train.csv", "size": len(payload)}

    resp = await test_client.get("/api/files")
    assert resp.status_code == 200, resp.text
    assert resp.json() == [{"filename": "train.csv", "size": len(payload)}]

    resp = await test_client.get("/api/files/download/train.csv")
    assert resp.status_code == 200, resp.text
    assert resp.content == payload
    assert "train.csv" in resp.headers.get("content-disposition", "")

    resp = await test_client.delete("/api/files/train.csv")
    assert resp.status_code == 200, resp.text
    assert not (data_files_dir / "train.csv").exists()

    resp = await test_client.get("/api/files/download/train.csv")
    assert resp.status_code == 404, resp.text
    resp = await test_client.get("/api/files")
    assert resp.json() == []


async def test_download_and_delete_of_a_missing_file_report_404(
    test_client, data_files_dir,
):
    """A name that is perfectly legal but absent is "not found", not
    "invalid" — the two answers must stay distinguishable, because a 400 here
    would send a learner hunting for a bad filename instead of a missing file.
    """
    resp = await test_client.get("/api/files/download/absent.csv")
    assert resp.status_code == 404, resp.text
    assert resp.json()["detail"] == "File not found: absent.csv"

    resp = await test_client.delete("/api/files/absent.csv")
    assert resp.status_code == 404, resp.text
    assert resp.json()["detail"] == "File not found: absent.csv"
