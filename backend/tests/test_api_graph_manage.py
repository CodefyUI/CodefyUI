"""Managing saved graphs: DELETE /{name}, POST /rename, /list's `modified`,
and POST /save's overwrite guard.

Conventions follow test_api_graph_run.py: `_graphs_dir` monkeypatches
`settings.GRAPHS_DIR` at a temp dir so no test can reach a real
`backend/data/graphs`. Project mode gets a fixture of its own (the shape
test_graph_save_load_project.py uses) because `LAYOUT_DIR` is derived from
`PROJECT_DIR` rather than settable on its own.

Traversal payloads are percent-encoded for the reason test_api_data_files.py
spells out at length: httpx removes `..` segments from a URL before sending
it, so an unencoded payload never reaches the handler and the test would pass
with `_sanitize_name` deleted. Only the backslash payload can be tested
through the URL at all -- `{name}` is a single path segment, so a decoded
forward slash is a router 404 -- which is why the `../` payloads live in the
rename tests, where names arrive in a JSON body that nothing normalises.
"""

import json
import os

import pytest


@pytest.fixture
def _graphs_dir(tmp_path, monkeypatch):
    """Isolate saved graphs per test (pattern: test_api_graph.py)."""
    monkeypatch.setattr("app.config.settings.PROJECT_DIR", None)
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def project_settings(tmp_path, monkeypatch):
    """Project mode, with GRAPHS_DIR one level below `tmp_path` -- which is
    also what gives the traversal tests somewhere OUTSIDE it to put a decoy.
    """
    monkeypatch.setattr("app.config.settings.PROJECT_DIR", tmp_path)
    monkeypatch.setattr("app.config.settings.GRAPHS_DIR", tmp_path / "graphs")
    (tmp_path / "graphs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "layout").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _graph(name="demo", x=10):
    """The smallest thing /save will write: one node, no edges."""
    return {
        "name": name,
        "description": "",
        "nodes": [
            {"id": "a", "type": "Dataset", "position": {"x": x, "y": 0},
             "data": {"params": {"name": "MNIST"}}},
        ],
        "edges": [],
        "presets": [],
        "segmentGroups": [],
    }


async def _save(client, graph):
    resp = await client.post("/api/graph/save", json=graph)
    assert resp.status_code == 200, resp.text


def _verbatim(name: str) -> str:
    """Encode *name* so the route receives it character for character.

    Only the dots need it (test_api_data_files.py): `.` is what triggers a
    client's dot-segment removal, and `%2e` is decoded back before the handler
    sees it.
    """
    return name.replace(".", "%2e")


# ── DELETE /api/graph/{name} ─────────────────────────────────────────────


async def test_delete_removes_the_file(test_client, _graphs_dir):
    await _save(test_client, _graph(name="demo"))
    assert (_graphs_dir / "demo.json").exists()

    resp = await test_client.delete("/api/graph/demo")
    assert resp.status_code == 200, resp.text
    assert not (_graphs_dir / "demo.json").exists()
    assert (await test_client.get("/api/graph/list")).json() == []


async def test_delete_404_when_there_was_nothing(test_client, _graphs_dir):
    """Not a quiet success: the panel showed the user a row, and "another tab
    deleted it first" is something they should get to see."""
    resp = await test_client.delete("/api/graph/never-saved")
    assert resp.status_code == 404, resp.text
    assert resp.json()["detail"] == "Graph 'never-saved' not found"


async def test_delete_removes_both_halves_in_project_mode(
    test_client, project_settings,
):
    await _save(test_client, _graph(name="demo"))
    logic = project_settings / "graphs" / "demo.graph.json"
    layout = project_settings / "layout" / "demo.layout.json"
    assert logic.exists() and layout.exists()

    resp = await test_client.delete("/api/graph/demo")
    assert resp.status_code == 200, resp.text
    assert not logic.exists()
    assert not layout.exists()


async def test_delete_removes_a_legacy_single_file(test_client, project_settings):
    """A project that has not saved since the split still stores `<name>.json`;
    resolve_graph_file hands the delete exactly that file, so there is no
    second unlink to forget."""
    legacy = project_settings / "graphs" / "old.json"
    legacy.write_text(json.dumps(_graph(name="old")))

    resp = await test_client.delete("/api/graph/old")
    assert resp.status_code == 200, resp.text
    assert not legacy.exists()


async def test_delete_of_an_ambiguous_pair_409s(test_client, project_settings):
    """Both forms of one base: delete refuses exactly as /load does, naming
    both files. Guessing here means destroying the wrong one."""
    (project_settings / "graphs" / "dup.graph.json").write_text("{}")
    (project_settings / "graphs" / "dup.json").write_text("{}")

    resp = await test_client.delete("/api/graph/dup")
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert "dup.graph.json" in detail
    assert "dup.json" in detail
    assert (project_settings / "graphs" / "dup.graph.json").exists()
    assert (project_settings / "graphs" / "dup.json").exists()


async def test_delete_traversal_never_leaves_graphs_dir(
    test_client, project_settings,
):
    """`..\\outside` reaches the handler on both platforms (no forward slash,
    so it matches `{name}`), and `_sanitize_name` reduces it to underscores
    inside GRAPHS_DIR. The echoed name in the detail is what proves the
    handler ran at all rather than the router answering first."""
    outside = project_settings / "outside.json"
    outside.write_text(json.dumps(_graph(name="outside")))

    resp = await test_client.delete(f"/api/graph/{_verbatim('..')}\\outside")
    assert resp.status_code == 404, resp.text
    assert resp.json()["detail"] == "Graph '..\\outside' not found"
    assert outside.exists()


# ── POST /api/graph/rename ───────────────────────────────────────────────


async def test_rename_moves_the_file(test_client, _graphs_dir):
    await _save(test_client, _graph(name="before"))

    resp = await test_client.post(
        "/api/graph/rename", json={"from": "before", "to": "after"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["file"] == "after"
    assert not (_graphs_dir / "before.json").exists()
    assert (_graphs_dir / "after.json").exists()


async def test_rename_rewrites_the_name_inside_the_file(test_client, _graphs_dir):
    """The list shows each file's own `name`, so a rename that moved only the
    file would leave a row still calling itself by the old title."""
    await _save(test_client, _graph(name="before"))

    resp = await test_client.post(
        "/api/graph/rename", json={"from": "before", "to": "after"})
    assert resp.status_code == 200, resp.text
    assert json.loads((_graphs_dir / "after.json").read_text())["name"] == "after"
    listed = (await test_client.get("/api/graph/list")).json()
    assert [(g["name"], g["file"]) for g in listed] == [("after", "after")]


async def test_rename_404_when_the_source_is_gone(test_client, _graphs_dir):
    resp = await test_client.post(
        "/api/graph/rename", json={"from": "never-saved", "to": "x"})
    assert resp.status_code == 404, resp.text
    assert resp.json()["detail"] == "Graph 'never-saved' not found"
    assert not (_graphs_dir / "x.json").exists()


async def test_rename_onto_an_existing_graph_409s(test_client, _graphs_dir):
    await _save(test_client, _graph(name="one"))
    await _save(test_client, _graph(name="two"))

    resp = await test_client.post(
        "/api/graph/rename", json={"from": "one", "to": "two"})
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"] == "Graph 'two' already exists"
    assert (_graphs_dir / "one.json").exists()
    assert (_graphs_dir / "two.json").exists()


async def test_rename_onto_a_legacy_file_of_the_same_base_409s(
    test_client, project_settings,
):
    """The collision check resolves the target the way a READ would, so
    renaming a canonical pair onto a base that exists only in the legacy form
    is refused too -- otherwise both forms of one base would end up on disk,
    which is the state GraphAmbiguityError exists to forbid."""
    await _save(test_client, _graph(name="pair"))
    (project_settings / "graphs" / "taken.json").write_text(
        json.dumps(_graph(name="taken")))

    resp = await test_client.post(
        "/api/graph/rename", json={"from": "pair", "to": "taken"})
    assert resp.status_code == 409, resp.text
    assert (project_settings / "graphs" / "pair.graph.json").exists()
    assert not (project_settings / "graphs" / "taken.graph.json").exists()


async def test_rename_moves_the_layout_half_in_project_mode(
    test_client, project_settings,
):
    await _save(test_client, _graph(name="before"))

    resp = await test_client.post(
        "/api/graph/rename", json={"from": "before", "to": "after"})
    assert resp.status_code == 200, resp.text
    graphs, layout = project_settings / "graphs", project_settings / "layout"
    assert not (graphs / "before.graph.json").exists()
    assert not (layout / "before.layout.json").exists()
    assert (graphs / "after.graph.json").exists()
    assert (layout / "after.layout.json").exists()
    # Positions live only in the layout file, so a graph that still loads at
    # x=10 is proof the layout half followed rather than being left orphaned.
    loaded = (await test_client.get("/api/graph/load/after")).json()
    assert loaded["name"] == "after"
    assert loaded["nodes"][0]["position"] == {"x": 10, "y": 0}


async def test_rename_onto_an_orphan_layout_409s(test_client, project_settings):
    """A layout file outliving its graph -- a `git checkout` of `graphs/`
    alone, a manual delete -- is still someone's positions. The logic half has
    nothing to collide with there, so only the layout check stands between the
    rename and `Path.replace` overwriting the orphan without a word."""
    await _save(test_client, _graph(name="before"))
    orphan = project_settings / "layout" / "after.layout.json"
    orphan.write_text(json.dumps(
        {"format_version": 1, "positions": {"a": {"x": 999, "y": 999}}}))
    before = orphan.read_text()

    resp = await test_client.post(
        "/api/graph/rename", json={"from": "before", "to": "after"})
    assert resp.status_code == 409, resp.text
    assert "layout" in resp.json()["detail"].lower()
    # Refused before anything moved: the source pair is whole and the orphan
    # still holds its own bytes rather than the renamed graph's.
    assert (project_settings / "graphs" / "before.graph.json").exists()
    assert (project_settings / "layout" / "before.layout.json").exists()
    assert not (project_settings / "graphs" / "after.graph.json").exists()
    assert orphan.read_text() == before


@pytest.mark.parametrize("reserved", ["weird.graph", "weird.layout"])
async def test_rename_to_a_reserved_name_is_refused(
    test_client, project_settings, reserved,
):
    """The guard /save applies, applied to the other way a name gets chosen."""
    await _save(test_client, _graph(name="demo"))

    resp = await test_client.post(
        "/api/graph/rename", json={"from": "demo", "to": reserved})
    assert resp.status_code == 400, resp.text
    assert "reserved" in resp.json()["detail"].lower()
    # Refused before anything moved: the graph is still where it was, and no
    # sanitized would-be target (weird_graph.graph.json) was written.
    assert [p.name for p in (project_settings / "graphs").iterdir()] == [
        "demo.graph.json"]


@pytest.mark.parametrize("payload", ["../outside", "..\\outside", "a/../../outside"])
async def test_rename_from_a_traversal_is_refused(
    test_client, project_settings, payload,
):
    """Names arrive in a JSON body here, so these reach the handler verbatim
    on every platform -- no encoding, no client-side normalisation."""
    outside = project_settings / "outside.json"
    outside.write_text(json.dumps(_graph(name="outside")))

    resp = await test_client.post(
        "/api/graph/rename", json={"from": payload, "to": "stolen"})
    assert resp.status_code == 404, resp.text
    assert outside.exists()
    assert not (project_settings / "graphs" / "stolen.graph.json").exists()


async def test_rename_to_a_traversal_stays_inside_graphs_dir(
    test_client, project_settings,
):
    """The write side of the same guard: the target is sanitized too, so
    `../escaped` becomes underscores inside GRAPHS_DIR rather than a file one
    directory up."""
    await _save(test_client, _graph(name="demo"))

    resp = await test_client.post(
        "/api/graph/rename", json={"from": "demo", "to": "../escaped"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["file"] == "___escaped"
    assert (project_settings / "graphs" / "___escaped.graph.json").exists()
    assert not (project_settings / "escaped.json").exists()
    assert not (project_settings / "escaped.graph.json").exists()


# ── GET /api/graph/list: `modified` ──────────────────────────────────────


async def test_list_reports_the_file_mtime(test_client, _graphs_dir):
    await _save(test_client, _graph(name="demo"))

    listed = (await test_client.get("/api/graph/list")).json()
    assert len(listed) == 1
    # `name` and `file` are what existing clients read; `modified` is additive.
    assert listed[0]["name"] == "demo"
    assert listed[0]["file"] == "demo"
    assert listed[0]["modified"] == (_graphs_dir / "demo.json").stat().st_mtime


async def test_list_reports_the_logic_mtime_not_the_layout(
    test_client, project_settings,
):
    """A drag rewrites only the layout half. A list sorted most-recent-first
    must not read that as an edit to the graph."""
    await _save(test_client, _graph(name="demo"))
    logic = project_settings / "graphs" / "demo.graph.json"
    layout = project_settings / "layout" / "demo.layout.json"
    newer = logic.stat().st_mtime + 3600
    os.utime(layout, (newer, newer))

    listed = (await test_client.get("/api/graph/list")).json()
    assert listed[0]["modified"] == logic.stat().st_mtime
    assert listed[0]["modified"] < newer


# ── POST /api/graph/save: the overwrite guard (issue #455) ───────────────
#
# A save that names no address is a Save As: the target stem is derived from
# the TITLE, by `_sanitize_name`, and the server is the only thing that knows
# that rule. The editor used to keep a replica of it and check for a
# collision itself -- which is how a graph titled "模型" plus U+2EBF0 (CJK
# Ext I) silently destroyed an existing "模型_". Node's ICU tables call
# U+2EBF0 a letter (Unicode 17) and CPython's `str.isalnum` does not (Unicode
# 14), so the replica computed the stem "模型" plus that character, matched no
# existing row, showed no confirmation, and the server then wrote "模型_".
# 14,049 code points disagree that way, always in the direction of the client
# keeping a character the server replaces. So the client stopped guessing:
# it posts `overwrite: false`, and a collision comes back as a 409 it can
# confirm and re-post.
#
# `overwrite` has THREE states, and the tests here cover all of them: `false`
# is the opt-in above, `true` is the confirmed retry, and the field left off
# entirely means "I have never heard of this guard" and writes -- which is
# what every dist up to 2.8.0 says, `file` having shipped only in 2.8.1.
#
# The tests below use two of those code points deliberately -- U+2EBF0 for
# the astral case and U+A7CB for the cheapest BMP one -- rather than an
# invented character, because the bug only exists where the two Unicode
# versions actually disagree.

#: The title whose stem NEITHER side of the old arrangement agreed on.
SKEWED_TITLE = "模型\U0002EBF0"
#: What the server resolves it to, and what the editor's replica did not.
SKEWED_STEM = "模型_"


def _save_as(graph: dict, **extra) -> dict:
    """The body the editor's FIRST Save As attempt puts on the wire.

    No `file`, because on that path the title is the address and resolving it
    is what the server is for -- and `overwrite` spelled out as False, which
    is the sender saying "I know this can come back 409, and I have a user to
    ask". Omitting the key says the opposite (see
    `test_a_client_that_never_heard_of_overwrite_saves_as_it_always_did`), so
    every test that expects the guard to fire has to send it, and so does
    every test asserting that something ELSE fires first.
    """
    return {**graph, "overwrite": False, **extra}


async def test_save_as_onto_an_occupied_stem_409s_instead_of_overwriting(
    test_client, _graphs_dir,
):
    """THE BUG (#455): no `file`, a title that sanitizes onto an existing
    graph's stem, and the existing graph must still be there afterwards."""
    await _save(test_client, _graph(name=SKEWED_STEM, x=10))
    target = _graphs_dir / f"{SKEWED_STEM}.json"
    before = target.read_text(encoding="utf-8")

    resp = await test_client.post(
        "/api/graph/save", json=_save_as(_graph(name=SKEWED_TITLE, x=999)))
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"] == {
        "error": "graph_exists",
        "file": SKEWED_STEM,
        "name": SKEWED_STEM,
    }
    # Not "a file is still there" -- the same bytes are still there. A guard
    # that 409'd after writing would pass the existence check and still have
    # destroyed the graph.
    assert target.read_text(encoding="utf-8") == before


async def test_save_as_with_overwrite_replaces_the_graph(
    test_client, _graphs_dir,
):
    """The other half of the round trip: the user confirmed, so the second
    POST does exactly what the first one refused to do."""
    await _save(test_client, _graph(name=SKEWED_STEM, x=10))
    target = _graphs_dir / f"{SKEWED_STEM}.json"

    resp = await test_client.post("/api/graph/save", json={
        **_graph(name=SKEWED_TITLE, x=999), "overwrite": True,
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["file"] == SKEWED_STEM
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["name"] == SKEWED_TITLE
    assert saved["nodes"][0]["position"] == {"x": 999, "y": 0}
    # `overwrite` is an answer this one request gave, not a property of the
    # graph: non-project mode writes the request payload verbatim, so a
    # missing pop would persist it into the file and into every git diff.
    assert "overwrite" not in saved
    assert '"overwrite"' not in target.read_text(encoding="utf-8")


async def test_the_guard_catches_the_cheapest_bmp_skew_too(
    test_client, _graphs_dir,
):
    """U+A7CB, the least exotic of the 16 disagreeing BMP code points: the
    bug needs no astral character, so neither does its test."""
    await _save(test_client, _graph(name="model_", x=10))
    target = _graphs_dir / "model_.json"
    before = target.read_text(encoding="utf-8")

    resp = await test_client.post(
        "/api/graph/save", json=_save_as(_graph(name="modelꟋ", x=999)))
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"] == {
        "error": "graph_exists", "file": "model_", "name": "model_",
    }
    assert target.read_text(encoding="utf-8") == before


async def test_save_as_to_a_free_stem_is_still_an_ordinary_save(
    test_client, _graphs_dir,
):
    """The guard answers "is anything already there?", so on a fresh name it
    has nothing to say. Saving a new graph must not need `overwrite`."""
    resp = await test_client.post(
        "/api/graph/save", json=_save_as(_graph(name=SKEWED_TITLE, x=10)))
    assert resp.status_code == 200, resp.text
    assert resp.json()["file"] == SKEWED_STEM
    assert (_graphs_dir / f"{SKEWED_STEM}.json").exists()


async def test_a_save_that_names_its_address_never_takes_the_guard(
    test_client, _graphs_dir,
):
    """A request carrying `file` is asserting where it wants to be written --
    that IS the in-place Save the editor does on every Ctrl+S from 2.8.1 on,
    and it lands on an existing file by definition. Asking "replace it?"
    about the file the tab is already sitting on is not a question worth a
    dialog, so the guard is only for the case where the SERVER, not the
    client, chose the target.

    `overwrite: False` is on the wire deliberately: without it the sentinel
    below would let this request through on its own and the test would prove
    nothing about `file`.
    """
    await _save(test_client, _graph(name="demo", x=10))

    resp = await test_client.post("/api/graph/save", json=_save_as(
        _graph(name="A New Title", x=999), file="demo",
    ))
    assert resp.status_code == 200, resp.text
    saved = json.loads((_graphs_dir / "demo.json").read_text(encoding="utf-8"))
    assert saved["name"] == "A New Title"
    assert saved["nodes"][0]["position"] == {"x": 999, "y": 0}


@pytest.mark.parametrize("says", [{}, {"overwrite": None}], ids=["omitted", "null"])
async def test_a_client_that_never_heard_of_overwrite_saves_as_it_always_did(
    test_client, _graphs_dir, says,
):
    """THE COMPATIBILITY HALF, and it is as load-bearing as the guard.

    `file` only shipped in 2.8.1. Every dist up to 2.8.0 -- and every script
    written against this route's documented contract, where the body is the
    graph and the address comes from `name` -- posts exactly this for an
    ordinary in-place Ctrl+S: no address, no `overwrite`, onto a stem that is
    occupied by definition because it holds the graph being re-saved.

    Read `overwrite` as a plain bool and all of those get a 409 they cannot
    answer: that build's `saveGraph` raises on any non-2xx and the string
    `overwrite` does not exist anywhere in it, so the edit is dropped under a
    "Save failed: Conflict" toast and in-place Save is simply gone. Hence the
    three-state field -- absent means "nobody on this end can be asked", and
    refusing a save so somebody can confirm it is only worth doing when
    somebody is there. An explicit `null` is the same statement.
    """
    await _save(test_client, _graph(name="demo", x=10))

    resp = await test_client.post(
        "/api/graph/save", json={**_graph(name="demo", x=999), **says})
    assert resp.status_code == 200, resp.text
    saved = json.loads((_graphs_dir / "demo.json").read_text(encoding="utf-8"))
    assert saved["nodes"][0]["position"] == {"x": 999, "y": 0}
    # And the sentinel is no more a property of the graph than `True` is.
    assert "overwrite" not in saved


async def test_an_old_client_still_saves_in_place_in_project_mode(
    test_client, project_settings,
):
    """The same pre-2.8.1 shape inside a project, where the re-save also has
    to keep doing the legacy-to-pair upgrade `write_graph_pair` performs. A
    409 here would strand a project on its legacy single files."""
    await _save(test_client, _graph(name="demo", x=10))
    logic = project_settings / "graphs" / "demo.graph.json"

    resp = await test_client.post(
        "/api/graph/save", json=_graph(name="demo", x=999))
    assert resp.status_code == 200, resp.text
    layout = project_settings / "layout" / "demo.layout.json"
    assert json.loads(
        layout.read_text(encoding="utf-8"))["positions"]["a"] == {"x": 999, "y": 0}
    assert logic.exists()


async def test_a_blank_file_counts_as_absent_and_takes_the_guard(
    test_client, _graphs_dir,
):
    """`file: ""` is how a client spells "this tab has no file yet", and
    /save already reads it as absent when it picks the target. It has to be
    read the same way one line later, or the tab that most needs the
    confirmation is the one that never gets it."""
    await _save(test_client, _graph(name="demo", x=10))
    before = (_graphs_dir / "demo.json").read_text(encoding="utf-8")

    resp = await test_client.post(
        "/api/graph/save", json=_save_as(_graph(name="demo", x=999), file=""))
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["file"] == "demo"
    assert (_graphs_dir / "demo.json").read_text(encoding="utf-8") == before


@pytest.mark.parametrize("raw", ["not json at all", "[]", '{"nodes": []}'])
async def test_the_existing_title_falls_back_to_the_stem(
    test_client, _graphs_dir, raw,
):
    """`name` in the 409 is there so the dialog can say WHICH graph it is
    about. A file that cannot be parsed, or that carries no title, still
    occupies the stem -- so the answer is the stem rather than a 500."""
    (_graphs_dir / "broken.json").write_text(raw, encoding="utf-8")

    resp = await test_client.post(
        "/api/graph/save", json=_save_as(_graph(name="broken", x=999)))
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"] == {
        "error": "graph_exists", "file": "broken", "name": "broken",
    }
    assert (_graphs_dir / "broken.json").read_text(encoding="utf-8") == raw


async def test_the_guard_reports_the_stored_title_not_the_stem(
    test_client, _graphs_dir,
):
    """When the file does have a title, that is what the dialog shows: the
    stem is a filename, and "replace 模型_?" is a worse question than
    "replace the graph called 我的模型?"."""
    await _save(test_client, {**_graph(name="我的模型"), "file": SKEWED_STEM})

    resp = await test_client.post(
        "/api/graph/save", json=_save_as(_graph(name=SKEWED_TITLE)))
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"] == {
        "error": "graph_exists", "file": SKEWED_STEM, "name": "我的模型",
    }


# ── The guard in project mode ────────────────────────────────────────────


async def test_the_guard_sees_the_canonical_half_in_project_mode(
    test_client, project_settings,
):
    """A project stores the logic half as `<stem>.graph.json`, so that is
    what "already there" means there."""
    await _save(test_client, _graph(name=SKEWED_STEM, x=10))
    logic = project_settings / "graphs" / f"{SKEWED_STEM}.graph.json"
    before = logic.read_text(encoding="utf-8")

    resp = await test_client.post(
        "/api/graph/save", json=_save_as(_graph(name=SKEWED_TITLE, x=999)))
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"] == {
        "error": "graph_exists", "file": SKEWED_STEM, "name": SKEWED_STEM,
    }
    assert logic.read_text(encoding="utf-8") == before


async def test_the_guard_sees_a_legacy_single_file_in_project_mode(
    test_client, project_settings,
):
    """A project that has not saved since the split still stores
    `<stem>.json`. That graph is every bit as destroyable as a canonical
    one -- more so, since the save that overwrote it would also unlink it as
    part of the upgrade."""
    legacy = project_settings / "graphs" / f"{SKEWED_STEM}.json"
    legacy.write_text(
        json.dumps(_graph(name="Legacy Title")), encoding="utf-8")

    resp = await test_client.post(
        "/api/graph/save", json=_save_as(_graph(name=SKEWED_TITLE, x=999)))
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"] == {
        "error": "graph_exists", "file": SKEWED_STEM, "name": "Legacy Title",
    }
    assert legacy.exists()
    assert not (
        project_settings / "graphs" / f"{SKEWED_STEM}.graph.json").exists()


async def test_the_guard_does_not_choke_when_both_forms_exist(
    test_client, project_settings,
):
    """Both forms of one base is a state a plain `git checkout` of an older
    commit produces, and /load and /list both refuse it outright. /save does
    not, and must not start to: asking "is anything already there?" has an
    unambiguous answer here (yes), and a save that works today has to keep
    working. So the guard answers, the confirmed re-save goes through, and
    write_graph_pair's upgrade removes the legacy half -- which is the one
    thing that actually resolves the ambiguity."""
    graphs = project_settings / "graphs"
    (graphs / "dup.graph.json").write_text(
        json.dumps(_graph(name="Canonical Title")), encoding="utf-8")
    (graphs / "dup.json").write_text(
        json.dumps(_graph(name="Legacy Title")), encoding="utf-8")

    resp = await test_client.post(
        "/api/graph/save", json=_save_as(_graph(name="dup", x=999)))
    assert resp.status_code == 409, resp.text
    # The structured refusal, NOT GraphAmbiguityError's prose: the canonical
    # half is the one a read would resolve to, so it is the one named.
    assert resp.json()["detail"] == {
        "error": "graph_exists", "file": "dup", "name": "Canonical Title",
    }

    resp = await test_client.post(
        "/api/graph/save",
        json={**_graph(name="dup", x=999), "overwrite": True})
    assert resp.status_code == 200, resp.text
    assert not (graphs / "dup.json").exists()
    assert json.loads(
        (graphs / "dup.graph.json").read_text(encoding="utf-8"))["name"] == "dup"


async def test_the_reserved_name_refusal_still_comes_first(
    test_client, project_settings,
):
    """Order matters, and only a stem that is BOTH reserved and occupied can
    prove it. `weird.graph` would be written to `weird_graph.graph.json`, so
    a graph already sitting there would make the guard fire -- but the name
    is refused outright, and a 400 saying "that name cannot be used" is the
    answer, not a 409 offering to overwrite something with it."""
    (project_settings / "graphs" / "weird_graph.graph.json").write_text(
        json.dumps(_graph(name="Bystander")), encoding="utf-8")

    resp = await test_client.post(
        "/api/graph/save", json=_save_as(_graph(name="weird.graph")))
    assert resp.status_code == 400, resp.text
    assert "reserved" in resp.json()["detail"].lower()
