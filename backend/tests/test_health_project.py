"""/api/health's additive `project` field (spec ID4), and the shape it sits in.

The original version of this file claimed a non-project response body "stays
identical to pre-Task-10 main.py". That has not been true for some time and
was never going to stay true: `version` was added unconditionally and openly
("this is a new capability rather than a refactor" -- see `health()` in
`app/main.py`), and `caches` followed for #135. Repeating a dead byte-for-byte
guarantee in a docstring is worse than not having one, because it reads as an
invariant somebody is enforcing (#193).

What IS still load-bearing, and what the tests below actually check:

* `project` must be ABSENT, not null, outside project mode. The frontend
  normalises with `data.project ?? null` and `isProjectMode` keys off a
  strict `!== null`, so a literal `null` would work but `undefined` from a
  MISSING key is what the normalisation exists to convert -- and the
  distinction is only testable from this side.
* The top-level key set is fixed. Additive is a decision, not a drift: a new
  key arriving here should break this test once, on purpose, and be added to
  the list by the person who added it.
"""

from app.api import routes_graph  # for the shared settings object

#: Every top-level key `/api/health` returns outside project mode. `project`
#: is the one conditional key and is asserted separately below.
_BASE_KEYS = {"status", "version", "boot_id", "nodes_loaded",
              "presets_loaded", "caches"}


async def test_health_project_key_absent_when_unset(test_client, monkeypatch):
    monkeypatch.setattr(routes_graph.settings, "PROJECT_DIR", None)
    r = await test_client.get("/api/health")
    body = r.json()
    assert body["status"] == "ok"
    assert "project" not in body


async def test_health_project_dir_in_project_mode(test_client, monkeypatch, tmp_path):
    monkeypatch.setattr(routes_graph.settings, "PROJECT_DIR", tmp_path)
    r = await test_client.get("/api/health")
    assert r.json()["project"] == str(tmp_path)


async def test_health_body_carries_exactly_the_documented_keys(
    test_client, monkeypatch, tmp_path,
):
    """The replacement for the dead "identical to pre-Task-10" claim (#193).

    Both modes, because the whole point of `project` being conditional is
    that it is the ONLY difference between them.
    """
    settings = routes_graph.settings

    monkeypatch.setattr(settings, "PROJECT_DIR", None)
    assert set((await test_client.get("/api/health")).json()) == _BASE_KEYS

    monkeypatch.setattr(settings, "PROJECT_DIR", tmp_path)
    assert set((await test_client.get("/api/health")).json()) == (
        _BASE_KEYS | {"project"})
