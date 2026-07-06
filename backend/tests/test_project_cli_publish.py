"""cdui project publish: local-only, refuses on a health.project mismatch,
sends git provenance, warns loudly on a dirty tree (spec 8 / ID4 / ID12)."""

import argparse

import project


def _init_with_publish_defaults(tmp_path):
    target = tmp_path / "svc"
    project.cmd_init(argparse.Namespace(dir=str(target), adopt=None, force=False))
    (target / "codefyui.project.toml").write_text(
        '[project]\nname = "svc"\nformat_version = 1\n\n'
        '[publish]\ngraph = "echo-graph"\nslug = "myslug"\n', encoding="utf-8")
    return target


def _pargs(proj, **kw):
    ns = argparse.Namespace(dir=str(proj), graph=None, slug=None, note=None)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def _patch_common(monkeypatch, proj, *, project_reported=None, provenance=("d" * 40, False)):
    monkeypatch.setattr(project, "_read_session_token", lambda: "tok")
    monkeypatch.setattr(project, "_server_base",
                        lambda: ("http://127.0.0.1:8000", "127.0.0.1:8000"))
    reported = str(proj.resolve()) if project_reported is None else project_reported
    monkeypatch.setattr(project, "_http_get_json",
                        lambda url, host: {"project": reported})
    monkeypatch.setattr("app.core.project.git_provenance", lambda p: provenance)


def test_publish_sends_provenance_and_manifest_defaults(tmp_path, monkeypatch):
    proj = _init_with_publish_defaults(tmp_path)
    _patch_common(monkeypatch, proj)
    posted = {}

    def fake_post(url, host, token, body):
        posted.update(body)
        posted["_url"] = url
        return {"version": 1, "_status": 200}

    monkeypatch.setattr(project, "_http_post_json", fake_post)
    assert project.cmd_publish(_pargs(proj, note="cut1")) == 0
    assert posted["slug"] == "myslug"          # from manifest [publish]
    assert posted["graph"] == "echo-graph"
    assert posted["git_commit"] == "d" * 40
    assert posted["git_dirty"] is False
    assert posted["note"] == "cut1"
    assert posted["_url"].endswith("/api/apps/myslug/publish")


def test_publish_refuses_on_project_mismatch(tmp_path, monkeypatch):
    proj = _init_with_publish_defaults(tmp_path)
    _patch_common(monkeypatch, proj, project_reported="/some/other/project")
    posted = []
    monkeypatch.setattr(project, "_http_post_json",
                        lambda *a, **k: posted.append(1) or {"_status": 200})
    assert project.cmd_publish(_pargs(proj)) == 1
    assert posted == []  # never published against a foreign project


def test_publish_warns_loudly_on_dirty(tmp_path, monkeypatch, capsys):
    proj = _init_with_publish_defaults(tmp_path)
    _patch_common(monkeypatch, proj, provenance=("e" * 40, True))
    monkeypatch.setattr(project, "_http_post_json",
                        lambda *a, **k: {"version": 2, "_status": 200})
    assert project.cmd_publish(_pargs(proj)) == 0
    assert "dirty" in capsys.readouterr().out.lower()


def test_publish_not_a_repo_null_provenance(tmp_path, monkeypatch):
    proj = _init_with_publish_defaults(tmp_path)
    _patch_common(monkeypatch, proj, provenance=(None, None))
    posted = {}
    monkeypatch.setattr(project, "_http_post_json",
                        lambda url, host, token, body: posted.update(body)
                        or {"version": 1, "_status": 200})
    assert project.cmd_publish(_pargs(proj)) == 0
    assert "git_commit" not in posted  # NULL provenance -> field omitted
