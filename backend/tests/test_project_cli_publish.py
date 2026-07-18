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
    ns = argparse.Namespace(dir=str(proj), graph=None, slug=None, note=None,
                            create=False)
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
    # slug targets via the URL path only -- PublishRequest declares no
    # "slug" field, so the body must not smuggle one (issue #86).
    assert "slug" not in posted
    assert posted["_url"].endswith("/api/apps/myslug/publish")
    assert posted["graph"] == "echo-graph"
    # The manifest's committed [publish].slug is a deliberate target:
    # first publish may create the app without extra flags.
    assert posted["create"] is True
    assert posted["git_commit"] == "d" * 40
    assert posted["git_dirty"] is False
    assert posted["note"] == "cut1"


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


def test_publish_unknown_dirty_sends_null(tmp_path, monkeypatch, capsys):
    """git status failed AFTER rev-parse resolved a commit: git_dirty must
    be null (= unknown, the schema's meaning), never a fabricated false
    (issue #86)."""
    proj = _init_with_publish_defaults(tmp_path)
    _patch_common(monkeypatch, proj, provenance=("f" * 40, None))
    posted = {}
    monkeypatch.setattr(project, "_http_post_json",
                        lambda url, host, token, body: posted.update(body)
                        or {"version": 1, "_status": 200})
    assert project.cmd_publish(_pargs(proj)) == 0
    assert posted["git_commit"] == "f" * 40
    assert "git_dirty" in posted and posted["git_dirty"] is None
    # The CLI says so instead of silently pretending the tree is clean.
    assert "unknown" in capsys.readouterr().out.lower()


def test_publish_explicit_slug_does_not_create(tmp_path, monkeypatch):
    """--slug on the command line must NOT auto-create: a typo has to hit
    the server's app_not_found 404 instead of minting a second app."""
    proj = _init_with_publish_defaults(tmp_path)
    _patch_common(monkeypatch, proj)
    posted = {}
    monkeypatch.setattr(project, "_http_post_json",
                        lambda url, host, token, body: posted.update(body)
                        or {"version": 3, "_status": 200})
    assert project.cmd_publish(_pargs(proj, slug="myslug")) == 0
    assert posted["create"] is False


def test_publish_explicit_slug_with_create_flag(tmp_path, monkeypatch):
    proj = _init_with_publish_defaults(tmp_path)
    _patch_common(monkeypatch, proj)
    posted = {}
    monkeypatch.setattr(project, "_http_post_json",
                        lambda url, host, token, body: posted.update(body)
                        or {"version": 1, "_status": 200})
    assert project.cmd_publish(_pargs(proj, slug="brand-new", create=True)) == 0
    assert posted["create"] is True


def test_publish_app_not_found_hints_create_flag(tmp_path, monkeypatch, capsys):
    """The server's 404 talks JSON ('pass \"create\": true'); the CLI must
    translate that to its own flag so the fix is discoverable."""
    proj = _init_with_publish_defaults(tmp_path)
    _patch_common(monkeypatch, proj)
    monkeypatch.setattr(
        project, "_http_post_json",
        lambda *a, **k: {"_status": 404, "detail": {
            "code": "app_not_found",
            "message": "app 'my-svc' does not exist", "details": None}})
    assert project.cmd_publish(_pargs(proj, slug="my-svc")) == 1
    captured = capsys.readouterr()
    assert "--create" in (captured.out + captured.err)


# -- Error paths before/around the HTTP hop (issue #88), monkeypatched at the
# -- same _read_session_token/_server_base/_http_* module boundaries as above.
# -- Check order under test: token -> health(GET) -> publish(POST).


def test_publish_missing_token_makes_no_http_calls(tmp_path, monkeypatch,
                                                   capsys):
    """No session token (server not running): fail BEFORE any network I/O."""
    proj = _init_with_publish_defaults(tmp_path)
    monkeypatch.setattr(project, "_read_session_token", lambda: None)
    calls = []
    monkeypatch.setattr(project, "_http_get_json",
                        lambda *a, **k: calls.append("get") or {"project": ""})
    monkeypatch.setattr(project, "_http_post_json",
                        lambda *a, **k: calls.append("post") or {"_status": 200})
    assert project.cmd_publish(_pargs(proj)) == 1
    assert calls == []  # neither the health GET nor the publish POST fired
    captured = capsys.readouterr()
    # Both locales name the missing session token.
    assert "token" in (captured.out + captured.err).lower()


def test_publish_health_unreachable_never_posts(tmp_path, monkeypatch, capsys):
    """/api/health unreachable: rc 1 naming the URL, and the publish POST is
    never attempted."""
    proj = _init_with_publish_defaults(tmp_path)
    monkeypatch.setattr(project, "_read_session_token", lambda: "tok")
    monkeypatch.setattr(project, "_server_base",
                        lambda: ("http://127.0.0.1:8000", "127.0.0.1:8000"))
    monkeypatch.setattr(project, "_http_get_json", lambda url, host: None)
    posted = []
    monkeypatch.setattr(project, "_http_post_json",
                        lambda *a, **k: posted.append(1) or {"_status": 200})
    assert project.cmd_publish(_pargs(proj)) == 1
    assert posted == []
    captured = capsys.readouterr()
    assert "/api/health" in (captured.out + captured.err)


def test_publish_non_200_post_reports_detail(tmp_path, monkeypatch, capsys):
    """A non-200 publish response (e.g. a 409 pre-flight refusal) is rc 1
    and the server's detail is echoed so CI logs say WHY."""
    proj = _init_with_publish_defaults(tmp_path)
    _patch_common(monkeypatch, proj)
    monkeypatch.setattr(
        project, "_http_post_json",
        lambda *a, **k: {"_status": 409, "detail": "secret_in_graph boom"})
    assert project.cmd_publish(_pargs(proj)) == 1
    captured = capsys.readouterr()
    assert "secret_in_graph boom" in (captured.out + captured.err)
