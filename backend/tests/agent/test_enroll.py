"""Agent-side enrollment (#780) against a fake control plane.

The fake is a plain ``http.server`` answering ``POST /api/workers/enroll``
the way ``routes_workers.enroll`` does, so these tests pin the agent's half
of the contract: what it sends, what it stores, when it reuses the stored
credential and when it enrolls again.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest

from app.agent import cli
from app.agent import enroll as agent_enroll

ENROLL_TOKEN = "abcdef.0123456789abcdef"


class FakeControlPlane:
    """Records each enrollment; answers with ``status`` and ``body``."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.status = 200
        self.body: dict[str, Any] = {"worker_id": "gpu-box",
                                     "worker_token": "w" * 43}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 -- http.server's name
                length = int(self.headers.get("Content-Length", 0))
                outer.requests.append({
                    "path": self.path,
                    "host": self.headers.get("Host"),
                    "session": self.headers.get("X-CodefyUI-Token"),
                    "body": json.loads(self.rfile.read(length))})
                payload = json.dumps(outer.body).encode()
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self) -> "FakeControlPlane":
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def cp():
    with FakeControlPlane() as server:
        yield server


@pytest.fixture
def user_data(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "_REPO_ROOT", tmp_path / "no-repo")
    monkeypatch.delenv(cli.TOKEN_ENV, raising=False)
    monkeypatch.setenv("CODEFYUI_LANG", "en")
    monkeypatch.setattr(cli, "_interactive", lambda: False)
    monkeypatch.setattr(cli, "_probe_facts", lambda: [SimpleNamespace(
        value="cpu", kind="cpu", index=None, name="CPU", total_bytes=None,
        free_bytes=None, source="none", uuid=None)])
    return tmp_path


@pytest.fixture
def started(monkeypatch):
    """Replace the agent loop: record what it would have connected with."""
    calls: list[dict[str, Any]] = []

    def fake_run_agent(args, control, token, facts, token_source=None):
        calls.append({"control": control, "token": token, "name": args.name,
                      "token_source": token_source})
        return 0

    monkeypatch.setattr(cli, "run_agent", fake_run_agent)
    monkeypatch.setattr(cli, "_setup_logging", lambda daemonized: None)
    return calls


def _start(*extra: str) -> int:
    return cli.main(["start", "--foreground", *extra])


# ── the module ────────────────────────────────────────────────────────────


def test_urls_normalize_so_one_control_plane_has_one_file(tmp_path):
    assert agent_enroll.normalize_url("HTTP://Box.LAN:8000/") == (
        "http://box.lan:8000")
    assert agent_enroll.normalize_url("https://cp.example") == (
        "https://cp.example:443")
    assert agent_enroll.normalize_url("http://[::1]:9/x/") == (
        "http://[::1]:9/x")
    with pytest.raises(ValueError):
        agent_enroll.normalize_url("box:8000")
    assert agent_enroll.credential_path(tmp_path, "http://Box.lan:8000/") == (
        tmp_path / "agent" / "box.lan-8000.token")


def test_enroll_sends_the_facts_and_returns_the_worker_token(cp):
    worker_id, token = agent_enroll.enroll(
        cp.url + "/", f" {ENROLL_TOKEN} ", name="gpu-box",
        platform={"system": "Linux"}, devices=[{"value": "cpu"}])
    assert (worker_id, token) == ("gpu-box", "w" * 43)
    [request] = cp.requests
    assert request["path"] == "/api/workers/enroll"
    assert request["session"] is None
    assert request["body"] == {"token": ENROLL_TOKEN, "name": "gpu-box",
                               "platform": {"system": "Linux"},
                               "devices": [{"value": "cpu"}], "mounts": {}}


@pytest.mark.parametrize("status, detail, expected", [
    (401, "enrollment token abcdef has already been used",
     "HTTP 401: enrollment token abcdef has already been used"),
    (421, "Misdirected Request", "CODEFYUI_EXTRA_ALLOWED_HOSTS"),
    (429, "too many enrollment attempts", "HTTP 429: too many"),
])
def test_a_refusal_carries_the_servers_reason(cp, status, detail, expected):
    cp.status, cp.body = status, {"detail": detail}
    with pytest.raises(agent_enroll.EnrollError) as refused:
        agent_enroll.enroll(cp.url, ENROLL_TOKEN, name="x", platform={},
                            devices=[])
    assert expected in str(refused.value)


def test_bad_answers_and_no_server_are_enroll_errors(cp):
    cp.body = {"nothing": True}
    with pytest.raises(agent_enroll.EnrollError, match="without a worker"):
        agent_enroll.enroll(cp.url, ENROLL_TOKEN, name="x", platform={},
                            devices=[])
    port = cp.server.server_address[1]
    cp.server.shutdown()
    cp.server.server_close()
    with pytest.raises(agent_enroll.EnrollError, match="could not reach"):
        agent_enroll.enroll(f"http://127.0.0.1:{port}", ENROLL_TOKEN,
                            name="x", platform={}, devices=[], timeout=2)


def test_the_credential_is_private_and_bound_to_its_url(tmp_path):
    path = agent_enroll.save_credential(tmp_path, "http://box:8000/",
                                        "gpu-box", "secret")
    if sys.platform != "win32":
        assert os.stat(path).st_mode & 0o777 == 0o600
        assert os.stat(path.parent).st_mode & 0o777 == 0o700
    stored = agent_enroll.load_credential(tmp_path, "HTTP://BOX:8000")
    assert stored is not None
    assert (stored.worker_id, stored.worker_token) == ("gpu-box", "secret")
    assert json.loads(path.read_text())["control_url"] == "http://box:8000"
    # Another control plane on the same host and port, under a path prefix:
    # same file name, different URL, so not this credential.
    assert agent_enroll.load_credential(tmp_path, "http://box:8000/sub") is None
    path.write_text("not json")
    assert agent_enroll.load_credential(tmp_path, "http://box:8000") is None
    assert agent_enroll.forget_credential(tmp_path, "http://box:8000") == path
    assert agent_enroll.forget_credential(tmp_path, "http://box:8000") is None


# ── cdui agent start / forget ─────────────────────────────────────────────


def test_start_enrolls_once_then_reuses_the_stored_credential(
        cp, user_data, started, capsys):
    assert _start("--control", cp.url, "--token", ENROLL_TOKEN,
                  "--name", "gpu-box") == 0
    assert len(cp.requests) == 1
    assert started[-1]["token"] == "w" * 43
    assert started[-1]["name"] == "gpu-box"
    assert "cdui agent forget" in started[-1]["token_source"]
    stored = agent_enroll.load_credential(user_data, cp.url)
    assert stored is not None and stored.worker_token == "w" * 43

    # A restart: no --token, no prompt, no second enrollment.
    assert _start("--control", cp.url) == 0
    assert len(cp.requests) == 1
    assert started[-1]["token"] == "w" * 43
    assert "stored" in started[-1]["token_source"]

    # The same command again: the stored credential wins, and says so.
    capsys.readouterr()
    assert _start("--control", cp.url, "--token", ENROLL_TOKEN) == 0
    assert len(cp.requests) == 1
    assert "cdui agent forget" in capsys.readouterr().out

    # Forget, and the next start enrolls again.
    assert cli.main(["forget", cp.url]) == 0
    assert agent_enroll.load_credential(user_data, cp.url) is None
    assert _start("--control", cp.url, "--token", ENROLL_TOKEN,
                  "--name", "gpu-box") == 0
    assert len(cp.requests) == 2


def test_the_server_names_the_worker(cp, user_data, started):
    cp.body = {"worker_id": "renamed", "worker_token": "t2"}
    assert _start("--control", cp.url, "--token", ENROLL_TOKEN,
                  "--name", "gpu-box") == 0
    assert started[-1]["name"] == "renamed"


def test_a_stored_credential_refuses_another_name(cp, user_data, started,
                                                  capsys):
    agent_enroll.save_credential(user_data, cp.url, "gpu-box", "secret")
    assert _start("--control", cp.url, "--name", "other") == 2
    assert "cdui agent forget" in capsys.readouterr().err
    assert started == []
    # The stored name itself is fine.
    assert _start("--control", cp.url, "--name", "gpu-box") == 0


def test_a_refused_enrollment_fails_the_start(cp, user_data, started, capsys):
    cp.status, cp.body = 401, {"detail": "already been used"}
    assert _start("--control", cp.url, "--token", ENROLL_TOKEN) == 1
    assert "already been used" in capsys.readouterr().err
    assert started == []
    assert agent_enroll.load_credential(user_data, cp.url) is None


def test_a_session_token_is_never_sent_to_enroll(cp, user_data, started):
    assert _start("--control", cp.url, "--token", "session-token") == 0
    assert cp.requests == []
    assert started[-1]["token"] == "session-token"


def test_the_background_agent_never_enrolls(cp, user_data, started, capsys):
    assert _start("--control", cp.url, "--token", ENROLL_TOKEN,
                  "--daemonized") == 2
    assert cp.requests == []
    assert "enrollment token" in capsys.readouterr().err


def test_a_prompted_enrollment_token_enrolls(cp, user_data, started,
                                             monkeypatch):
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: ENROLL_TOKEN)
    resolve = cli.resolve_token
    monkeypatch.setattr(cli, "resolve_token", lambda explicit, interactive: (
        resolve(explicit, interactive=interactive, ask=lambda p: ENROLL_TOKEN)))
    assert _start("--control", cp.url, "--name", "gpu-box") == 0
    assert len(cp.requests) == 1
    assert started[-1]["token"] == "w" * 43


def test_forget_without_a_credential_and_with_a_bad_url(user_data, capsys):
    assert cli.main(["forget", "http://nowhere:1"]) == 0
    assert "No credential" in capsys.readouterr().out
    assert cli.main(["forget", "nowhere"]) == 2


def test_forget_leaves_another_control_planes_credential(tmp_path):
    path = agent_enroll.save_credential(tmp_path, "http://box:8000/sub",
                                        "gpu-box", "secret")
    assert agent_enroll.forget_credential(tmp_path, "http://box:8000") is None
    assert path.exists()
    assert agent_enroll.forget_credential(tmp_path,
                                          "http://BOX:8000/sub/") == path
    assert not path.exists()


def test_a_running_agent_is_reported_before_any_enrollment(cp, user_data,
                                                          monkeypatch,
                                                          capsys):
    monkeypatch.setattr(cli, "running_agent_pid", lambda: 4242)
    assert cli.main(["start", "--control", cp.url, "--token",
                     ENROLL_TOKEN]) == 0
    assert cp.requests == []
    assert "already running (PID 4242)" in capsys.readouterr().out
