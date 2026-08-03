"""Tests for `cdui run` — the CLI that submits a graph to the server (#123).

The command is a REST CLIENT, so it is tested at the transport boundary:
``dev.urlopen`` is replaced with a scripted fake (the ``test_status_dashboard``
precedent) and the tests assert what went onto the wire, what came back off
it, and what the process exits with.

One test deliberately goes further and pushes the CLI's own envelope through
the SERVER's validators (``SubmitRunRequest`` + ``normalize_options``). That
is the drift this file exists to catch: a CLI that builds an envelope the API
rejects fails at the worst possible moment — on a user's machine, after they
typed the command — and a mock that agrees with the client cannot see it.
"""

from __future__ import annotations

import json
import sys
from urllib.error import HTTPError, URLError

import pytest

import dev  # scripts/dev.py — put on sys.path by conftest

from app.api.routes_runs import SubmitRunRequest
from app.core.run_service import normalize_graph, normalize_name, normalize_options

BASE = "http://127.0.0.1:8000"


# ── plumbing ──────────────────────────────────────────────────────────────


class _FakeResp:
    def __init__(self, status: int, payload) -> None:
        self.status = status
        self._body = (b"" if payload is None
                      else json.dumps(payload).encode("utf-8"))

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeServer:
    """Answers by URL substring, in the order the answers were registered.

    A list per route rather than one response, because ``--wait`` polls the
    same URL repeatedly and the interesting behaviour is what the CLI does
    ACROSS those polls. The last answer repeats once the script runs out, so a
    test only scripts the steps it cares about.
    """

    def __init__(self) -> None:
        self.routes: dict[str, list] = {}
        self.requests: list = []

    def on(self, pattern: str, *responses) -> "_FakeServer":
        """Register answers. A leading ``=`` means "URL ENDS with this".

        Submit (``/api/runs``), detail (``/api/runs/r1``) and events
        (``/api/runs/r1/events?...``) share a prefix, so substring matching
        alone would route them to each other. Registration order decides,
        and ``=`` is what makes the collision-free case expressible.
        """
        self.routes.setdefault(pattern, []).extend(responses)
        return self

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        url = req.full_url if hasattr(req, "full_url") else str(req)
        for pattern, answers in self.routes.items():
            hit = (url.endswith(pattern[1:]) if pattern.startswith("=")
                   else pattern in url)
            if hit:
                status, payload = (answers.pop(0) if len(answers) > 1
                                   else answers[0])
                # BaseException, not Exception: KeyboardInterrupt is one of
                # the things a test needs to be able to inject here, and it
                # is deliberately outside the Exception hierarchy.
                if isinstance(payload, BaseException):
                    raise payload
                return _FakeResp(status, payload)
        raise AssertionError(f"unexpected request to {url}")


@pytest.fixture(autouse=True)
def plain_output(monkeypatch, tmp_path):
    """English, no ANSI, and every filesystem touch inside tmp_path."""
    monkeypatch.setattr(dev, "LANG", "en")
    monkeypatch.setattr(dev, "USE_COLOR", False)
    for name in ("RESET", "BOLD", "DIM", "RED", "GREEN", "YELLOW", "BLUE",
                 "MAGENTA", "CYAN", "GRAY"):
        monkeypatch.setattr(dev, name, "")
    # _apply_dev_env() mkdirs this and setdefault()s the env var; both must
    # land in tmp_path, never in the developer's real dev clone.
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path / "userdata"))
    monkeypatch.setattr(dev, "DEV_USER_DATA_DIR", tmp_path / "userdata")
    monkeypatch.setattr(dev, "SERVER_ADDRFILE", tmp_path / "server.addr")
    monkeypatch.setattr(dev, "_session_token", lambda: "test-token")


@pytest.fixture
def graph_file(tmp_path):
    path = tmp_path / "g.json"
    path.write_text(json.dumps({
        "nodes": [{"id": "start", "type": "Start", "data": {"params": {}}}],
        "edges": [],
    }), encoding="utf-8")
    return path


def _argv(graph_file, *flags):
    return ["cdui", "run", str(graph_file), *flags]


def _events(*types, status: str, cursor: int = 1):
    return {"run_id": "r1", "status": status, "active": True,
            "events": [{"cursor": cursor + i, "type": kind, "payload": {},
                        "ts": "now"} for i, kind in enumerate(types)],
            "cursor": cursor + len(types) - 1}


# ── the submit envelope ───────────────────────────────────────────────────


def test_submit_body_maps_the_flags(graph_file):
    args = dev._parse_run_args([str(graph_file), "--name", "nightly",
                                "--device", "cuda:0", "--seed", "7",
                                "--record-outputs"])
    body = dev._run_submit_body(args)
    assert body["name"] == "nightly"
    assert body["options"] == {"device": "cuda:0", "record_outputs": True,
                               "seed": 7}
    assert body["graph"]["nodes"][0]["type"] == "Start"


def test_submit_body_omits_what_was_not_asked_for(graph_file):
    body = dev._run_submit_body(dev._parse_run_args([str(graph_file)]))
    assert "name" not in body                 # unnamed, not an empty string
    assert "seed" not in body["options"]      # null seed, not seed 0
    # The lane is the SERVER's default. Naming it here would hardcode a
    # policy the CLI has no opinion about.
    assert "lane" not in body["options"]
    # `auto` is what the signature promises; the server maps it to cpu, and
    # the CLI does not second-guess that mapping locally.
    assert body["options"]["device"] == "auto"


def test_deterministic_flag_reaches_the_options(graph_file):
    """core#134: the other half of a reproducible run."""
    args = dev._parse_run_args([str(graph_file), "--seed", "7",
                                "--deterministic"])
    body = dev._run_submit_body(args)
    assert body["options"]["deterministic"] is True
    # And it survives the server's own validator, which rejects unknown keys.
    assert normalize_options(body["options"])["deterministic"] is True


def test_deterministic_is_omitted_when_not_asked_for(graph_file):
    body = dev._run_submit_body(dev._parse_run_args([str(graph_file)]))
    assert "deterministic" not in body["options"]


def test_submit_body_is_accepted_by_the_server_contract(graph_file):
    """The envelope the CLI builds must survive the API's own validators."""
    args = dev._parse_run_args([str(graph_file), "--name", "nightly",
                                "--device", "cuda:0", "--seed", "7"])
    body = dev._run_submit_body(args)

    request = SubmitRunRequest(**body)
    options = normalize_options(request.options)
    assert options["device"] == "cuda:0" and options["seed"] == 7
    assert options["lane"] == "queued"        # the queue's lane, by default
    assert normalize_name(request.name) == "nightly"
    assert normalize_graph(request.graph)["nodes"]


def test_every_device_the_cli_advertises_is_one_the_server_accepts():
    """`--device`'s help text and DEVICE_PATTERN must not drift apart."""
    for device in ("cpu", "auto", "cuda", "cuda:1", "mps"):
        assert normalize_options({"device": device})["device"] == device


def test_detach_and_wait_are_mutually_exclusive(graph_file, capsys):
    assert dev._parse_run_args([str(graph_file)]).wait is True
    assert dev._parse_run_args([str(graph_file), "--detach"]).wait is False
    with pytest.raises(SystemExit):
        dev._parse_run_args([str(graph_file), "--wait", "--detach"])


# ── the HTTP layer ────────────────────────────────────────────────────────


def test_api_request_authenticates_and_sets_the_host(monkeypatch):
    server = _FakeServer().on("/api/runs", (200, {"run_id": "r1"}))
    monkeypatch.setattr(dev, "urlopen", server)

    status, body = dev._api_request(f"{BASE}/api/runs", "127.0.0.1:8000",
                                    token="tok", body={"graph": {}})
    assert (status, body) == (200, {"run_id": "r1"})
    request = server.requests[0]
    assert request.method == "POST"
    assert request.get_header("X-codefyui-token") == "tok"
    # The Host whitelist is why this header is explicit rather than implied.
    assert request.get_header("Host") == "127.0.0.1:8000"
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(request.data)["graph"] == {}


def test_api_request_returns_the_servers_own_error_detail(monkeypatch):
    error = HTTPError(f"{BASE}/api/runs", 400, "Bad Request", {},
                      _Body(json.dumps({"detail": "graph.nodes is empty"})))
    monkeypatch.setattr(dev, "urlopen", _FakeServer().on("/api/runs",
                                                         (0, error)))
    status, body = dev._api_request(f"{BASE}/api/runs", "h", body={})
    assert status == 400
    assert body["detail"] == "graph.nodes is empty"


def test_api_request_reports_a_dead_server_as_zero(monkeypatch):
    monkeypatch.setattr(dev, "urlopen", _FakeServer().on(
        "/api/runs", (0, URLError("connection refused"))))
    assert dev._api_request(f"{BASE}/api/runs", "h") == (0, None)


class _Body:
    """Minimal file-like for HTTPError's body argument."""

    def __init__(self, text: str) -> None:
        self._text = text.encode("utf-8")

    def read(self) -> bytes:
        return self._text


# ── --detach ──────────────────────────────────────────────────────────────


def test_detach_submits_and_returns_without_tailing(monkeypatch, graph_file,
                                                    capsys):
    server = _FakeServer().on("/api/runs",
                              (200, {"run_id": "r1", "status": "queued"}))
    monkeypatch.setattr(dev, "urlopen", server)
    monkeypatch.setattr(sys, "argv", _argv(graph_file, "--detach"))

    dev.run_graph()                            # no SystemExit == exit 0

    out = capsys.readouterr().out
    assert "r1" in out and "queued" in out
    assert len(server.requests) == 1           # submitted, never polled


# ── --wait ────────────────────────────────────────────────────────────────


def test_wait_streams_progress_and_exits_zero(monkeypatch, graph_file, capsys):
    server = (_FakeServer()
              .on("=/api/runs", (200, {"run_id": "r1", "status": "running"}))
              .on("/events",
                  (200, _events("execution_start", "node_status",
                                status="running")),
                  (200, {"run_id": "r1", "status": "succeeded", "active": False,
                         "events": [{"cursor": 3, "type": "execution_complete",
                                     "payload": {}, "ts": "now"}],
                         "cursor": 3})))
    monkeypatch.setattr(dev, "urlopen", server)
    monkeypatch.setattr(sys, "argv", _argv(graph_file))

    dev.run_graph()

    out = capsys.readouterr().out
    assert "started" in out
    assert "run complete" in out
    assert "succeeded" in out
    # The cursor advances, so the second poll asks for what it has not seen.
    assert "cursor=0" in server.requests[1].full_url
    assert "cursor=2" in server.requests[2].full_url


def test_wait_exits_one_on_a_failed_run(monkeypatch, graph_file, capsys):
    server = (_FakeServer()
              .on("=/api/runs", (200, {"run_id": "r1", "status": "running"}))
              .on("/events", (200, {
                  "run_id": "r1", "status": "failed", "active": False,
                  "events": [{"cursor": 1, "type": "execution_error",
                              "payload": {"error": "boom"}, "ts": "now"}],
                  "cursor": 1})))
    monkeypatch.setattr(dev, "urlopen", server)
    monkeypatch.setattr(sys, "argv", _argv(graph_file))

    with pytest.raises(SystemExit) as exc:
        dev.run_graph()
    assert exc.value.code == 1
    assert "boom" in capsys.readouterr().out


@pytest.mark.parametrize("status", ["cancelled", "interrupted"])
def test_wait_exits_one_when_the_run_was_stopped(monkeypatch, graph_file,
                                                 capsys, status):
    """A run that was stopped did not do what was asked; 0 would be a lie."""
    server = (_FakeServer()
              .on("=/api/runs", (200, {"run_id": "r1", "status": "running"}))
              .on("/events", (200, {
                  "run_id": "r1", "status": status, "active": False,
                  "events": [{"cursor": 1, "type": "execution_stopped",
                              "payload": {"reason": status}, "ts": "now"}],
                  "cursor": 1})))
    monkeypatch.setattr(dev, "urlopen", server)
    monkeypatch.setattr(sys, "argv", _argv(graph_file))

    with pytest.raises(SystemExit) as exc:
        dev.run_graph()
    assert exc.value.code == 1
    assert status in capsys.readouterr().out


def test_wait_reports_the_queue_position_once(monkeypatch, graph_file, capsys):
    """A queued run is not a hung run, and the CLI has to say so."""
    server = (_FakeServer()
              .on("=/api/runs", (200, {"run_id": "r1", "status": "queued"}))
              .on("/events",
                  (200, {"run_id": "r1", "status": "queued", "active": False,
                         "events": [], "cursor": 0}),
                  (200, {"run_id": "r1", "status": "queued", "active": False,
                         "events": [], "cursor": 0}),
                  (200, {"run_id": "r1", "status": "succeeded",
                         "active": False, "events": [], "cursor": 0}))
              .on("/api/runs/r1", (200, {"id": "r1", "status": "queued",
                                         "queue_position": 3,
                                         "queue_key": "cuda:0"})))
    monkeypatch.setattr(dev, "urlopen", server)
    monkeypatch.setattr(sys, "argv", _argv(graph_file))

    dev.run_graph()

    out = capsys.readouterr().out
    assert out.count("position 3 on cuda:0") == 1   # announced, not repeated


def test_wait_gives_up_at_the_timeout_without_killing_the_run(
        monkeypatch, graph_file, capsys):
    server = (_FakeServer()
              .on("=/api/runs", (200, {"run_id": "r1", "status": "running"}))
              .on("/events", (200, {"run_id": "r1", "status": "running",
                                    "active": True, "events": [],
                                    "cursor": 0})))
    monkeypatch.setattr(dev, "urlopen", server)
    monkeypatch.setattr(sys, "argv", _argv(graph_file, "--timeout", "0.01"))

    with pytest.raises(SystemExit) as exc:
        dev.run_graph()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "stopped waiting" in captured.err
    assert "the run continues on the server" in captured.err
    # No cancel was sent: a client giving up must not stop a server-owned run.
    assert not any("/cancel" in r.full_url for r in server.requests)


def test_ctrl_c_while_waiting_leaves_the_run_alone_and_exits_130(
        monkeypatch, graph_file, capsys):
    """Ctrl+C stops WATCHING. It must not print a socket traceback over the
    progress output, and it must not read as a failed run."""
    server = (_FakeServer()
              .on("=/api/runs", (200, {"run_id": "r1", "status": "running"}))
              .on("/events", (0, KeyboardInterrupt())))
    monkeypatch.setattr(dev, "urlopen", server)
    monkeypatch.setattr(sys, "argv", _argv(graph_file))

    with pytest.raises(SystemExit) as exc:
        dev.run_graph()
    assert exc.value.code == dev.EXIT_INTERRUPTED == 130
    captured = capsys.readouterr()
    assert "stopped following" in captured.err
    assert "the run continues on the server" in captured.err
    # No cancel was sent: interrupting the client must not stop the run.
    assert not any("/cancel" in r.full_url for r in server.requests)


def test_a_terminal_page_is_drained_so_the_error_text_prints(monkeypatch,
                                                             graph_file,
                                                             capsys):
    """A page is byte-bounded, so the poll that first sees the terminal
    status can be cut short — often right before the traceback."""
    server = (_FakeServer()
              .on("=/api/runs", (200, {"run_id": "r1", "status": "running"}))
              .on("/events",
                  # Terminal status, but the page stopped before the error.
                  (200, {"run_id": "r1", "status": "failed", "active": False,
                         "events": [{"cursor": 1, "type": "node_status",
                                     "payload": {"node_id": "n1",
                                                 "status": "completed"},
                                     "ts": "now"}],
                         "cursor": 1}),
                  (200, {"run_id": "r1", "status": "failed", "active": False,
                         "events": [{"cursor": 2, "type": "execution_error",
                                     "payload": {"error": "the real reason"},
                                     "ts": "now"}],
                         "cursor": 2}),
                  (200, {"run_id": "r1", "status": "failed", "active": False,
                         "events": [], "cursor": 2})))
    monkeypatch.setattr(dev, "urlopen", server)
    monkeypatch.setattr(sys, "argv", _argv(graph_file))

    with pytest.raises(SystemExit) as exc:
        dev.run_graph()
    assert exc.value.code == 1
    assert "the real reason" in capsys.readouterr().out
    # The drain does not long-poll: the log is already final.
    assert any("wait=0" in r.full_url for r in server.requests)


def test_the_drain_is_bounded(monkeypatch, graph_file, capsys):
    """A server that never stops producing must not park the command."""
    server = (_FakeServer()
              .on("=/api/runs", (200, {"run_id": "r1", "status": "running"}))
              .on("/events", (200, {
                  "run_id": "r1", "status": "succeeded", "active": False,
                  "events": [{"cursor": 1, "type": "node_status",
                              "payload": {"node_id": "n", "status": "completed"},
                              "ts": "now"}],
                  "cursor": 1})))
    monkeypatch.setattr(dev, "urlopen", server)
    monkeypatch.setattr(sys, "argv", _argv(graph_file))

    dev.run_graph()
    polls = [r for r in server.requests if "/events" in r.full_url]
    assert len(polls) <= dev.RUN_DRAIN_MAX_PAGES + 2


def test_a_dropped_connection_while_tailing_does_not_crash(monkeypatch,
                                                           graph_file, capsys):
    server = (_FakeServer()
              .on("=/api/runs", (200, {"run_id": "r1", "status": "running"}))
              .on("/events", (0, URLError("gone"))))
    monkeypatch.setattr(dev, "urlopen", server)
    monkeypatch.setattr(sys, "argv", _argv(graph_file))

    with pytest.raises(SystemExit) as exc:
        dev.run_graph()
    assert exc.value.code == 1
    assert "lost the connection" in capsys.readouterr().err


# ── failure before anything is submitted ──────────────────────────────────


def test_a_missing_graph_file_exits_one_before_any_request(monkeypatch,
                                                           tmp_path, capsys):
    calls = []
    monkeypatch.setattr(dev, "urlopen", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(sys, "argv", ["cdui", "run", str(tmp_path / "nope.json")])

    with pytest.raises(SystemExit) as exc:
        dev.run_graph()
    assert exc.value.code == 1
    assert "no such graph file" in capsys.readouterr().err
    assert calls == []


def test_unparseable_json_exits_one(monkeypatch, tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(dev, "urlopen",
                        lambda *a, **k: pytest.fail("must not reach the server"))
    monkeypatch.setattr(sys, "argv", ["cdui", "run", str(bad)])

    with pytest.raises(SystemExit) as exc:
        dev.run_graph()
    assert exc.value.code == 1
    assert "could not read the graph file" in capsys.readouterr().err


def test_a_missing_token_says_the_server_is_not_running(monkeypatch,
                                                        graph_file, capsys):
    monkeypatch.setattr(dev, "_session_token", lambda: None)
    monkeypatch.setattr(dev, "urlopen",
                        lambda *a, **k: pytest.fail("must not reach the server"))
    monkeypatch.setattr(sys, "argv", _argv(graph_file))

    with pytest.raises(SystemExit) as exc:
        dev.run_graph()
    assert exc.value.code == 1
    assert "is the server running" in capsys.readouterr().err


def test_a_rejected_submit_shows_the_servers_reason(monkeypatch, graph_file,
                                                    capsys):
    error = HTTPError(f"{BASE}/api/runs", 400, "Bad Request", {},
                      _Body(json.dumps({"detail": "graph.nodes is empty"})))
    monkeypatch.setattr(dev, "urlopen",
                        _FakeServer().on("/api/runs", (0, error)))
    monkeypatch.setattr(sys, "argv", _argv(graph_file))

    with pytest.raises(SystemExit) as exc:
        dev.run_graph()
    assert exc.value.code == 1
    assert "graph.nodes is empty" in capsys.readouterr().err


def test_an_unreachable_server_names_the_address(monkeypatch, graph_file,
                                                 capsys):
    monkeypatch.setattr(dev, "urlopen", _FakeServer().on(
        "/api/runs", (0, URLError("refused"))))
    monkeypatch.setattr(sys, "argv",
                        _argv(graph_file, "--port", "9123"))

    with pytest.raises(SystemExit) as exc:
        dev.run_graph()
    assert exc.value.code == 1
    assert "http://127.0.0.1:9123" in capsys.readouterr().err


# ── rendering ─────────────────────────────────────────────────────────────


def test_progress_renders_counters_before_metrics():
    line = dev._format_progress({"epoch": 3, "total_epochs": 10,
                                 "loss": 0.12345678, "event": "epoch",
                                 "note": "ignored", "flag": True})
    assert line == "epoch 3/10  loss=0.1235"


def test_progress_with_nothing_to_say_renders_nothing():
    assert dev._format_progress({"event": "batch"}) == ""


def test_node_status_lines_use_ascii_and_the_house_glyphs(capsys):
    for status in ("completed", "cached", "skipped", "error", "progress"):
        dev._render_node_status({"node_id": "n1", "status": status,
                                 "error": "bad", "epoch": 1})
    out = capsys.readouterr().out
    assert "n1" in out
    # The approved functional glyphs only — no pictographic emoji anywhere.
    assert not any(ord(ch) > 0x2800 for ch in out)


def test_run_command_is_registered_under_its_own_name():
    """`run` is taken by the subprocess helper; the command is run_graph."""
    assert dev.COMMANDS["run"] is dev.run_graph
    assert dev.run is not dev.run_graph
    # It needs the venv (platformdirs), so it must not skip the venv hop.
    assert "run" not in dev._SKIP_VENV_EXEC
