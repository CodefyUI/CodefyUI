"""The agent's verification of the control plane over TLS (#786).

Self-signed certificates are made here with ``trustme`` (a test-only
dependency): an HTTPS server answers the enrollment request and a ``wss``
server speaks ``/ws/worker``, both from one server ``SSLContext`` whose
certificate a test can rotate (``load_cert_chain`` again; new handshakes
get the new key). The same :class:`ControlPlaneTrust` serves both paths.

The model (#786 review): a pin is only ever the user's (``--ca-fingerprint``)
and matches the leaf key only; without one the system CAs decide and
nothing is recorded; an untrusted certificate without a pin is retried, a
pin mismatch is final.
"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest
import trustme
import websockets
from cryptography import x509
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)

from app.agent import cli
from app.agent import enroll as agent_enroll
from app.agent import trust as agent_trust
from app.agent.connection import AgentConnection
from app.agent.runner import Runner
from app.agent.trust import ControlPlaneTrust, FingerprintMismatch
from app.core import tls_fingerprint
from app.core import worker_protocol as wp

from .test_connection import FakeControlPlane, _cpu

ENROLL_TOKEN = "abcdef.0123456789abcdef"
WORKER_TOKEN = "w" * 43


# ── certificates ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ca() -> trustme.CA:
    return trustme.CA()


def _fingerprint(cert: trustme.LeafCert) -> str:
    der = tls_fingerprint.pem_certificates(cert.cert_chain_pems[0].bytes())[0]
    return tls_fingerprint.spki_fingerprint(der)


class Certs:
    """A server context with a certificate that can be rotated."""

    def __init__(self, ca: trustme.CA) -> None:
        self.ca = ca
        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.leaf = ca.issue_cert("127.0.0.1")
        self.leaf.configure_cert(self.context)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.leaf)

    def rotate(self) -> str:
        """A new certificate with a new key, from the same CA."""
        self.leaf = self.ca.issue_cert("127.0.0.1")
        self.leaf.configure_cert(self.context)
        return self.fingerprint


@pytest.fixture
def certs(ca) -> Certs:
    return Certs(ca)


@pytest.fixture
def trusted_ca(ca, monkeypatch):
    """Make first use trust the test CA (as if it were this machine's)."""
    def context() -> ssl.SSLContext:
        ctx = ssl.create_default_context()
        ca.configure_trust(ctx)
        return ctx

    monkeypatch.setattr(agent_trust, "first_use_context", context)


# ── an HTTPS enrollment endpoint ──────────────────────────────────────────


class HttpsControlPlane:
    def __init__(self, certs: Certs) -> None:
        self.requests: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 -- http.server's name
                length = int(self.headers.get("Content-Length", 0))
                outer.requests.append(json.loads(self.rfile.read(length)))
                payload = json.dumps({"worker_id": "gpu-box",
                                      "worker_token": WORKER_TOKEN}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.socket = certs.context.wrap_socket(self.server.socket,
                                                       server_side=True)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)

    @property
    def url(self) -> str:
        return f"https://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self) -> "HttpsControlPlane":
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def https_cp(certs):
    with HttpsControlPlane(certs) as server:
        yield server


def _enroll(url: str, trust: ControlPlaneTrust | None) -> tuple[str, str]:
    return agent_enroll.enroll(url, ENROLL_TOKEN, name="gpu-box",
                               platform={}, devices=[], timeout=10,
                               trust=trust)


# ── a wss /ws/worker endpoint ─────────────────────────────────────────────


class WssControlPlane(FakeControlPlane):
    def __init__(self, certs: Certs) -> None:
        super().__init__()
        self.certs = certs

    @property
    def url(self) -> str:
        port = self.server.sockets[0].getsockname()[1]
        return f"https://127.0.0.1:{port}"

    async def __aenter__(self) -> "WssControlPlane":
        self.server = await websockets.serve(
            self._handle, "127.0.0.1", 0, ssl=self.certs.context)
        return self


class TlsAgent:
    """An ``AgentConnection`` with *trust*, serving in a task."""

    def __init__(self, tmp_path: Any, control: str,
                 trust: ControlPlaneTrust | None) -> None:
        self.recorded: list[str] = []
        self.connection = AgentConnection(
            runner=Runner(runs_dir=tmp_path / "agent-runs",
                          offered_devices={"cpu"}),
            name="gpu-box", version="9.9.9", control_url=control,
            token=WORKER_TOKEN, probe_devices=_cpu, backoff_min_s=0.05,
            trust=trust, on_trusted=self.recorded.append)
        self.stop = asyncio.Event()
        self.task: asyncio.Task | None = None

    async def __aenter__(self) -> "TlsAgent":
        self.task = asyncio.create_task(self.connection.run(self.stop))
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.stop.set()
        await asyncio.wait_for(self.task, 30)

    async def exit_code(self) -> int:
        return await asyncio.wait_for(self.task, 30)


def _hello(_number: int, frame: Any) -> bool:
    return isinstance(frame, wp.Hello)


# ── the fingerprint itself ────────────────────────────────────────────────


def test_the_fingerprint_is_the_sha256_of_the_subject_public_key_info(ca):
    leaf = ca.issue_cert("127.0.0.1")
    der = tls_fingerprint.pem_certificates(leaf.cert_chain_pems[0].bytes())[0]
    spki = x509.load_der_x509_certificate(der).public_key().public_bytes(
        Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
    assert tls_fingerprint.spki_der(der) == spki
    import hashlib

    assert tls_fingerprint.spki_fingerprint(der) == (
        "sha256:" + hashlib.sha256(spki).hexdigest())
    # The CA certificate has a fingerprint of its own.
    ca_der = tls_fingerprint.pem_certificates(ca.cert_pem.bytes())[0]
    assert tls_fingerprint.spki_fingerprint(ca_der) != (
        tls_fingerprint.spki_fingerprint(der))
    with pytest.raises(ValueError):
        tls_fingerprint.spki_der(b"\x30\x03\x02\x01\x01")


def test_fingerprints_are_normalized_and_checked():
    digits = "ab" * 32
    spaced = ":".join(digits[i:i + 2] for i in range(0, 64, 2)).upper()
    assert tls_fingerprint.normalize(f"SHA256:{spaced}") == f"sha256:{digits}"
    assert tls_fingerprint.normalize(f" {digits} ") == f"sha256:{digits}"
    for bad in ("sha256:abc", "md5:" + digits, digits + "00", "zz" * 32):
        with pytest.raises(ValueError):
            tls_fingerprint.normalize(bad)


# ── enrollment over HTTPS ─────────────────────────────────────────────────


def test_enroll_with_the_right_fingerprint_connects(https_cp, certs):
    trust = ControlPlaneTrust(certs.fingerprint)
    assert _enroll(https_cp.url, trust) == ("gpu-box", WORKER_TOKEN)
    assert trust.observed == certs.fingerprint
    assert len(https_cp.requests) == 1


def test_enroll_with_a_wrong_fingerprint_is_refused_before_sending(
        https_cp, certs):
    wrong = "sha256:" + "0" * 64
    with pytest.raises(agent_enroll.EnrollError) as refused:
        _enroll(https_cp.url, ControlPlaneTrust(wrong))
    assert wrong in str(refused.value)
    assert certs.fingerprint in str(refused.value)
    # The enrollment token never left the agent.
    assert https_cp.requests == []


def test_only_the_leaf_key_matches_a_pin(https_cp, ca):
    """The CA's own key is not the server's: pinning it is refused, on
    every Python version (no chain matching)."""
    ca_print = tls_fingerprint.spki_fingerprint(
        tls_fingerprint.pem_certificates(ca.cert_pem.bytes())[0])
    with pytest.raises(agent_enroll.EnrollError, match="does not match"):
        _enroll(https_cp.url, ControlPlaneTrust(ca_print))
    assert https_cp.requests == []


def test_without_a_pin_the_system_cas_decide(https_cp, certs, trusted_ca):
    trust = ControlPlaneTrust()
    _enroll(https_cp.url, trust)
    assert trust.expected is None  # nothing learned


def test_an_untrusted_certificate_without_a_pin_shows_its_print(
        https_cp, certs):
    # This machine does not trust the test CA: no connection, and the
    # message says what to pin after checking it.
    with pytest.raises(agent_enroll.EnrollError) as refused:
        _enroll(https_cp.url, ControlPlaneTrust())
    assert f"--ca-fingerprint {certs.fingerprint}" in str(refused.value)
    assert https_cp.requests == []


# ── the worker socket over wss ────────────────────────────────────────────


async def _until(predicate: Any, timeout: float = 30.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "timed out"
        await asyncio.sleep(0.02)


async def test_wss_with_the_right_fingerprint_connects(tmp_path, certs):
    async with WssControlPlane(certs) as cp, \
            TlsAgent(tmp_path, cp.url,
                     ControlPlaneTrust(certs.fingerprint)) as agent:
        await cp.wait_for(_hello)
        # The pin, once per connection.
        await _until(lambda: agent.recorded == [certs.fingerprint])


async def test_wss_with_a_wrong_fingerprint_is_refused(tmp_path, certs,
                                                      caplog):
    wrong = "sha256:" + "1" * 64
    async with WssControlPlane(certs) as cp:
        agent = TlsAgent(tmp_path, cp.url, ControlPlaneTrust(wrong))
        async with agent:
            assert await agent.exit_code() == 2
    assert cp.frames == []
    assert wrong in caplog.text and certs.fingerprint in caplog.text


async def test_a_pinned_certificate_rotated_without_forget_is_refused(
        tmp_path, certs, caplog):
    async with WssControlPlane(certs) as cp:
        agent = TlsAgent(tmp_path, cp.url, ControlPlaneTrust(
            certs.fingerprint, hint="run cdui agent forget"))
        async with agent:
            await cp.wait_for(_hello)
            first = certs.fingerprint
            rotated = certs.rotate()
            await cp.sockets[-1].close()
            assert await agent.exit_code() == 2
    assert len(cp.of_type(wp.Hello)) == 1
    assert first in caplog.text and rotated in caplog.text
    assert "cdui agent forget" in caplog.text


async def test_without_a_pin_a_renewed_key_keeps_working(
        tmp_path, certs, trusted_ca):
    """The system CAs validate the certificate: nothing is pinned or
    recorded, so a renewal with a new key reconnects."""
    async with WssControlPlane(certs) as cp:
        agent = TlsAgent(tmp_path, cp.url, ControlPlaneTrust())
        async with agent:
            await cp.wait_for(_hello)
            certs.rotate()
            await cp.sockets[-1].close()
            await cp.wait_for(lambda n, f: n == 1 and isinstance(f, wp.Hello))
            assert not agent.task.done()
    assert agent.recorded == []
    assert agent.connection.trust.expected is None


async def test_an_untrusted_certificate_is_logged_and_retried(
        tmp_path, certs, caplog):
    async with WssControlPlane(certs) as cp:
        agent = TlsAgent(tmp_path, cp.url, None)
        async with agent:
            await _until(lambda: caplog.text.count(
                "not trusted by this machine") >= 3)
            assert not agent.task.done()
        assert await agent.exit_code() == 0
    errors = [r for r in caplog.records if r.levelno == logging.ERROR
              and "not trusted" in r.getMessage()]
    assert all("retrying" in r.getMessage() for r in errors)
    # The fingerprint to pin is shown once, not on every retry.
    assert caplog.text.count(f"--ca-fingerprint {certs.fingerprint}") == 1


def test_the_mismatch_is_an_ssl_error_with_both_values():
    exc = FingerprintMismatch("sha256:aa", "sha256:bb", "--ca-fingerprint",
                              "hint")
    assert isinstance(exc, ssl.SSLError)
    assert (exc.expected, exc.actual) == ("sha256:aa", "sha256:bb")
    assert str(exc).endswith(". hint")


# ── cdui agent start ──────────────────────────────────────────────────────


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
    calls: list[dict[str, Any]] = []

    def fake_run_agent(args, control, token, facts, token_source=None,
                       mounts=None, trust=None, worker_credential=False):
        calls.append({"control": control, "token": token, "trust": trust})
        return 0

    monkeypatch.setattr(cli, "run_agent", fake_run_agent)
    monkeypatch.setattr(cli, "_setup_logging", lambda daemonized: None)
    return calls


def _start(*extra: str) -> int:
    return cli.main(["start", "--foreground", *extra])


def test_start_pins_the_flag_records_it_and_keeps_it(
        https_cp, certs, user_data, started):
    assert _start("--control", https_cp.url, "--token", ENROLL_TOKEN,
                  "--ca-fingerprint", certs.fingerprint.upper()) == 0
    stored = agent_enroll.load_credential(user_data, https_cp.url)
    assert stored is not None and stored.ca_fingerprint == certs.fingerprint
    assert started[-1]["trust"].expected == certs.fingerprint

    # A restart without the flag keeps the user's pin; the hint names both
    # ways out.
    assert _start("--control", https_cp.url) == 0
    trust = started[-1]["trust"]
    assert trust.expected == certs.fingerprint
    assert str(stored.path) in trust.source
    assert "--ca-fingerprint" in trust.hint
    assert f"cdui agent forget {https_cp.url}" in trust.hint

    # Rotated without forget: refused (the enrollment path, with the trust
    # the restart built; the wss path is tested above).
    certs.rotate()
    with pytest.raises(agent_enroll.EnrollError, match="does not match"):
        _enroll(https_cp.url, trust)

    # forget drops the credential and its pin.
    assert cli.main(["forget", https_cp.url]) == 0
    assert agent_enroll.load_credential(user_data, https_cp.url) is None


def test_start_without_a_pin_records_nothing(https_cp, certs, user_data,
                                            started, trusted_ca, capsys):
    assert _start("--control", https_cp.url, "--token", ENROLL_TOKEN) == 0
    stored = agent_enroll.load_credential(user_data, https_cp.url)
    assert stored is not None and stored.ca_fingerprint is None
    assert "ca_fingerprint" not in stored.path.read_text()
    assert started[-1]["trust"].expected is None
    assert _start("--control", https_cp.url) == 0
    assert started[-1]["trust"].expected is None


def test_a_new_flag_overrides_the_recorded_pin(user_data, started):
    url = "https://cp.example:8443"
    old, new = "sha256:" + "a" * 64, "sha256:" + "b" * 64
    agent_enroll.save_credential(user_data, url, "gpu-box", WORKER_TOKEN,
                                 ca_fingerprint=old)
    assert _start("--control", url, "--ca-fingerprint", new) == 0
    assert started[-1]["trust"].expected == new
    # Recorded once a connection verified it; the same pin again is no write.
    record = cli.fingerprint_recorder(url)
    record(new)
    stored = agent_enroll.load_credential(user_data, url)
    assert stored.ca_fingerprint == new
    assert stored.worker_token == WORKER_TOKEN
    before = stored.path.stat().st_mtime_ns
    stored.path.chmod(0o400)  # a write would now fail
    try:
        record(new)
    finally:
        stored.path.chmod(0o600)
    assert stored.path.stat().st_mtime_ns == before
    assert agent_enroll.record_fingerprint(user_data, url, new) is None
    # Nothing to record into without a credential.
    cli.fingerprint_recorder("https://other:1")(new)
    assert agent_enroll.load_credential(user_data, "https://other:1") is None


def test_a_malformed_stored_pin_is_reported_and_the_token_kept(
        user_data, started, caplog):
    url = "https://cp.example:8443"
    path = agent_enroll.save_credential(user_data, url, "gpu-box",
                                        WORKER_TOKEN)
    data = json.loads(path.read_text())
    data["ca_fingerprint"] = "sha256:not-hex"
    path.write_text(json.dumps(data))
    stored = agent_enroll.load_credential(user_data, url)
    assert stored is not None and stored.worker_token == WORKER_TOKEN
    assert stored.ca_fingerprint is None
    assert "malformed ca_fingerprint" in caplog.text and str(path) in (
        caplog.text)
    assert _start("--control", url) == 0
    assert started[-1]["token"] == WORKER_TOKEN
    assert started[-1]["trust"].expected is None


@pytest.mark.parametrize("extra, message", [
    (("--control", "http://10.0.0.5:8000", "--ca-fingerprint",
      "sha256:" + "a" * 64), "needs an https://"),
    (("--control", "https://10.0.0.5:8000", "--ca-fingerprint", "nope"),
     "sha256:<64 hex digits>"),
])
def test_bad_fingerprint_flags_are_refused(user_data, started, capsys,
                                           extra, message):
    assert _start(*extra, "--token", WORKER_TOKEN) == 2
    assert message in capsys.readouterr().err
    assert started == []


def test_the_flag_help_says_it_is_the_server_key(capsys):
    with pytest.raises(SystemExit):
        cli.main(["start", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "server (leaf) certificate public key" in help_text


# ── plain http ────────────────────────────────────────────────────────────


def test_plain_http_to_another_machine_warns_exactly_once(user_data, started,
                                                          capsys, caplog):
    caplog.set_level(logging.INFO)
    assert _start("--control", "http://10.0.0.5:8000", "--token",
                  WORKER_TOKEN) == 0
    out = capsys.readouterr()
    text = out.out + out.err + caplog.text
    assert text.count("is plain HTTP") == 1
    assert "secrets in every assignment travel unencrypted" in out.err
    assert started[-1]["trust"] is None


@pytest.mark.parametrize("control", [
    "http://127.0.0.1:8000", "http://localhost:18001", "http://[::1]:8000",
    "https://10.0.0.5:8000"])
def test_no_plain_http_warning_for_loopback_or_tls(user_data, started, capsys,
                                                   control):
    # Loopback is this machine or an SSH tunnel's local end (#784).
    assert _start("--control", control, "--token", WORKER_TOKEN) == 0
    assert "plain HTTP" not in capsys.readouterr().err


def test_the_background_agent_does_not_warn_again(user_data, started, capsys):
    assert cli.main(["start", "--foreground", "--daemonized", "--control",
                     "http://10.0.0.5:8000"]) == 0
    assert "plain HTTP" not in capsys.readouterr().err


def test_the_background_launch_passes_the_pin_on(user_data, monkeypatch):
    argv: list[list[str]] = []

    class Proc:
        pid = 4242

        def poll(self) -> int:
            return 0

    monkeypatch.setattr(cli, "running_agent_pid", lambda: None)
    monkeypatch.setattr(cli.subprocess, "Popen",
                        lambda args, **_: argv.append(args) or Proc())
    pin = "sha256:" + "d" * 64
    cli.main(["start", "--control", "https://cp.example:8443", "--token",
              WORKER_TOKEN, "--ca-fingerprint", pin])
    assert argv and argv[0][argv[0].index("--ca-fingerprint") + 1] == pin
