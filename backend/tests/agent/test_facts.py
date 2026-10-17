"""``app.agent.facts``: the fingerprint and node catalog for ``hello`` (#781)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from app.agent import facts
from app.core.env_fingerprint import fingerprint_digest
from app.core.node_registry import node_catalog

BACKEND_DIR = Path(__file__).resolve().parents[2]


def test_collect_reports_this_process_catalog_and_fingerprint():
    collected = facts.collect(discover=False)
    catalog = node_catalog()
    assert collected["node_catalog_hash"] == catalog.hash
    assert collected["node_types"] == list(catalog.node_types)
    assert collected["fingerprint_digest"] == fingerprint_digest(
        collected["fingerprint"])


def test_a_helper_that_dies_without_reporting_gives_no_facts(caplog):
    # ``os.getpid(conn)`` raises in the child before it sends anything.
    assert facts.gather(target=os.getpid, timeout_s=60) == facts.AgentFacts()
    assert "exited without reporting" in caplog.text


class _Conn:
    def __init__(self) -> None:
        self.sent = []
        self.closed = False

    def send(self, value):
        self.sent.append(value)

    def close(self):
        self.closed = True


def test_helper_main_reports_a_failure(monkeypatch):
    def boom(**_):
        raise RuntimeError("no torch")

    monkeypatch.setattr(facts, "collect", boom)
    conn = _Conn()
    facts.helper_main(conn)
    assert conn.sent == [("error", "RuntimeError: no torch")] and conn.closed


def test_the_agent_process_stays_torch_free():
    """Importing the agent and gathering nothing loads no torch: the facts
    come from the helper subprocess."""
    code = ("import sys, app.agent.cli, app.agent.connection, "
            "app.agent.facts; "
            "print(any(m == 'torch' or m.startswith('torch.') "
            "for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=BACKEND_DIR,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def _never_reports(conn) -> None:
    """A helper that hangs (the spawn child imports it from here)."""
    import time

    time.sleep(120)


def test_a_stop_while_gathering_kills_the_helper_at_once():
    import threading
    import time

    cancel = threading.Event()
    threading.Timer(1.0, cancel.set).start()
    started = time.monotonic()
    assert facts.gather(target=_never_reports, timeout_s=120,
                        cancel=cancel) is None
    assert time.monotonic() - started < 30


def test_a_helper_that_never_reports_times_out(caplog):
    assert facts.gather(target=_never_reports, timeout_s=1) == \
        facts.AgentFacts()
    assert "did not report within" in caplog.text
