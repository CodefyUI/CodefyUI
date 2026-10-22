"""The agent works in the install's ``backend/``, as the server does (#839).

``cdui agent start --foreground`` (what an SSH worker runs) used to keep the
directory it was started in, the home directory over SSH, and its run
subprocesses inherited it: a graph reading the bundled
``data/samples/iris.csv`` failed there with ``CSVReader: file not found``.
The agent now changes to ``backend/`` once at start, as the background
agent always did (``_start_background`` starts it with that ``cwd``), and
relative path settings given at start keep naming the directory they were
given in.
"""

from __future__ import annotations

import contextlib
import os
import threading
from pathlib import Path

import pytest

from app.agent import child
from app.agent import cli as agent_cli
from app.agent.runner import Runner
from app.core import worker_protocol as wp

from . import _helpers as h


@contextlib.contextmanager
def _environ_restored():
    """Put ``os.environ`` back whatever the code under test rewrote."""
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


@pytest.fixture
def saved_environ():
    with _environ_restored():
        yield


def test_path_settings_come_from_the_settings_model():
    names = agent_cli.path_setting_envs()
    for name in ("CODEFYUI_USER_DATA_DIR", "CODEFYUI_MODELS_DIR",
                 "CODEFYUI_DATA_FILES_DIR", "CODEFYUI_PROJECT_DIR",
                 "CODEFYUI_DB_PATH", "CODEFYUI_LOG_DIR"):
        assert name in names
    # Flags and labels whose names merely end like a path setting.
    for name in ("CODEFYUI_PREFER_WORKING_DIR", "CODEFYUI_MOUNT_LABEL",
                 "CODEFYUI_HOST"):
        assert name not in names


def test_pin_path_settings_only_touches_relative_path_settings(tmp_path):
    env = {"CODEFYUI_MODELS_DIR": "rel/models",
           "CODEFYUI_IMAGES_DIR": str(tmp_path / "abs"),
           "CODEFYUI_MEDIA_DIR": "",
           "CODEFYUI_PREFER_WORKING_DIR": "true",
           "CODEFYUI_MOUNT_LABEL": "data"}
    agent_cli.pin_path_settings(env, tmp_path)
    assert env == {"CODEFYUI_MODELS_DIR": str(tmp_path / "rel" / "models"),
                   "CODEFYUI_IMAGES_DIR": str(tmp_path / "abs"),
                   "CODEFYUI_MEDIA_DIR": "",
                   "CODEFYUI_PREFER_WORKING_DIR": "true",
                   "CODEFYUI_MOUNT_LABEL": "data"}


def test_enter_backend_dir(monkeypatch, tmp_path, saved_environ):
    monkeypatch.chdir(tmp_path)
    os.environ["CODEFYUI_MODELS_DIR"] = "rel/models"
    os.environ["CODEFYUI_PREFER_WORKING_DIR"] = "true"
    assert agent_cli.enter_backend_dir() is True
    assert Path.cwd().resolve() == agent_cli._BACKEND_DIR.resolve()
    assert os.environ["CODEFYUI_MODELS_DIR"] == str(
        tmp_path.resolve() / "rel" / "models")
    assert os.environ["CODEFYUI_PREFER_WORKING_DIR"] == "true"


def test_a_directory_that_cannot_be_entered_is_reported(
        monkeypatch, tmp_path, capsys, saved_environ):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEFYUI_LANG", "en")

    def refuse(path):
        raise PermissionError("denied")

    monkeypatch.setattr(agent_cli.os, "chdir", refuse)
    assert agent_cli.enter_backend_dir() is False
    assert "cannot change to the install's directory" in capsys.readouterr().err


def test_a_vanished_start_directory_is_reported(monkeypatch, capsys,
                                                saved_environ):
    monkeypatch.setenv("CODEFYUI_LANG", "en")

    def gone():
        raise FileNotFoundError("no such directory")

    monkeypatch.setattr(agent_cli.Path, "cwd", staticmethod(gone))
    assert agent_cli.enter_backend_dir() is False
    assert "cannot read the current working directory" in \
        capsys.readouterr().err


async def test_a_run_subprocess_inherits_the_directory_and_flags_unchanged(
        monkeypatch, tmp_path, saved_environ):
    """First review item of round 2: a boolean ``CODEFYUI_*`` setting must
    reach the run subprocess as given (it used to become ``<cwd>/true``
    and crash every run), and the subprocess works in ``backend/``."""
    monkeypatch.chdir(tmp_path)
    os.environ["CODEFYUI_PREFER_WORKING_DIR"] = "true"
    os.environ["CODEFYUI_MODELS_DIR"] = "rel/models"
    assert agent_cli.enter_backend_dir()
    sink = h.FrameSink()
    runner = Runner(runs_dir=tmp_path / "agent-runs", offered_devices={"cpu"},
                    send=sink, child_entry=h.workdir_child)
    await runner.assign(wp.Assign(run_id="w1", graph={"nodes": []}))
    event = await sink.wait_for(lambda f: isinstance(f, wp.Event)
                                and f.run_id == "w1")
    finished = await sink.wait_for(lambda f: isinstance(f, wp.RunFinished)
                                   and f.run_id == "w1")
    assert finished.status == "succeeded"
    assert Path(event.payload["cwd"]).resolve() == \
        agent_cli._BACKEND_DIR.resolve()
    assert event.payload["prefer_env"] == "true"
    assert event.payload["prefer"] is True
    assert event.payload["models_dir"] == str(
        tmp_path.resolve() / "rel" / "models")


def _run(spec) -> list[tuple]:
    sent: list[tuple] = []
    thread = threading.Thread(target=child.run_assignment,
                              args=(spec, sent.append),
                              kwargs={"discover": False})
    thread.start()
    thread.join(60)
    assert not thread.is_alive()
    return sent


def _sample_spec() -> dict:
    graph = {"nodes": [h.node("start", "Start"),
                       h.node("csv", "CSVReader",
                              path="data/samples/iris.csv")],
             "edges": [h.trigger("start", "csv")]}
    return {"run_id": "s1", "graph": graph, "options": {}, "device": "cpu"}


def test_a_bundled_sample_reads_once_the_agent_is_in_backend(
        monkeypatch, tmp_path, saved_environ):
    """B2 of #839, end to end in the run code: from the home directory the
    sample is not found; after the agent's start-up change it reads."""
    monkeypatch.chdir(tmp_path)
    missed = _run(_sample_spec())
    assert missed[-1][1]["status"] == "failed"
    assert "file not found" in missed[-1][1]["error"]
    assert agent_cli.enter_backend_dir()
    sent = _run(_sample_spec())
    assert sent[-1] == (child.MSG_FINISHED, {
        "status": "succeeded", "error": None, "peak_memory": None})
