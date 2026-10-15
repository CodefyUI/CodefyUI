"""The run subprocess's data directories come from the agent's mount (#782).

``--mount data=<path>`` makes ``<path>`` the run's data directory, laid out
as ``config.py`` lays out ``backend/data`` (or, for a project directory,
``CODEFYUI_PROJECT_DIR``). These tests start real spawned subprocesses and
read what ``app.config`` derives inside them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent import agent_config
from app.agent import cli as agent_cli
from app.agent.child import artifact_location
from app.agent.runner import Runner, child_environment
from app.core import mounts
from app.core import worker_protocol as wp

from . import _helpers as h


def _runner(tmp_path: Path, sink: h.FrameSink, **kw) -> Runner:
    return Runner(runs_dir=tmp_path / "agent-runs", offered_devices={"cpu"},
                  send=sink, child_entry=h.data_roots_child, **kw)


async def _roots(runner: Runner, sink: h.FrameSink, run_id: str,
                 storage: wp.Storage | None) -> dict:
    await runner.assign(wp.Assign(run_id=run_id, graph={"nodes": []},
                                  storage=storage))
    event = await sink.wait_for(lambda f: isinstance(f, wp.Event)
                                and f.run_id == run_id)
    await sink.wait_for(lambda f: isinstance(f, wp.RunFinished)
                        and f.run_id == run_id)
    return event.payload


async def test_a_data_mount_derives_every_data_directory(tmp_path,
                                                         monkeypatch):
    # An agent environment that pointed elsewhere must not win.
    monkeypatch.setenv("CODEFYUI_MODELS_DIR", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("CODEFYUI_PROJECT_DIR", str(tmp_path / "proj"))
    nas = tmp_path / "nas" / "data"
    nas.mkdir(parents=True)
    sink = h.FrameSink()
    runner = _runner(tmp_path, sink, mounts={"data": str(nas)})
    payload = await _roots(runner, sink, "m1", wp.Storage(mount="data"))
    assert payload["roots"] == {
        "DATA_FILES_DIR": str(nas / "files"),
        "IMAGES_DIR": str(nas / "images"),
        "MODELS_DIR": str(nas / "models"),
        "MEDIA_DIR": str(nas / "media"),
        "GRAPHS_DIR": str(nas / "graphs"),
        "USER_PRESETS_DIR": str(nas / "presets"),
        "PROJECT_DIR": None,
    }
    assert payload["mount"] == {"label": "data", "root": str(nas)}
    # A file the run writes under the mount is reported relative to it.
    assert payload["artifact"] == ["models/periodic/r-n-e1.pt", "data"]


async def test_a_project_directory_mount_sets_the_project(tmp_path,
                                                          monkeypatch):
    monkeypatch.setenv("CODEFYUI_IMAGES_DIR", str(tmp_path / "elsewhere"))
    project = tmp_path / "nas" / "proj"
    project.mkdir(parents=True)
    (project / mounts.PROJECT_MANIFEST).write_text("[project]\n")
    sink = h.FrameSink()
    runner = _runner(tmp_path, sink, mounts={"data": str(project)})
    payload = await _roots(runner, sink, "p1", wp.Storage(mount="data"))
    resolved = project.resolve()
    assert payload["roots"]["PROJECT_DIR"] == str(resolved)
    assert payload["roots"]["IMAGES_DIR"] == str(resolved / "assets" /
                                                 "images")
    assert payload["roots"]["MODELS_DIR"] == str(resolved / "assets" /
                                                 "models")
    assert payload["roots"]["GRAPHS_DIR"] == str(resolved / "graphs")
    assert payload["artifact"] == ["assets/models/periodic/r-n-e1.pt", "data"]


async def test_the_control_planes_label_picks_the_mount(tmp_path):
    """The label comes from the control plane (``assign.storage``): an
    agent whose only mount has another label derives nothing."""
    nas = tmp_path / "nas"
    nas.mkdir()
    sink = h.FrameSink()
    runner = _runner(tmp_path, sink, mounts={"lab": str(nas)})
    labelled = await _roots(runner, sink, "l1", wp.Storage(mount="lab"))
    assert labelled["roots"]["MODELS_DIR"] == str(nas / "models")
    other = await _roots(runner, sink, "l2", wp.Storage(mount="data"))
    assert other["mount"] is None
    assert other["roots"]["MODELS_DIR"] != str(nas / "models")
    assert other["artifact"][1] is None


async def test_a_same_host_worker_borrows_the_control_planes_directories(
        tmp_path):
    """The lent directories are set exactly as sent, whatever layout they
    have, and a same-host run reports its files with absolute paths."""
    lent = {"CODEFYUI_DATA_FILES_DIR": str(tmp_path / "up"),
            "CODEFYUI_IMAGES_DIR": str(tmp_path / "pics"),
            "CODEFYUI_MODELS_DIR": str(tmp_path / "elsewhere" / "models"),
            "CODEFYUI_MEDIA_DIR": str(tmp_path / "media")}
    sink = h.FrameSink()
    runner = _runner(tmp_path, sink)
    payload = await _roots(runner, sink, "s1", wp.Storage(
        mount="data", dirs=lent, relative_artifacts=False))
    assert payload["roots"]["DATA_FILES_DIR"] == str(tmp_path / "up")
    assert payload["roots"]["IMAGES_DIR"] == str(tmp_path / "pics")
    assert payload["roots"]["MODELS_DIR"] == str(
        tmp_path / "elsewhere" / "models")
    assert payload["mount"] is None
    assert payload["artifact"][1] is None  # absolute: a local row
    # Its own --mount wins over the lent directories, and still reports
    # absolute paths when the control plane asks for them.
    own = tmp_path / "own"
    own.mkdir()
    runner.mounts = {"data": str(own)}
    payload = await _roots(runner, sink, "s2", wp.Storage(
        mount="data", dirs=lent, relative_artifacts=False))
    assert payload["roots"]["DATA_FILES_DIR"] == str(own / "files")
    assert payload["artifact"][1] is None


async def test_a_missing_mount_directory_is_not_used(tmp_path):
    """A saved share that is not mounted: nothing derived from it, and it
    is used as soon as the directory appears."""
    nas = tmp_path / "nas"
    sink = h.FrameSink()
    runner = _runner(tmp_path, sink, mounts={"data": str(nas)})
    assert runner.available_mounts() == {}
    payload = await _roots(runner, sink, "x1", wp.Storage(mount="data"))
    assert payload["mount"] is None
    nas.mkdir()
    assert runner.available_mounts() == {"data": str(nas)}
    payload = await _roots(runner, sink, "x2", wp.Storage(mount="data"))
    assert payload["mount"] == {"label": "data", "root": str(nas)}


async def test_without_storage_nothing_is_derived(tmp_path):
    """An assignment from before #782 and an agent without a mount: the
    child keeps the agent's data settings, as in #771."""
    sink = h.FrameSink()
    payload = await _roots(_runner(tmp_path, sink), sink, "n1", None)
    assert payload["mount"] is None
    assert child_environment("cpu", [])["set"] == {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": ""}


def test_artifact_location(tmp_path):
    root = tmp_path / "nas"
    (root / "models").mkdir(parents=True)
    mount = {"label": "data", "root": str(root)}
    inside = str(root / "models" / "x.pt")
    assert artifact_location(inside, mount) == ("models/x.pt", "data")
    assert artifact_location("/elsewhere/x.pt", mount) == (
        "/elsewhere/x.pt", None)
    assert artifact_location(inside, None) == (inside, None)
    # A symlinked mount point reports the same relative path.
    link = tmp_path / "link"
    link.symlink_to(root)
    assert artifact_location(str(link / "models" / "x.pt"), mount) == (
        "models/x.pt", "data")


# ── --mount: parsed, checked and saved with the agent's config ───────────


@pytest.fixture
def user_data(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEFYUI_USER_DATA_DIR", str(tmp_path / "ud"))
    return tmp_path / "ud"


def test_mounts_are_saved_and_reused(user_data, tmp_path):
    nas = tmp_path / "nas"
    nas.mkdir()
    assert agent_cli.resolve_mounts([f"data={nas}"]) == {"data": str(nas)}
    saved = agent_config.load(user_data)
    assert saved == {"mounts": {"data": str(nas)}}
    # A later start with no flag keeps them; other keys in the file survive.
    agent_config.save(user_data, {**saved, "other": 1})
    assert agent_cli.resolve_mounts(None) == {"data": str(nas)}
    # LABEL= removes one.
    assert agent_cli.resolve_mounts(["data="]) == {}
    assert agent_config.load(user_data) == {"mounts": {}, "other": 1}


@pytest.mark.parametrize("flag", ["data", "Bad Label=/x", "=/x"])
def test_a_malformed_mount_is_refused(user_data, flag):
    with pytest.raises(ValueError):
        agent_cli.resolve_mounts([flag])


def test_a_mount_that_is_not_a_directory_is_refused(user_data, tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        agent_cli.resolve_mounts([f"data={tmp_path / 'missing'}"])
    assert agent_config.load(user_data) == {}


def test_a_saved_mount_that_is_missing_now_is_only_warned_about(
        user_data, tmp_path, capsys):
    gone = tmp_path / "share"
    agent_config.save(user_data, {"mounts": {"data": str(gone)}})
    assert agent_cli.resolve_mounts(None) == {"data": str(gone)}
    assert "not a directory" in capsys.readouterr().err


def test_mounts_cannot_change_while_the_agent_runs(user_data, tmp_path,
                                                   monkeypatch, capsys):
    nas = tmp_path / "nas"
    nas.mkdir()
    monkeypatch.setattr(agent_cli, "running_agent_pid", lambda: 4242)
    code = agent_cli.main(["start", "--mount", f"data={nas}"])
    assert code == 1
    assert "stop the agent first" in capsys.readouterr().err
    assert agent_config.load(user_data) == {}


def test_save_leaves_no_temp_file_behind_on_failure(tmp_path, monkeypatch):
    def refuse(*_):
        raise OSError("disk full")

    monkeypatch.setattr(agent_config.os, "replace", refuse)
    with pytest.raises(OSError):
        agent_config.save(tmp_path, {"mounts": {}})
    assert list(tmp_path.iterdir()) == []


def test_the_parser_takes_repeated_mounts():
    args = agent_cli.build_parser().parse_args(
        ["start", "--mount", "data=/a", "--mount", "lab=/b"])
    assert args.mount == ["data=/a", "lab=/b"]
    assert agent_cli.build_parser().parse_args(["start"]).mount is None


def test_mount_paths_are_made_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert agent_config.merge_mounts({}, ["data=rel"]) == {
        "data": str(tmp_path / "rel")}

