"""The WebSocket message ceiling is CodefyUI's number, not uvicorn's (core#274).

`WS /ws/execution` carries a whole graph in one text frame. It was never
uncapped -- uvicorn's `ws_max_size` defaults to 16 MB and the `websockets`
library enforces it WHILE assembling fragments, so memory was already bounded
-- but that 16 MB was an inherited library default: nothing in this repo chose
it, no launch path passed it, no document mentioned it, and it was four times
STRICTER than the 64 MB `MAX_RUN_BODY_BYTES` the HTTP routes use. A graph
between the two was accepted by `POST /api/graph/run/{name}` and refused by the
socket the editor actually uses.

The cap therefore cannot be enforced in application code: by the time
`receive_text()` returns, the message is already assembled, and `len()` on the
returned `str` counts characters rather than bytes -- a weaker bound than the
one the transport already applies. So the setting exists to be handed to the
transport, and these tests pin the two properties that makes load-bearing:

1. `Settings.WS_MAX_MESSAGE_BYTES` defaults to `MAX_RUN_BODY_BYTES` rather than
   to a literal of its own, so one graph ceiling covers both transports and
   raising either env var cannot silently split them.
2. `scripts/dev.py` derives the same number and passes it as `--ws-max-size` on
   every launch path. dev.py must not import the backend (it runs on a bare
   interpreter before the venv exists), so the constant is duplicated -- and
   the drift test below is the only thing standing between that duplicate and
   a ceiling that disagrees with its own documentation.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import dev  # scripts/dev.py -- conftest puts scripts/ on sys.path
from app.config import Settings

_WS_ENV = "CODEFYUI_WS_MAX_MESSAGE_BYTES"
_BODY_ENV = "CODEFYUI_MAX_RUN_BODY_BYTES"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Both knobs unset unless a test sets them.

    A developer with either exported would otherwise silently change what
    these assertions mean.
    """
    monkeypatch.delenv(_WS_ENV, raising=False)
    monkeypatch.delenv(_BODY_ENV, raising=False)


# -- Settings precedence ---------------------------------------------------


def test_ws_ceiling_defaults_to_the_body_ceiling():
    """Unset, the two transports agree. This is the whole point of #274."""
    settings = Settings()
    assert settings.WS_MAX_MESSAGE_BYTES == settings.MAX_RUN_BODY_BYTES
    # And specifically NOT uvicorn's inherited default, which is what made
    # the socket stricter than HTTP.
    assert settings.WS_MAX_MESSAGE_BYTES != 16 * 1024 * 1024


def test_raising_the_body_ceiling_raises_the_ws_ceiling_with_it(monkeypatch):
    """An operator who lifts the HTTP cap must not have to know about a
    second one to make the canvas usable."""
    monkeypatch.setenv(_BODY_ENV, str(128 * 1024 * 1024))
    settings = Settings()
    assert settings.MAX_RUN_BODY_BYTES == 128 * 1024 * 1024
    assert settings.WS_MAX_MESSAGE_BYTES == 128 * 1024 * 1024


def test_an_explicit_ws_ceiling_wins_over_the_body_ceiling(monkeypatch):
    """The two CAN be split apart -- deliberately, never by accident."""
    monkeypatch.setenv(_WS_ENV, str(4 * 1024 * 1024))
    settings = Settings()
    assert settings.WS_MAX_MESSAGE_BYTES == 4 * 1024 * 1024
    assert settings.MAX_RUN_BODY_BYTES == 64 * 1024 * 1024


def test_both_set_are_both_honoured(monkeypatch):
    monkeypatch.setenv(_BODY_ENV, "8388608")
    monkeypatch.setenv(_WS_ENV, "2097152")
    settings = Settings()
    assert settings.MAX_RUN_BODY_BYTES == 8388608
    assert settings.WS_MAX_MESSAGE_BYTES == 2097152


# -- dev.py must not drift from Settings -----------------------------------


@pytest.mark.parametrize(
    "env",
    [
        {},
        {_BODY_ENV: str(128 * 1024 * 1024)},
        {_WS_ENV: str(4 * 1024 * 1024)},
        {_BODY_ENV: "8388608", _WS_ENV: "2097152"},
        # Body set to something the WS knob must inherit exactly, including
        # an awkward non-power-of-two.
        {_BODY_ENV: "1234567"},
    ],
)
def test_dev_py_derives_the_same_ceiling_as_settings(env, monkeypatch):
    """The duplicated constant and precedence in dev.py agree with pydantic.

    dev.py cannot import `app.config` -- it runs before the venv exists -- so
    it reimplements this precedence by hand. Without this test the launcher
    could pass a ceiling that disagreed with the number the docs quote and
    the number the editor's "graph too large" message is about, and nothing
    would notice.
    """
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert dev._ws_max_size() == Settings().WS_MAX_MESSAGE_BYTES


def test_dev_py_ignores_a_malformed_value_and_lets_pydantic_rule(monkeypatch):
    """A bad value is the child's error to report, not the launcher's.

    dev.py falling back here (rather than exiting) keeps ONE authority on
    what a valid ceiling is. The server then refuses to start with pydantic's
    own message, which names the field and the offending value.
    """
    monkeypatch.setenv(_WS_ENV, "not-a-number")
    assert dev._ws_max_size() == 64 * 1024 * 1024
    with pytest.raises(Exception):
        Settings()


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_dev_py_ignores_a_non_positive_value(bad, monkeypatch):
    """`--ws-max-size 0` would be a socket that refuses every message."""
    monkeypatch.setenv(_WS_ENV, bad)
    assert dev._ws_max_size() == 64 * 1024 * 1024


# -- every launch path passes the flag -------------------------------------


def _uvicorn_command_literals() -> list[list[str]]:
    """Every list literal in dev.py that spawns `app.main:app`.

    AST rather than a regex so this reads the real command lists and cannot
    be fooled by the string appearing in a comment, a docstring or one of the
    `cdui stop` process matchers.
    """
    tree = ast.parse(Path(dev.__file__).read_text(encoding="utf-8"))
    commands = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.List):
            continue
        constants = [e.value for e in node.elts
                     if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        if "app.main:app" in constants:
            commands.append(constants)
    return commands


def test_every_uvicorn_launch_passes_ws_max_size():
    """`cdui start` AND `cdui dev` -- a ceiling that only holds in production
    is one developers discover from a bug report."""
    commands = _uvicorn_command_literals()
    # start() and dev(); if a third launch path appears it is covered too.
    assert len(commands) >= 2, commands
    for cmd in commands:
        assert "--ws-max-size" in cmd, cmd


def test_ws_max_size_is_a_flag_cdui_owns():
    """Forwarding it after `--` would desync the real ceiling from the
    documented one, so it is refused like --host and --port."""
    assert "--ws-max-size" in dev._UVICORN_FLAGS_CDUI_OWNS


@pytest.mark.parametrize(
    "extra",
    [["--ws-max-size", "1024"], ["--ws-max-size=1024"]],
)
def test_forwarding_ws_max_size_is_refused_and_names_the_env_var(extra, capsys):
    with pytest.raises(SystemExit) as exc:
        dev._reject_owned_uvicorn_flags(extra)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    # The refusal has to point at the knob that DOES work, or it is just a
    # dead end for the operator who reached for the flag.
    assert _WS_ENV in err
