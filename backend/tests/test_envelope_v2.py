"""Unit pins for the Stage-2 envelope additions (spec Section 6): nine
always-present keys, app/version kwargs, and error_response's headers=
passthrough (the invoke 401 carries WWW-Authenticate: Bearer)."""

from __future__ import annotations

import json

from app.api.routes_graph_run import build_envelope, error_response

ENVELOPE_KEYS = {
    "status", "run_id", "graph", "app", "version",
    "device", "outputs", "error", "timing",
}


def test_build_envelope_emits_nine_keys_with_null_app_version_by_default():
    env = build_envelope(status="ok", run_id="r1", graph="g",
                         outputs={"y": 1})
    assert set(env.keys()) == ENVELOPE_KEYS
    assert env["app"] is None       # editor route: always null
    assert env["version"] is None


def test_build_envelope_carries_app_and_version_when_given():
    env = build_envelope(status="ok", run_id="r1", graph="my-app",
                         app="my-app", version=3, outputs={})
    assert env["app"] == "my-app"
    assert env["version"] == 3


def test_error_response_carries_app_version_and_headers():
    resp = error_response(
        401, run_id="r2", graph="my-app", app="my-app",
        code="invalid_key", message="nope",
        headers={"WWW-Authenticate": "Bearer"},
    )
    assert resp.status_code == 401
    assert resp.headers["WWW-Authenticate"] == "Bearer"
    body = json.loads(resp.body)
    assert set(body.keys()) == ENVELOPE_KEYS
    assert body["app"] == "my-app"
    assert body["version"] is None   # pre-resolution: version stays null


def test_error_response_headers_default_adds_nothing():
    resp = error_response(500, run_id="r", graph="g", code="timeout",
                          message="m")
    assert "WWW-Authenticate" not in resp.headers
