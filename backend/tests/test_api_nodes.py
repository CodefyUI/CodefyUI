"""Tests for the nodes API endpoints."""

import pytest


@pytest.mark.asyncio
async def test_list_nodes(test_client):
    resp = await test_client.get("/api/nodes")
    assert resp.status_code == 200
    nodes = resp.json()
    assert isinstance(nodes, list)
    assert len(nodes) >= 1
    # Check structure of first node
    node = nodes[0]
    assert "node_name" in node
    assert "category" in node
    assert "inputs" in node
    assert "outputs" in node
    assert "params" in node


@pytest.mark.asyncio
async def test_get_specific_node(test_client):
    resp = await test_client.get("/api/nodes/Conv2d")
    assert resp.status_code == 200
    node = resp.json()
    assert node["node_name"] == "Conv2d"
    assert node["category"] == "CNN"


@pytest.mark.asyncio
async def test_get_nonexistent_node(test_client):
    resp = await test_client.get("/api/nodes/DoesNotExist")
    assert resp.status_code == 404


# ── Script validation (core#131) ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_validate_script_accepts_an_allowlisted_script(test_client):
    resp = await test_client.post(
        "/api/nodes/script/validate",
        json={"code": "import numpy\n\ndef run(inputs, params):\n    return 1\n"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["error"] is None
    assert body["defines_run"] is True
    assert "numpy" in body["allowed_modules"]


@pytest.mark.asyncio
async def test_validate_script_rejects_a_network_import_with_the_policy_message(test_client):
    resp = await test_client.post(
        "/api/nodes/script/validate",
        json={"code": "import requests\n\ndef run(inputs, params):\n    return 1\n"},
    )
    # A rejection is a normal answer while typing, not an HTTP failure.
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert "requests" in body["error"]
    assert "custom node" in body["error"].lower()
    assert body["line"] == 1


@pytest.mark.asyncio
async def test_validate_script_reports_the_offending_line(test_client):
    resp = await test_client.post(
        "/api/nodes/script/validate",
        json={"code": "def run(inputs, params):\n    x = 1\n    return eval('2')\n"},
    )
    body = resp.json()
    assert body["ok"] is False
    assert body["line"] == 3


@pytest.mark.asyncio
async def test_validate_script_reports_a_syntax_error_line(test_client):
    resp = await test_client.post(
        "/api/nodes/script/validate",
        json={"code": "def run(inputs, params)\n    return 1\n"},
    )
    body = resp.json()
    assert body["ok"] is False
    assert body["line"] == 1


@pytest.mark.asyncio
async def test_validate_script_flags_a_missing_entry_point(test_client):
    resp = await test_client.post(
        "/api/nodes/script/validate",
        json={"code": "x = 1\n"},
    )
    body = resp.json()
    assert body["ok"] is True          # nothing about it violates the policy
    assert body["defines_run"] is False  # but it would fail at run time


@pytest.mark.asyncio
async def test_validate_script_accepts_an_empty_body(test_client):
    resp = await test_client.post("/api/nodes/script/validate", json={})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


@pytest.mark.asyncio
async def test_validate_script_refuses_an_oversized_script(test_client):
    from app.core.script_policy import MAX_SCRIPT_CHARS

    resp = await test_client.post(
        "/api/nodes/script/validate",
        json={"code": "x = 1\n" * (MAX_SCRIPT_CHARS // 4)},
    )
    body = resp.json()
    assert body["ok"] is False
    assert "limit" in body["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("label", "code"),
    [
        (
            "frame walk to the host's globals",
            "def run(inputs, params):\n"
            "    try:\n"
            "        raise ValueError()\n"
            "    except ValueError as e:\n"
            "        g = e.__traceback__.tb_frame.f_back.f_globals\n"
            "        return getattr(g['importlib'].import_module('os'), 'getcwd')()\n",
        ),
        (
            "frame walk via a generator",
            "def run(inputs, params):\n"
            "    def gen():\n"
            "        yield 1\n"
            "    return sorted(gen().gi_frame.f_globals)\n",
        ),
        (
            "os through an allowlisted module",
            "import torch\n\ndef run(inputs, params):\n    return torch.os.getcwd()\n",
        ),
        (
            "pickle loader with a laundered receiver",
            "import torch\n\ndef run(inputs, params):\n"
            "    b = torch\n    return b.load('x.pt')\n",
        ),
    ],
)
async def test_validate_script_says_no_to_the_round_two_escapes(test_client, label, code):
    """The editor's own endpoint answered ``ok: true`` for every one of these
    while the escape worked, which is the answer that mattered -- the user was
    being told the script was within policy as it read a file off disk."""
    resp = await test_client.post("/api/nodes/script/validate", json={"code": code})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False, f"endpoint still approves: {label}"
    assert body["line"], "a rejection must point the editor at a line"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("label", "code"),
    [
        (
            "the real sys under a private alias",
            "def run(inputs, params):\n    return str(collections._sys)\n",
        ),
        (
            "sys.modules subscript to os.getcwd()",
            "def run(inputs, params):\n"
            "    s = collections._sys\n"
            "    return s.modules['os'].getcwd()\n",
        ),
        (
            "os.system bound to a local first",
            "def run(inputs, params):\n"
            "    f = collections._sys.modules['os'].system\n"
            "    return f('cmd /c ver')\n",
        ),
        (
            "os two private hops from statistics",
            "def run(inputs, params):\n    return statistics.random._os.getcwd()\n",
        ),
        (
            "the pickle loader read rather than called",
            "import torch\n\ndef run(inputs, params):\n    f = torch.load\n    return f\n",
        ),
        (
            "poisoning a shared library module",
            "import torch\n\ndef run(inputs, params):\n    torch.zeros = None\n",
        ),
    ],
)
async def test_validate_script_says_no_to_the_round_three_escapes(test_client, label, code):
    """The editor answered ``ok: true`` for every one of these while they ran
    ``os.system`` and ``subprocess.run`` through the shipped node."""
    resp = await test_client.post("/api/nodes/script/validate", json={"code": code})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False, f"endpoint still approves: {label}"
    assert body["line"], "a rejection must point the editor at a line"


@pytest.mark.asyncio
async def test_the_endpoint_can_approve_what_the_runtime_then_refuses(test_client):
    """Stated as a test so nobody reads ``ok: true`` as "this will run".

    The endpoint runs the AST gate, which is keyed on names: ``json.codecs``
    is an unlisted module reached through an allowed one, so the gate has
    nothing to object to. The runtime proxy refuses it on what it RESOLVES
    to. That split is the architecture, not an oversight -- but a user who
    reads the green badge as a guarantee would be wrong, and the docs say so.
    """
    code = (
        "def run(inputs, params):\n"
        "    b = json.codecs.builtins\n"
        "    f = b.eval\n"
        "    return f('6*7')\n"
    )
    body = (
        await test_client.post("/api/nodes/script/validate", json={"code": code})
    ).json()
    assert body["ok"] is True

    from app.nodes.utility.python_script_node import PythonScriptNode

    with pytest.raises(RuntimeError, match="not on the Tier-0 list"):
        PythonScriptNode().execute({}, {"code": code})


@pytest.mark.asyncio
async def test_validate_script_still_approves_ordinary_statistics_work(test_client):
    """The round-2 rules add 25 denied attribute names and invert the pickle
    rule; the endpoint must still say yes to the scripts the docs recommend."""
    code = (
        "import numpy as np\n"
        "import json\n"
        "\n"
        "\n"
        "def run(inputs, params):\n"
        "    x = inputs['in1']\n"
        "    summary = {'mean': float(np.mean(x)), 'std': float(np.std(x))}\n"
        "    print(json.dumps(summary))\n"
        "    return {'out1': summary}\n"
    )
    body = (
        await test_client.post("/api/nodes/script/validate", json={"code": code})
    ).json()
    assert body["ok"] is True
    assert body["error"] is None
    assert body["defines_run"] is True


@pytest.mark.asyncio
async def test_python_script_node_is_registered_with_a_code_param(test_client):
    resp = await test_client.get("/api/nodes/PythonScript")
    assert resp.status_code == 200
    node = resp.json()
    assert node["category"] == "Utility"
    params = {p["name"]: p for p in node["params"]}
    assert params["code"]["param_type"] == "code"
    assert params["code"]["default"].startswith("def run(")
    # The palette template shows the default one-in / one-out shape.
    assert [p["name"] for p in node["inputs"]] == ["in1"]
    assert [p["name"] for p in node["outputs"]] == ["out1"]
