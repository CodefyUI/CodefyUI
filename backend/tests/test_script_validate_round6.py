"""Validate reports what the run is sure to refuse (round 6, H3).

Measured with a real model: a PythonScript importing ``sklearn.datasets``
passed ``/api/graph/validate`` and was refused by the Tier-0 policy only at
run time. Validate now runs the same gate (``validate_script_source``) through
``BaseNode.static_issues``.
"""

from __future__ import annotations

from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core.auth import TOKEN_HEADER, session_token
from app.core.graph_engine import validate_graph
from app.core.script_policy import TIER0_MODULES
from app.main import app
from app.nodes.data.csv_reader_node import CSVReaderNode
from app.nodes.utility.python_script_node import PythonScriptNode

BASE_URL = f"http://127.0.0.1:{settings.PORT}"

BLOCKED = "import numpy\nfrom sklearn.datasets import load_iris\n\ndef run(inputs, params):\n    return {}\n"
FINE = "import numpy\n\ndef run(inputs, params):\n    return {'out1': numpy.zeros(2)}\n"


def _graph(code: str) -> tuple[list[dict], list[dict]]:
    nodes = [{"id": "start", "type": "Start", "data": {"params": {}}},
             {"id": "py", "type": "PythonScript", "data": {"params": {"code": code}}}]
    edges = [{"id": "e1", "source": "start", "target": "py", "sourceHandle": "trigger",
              "targetHandle": "trigger", "type": "trigger"}]
    return nodes, edges


def _policy_issues(errors):
    return [e for e in errors if getattr(e, "code", None) == "script_policy"]


def test_a_blocked_import_is_a_validation_error_with_node_line_and_modules():
    issues = _policy_issues(validate_graph(*_graph(BLOCKED)))
    assert len(issues) == 1
    issue = issues[0]
    assert issue.node_id == "py"
    assert issue.params["line"] == 2
    assert issue.params["allowed_modules"] == list(TIER0_MODULES)
    message = str(issue)
    assert message.startswith("PythonScript node py: line 2: ")
    assert "sklearn.datasets" in message
    assert ", ".join(TIER0_MODULES) in message


def test_an_allowed_script_has_no_policy_issue():
    assert _policy_issues(validate_graph(*_graph(FINE))) == []


def test_the_check_is_the_run_gate_itself():
    """Not a second list: whatever the run's gate refuses, validate refuses."""
    for code in (BLOCKED, "import os\n\ndef run(i, p):\n    return {}\n",
                 "async def run(inputs, params):\n    return {}\n",
                 "x = eval('1')\n\ndef run(i, p):\n    return {}\n"):
        assert PythonScriptNode.static_issues({"code": code}), code
    assert PythonScriptNode.static_issues({"code": FINE}) == []


async def test_the_validate_route_says_invalid():
    nodes, edges = _graph(BLOCKED)
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL,
                           headers={TOKEN_HEADER: session_token()}) as http:
        resp = await http.post("/api/graph/validate", json={"nodes": nodes, "edges": edges})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is False
    issue = next(i for i in body["issues"] if i["code"] == "script_policy")
    assert issue["node_id"] == "py" and issue["params"]["line"] == 2


def test_the_code_param_names_every_allowed_module():
    code = next(p for p in PythonScriptNode.define_params() if p.name == "code")
    assert ", ".join(TIER0_MODULES) in code.description
    assert ", ".join(TIER0_MODULES) in PythonScriptNode.DETAILS


def test_the_reader_paths_say_how_to_name_a_bundled_sample():
    from app.nodes.llm.document_loader_node import DocumentLoaderNode
    from app.nodes.llm.text_corpus_dataset_node import TextCorpusDatasetNode

    for cls, name in ((CSVReaderNode, "path"), (DocumentLoaderNode, "directory"),
                      (TextCorpusDatasetNode, "local_path")):
        param = next(p for p in cls.define_params() if p.name == name)
        assert "data/samples/" in param.description, cls.__name__
