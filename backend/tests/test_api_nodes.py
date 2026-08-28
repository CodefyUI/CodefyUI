"""Tests for the nodes API endpoints."""

from typing import Any

import pytest

from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


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

    The endpoint runs the AST gate, which is keyed on names: ``numpy.f2py``
    and ``torch.utils`` are numpy's and torch's own subpackages reached
    through an allowed module, so the gate has nothing to object to. The
    runtime proxy refuses them on what they RESOLVE to. That split is the
    architecture, not an oversight -- but a user who reads the green badge as
    a guarantee would be wrong, and the docs say so.
    """
    from app.nodes.utility.python_script_node import PythonScriptNode

    cases = [
        # A literal eval(), inside numpy.
        "def run(inputs, params):\n"
        "    f = numpy.f2py.crackfortran.myeval\n"
        "    return f('1+1')\n",
        # pathlib.Path re-exported by numpy: arbitrary file read and write.
        "def run(inputs, params):\n"
        "    P = numpy.f2py.crackfortran.Path\n"
        "    P('round5.txt').write_text('pwned')\n"
        "    return 1\n",
        # torch's own subprocess wrapper, shell=True.
        "def run(inputs, params):\n"
        "    return str(torch.utils.collect_env.run('cmd /c ver'))\n",
    ]
    for code in cases:
        body = (
            await test_client.post("/api/nodes/script/validate", json={"code": code})
        ).json()
        assert body["ok"] is True
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


# ── two-tier parameter schema (core#134) ──────────────────────────────────


@pytest.mark.asyncio
async def test_every_param_carries_the_advanced_flag(test_client):
    """The flag is on the WIRE, not just on the Python dataclass.

    ``ParamDefinitionSchema`` is a hand-written mirror of
    ``ParamDefinition`` and ``_node_to_definition`` copies field by field, so
    a new attribute is dropped silently unless both are updated. This is the
    test that notices.
    """
    nodes = (await test_client.get("/api/nodes")).json()
    params = [p for node in nodes for p in node["params"]]

    assert params, "no node declares a parameter"
    assert all("advanced" in p for p in params)
    assert all(isinstance(p["advanced"], bool) for p in params)


@pytest.mark.asyncio
async def test_advanced_and_basic_params_are_both_present(test_client):
    """A flag nothing sets is a flag that proves nothing."""
    node = (await test_client.get("/api/nodes/Optimizer")).json()
    params = {p["name"]: p for p in node["params"]}

    assert params["lr"]["advanced"] is False
    assert params["momentum"]["advanced"] is False
    assert params["betas"]["advanced"] is True
    assert params["amsgrad"]["advanced"] is True


@pytest.mark.asyncio
async def test_visible_when_supports_a_list_of_accepted_values(test_client):
    """Per-type visibility needs "any of", not just equality.

    Four optimizers take ``betas``; a single-value rule could only name one
    of them.
    """
    node = (await test_client.get("/api/nodes/Optimizer")).json()
    params = {p["name"]: p for p in node["params"]}

    assert params["betas"]["visible_when"] == {
        "type": ["Adam", "AdamW", "NAdam", "RAdam"]}
    assert params["nesterov"]["visible_when"] == {"type": ["SGD"]}


@pytest.mark.asyncio
async def test_loss_and_dataloader_advertise_their_new_params(test_client):
    loss = {p["name"]: p for p in
            (await test_client.get("/api/nodes/Loss")).json()["params"]}
    assert loss["label_smoothing"]["advanced"] is False
    assert loss["reduction"]["advanced"] is True
    assert loss["reduction"]["options"] == ["mean", "sum", "none"]

    loader = {p["name"]: p for p in
              (await test_client.get("/api/nodes/DataLoader")).json()["params"]}
    assert loader["pin_memory"]["advanced"] is False
    assert loader["drop_last"]["advanced"] is True
    assert loader["prefetch_factor"]["advanced"] is True


# ── Package Center metadata (requires_pack / option_packs) ───────────────


class _PackedNode(BaseNode):
    """Stand-in for a node that only runs once an optional pack is there.

    Synthetic rather than a real node so these tests keep passing whichever
    way the catalog is edited: what is under test is the plumbing from the
    class attribute to the wire, not any particular node's needs.
    """

    NODE_NAME = "_PackedTest"
    CATEGORY = "Test"
    DESCRIPTION = "Needs an optional pack"
    REQUIRES_PACK = "rag"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="table",
                param_type=ParamType.SELECT,
                default="demo-16d",
                options=["demo-16d", "glove-50d"],
                option_packs={"glove-50d": "word-vectors"},
            ),
            ParamDefinition(name="label", param_type=ParamType.STRING, default=""),
        ]

    def execute(self, inputs: dict[str, Any],
                params: dict[str, Any]) -> dict[str, Any]:
        return {"value": None}


@pytest.fixture
def packed_node():
    """Register ``_PackedTest`` for one test, then take it back out.

    Left behind it would show up in every later registry-wide assertion --
    the zh-TW ratchet at the bottom of this file included -- as a node core
    does not actually ship.
    """
    from app.core.node_registry import registry

    registry._nodes[_PackedNode.NODE_NAME] = _PackedNode
    yield _PackedNode
    registry._nodes.pop(_PackedNode.NODE_NAME, None)


@pytest.mark.asyncio
async def test_node_definition_exposes_requires_pack(test_client, packed_node):
    """A node the base install cannot run says so before anyone runs it."""
    packed = (await test_client.get("/api/nodes/_PackedTest")).json()
    assert packed["requires_pack"] == "rag"

    plain = (await test_client.get("/api/nodes/Linear")).json()
    assert plain["requires_pack"] is None

    nodes = (await test_client.get("/api/nodes")).json()
    assert all("requires_pack" in node for node in nodes)


@pytest.mark.asyncio
async def test_param_option_packs_round_trip(test_client, packed_node):
    """Per-OPTION gating: one SELECT, some options installed, some not.

    The map arrives verbatim so the editor can grey out exactly the options
    whose pack is missing instead of disabling the whole parameter.
    """
    params = {p["name"]: p for p in
              (await test_client.get("/api/nodes/_PackedTest")).json()["params"]}

    assert params["table"]["option_packs"] == {"glove-50d": "word-vectors"}
    assert params["label"]["option_packs"] is None


@pytest.mark.asyncio
async def test_text_embedding_declares_its_pack_on_the_wire(test_client):
    """The real node the synthetic one above stands in for.

    ``_PackedTest`` proves the plumbing exists; this proves a node a learner
    actually sees is plugged into it. Both halves matter and they are read
    by different parts of the editor: ``requires_pack`` greys the node out
    in the palette, and the per-option map greys out the models THIS install
    has not downloaded inside a node it can otherwise run.
    """
    node = (await test_client.get("/api/nodes/TextEmbedding")).json()
    assert node["requires_pack"] == "sentence-embeddings"

    listed = {n["node_name"]: n
              for n in (await test_client.get("/api/nodes")).json()}
    assert listed["TextEmbedding"]["requires_pack"] == "sentence-embeddings"

    model = {p["name"]: p for p in node["params"]}["model"]
    assert set(model["option_packs"]) == set(model["options"])
    assert all(value.startswith("sentence-embeddings:")
               for value in model["option_packs"].values())


@pytest.mark.asyncio
async def test_every_param_carries_the_option_packs_key(test_client):
    """The key is on the WIRE, not just on the Python dataclass -- the same
    hand-copied mirror that ``advanced`` had to be threaded through."""
    nodes = (await test_client.get("/api/nodes")).json()
    params = [p for node in nodes for p in node["params"]]

    assert params, "no node declares a parameter"
    assert all("option_packs" in p for p in params)
    assert all(p["option_packs"] is None or isinstance(p["option_packs"], dict)
               for p in params)


# ── zh-TW coverage (#188 review M8; widened by the core#136 review) ──────

#: Nodes whose params are pinned as fully translated. Anything here that
#: gains a param without a zh-TW entry fails below.
#:
#: The four training nodes come from #188. The eleven after them are
#: core#136's: the review found that this test was still hardcoded to the
#: original four, so eleven new nodes and twenty-three new param keys were
#: translated correctly and pinned by nothing -- the same gap the test was
#: written for. ``Conv2dExplicit`` is #367's: #362 gave it three new
#: user-facing params in English on the node that C1-3 §C1.3.4.1 teaches
#: from, so it was translated and moved up here out of the debt list below.
#: ``WordVector`` was in NEITHER list: it had a zh-TW block, so the ratchet
#: below was satisfied and nothing checked its params -- and its four
#: translations were rewritten for the real GloVe and sentence backends
#: without a test that would have noticed a fifth param arriving in English.
#: ``TextEmbedding`` ships in the same PR and is pinned from birth rather
#: than translated and then forgotten: all nine of its params are knobs a
#: learner turns, and the ratchet below would have been satisfied by the
#: block alone.
TRANSLATED_NODES = (
    "Conv2dExplicit",
    "WordVector",
    "TextEmbedding",
    "Optimizer",
    "Loss",
    "DataLoader",
    "TrainingLoop",
    "EvaluateModel",
    "ImageFolderDataset",
    "Transform",
    "ComposeTransform",
    "ColorJitter",
    "NormalizeTransform",
    "RandAugment",
    "RandomCrop",
    "RandomHorizontalFlip",
    "RandomRotation",
    "ResizeTransform",
    "ToTensorTransform",
)

#: Built-in nodes with no zh-TW block at all. PRE-EXISTING debt, all of it
#: older than core#136 -- listed rather than fixed because translating
#: sixteen nodes is its own change. The point of the list is the ratchet in
#: ``test_no_new_node_ships_without_a_zh_tw_entry``: this set may shrink,
#: never grow. Delete a name from here when you translate it. Two are gone
#: already: ``Conv2dKernel`` with the node itself (#362), and
#: ``Conv2dExplicit`` by being translated (#367) -- thirteen left.
UNTRANSLATED_NODES = frozenset({
    "Argmax",
    "DatasetBatch",
    "DecisionBoundary",
    "DiffusionTrainingLoop",
    "GraphInput",
    "GraphOutput",
    "LLMChat",
    "RandomForestClassifier",
    "RowSelector",
    "ScalarMultiply",
    "ScatterPlot2D",
    "SyntheticSegmentation",
    "SyntheticShapes",
})


def _node_catalog() -> str:
    from pathlib import Path

    return (Path(__file__).resolve().parents[2] / "frontend" / "src" /
            "i18n" / "nodeLocales" / "zh-TW.ts").read_text(encoding="utf-8")


def _builtin_node_names() -> list[str]:
    """Registry names defined by core, excluding plugins and custom nodes.

    A developer machine with a plugin installed must not fail this file over
    a translation core does not own.
    """
    from app.core.node_registry import registry

    return sorted(
        name for name in registry.nodes
        if registry.get(name).__module__.startswith("app.nodes")
    )


def test_training_node_params_all_have_a_zh_tw_description():
    """Every param of every node in ``TRANSLATED_NODES`` is translated.

    This project mirrors user-facing strings in both locales and its primary
    audience reads Chinese, but nothing enforced it for NODE catalogs — so
    #134 shipped 17 English descriptions rendering directly beneath
    translated ones. The source of truth is ``define_params()``, so the
    check is against the real schema rather than a second hand-kept list.
    """
    import re

    from app.core.node_registry import registry

    catalog = _node_catalog()

    missing: list[str] = []
    for node_name in TRANSLATED_NODES:
        node_cls = registry.get(node_name)
        assert node_cls is not None, node_name
        block = re.search(
            rf"\n  {node_name}: \{{(.*?)\n  \}},", catalog, re.DOTALL)
        assert block, f"{node_name} has no zh-TW entry at all"
        body = block.group(1)
        for param in node_cls.define_params():
            if not re.search(rf"\n      {re.escape(param.name)}: ", body):
                missing.append(f"{node_name}.{param.name}")

    assert not missing, (
        "these params render in English under zh-TW; add them to "
        f"frontend/src/i18n/nodeLocales/zh-TW.ts: {missing}")


def test_no_new_node_ships_without_a_zh_tw_entry():
    """A ratchet, so the untranslated set can shrink but never grow.

    The scoped test above only guards nodes someone remembered to add to
    ``TRANSLATED_NODES`` -- which is exactly how core#136's eleven nodes
    ended up unpinned. This one is derived from the registry, so a node
    added tomorrow with no translation fails without anyone editing a list.
    """
    import re

    catalog = _node_catalog()
    untranslated = {
        name for name in _builtin_node_names()
        if not re.search(rf"\n  {re.escape(name)}: \{{", catalog)
    }

    new = sorted(untranslated - UNTRANSLATED_NODES)
    assert not new, (
        "these nodes have no zh-TW entry and render entirely in English; "
        "add them to frontend/src/i18n/nodeLocales/zh-TW.ts: " + str(new))

    fixed = sorted(UNTRANSLATED_NODES - untranslated)
    assert not fixed, (
        "these are translated now -- delete them from UNTRANSLATED_NODES so "
        "the ratchet keeps holding: " + str(fixed))


def test_the_zh_tw_catalog_has_no_entries_for_nodes_that_do_not_exist():
    """A renamed or deleted node leaves a dead translation behind."""
    import re

    from app.core.node_registry import registry

    catalog = _node_catalog()
    entries = set(re.findall(r"\n  ([A-Za-z][\w:]*): \{", catalog))
    dead = sorted(entry for entry in entries if registry.get(entry) is None)
    assert not dead, (
        "these zh-TW entries name nodes the registry does not have: "
        + str(dead))
