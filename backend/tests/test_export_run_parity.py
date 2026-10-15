"""Exported examples run, and agree with the engine (#765).

Course exams grade the file Export Python produces, and that file imports
``app.core.*`` when it runs: ``graph_engine.invoke_node``,
``ExecutionContext``, ``device_utils.resolve_device``, ``seeding``,
``runtime`` and ``api_contract``. ``test_codegen.py`` exports every example
and compiles it; nothing ran one. This module does, for every example the
in-process smoke suite (``test_builtin_examples.py``) executes:

1. export it through ``POST /api/graph/export``, the route the editor's
   button calls, and write the script into a fresh, otherwise empty folder;
2. run it the way a grader does (``test_export_contract.py``'s harness:
   ``python -I``, no ``CODEFYUI_*`` variable inherited, a dead proxy) with
   ``--device cpu`` and ``--verbose``;
3. run the same graph in-process with ``execute_graph`` on the same device
   and seed, and compare:

   * the order nodes ran in: ``--verbose`` prints ``[<id>] running`` per
     node in the script's order, which ``topological_sort`` decides at
     export time, and the engine reports ``running`` in its level order;
   * the answer: with a GraphOutput, the one JSON line on stdout against
     ``api_contract.serialize_output`` of the engine's outputs; without one,
     everything Print wrote to stdout, byte for byte.

The seed is the example's ``settings.seed``, baked into the script as its
default. An example without one runs with ``--seed 0`` on both sides: a
seeded run is serial, so its order is fixed, and its values reproduce.

Files an example reads by a relative name are copied beside the script, the
way a grader puts the test files in the working directory.

The set of examples is spelled out below, and a test keeps it equal to what
the skip lists allow, so a new example is a reviewed one-line change here.
Budget: the whole module stays under five minutes on CI. An example that
cannot is listed in ``TOO_SLOW_FOR_CI`` with a reason and skipped, rather
than dropped.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core import api_contract
from app.core.execution_context import ExecutionContext
from app.core.graph_engine import build_preset_fallback, execute_graph
from tests._example_graphs import graph_nodes
from tests.test_builtin_examples import _SLOW_NODE_TYPES
from tests.test_export_contract import RUN_TIMEOUT_S, _answer_line, _export, _grader_env

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_EXAMPLES_ROOT = _BACKEND_DIR.parent / "examples"

SCRIPT_NAME = "exported_example.py"

#: Node types that reach the network even though ``_SLOW_NODE_TYPES`` lets
#: them through. Empty today: every example the smoke suite runs works
#: offline, which the dead proxy in the grader environment holds us to.
NETWORK_NODE_TYPES: frozenset[str] = frozenset()

#: Every example this module exports and runs, by folder under ``examples/``.
#: Equal to the examples ``test_builtin_examples.py`` executes in-process
#: minus any using ``NETWORK_NODE_TYPES``; see
#: ``test_the_parity_set_is_what_the_skip_lists_allow``.
PARITY_EXAMPLES = (
    "Classical/Classifier-Showdown-Moons",
    "Classical/Iris-Sklearn-KNN",
    "Classical/Tabular-Iris-Pipeline",
    "Diffusion/Forward-Process",
    "Diffusion/Mini-UNet-Compact",
    "Diffusion/Toy-Sampling",
    "Diffusion/Train-DDPM-Blobs",
    "LLM/Word-Embedding-Analogy",
    "Model_Architecture/BERT-Encoder-Transformer",
    "Model_Architecture/BiGRU-SpeechRecognition-RNN",
    "Model_Architecture/ConvNeXt-CNN",
    "Model_Architecture/DQN-Atari-RL",
    "Model_Architecture/DiT-Diffusion-Transformer",
    "Model_Architecture/EfficientNet-CNN",
    "Model_Architecture/GPT-DecoderOnly-Transformer",
    "Model_Architecture/LLaMA-Decoder-Transformer",
    "Model_Architecture/PPO-Robotics-RL",
    "Model_Architecture/ResNet-SkipConnection-CNN",
    "Model_Architecture/Seq2Seq-Attention-RNN",
    "Model_Architecture/SwinTransformer-Transformer",
    "Model_Architecture/TimeSeries-LSTM-RNN",
    "Model_Architecture/UNet-Segmentation-CNN",
    "Model_Architecture/ViT-ImageClassifier-Transformer",
    "RL/GRPO-Step-GridWorld",
    "RL/RLHF-Reward-and-KL",
    "RL/Reward-Hacking-Shortcut",
    "RNN/RNN-OneStep",
    "Transformer/Causal-Attention-From-Scratch",
    "Transformer/MoE-TopK-Routing",
    "Usage_Example/Api-Function",
)

#: ``{example: reason}`` for an example too slow for the CI budget. It stays
#: in ``PARITY_EXAMPLES`` and is reported as skipped with the reason.
TOO_SLOW_FOR_CI: dict[str, str] = {}

#: Files the examples name by a relative path, as ``{path in the working
#: directory: the repository copy}``. Every example gets all of them.
FILES_BESIDE_THE_SCRIPT = {
    "data/samples/iris.csv": _BACKEND_DIR / "data" / "samples" / "iris.csv",
    "test_digit.png": _BACKEND_DIR / "data" / "images" / "test_digit.png",
}

#: Used on both sides when an example carries no seed of its own.
FALLBACK_SEED = 0

_RUNNING_LINE = re.compile(r"^\[(?P<node_id>.+)\] running$", re.MULTILINE)


def _graph_dir(example: str) -> Path:
    return _EXAMPLES_ROOT / example


def _load(example: str) -> dict:
    return json.loads((_graph_dir(example) / "graph.json").read_text(encoding="utf-8"))


def _node_types(graph: dict) -> set[str]:
    return {node["type"] for node in graph_nodes(graph)}


def test_the_parity_set_is_what_the_skip_lists_allow():
    """A new example that the smoke suite runs must be added here too."""
    derived = set()
    for path in sorted(_EXAMPLES_ROOT.rglob("graph.json")):
        graph = json.loads(path.read_text(encoding="utf-8"))
        types = _node_types(graph)
        if types & _SLOW_NODE_TYPES or types & NETWORK_NODE_TYPES:
            continue
        derived.add(path.parent.relative_to(_EXAMPLES_ROOT).as_posix())
    assert set(PARITY_EXAMPLES) == derived
    assert len(PARITY_EXAMPLES) == len(set(PARITY_EXAMPLES))
    assert set(TOO_SLOW_FOR_CI) <= set(PARITY_EXAMPLES)


def _stage_folder(folder: Path) -> None:
    for relative, source in FILES_BESIDE_THE_SCRIPT.items():
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _run_exported(script: str, folder: Path, home: Path,
                  extra_args: list[str]) -> subprocess.CompletedProcess:
    (folder / SCRIPT_NAME).write_bytes(script.encode("utf-8"))
    return subprocess.run(
        # -I and the scrubbed environment: see test_export_contract.py.
        [sys.executable, "-I", SCRIPT_NAME, "--device", "cpu", "--verbose",
         *extra_args],
        cwd=folder,
        env=_grader_env(home),
        capture_output=True,
        timeout=RUN_TIMEOUT_S,
        check=False,
    )


def _engine_run(graph: dict, seed: int) -> tuple[dict, list[str]]:
    """``execute_graph`` on the graph as the export route sees it."""
    nodes = graph_nodes(graph)
    kept = {node["id"] for node in nodes}
    edges = [edge for edge in graph.get("edges", [])
             if edge["source"] in kept and edge["target"] in kept]
    order: list[str] = []

    def on_progress(node_id, status, data=None):
        if status == "running":
            order.append(node_id)

    results = asyncio.run(execute_graph(
        nodes,
        edges,
        on_progress=on_progress,
        context=ExecutionContext(device="cpu", seed=seed),
        error_mode="fail_fast",
        preset_fallback=build_preset_fallback(graph.get("presets") or []),
        subgraphs=graph.get("subgraphs") or None,
    ))
    return results, order


def _engine_answer(graph: dict, results: dict) -> dict:
    contract = api_contract.derive_contract(graph_nodes(graph))
    collected, missing = api_contract.collect_outputs(contract, results)
    assert not missing, f"the engine produced no value for {missing}"
    serialized = {name: api_contract.serialize_output(value)
                  for name, value in collected.items()}
    # Through JSON once, as the script does, so tuples and float spelling
    # compare the way they arrive on the script's stdout.
    return json.loads(json.dumps(serialized, ensure_ascii=False, sort_keys=True))


@pytest.mark.parametrize("example", PARITY_EXAMPLES)
async def test_exported_example_matches_the_engine(
        example, test_client, tmp_path, monkeypatch, capsys):
    if example in TOO_SLOW_FOR_CI:
        pytest.skip(TOO_SLOW_FOR_CI[example])

    graph = _load(example)
    has_output = "GraphOutput" in _node_types(graph)
    own_seed = (graph.get("settings") or {}).get("seed")
    seed = FALLBACK_SEED if own_seed is None else own_seed

    script = await _export(test_client, graph)
    folder, home = tmp_path / "grader", tmp_path / "home"
    _stage_folder(folder)
    run = await asyncio.to_thread(
        _run_exported, script, folder, home,
        [] if own_seed is not None else ["--seed", str(FALLBACK_SEED)])
    stdout = run.stdout.decode("utf-8")
    stderr = run.stderr.decode("utf-8", errors="replace")
    assert run.returncode == 0, f"exit {run.returncode}\nstderr:\n{stderr}"
    assert re.search(r"completed on cpu \(\d+ node results\)\.$", stderr,
                     re.MULTILINE), stderr

    # Same folder as the script, so relative names resolve the same way.
    monkeypatch.chdir(folder)
    capsys.readouterr()
    results, engine_order = await asyncio.to_thread(_engine_run, graph, seed)
    engine_stdout = capsys.readouterr().out

    exported_order = _RUNNING_LINE.findall(stderr)
    assert exported_order == engine_order, (
        "the exported script runs its nodes in another order than the engine")

    if has_output:
        answer = json.loads(
            _answer_line(SimpleNamespace(stdout=stdout, stderr=stderr)))
        assert answer == _engine_answer(graph, results)
    else:
        assert stdout == engine_stdout

