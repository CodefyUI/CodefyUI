"""Memory estimates for a run's default budget (#794).

#788 gave a run on a card shared by memory a default budget from the last
peak of the same graph (``memory_admission.HistoryBudgetSource``). This
module adds what that needs and what it lacks:

- :func:`structural_hash`, the history key: what the executable graph
  computes (node types, the params that shape it, the wiring), with node
  ids, positions, labels, seeds, names and the knobs that only change how
  long a run takes (``epochs``, ``max_steps``, ``lr``) left out, so a rerun
  with another seed or more epochs finds its history.
- :func:`estimate_static`, for a graph with no history: parameter counts of
  its model nodes, times what training holds per parameter (weights,
  gradients, optimizer state: the usual 4x for Adam), plus a rough
  activation term for its batch size, plus a fixed allowance for the CUDA
  libraries' workspaces. **An estimate, never a measurement**: the API and
  the dialog say which one a suggestion is (``source: "estimate"``).
- :class:`EstimateBudgetSource`, the default ``BudgetSource``: the history
  first, then the static estimate when it is usable (``confidence:
  "medium"``).

How the static estimate counts, without building a real model
--------------------------------------------------------------
A node that holds weights through :class:`StatefulModuleMixin` builds its
module from its params alone (``build_module(params)``). That build runs
on PyTorch's ``meta`` device (shapes, no storage), in a single worker
thread off the event loop, under :data:`STATIC_TIMEOUT_S`; torch is
imported there, never on the request path. The same meta module then runs
one forward pass on a meta input of the run's batch size, with a hook on
every leaf layer adding up the size of what it outputs: the activations a
training step keeps for backward. The input's shape comes from the model's
first layer (``in_features``, ``in_channels``, ...) and the graph's data
nodes (``image_size``, ``seq_len``, a known dataset's resolution); when no
shape fits, or the node has no recipe for its inputs, the activation term is
missing and the estimate is ``low`` confidence (shown, never used as a
default). A few small built-in models that do not use the mixin (DQN, PPO,
RewardModel) are counted by formula. A model node neither path can count
is listed in ``uncounted`` and makes the estimate ``low`` as well.

The constants below were tuned against measured peaks of built-in examples
on an RTX PRO 6000 (#794's PR has the table), so the estimate sits above
the measured ``reserved_bytes`` peak for those graphs without being several
times it.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Mapping

from . import memory_admission as mem

logger = logging.getLogger(__name__)

# ── the structural hash ──────────────────────────────────────────────────

#: Params that never change how much device memory a run needs: identity
#: and display text, seeds, how long it runs, and the host-side loader
#: workers.
IGNORED_PARAMS = frozenset({
    "seed", "name", "label", "title", "description", "comment", "note",
    "notes", "tag", "tags", "color", "lr", "learning_rate", "epochs",
    "episodes", "max_steps", "log_interval", "early_stopping_patience",
    "patience", "tensorboard", "checkpoint_every", "checkpoint_every_steps",
    "val_every_steps",
    # How the batches are fed, on the host: the device sees the same ones.
    "num_workers", "persistent_workers", "prefetch_factor", "shuffle",
})


def _ignored(key: str) -> bool:
    return key in IGNORED_PARAMS or key.endswith("_seed")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      default=str)


def _structural_value(value: Any) -> Any:
    """A param value as the hash sees it: a layer graph (SequentialModel's
    ``layers``, a JSON string) by its own structure, so moving a box in the
    architecture editor or renaming a layer is the same model."""
    if isinstance(value, str) and value[:1] == "{" and '"nodes"' in value:
        try:
            spec = json.loads(value)
        except ValueError:
            return value
        if isinstance(spec, Mapping):
            return _structure(list(spec.get("nodes") or ()),
                              list(spec.get("edges") or ()), layer=True)
    return value


def _node_label(node: Mapping[str, Any], *, layer: bool) -> str:
    if layer:
        # A layer-graph node: ``{id, type, params, ports}``; port names
        # are the model's input/output names, which the wiring reads.
        params = node.get("params") or {}
        extra = [port.get("name") for port in node.get("ports") or ()
                 if isinstance(port, Mapping)]
    else:
        params = (node.get("data") or {}).get("params") or {}
        # A preset card read as drawn (the graph did not prepare): its
        # inner nodes' params are what tells two cards apart.
        inner = (node.get("data") or {}).get("internalParams") or {}
        extra = sorted(
            _canonical([name, {key: value for key, value in values.items()
                               if not _ignored(str(key))}])
            for name, values in inner.items() if isinstance(values, Mapping)
        ) if isinstance(inner, Mapping) else []
    kept = {key: _structural_value(value) for key, value in params.items()
            if isinstance(key, str) and not _ignored(key)}
    return hashlib.sha256(_canonical(
        [node.get("type"), kept, extra]).encode("utf-8")).hexdigest()


def _structure(nodes: list[Any], edges: list[Any], *,
               layer: bool = False) -> dict[str, Any]:
    """Nodes as the sorted multiset of their labels (type + kept params) and
    edges as the multiset of ``(source label, handle, target label,
    handle)``: no node id appears, so a copy-pasted graph with fresh ids
    hashes the same. Two nodes with one label are interchangeable for
    memory, which is all the hash has to tell apart."""
    labels = {node.get("id"): _node_label(node, layer=layer)
              for node in nodes if isinstance(node, Mapping)}
    wires = sorted(
        _canonical([labels.get(edge.get("source")), edge.get("sourceHandle"),
                    labels.get(edge.get("target")), edge.get("targetHandle")])
        for edge in edges if isinstance(edge, Mapping))
    return {"nodes": sorted(labels.values()), "edges": wires}


def executable(graph: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
    """The graph as it executes: presets and subgraphs expanded, bypassed
    and note nodes gone, params filled with their defaults
    (``prepare_executable_graph``). A graph that does not validate is read
    as drawn: a suggestion is never an error."""
    from .graph_engine import build_preset_fallback, prepare_executable_graph

    nodes = [node for node in graph.get("nodes") or ()
             if isinstance(node, dict)]
    edges = [edge for edge in graph.get("edges") or ()
             if isinstance(edge, dict)]
    try:
        nodes, edges, _ = prepare_executable_graph(
            nodes, edges,
            preset_fallback=build_preset_fallback(graph.get("presets") or []),
            subgraphs=graph.get("subgraphs") or [], fill_defaults=True)
    except Exception:  # noqa: BLE001 -- read it as drawn instead
        logger.debug("memory estimate: the graph does not prepare; "
                     "reading it as drawn", exc_info=True)
    return list(nodes), list(edges)


def structural_hash(graph: Mapping[str, Any],
                    options: Mapping[str, Any] | None = None) -> str:
    """The history key of a run of *graph* with *options* (#794 spec 1):
    the structure of the executable graph (:func:`_structure`) and the
    run's device kind and count (``memory_admission.run_shape``)."""
    nodes, edges = executable(graph)
    document = {"graph": _structure(nodes, edges),
                "shape": mem.run_shape(options)}
    return hashlib.sha256(_canonical(document).encode("utf-8")).hexdigest()


# ── the static estimate ──────────────────────────────────────────────────

#: Bytes of one parameter (weights are kept in float32, mixed precision
#: included: autocast casts per op).
PARAM_BYTES = 4
#: Optimizer state tensors per trainable parameter, by ``Optimizer.type``
#: (lower case). Weights + gradients + these = the multiplier: Adam 4x,
#: SGD with momentum 3x, plain SGD 2x.
OPTIMIZER_STATES = {
    "adam": 2, "adamw": 2, "nadam": 2, "radam": 2, "adamax": 2,
    "sgd": 1, "rmsprop": 1, "adagrad": 1, "adadelta": 2, "lbfgs": 2,
}
#: An optimizer whose type is not in the table.
DEFAULT_OPTIMIZER_STATES = 2
#: Nodes whose presence means a training step (gradients) runs on a model.
TRAINING_NODES = frozenset({
    "TrainingLoop", "DiffusionTrainingLoop", "RLTrainingLoop",
    "BackwardOnce", "BradleyTerryTrain",
})
#: Nodes that run a model without gradients, and their batch params.
EVAL_NODES = frozenset({
    "EvaluateModel", "PerplexityEvaluate", "Inference", "VLAActionEval",
})
#: A training run's activations as a multiple of what its leaf layers
#: output in one forward pass: what backward keeps, plus the gradient
#: buffers alive at its peak.
TRAIN_ACTIVATION_FACTOR = 2.0
#: An evaluation pass (no gradients) at its own batch, in float32 units.
#: More than "a few layers' outputs": a layer graph (``GraphModelModule``)
#: keeps every layer's output until its forward returns, so an evaluation
#: at a large batch is often a run's peak (measured: ResNet-18, 32 px,
#: batch 512 under ``no_grad``, 4.0 GiB against 0.6 GiB for training at
#: batch 128).
EVAL_ACTIVATION_FACTOR = 1.2
#: Bytes per activation element in training: float32, or under bf16/fp16
#: autocast a mix of half-precision outputs and the float32 ones norms and
#: losses keep. Evaluation is counted in float32.
ACTIVATION_BYTES = {"fp32": 4.0, "bf16": 2.5, "fp16": 2.5}
#: The loss of a model's final output (log-softmax, its gradient), as
#: float32 copies of that output.
LOSS_COPIES = 3
#: cuBLAS/cuDNN workspaces and the allocator's block rounding, which every
#: CUDA run holds beyond its tensors (the CUDA context itself is counted by
#: admission, ``CODEFYUI_EXECUTION_CUDA_CONTEXT_MIB``).
WORKSPACE_BYTES = 128 * 1024 ** 2
#: What the caching allocator reserves beyond what is allocated
#: (fragmentation), applied to the tensors' total (measured 1.05-1.25 for
#: runs above a GiB).
RESERVE_FACTOR = 1.2
#: A batch size when the graph names none (``low`` confidence).
DEFAULT_BATCH = 32
#: The probe's own batch: activations are linear in it, and 2 keeps a
#: train-mode BatchNorm from refusing a batch of one.
PROBE_BATCH = 2
#: How long one static estimate may take (the first one imports torch).
STATIC_TIMEOUT_S = 30.0

#: Image sides tried for a model whose first layer is a 2-D conv, after the
#: graph's own hints (largest first).
IMAGE_SIDES = (28, 32, 64, 16, 96, 128, 224, 256, 8)
#: Sequence lengths tried for a sequence model, after the graph's hints.
SEQUENCE_LENGTHS = (128, 64, 32, 256, 16, 512, 10, 1024)
#: Known datasets' image side (``Dataset.name``).
DATASET_SIDES = {
    "mnist": 28, "fashionmnist": 28, "kmnist": 28, "emnist": 28,
    "cifar10": 32, "cifar100": 32, "svhn": 32, "stl10": 96,
}
#: Param names whose integer values are image sides / sequence lengths.
IMAGE_HINT_PARAMS = ("image_size", "resize", "size", "crop_size", "img_size")
SEQUENCE_HINT_PARAMS = ("seq_len", "block_size", "context_length",
                        "max_len", "sequence_length")

CONFIDENCE_MEASURED = "measured"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"


@dataclass
class ModelCount:
    """One model node's share of the estimate."""

    node_type: str
    parameters: int
    trainable: int
    #: Elements its leaf layers output for one sample (forward pass), or
    #: None when the probe found no input shape.
    activation_elements: float | None = None
    #: Elements of the model's final output for one sample.
    output_elements: float = 0.0
    #: The input shape the probe used, for the breakdown (one sample).
    probe_input: list[list[int]] | None = None


@dataclass
class StaticEstimate:
    """A static estimate and how it was reached (``describe()`` is the API's
    ``static`` object)."""

    bytes: int
    confidence: str
    parameters: int
    trainable_parameters: int
    training: bool
    optimizer: str | None
    multiplier: int
    batch_size: int
    eval_batch_size: int
    precision: str
    weights_bytes: int
    activation_bytes: int
    workspace_bytes: int
    models: list[ModelCount] = field(default_factory=list)
    #: Model node types that could not be counted.
    uncounted: list[str] = field(default_factory=list)
    #: Why the confidence is low, in words (English, for logs and the
    #: API; the dialog builds its own sentence from the fields).
    notes: list[str] = field(default_factory=list)
    #: No batch size in the graph: :data:`DEFAULT_BATCH` was assumed.
    batch_assumed: bool = False
    #: Which pass the activation term is mostly from: ``training`` (at
    #: ``batch_size``), ``evaluation`` (at ``eval_batch_size``), or
    #: ``none`` (no activation term: only models counted by formula).
    activation_phase: str = "training"

    def describe(self) -> dict[str, Any]:
        return {
            "bytes": self.bytes, "confidence": self.confidence,
            "parameters": self.parameters,
            "trainable_parameters": self.trainable_parameters,
            "training": self.training, "optimizer": self.optimizer,
            "multiplier": self.multiplier, "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "precision": self.precision,
            "weights_bytes": self.weights_bytes,
            "activation_bytes": self.activation_bytes,
            "workspace_bytes": self.workspace_bytes,
            "models": [{"type": m.node_type, "parameters": m.parameters,
                        "activations_known": m.activation_elements
                        is not None, "probe_input": m.probe_input}
                       for m in self.models],
            "uncounted": self.uncounted, "notes": self.notes,
            "batch_assumed": self.batch_assumed,
            "activation_phase": self.activation_phase,
        }


def _params(node: Mapping[str, Any]) -> dict[str, Any]:
    params = (node.get("data") or {}).get("params")
    return params if isinstance(params, dict) else {}


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _hints(nodes: Iterable[Mapping[str, Any]]) -> tuple[list[int], list[int]]:
    """Image sides and sequence lengths the graph names, largest first."""
    sides: set[int] = set()
    lengths: set[int] = set()
    for node in nodes:
        params = _params(node)
        if node.get("type") == "Dataset":
            side = DATASET_SIDES.get(
                str(params.get("name") or "").replace("-", "").replace(
                    "_", "").lower())
            if side:
                sides.add(side)
        for key in IMAGE_HINT_PARAMS:
            value = _int(params.get(key))
            if value and 1 < value <= 4096:
                sides.add(value)
        for key in SEQUENCE_HINT_PARAMS:
            value = _int(params.get(key))
            if value and 1 < value <= 1_000_000:
                lengths.add(value)
    return sorted(sides, reverse=True), sorted(lengths, reverse=True)


def _batch_sizes(nodes: list[Mapping[str, Any]], edges: list[Mapping[str, Any]]
                 ) -> tuple[int | None, int | None]:
    """``(training batch, evaluation batch)``: a DataLoader wired into a
    training node's ``dataloader`` input is a training batch (as is a
    training node's own ``batch_size``); every other batch is evaluation."""
    by_id = {node.get("id"): node for node in nodes}
    train: list[int] = []
    other: list[int] = []
    feeds_training: set[Any] = set()
    for edge in edges:
        target = by_id.get(edge.get("target"))
        if target is not None and target.get("type") in TRAINING_NODES \
                and edge.get("targetHandle") in ("dataloader", None):
            feeds_training.add(edge.get("source"))
    for node in nodes:
        size = _int(_params(node).get("batch_size"))
        if not size or size <= 0:
            continue
        if node.get("type") in TRAINING_NODES or (
                node.get("type") == "DataLoader"
                and node.get("id") in feeds_training):
            train.append(size)
        else:
            other.append(size)
    return (max(train) if train else None, max(other) if other else None)


def _optimizer(nodes: Iterable[Mapping[str, Any]]) -> tuple[str | None, int]:
    """The graph's optimizer type and its state tensors per parameter (the
    most of any optimizer node)."""
    found: str | None = None
    states = -1
    for node in nodes:
        if node.get("type") != "Optimizer":
            continue
        kind = str(_params(node).get("type") or "Adam")
        count = OPTIMIZER_STATES.get(kind.lower(), DEFAULT_OPTIMIZER_STATES)
        if kind.lower() == "sgd" and not (
                _float(_params(node).get("momentum")) or 0.0) > 0:
            count = 0
        if kind.lower() in ("adam", "adamw") and _params(node).get("amsgrad"):
            count += 1
        if count > states:
            found, states = kind, count
    return found, max(states, 0)


def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def _precision(nodes: Iterable[Mapping[str, Any]]) -> str:
    for node in nodes:
        value = str(_params(node).get("precision") or "").lower()
        if node.get("type") in TRAINING_NODES and value in ("bf16", "fp16"):
            return value
    return "fp32"


# Formulas for built-in models that do not use StatefulModuleMixin.
def _mlp(*widths: int) -> int:
    return sum(a * b + b for a, b in zip(widths, widths[1:]))


def _formula(node_type: str, params: Mapping[str, Any]) -> int | None:
    def get(key: str, default: int) -> int:
        value = _int(params.get(key))
        return value if value and value > 0 else default

    if node_type == "DQN":
        state, hidden, action = (get("state_dim", 4), get("hidden_dim", 128),
                                 get("action_dim", 2))
        return _mlp(state, hidden, hidden, action)
    if node_type == "PPO":
        state, hidden, action = (get("state_dim", 4), get("hidden_dim", 64),
                                 get("action_dim", 2))
        return _mlp(state, hidden, hidden) + _mlp(hidden, action) \
            + _mlp(hidden, 1)
    if node_type == "RewardModel":
        return _mlp(get("input_dim", 128), get("hidden_dim", 64), 1)
    return None


FORMULA_TYPES = frozenset({"DQN", "PPO", "RewardModel"})


def _model_source(node_cls: Any) -> bool:
    """Whether a node class makes a model of its own: it outputs a MODEL
    and takes none (a training or evaluation node passes one through)."""
    try:
        outputs = node_cls.define_outputs()
        inputs = node_cls.define_inputs()
    except Exception:  # noqa: BLE001 -- a broken plugin node
        return False

    def kind(port: Any) -> str:
        value = getattr(port, "data_type", "")
        return str(getattr(value, "value", value))

    return any(kind(port) == "MODEL" for port in outputs) and not any(
        kind(port) == "MODEL" for port in inputs)


# ── the meta-device probe (worker thread only) ───────────────────────────

_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None
_EXECUTOR_LOCK = threading.Lock()
_CACHE: dict[str, ModelCount | None] = {}
_CACHE_MAX = 256


def _executor() -> concurrent.futures.ThreadPoolExecutor:
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="memory-estimate")
        return _EXECUTOR


def _first_layers(spec: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """For each input port of a layer graph, the first layer with
    parameters it reaches (breadth first): what decides the input shape."""
    nodes = {node.get("id"): node for node in spec.get("nodes") or ()
             if isinstance(node, Mapping)}
    out: dict[Any, list[tuple[Any, Any]]] = {}
    for edge in spec.get("edges") or ():
        if isinstance(edge, Mapping):
            out.setdefault(edge.get("source"), []).append(
                (edge.get("sourceHandle"), edge.get("target")))
    found: dict[str, Mapping[str, Any]] = {}
    for node in nodes.values():
        if node.get("type") != "Input":
            continue
        for port in node.get("ports") or ():
            if not isinstance(port, Mapping):
                continue
            queue = [target for handle, target in out.get(node.get("id"), [])
                     if handle in (port.get("id"), None)]
            seen: set[Any] = set()
            while queue:
                current = queue.pop(0)
                if current in seen or current not in nodes:
                    continue
                seen.add(current)
                layer = nodes[current]
                if layer.get("type") in _SHAPED_LAYERS:
                    found[str(port.get("name"))] = layer
                    break
                queue.extend(target for _, target in out.get(current, []))
    return found


#: Layer types whose params say what input they take.
_SHAPED_LAYERS = frozenset({
    "Linear", "Conv1d", "Conv2d", "ConvTranspose2d", "BatchNorm2d",
    "BatchNorm1d", "InstanceNorm2d", "GroupNorm", "LayerNorm", "Embedding",
    "LSTM", "GRU", "RNN", "TransformerEncoder", "TransformerDecoder",
    "MultiHeadAttention", "PositionalEncoding",
})


def _layer_inputs(layer: Mapping[str, Any], sides: list[int],
                  lengths: list[int]) -> list[tuple[str, tuple[int, ...]]]:
    """Candidate one-sample input shapes for a layer-graph input port whose
    first shaped layer is *layer* (``("float"|"long", shape)``, no batch)."""
    kind = layer.get("type")
    params = layer.get("params") or {}

    def get(*keys: str) -> int | None:
        for key in keys:
            value = _int(params.get(key))
            if value:
                return value
        return None

    image = list(dict.fromkeys([*sides, *IMAGE_SIDES]))
    seqs = list(dict.fromkeys([*lengths, *SEQUENCE_LENGTHS]))
    if kind == "Linear":
        width = get("in_features") or 1
        return [("float", (width,))] + [("float", (t, width)) for t in seqs]
    if kind == "Conv1d":
        channels = get("in_channels") or 1
        return [("float", (channels, t)) for t in seqs]
    if kind in ("Conv2d", "ConvTranspose2d", "BatchNorm2d", "InstanceNorm2d",
                "GroupNorm"):
        channels = get("in_channels", "num_features", "num_channels") or 1
        return [("float", (channels, s, s)) for s in image]
    if kind == "BatchNorm1d":
        width = get("num_features") or 1
        return [("float", (width,))] + [("float", (width, t)) for t in seqs]
    if kind == "Embedding":
        return [("long", (t,)) for t in seqs]
    width = get("normalized_shape", "input_size", "d_model", "embed_dim") or 1
    shapes = [("float", (t, width)) for t in seqs]
    if kind == "LayerNorm":
        shapes.insert(0, ("float", (width,)))
    return shapes


def _probe_inputs(node_type: str, params: Mapping[str, Any], module: Any,
                  sides: list[int], lengths: list[int]
                  ) -> list[list[tuple[str, tuple[int, ...]]]] | None:
    """Candidate argument lists (one sample each) for *module*'s forward, or
    None when there is no recipe for this node type."""
    if node_type == "SequentialModel":
        try:
            spec = json.loads(params.get("layers") or "{}")
        except (TypeError, ValueError):
            return None
        firsts = _first_layers(spec)
        ports = [name for _, name in getattr(module, "input_ports", ())]
        if not ports or any(port not in firsts for port in ports):
            return None
        options = [_layer_inputs(firsts[port], sides, lengths)
                   for port in ports]
        if len(options) == 1:
            return [[shape] for shape in options[0]]
        # Several inputs: try each candidate index across all of them (the
        # same sequence length / side for every input), not the product.
        width = min(len(option) for option in options)
        return [[option[i] for option in options] for i in range(width)]
    if node_type == "CausalLMModel":
        longest = int(getattr(module, "max_seq_len", 0) or 1024)
        fitting = [t for t in lengths if t <= longest]
        return [[("long", (fitting[0] if fitting else longest,))]]
    if node_type == "DiffusionUNet":
        channels = _int(params.get("in_channels")) or 3
        image = list(dict.fromkeys([*sides, 32, 16, 64, 28]))
        return [[("float", (channels, s, s)), ("long", ())] for s in image]
    if node_type == "VLAModel":
        side = int(getattr(module, "image_size", 0) or 96)
        text = int(getattr(module, "max_text_len", 0) or 48)
        chunk = int(getattr(module, "chunk", 0) or 8)
        action = int(getattr(module, "action_dim", 0) or 2)
        return [[("tuple", (("float", (3, side, side)), ("long", (text,)),
                            ("float", (chunk, action))))]]
    return None


def _meta_tensor(torch: Any, spec: tuple[str, Any], batch: int) -> Any:
    kind, shape = spec
    if kind == "tuple":
        return tuple(_meta_tensor(torch, part, batch) for part in shape)
    full = (batch, *shape)
    if kind == "long":
        return torch.zeros(full, dtype=torch.long, device="meta")
    return torch.zeros(full, dtype=torch.float32, device="meta")


def _numel(value: Any) -> int:
    numel = getattr(value, "numel", None)
    if callable(numel) and getattr(value, "shape", None) is not None:
        return int(numel())
    if isinstance(value, (tuple, list)):
        return sum(_numel(part) for part in value)
    if isinstance(value, Mapping):
        return sum(_numel(part) for part in value.values())
    return 0


def _forward_probe(torch: Any, module: Any,
                   candidates: list[list[tuple[str, Any]]]
                   ) -> tuple[float, float, list[list[int]]] | None:
    """One sample's ``(leaf outputs, final output, input shape)`` from the
    first candidate input *module* accepts, or None."""
    nn = torch.nn
    totals = {"leaf": 0}
    handles = []

    def leaf_hook(layer: Any, inputs: Any, kwargs: Any, output: Any) -> None:
        totals["leaf"] += _numel(output)
        if isinstance(layer, nn.MultiheadAttention) and inputs \
                and kwargs.get("need_weights", True) is not False:
            # The attention weights (batch x heads x queries x keys), which
            # a call that asks for them materialises and keeps for backward
            # (``need_weights=False``, what TransformerEncoderLayer passes,
            # runs the fused kernel instead).
            query = inputs[0]
            key = inputs[1] if len(inputs) > 1 else query
            shape, other = getattr(query, "shape", ()), getattr(key, "shape", ())
            if len(shape) == 3 and len(other) == 3:
                batch_first = getattr(layer, "batch_first", False)
                b, t = (shape[0], shape[1]) if batch_first else (shape[1],
                                                                 shape[0])
                s = other[1] if batch_first else other[0]
                totals["leaf"] += 2 * b * layer.num_heads * t * s

    for layer in module.modules():
        if isinstance(layer, nn.MultiheadAttention) or not any(
                True for _ in layer.children()):
            handles.append(layer.register_forward_hook(leaf_hook,
                                                       with_kwargs=True))
    try:
        module.train()
        for candidate in candidates:
            totals["leaf"] = 0
            try:
                args = [_meta_tensor(torch, spec, PROBE_BATCH)
                        for spec in candidate]
                with torch.no_grad():
                    output = module(*args)
            except Exception:  # noqa: BLE001 -- the next shape, then none
                continue
            shapes = [list(_shape_of(spec)) for spec in candidate]
            return (totals["leaf"] / PROBE_BATCH,
                    _numel(output) / PROBE_BATCH, shapes)
        return None
    finally:
        for handle in handles:
            handle.remove()


def _shape_of(spec: tuple[str, Any]) -> tuple[int, ...]:
    kind, shape = spec
    if kind == "tuple":
        return tuple(dim for part in shape for dim in _shape_of(part))
    return tuple(shape)


def _count_module(node_cls: Any, node_type: str, params: dict[str, Any],
                  sides: list[int], lengths: list[int]) -> ModelCount | None:
    """Build *node_cls*'s module on the meta device and count it (worker
    thread). None when the build fails."""
    key = _canonical([node_type, params, sides, lengths])
    if key in _CACHE:
        return _CACHE[key]
    import torch

    result: ModelCount | None = None
    try:
        with torch.device("meta"):
            module = node_cls().build_module(dict(params))
        parameters = list({id(p): p for p in module.parameters()}.values())
        count = ModelCount(
            node_type=node_type,
            parameters=sum(int(p.numel()) for p in parameters),
            trainable=sum(int(p.numel()) for p in parameters
                          if p.requires_grad))
        if node_type == "CausalLMModel":
            # Its first forward checks the ids against the vocabulary with
            # ``.item()``, which a meta tensor cannot answer.
            module._ids_checked = True
        candidates = _probe_inputs(node_type, params, module, sides, lengths)
        if candidates:
            probed = _forward_probe(torch, module, candidates)
            if probed is not None:
                (count.activation_elements, count.output_elements,
                 count.probe_input) = probed
        result = count
    except Exception:  # noqa: BLE001 -- counted as not counted
        logger.debug("memory estimate: %s did not build on the meta device",
                     node_type, exc_info=True)
        result = None
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.clear()
    _CACHE[key] = result
    return result


def estimate_static(graph: Mapping[str, Any]) -> StaticEstimate | None:
    """The static estimate of *graph* (#794), or None when it has no model
    to count. Synchronous and may import torch: call it through
    :func:`estimate_static_async`, never on the event loop."""
    from .node_registry import registry
    from .stateful_module import StatefulModuleMixin

    nodes, edges = executable(graph)
    sides, lengths = _hints(nodes)
    models: list[ModelCount] = []
    uncounted: list[str] = []
    notes: list[str] = []
    for node in nodes:
        node_type = str(node.get("type") or "")
        node_cls = registry.get(node_type)
        if node_cls is None:
            continue
        if node_type in FORMULA_TYPES:
            count = _formula(node_type, _params(node)) or 0
            models.append(ModelCount(node_type, count, count, 0.0, 0.0))
            continue
        if isinstance(node_cls, type) and issubclass(node_cls,
                                                     StatefulModuleMixin):
            counted = _count_module(node_cls, node_type, _params(node),
                                    sides, lengths)
            if counted is None:
                uncounted.append(node_type)
            else:
                models.append(counted)
            continue
        if _model_source(node_cls):
            uncounted.append(node_type)
    if not models and not uncounted:
        return None

    types = {str(node.get("type")) for node in nodes}
    training = bool(types & TRAINING_NODES)
    optimizer, states = _optimizer(nodes)
    if training and optimizer is None:
        states = DEFAULT_OPTIMIZER_STATES
    precision = _precision(nodes)
    train_batch, eval_batch = _batch_sizes(nodes, edges)
    batch_assumed = training and train_batch is None
    if batch_assumed:
        notes.append(f"no batch size found; assumed {DEFAULT_BATCH}")
        train_batch = DEFAULT_BATCH
    eval_batch = eval_batch or (train_batch if training else None)
    if eval_batch is None:
        eval_batch = 1
    train_batch = train_batch or 0

    parameters = sum(m.parameters for m in models)
    trainable = sum(m.trainable for m in models)
    multiplier = (2 + states) if training else 1
    weights = parameters * PARAM_BYTES + (
        trainable * PARAM_BYTES * (1 + states) if training else 0)
    if training and precision != "fp32":
        # Autocast's half-precision copies of the weights.
        weights += parameters * 2

    element = ACTIVATION_BYTES.get(precision, 4.0)
    activations = 0.0
    by_phase = {"training": 0.0, "evaluation": 0.0}
    for model in models:
        if model.node_type in FORMULA_TYPES or not model.parameters:
            continue
        if model.activation_elements is None:
            notes.append(f"{model.node_type}: no input shape for an "
                         f"activation estimate")
            continue
        per_sample_train = (model.activation_elements * element
                            * TRAIN_ACTIVATION_FACTOR
                            + model.output_elements * 4 * LOSS_COPIES)
        per_sample_eval = (model.activation_elements * 4.0
                           * EVAL_ACTIVATION_FACTOR
                           + model.output_elements * 4)
        train_term = train_batch * per_sample_train if training else 0.0
        eval_term = eval_batch * per_sample_eval
        activations += max(train_term, eval_term)
        by_phase["training" if train_term >= eval_term
                 else "evaluation"] += max(train_term, eval_term)
    if uncounted:
        notes.append("not counted: " + ", ".join(sorted(set(uncounted))))
    total = int(math.ceil((weights + activations) * RESERVE_FACTOR
                          + WORKSPACE_BYTES))
    return StaticEstimate(
        bytes=total,
        confidence=CONFIDENCE_LOW if notes else CONFIDENCE_MEDIUM,
        parameters=parameters, trainable_parameters=trainable,
        training=training, optimizer=optimizer, multiplier=multiplier,
        batch_size=train_batch, eval_batch_size=eval_batch,
        precision=precision, weights_bytes=int(weights),
        activation_bytes=int(activations), workspace_bytes=WORKSPACE_BYTES,
        models=models, uncounted=sorted(set(uncounted)), notes=notes,
        batch_assumed=batch_assumed,
        activation_phase=(
            "none" if activations <= 0
            else "evaluation" if by_phase["evaluation"] > by_phase["training"]
            else "training"))


async def estimate_static_async(graph: Mapping[str, Any], *,
                                timeout: float | None = None
                                ) -> StaticEstimate | None:
    """:func:`estimate_static` in the estimate thread, under *timeout*
    (:data:`STATIC_TIMEOUT_S`). None when it fails or takes too long: a
    suggestion is never an error and never holds a request up for long."""
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(_executor(), estimate_static, graph)
    try:
        return await asyncio.wait_for(
            future, STATIC_TIMEOUT_S if timeout is None else timeout)
    except asyncio.TimeoutError:
        logger.warning("memory estimate: the static estimate took longer "
                       "than %.0f s; none offered", STATIC_TIMEOUT_S
                       if timeout is None else timeout)
        return None
    except Exception:  # noqa: BLE001 -- a suggestion, never an error
        logger.warning("memory estimate: the static estimate failed",
                       exc_info=True)
        return None


# ── the budget source ────────────────────────────────────────────────────


@dataclass
class Estimate:
    """Everything ``POST /api/runs/estimate`` reports (#794 spec 2)."""

    suggestion: mem.Suggestion | None
    #: Newest first: ``{run_id, peak_bytes, finished_at}``.
    history: list[dict[str, Any]] = field(default_factory=list)
    static: StaticEstimate | None = None

    @property
    def confidence(self) -> str | None:
        if self.suggestion is not None and \
                self.suggestion.budget.source == mem.SOURCE_HISTORY:
            return CONFIDENCE_MEASURED
        if self.static is not None:
            return self.static.confidence
        return None


class EstimateBudgetSource:
    """The default budget source from #794: the history of the graph's
    structural hash (#788's rule, the newest peak x 1.2), else a static
    estimate of ``medium`` confidence (``source: "estimate"``). A ``low``
    one is reported by :meth:`explain` but never applied: a run would be
    capped at a figure that skipped part of its model.

    *history* is ``HistoryBudgetSource``; *peaks* is
    ``RunStore.peak_history`` (``hash, limit -> [(run_id, finished_at,
    peak_memory)]``), optional."""

    def __init__(self, history: mem.BudgetSource,
                 peaks: Callable[[str, int], Awaitable[Any]] | None = None,
                 *, static: Callable[[Mapping[str, Any]],
                                     Awaitable[StaticEstimate | None]]
                 = estimate_static_async) -> None:
        self._history = history
        self._peaks = peaks
        self._static = static

    async def suggest(self, graph: Mapping[str, Any],
                      options: Mapping[str, Any]) -> mem.Suggestion | None:
        suggestion = await self._history.suggest(graph, options)
        if suggestion is not None:
            return suggestion
        return _from_static(await self._static(graph))

    async def explain(self, graph: Mapping[str, Any],
                      options: Mapping[str, Any]) -> Estimate:
        suggestion = await self._history.suggest(graph, options)
        static = await self._static(graph)
        if suggestion is None:
            suggestion = _from_static(static)
        return Estimate(suggestion=suggestion,
                        history=await self._history_rows(graph, options),
                        static=static)

    async def _history_rows(self, graph: Mapping[str, Any],
                            options: Mapping[str, Any]
                            ) -> list[dict[str, Any]]:
        if self._peaks is None:
            return []
        try:
            rows = await self._peaks(mem.graph_hash(graph, options),
                                     HISTORY_ROWS)
        except Exception:  # noqa: BLE001 -- a suggestion, never an error
            logger.warning("could not read this graph's memory history",
                           exc_info=True)
            return []
        history = []
        for run_id, finished_at, peak_memory in rows or ():
            peak = mem.peak_bytes(peak_memory)
            if peak is not None:
                history.append({"run_id": run_id, "peak_bytes": peak,
                                "finished_at": finished_at})
        return history


#: How many earlier peaks the estimate lists.
HISTORY_ROWS = 5


def _from_static(static: StaticEstimate | None) -> mem.Suggestion | None:
    if static is None or static.confidence != CONFIDENCE_MEDIUM:
        return None
    return mem.Suggestion(mem.Budget(static.bytes, mem.SOURCE_ESTIMATE))
