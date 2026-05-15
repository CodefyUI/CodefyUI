"""Generate a standalone Python script from a CodefyUI graph."""

from __future__ import annotations

import json
from collections import deque
from typing import Any


# ── Per-layer source rendering (shared between legacy + v2 builders) ──

_ACTIVATIONS = {
    "ReLU", "GELU", "Sigmoid", "Tanh", "LeakyReLU",
    "ELU", "SiLU", "Mish", "SELU", "PReLU", "Hardswish",
}
_INPLACE_ACTIVATIONS = {"ReLU", "LeakyReLU", "ELU", "SiLU", "Mish", "SELU", "Hardswish"}


def _var(nid: str) -> str:
    return nid.replace("-", "_")


def _layer_to_source(layer_type: str, params: dict) -> str:
    """Render a single layer dict as an ``nn.X(...)`` Python expression."""
    if layer_type == "Softmax":
        return "nn.Softmax(dim=-1)"
    if layer_type in _ACTIVATIONS:
        if layer_type in _INPLACE_ACTIVATIONS:
            return f"nn.{layer_type}(inplace=True)"
        return f"nn.{layer_type}()"
    args = ", ".join(f"{k}={v!r}" for k, v in params.items())
    return f"nn.{layer_type}({args})"


# ── SequentialModel codegen (legacy + v2) ────────────────────────────


def _gen_sequential_model(var: str, params: dict) -> list[str]:
    layers_json = params.get("layers", "[]")
    spec = json.loads(layers_json) if isinstance(layers_json, str) else layers_json

    # v2 graph spec: {"version":2, "nodes":[...], "edges":[...]}.  Older
    # graphs still ship a flat list of layer dicts.
    if isinstance(spec, dict) and spec.get("version") == 2:
        return _gen_v2_sequential(var, spec)
    return _gen_legacy_sequential(var, spec)


def _gen_legacy_sequential(var: str, layers: list[dict]) -> list[str]:
    layer_strs = [_layer_to_source(l.get("type", ""), {k: v for k, v in l.items() if k != "type"}) for l in layers]
    lines = ["# Build nn.Sequential model", f"{var} = nn.Sequential("]
    for s in layer_strs:
        lines.append(f"    {s},")
    lines.append(")")
    return lines


def _gen_v2_sequential(var: str, spec: dict) -> list[str]:
    """Emit an ``nn.Sequential`` when the v2 graph is a simple chain.

    For DAGs with merges/branches we fall back to a clear TODO comment;
    callers can wire up a custom ``nn.Module`` by hand. (Most teaching
    examples — CNN-MNIST, GPT-Mini, etc. — are simple chains.)
    """
    nodes = spec.get("nodes", [])
    edges = spec.get("edges", [])
    nodes_by_id = {n["id"]: n for n in nodes}

    inputs = [n for n in nodes if n["type"] == "Input"]
    outputs = [n for n in nodes if n["type"] == "Output"]
    if len(inputs) != 1 or len(outputs) != 1:
        return [
            f"# {var}: SequentialModel has {len(inputs)} input(s) and {len(outputs)} output(s)",
            f"# Custom forward needed — define a nn.Module subclass manually.",
            f"{var} = None",
        ]

    incoming: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    outgoing: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    for e in edges:
        src, tgt = e["source"], e["target"]
        if src in nodes_by_id and tgt in nodes_by_id:
            incoming[tgt].append(src)
            outgoing[src].append(tgt)

    # Topological order (Kahn's algorithm).
    in_degree = {nid: len(incoming[nid]) for nid in nodes_by_id}
    queue = deque(nid for nid, d in in_degree.items() if d == 0)
    topo: list[str] = []
    while queue:
        nid = queue.popleft()
        topo.append(nid)
        for tgt in outgoing[nid]:
            in_degree[tgt] -= 1
            if in_degree[tgt] == 0:
                queue.append(tgt)

    # Detect "simple chain": no merge nodes, every non-boundary layer has
    # exactly 1 in and 1 out edge.
    merge_types = {"Add", "Concat", "Multiply", "Subtract", "Mean", "Stack"}
    is_chain = True
    for n in nodes:
        if n["type"] in ("Input", "Output"):
            continue
        if n["type"] in merge_types:
            is_chain = False
            break
        if len(incoming[n["id"]]) > 1 or len(outgoing[n["id"]]) > 1:
            is_chain = False
            break

    if not is_chain:
        return [
            f"# {var}: SequentialModel contains merges or branches",
            f"# (Add/Concat/Multiply/skip-connections) — define a nn.Module subclass manually.",
            f"{var} = None",
        ]

    layer_strs: list[str] = []
    for nid in topo:
        n = nodes_by_id[nid]
        if n["type"] in ("Input", "Output"):
            continue
        layer_strs.append(_layer_to_source(n["type"], n.get("params", {})))

    lines = ["# Build nn.Sequential model (compiled from v2 graph spec)", f"{var} = nn.Sequential("]
    for s in layer_strs:
        lines.append(f"    {s},")
    lines.append(")")
    return lines


# ── Other node codegens ──────────────────────────────────────────────


def _gen_dataset(var: str, params: dict) -> list[str]:
    name = params.get("name", "MNIST")
    split = params.get("split", "train")
    data_dir = params.get("data_dir", "./data")
    is_train = split == "train"
    return [
        f"transform_{var} = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])",
        f"{var} = datasets.{name}({data_dir!r}, train={is_train}, download=True, transform=transform_{var})",
    ]


def _gen_dataloader(var: str, params: dict, inputs: dict[str, str]) -> list[str]:
    bs = params.get("batch_size", 64)
    shuffle = params.get("shuffle", True)
    nw = params.get("num_workers", 0)
    dataset_var = inputs.get("dataset", "dataset")
    return [f"{var} = DataLoader({dataset_var}, batch_size={bs}, shuffle={shuffle}, num_workers={nw})"]


def _gen_optimizer(var: str, params: dict, inputs: dict[str, str]) -> list[str]:
    opt_type = params.get("type", "Adam")
    lr = params.get("lr", 0.001)
    wd = params.get("weight_decay", 0.0)
    model_var = inputs.get("model", "model")
    args = f"{model_var}.parameters(), lr={lr}"
    if wd:
        args += f", weight_decay={wd}"
    return [f"{var} = optim.{opt_type}({args})"]


def _gen_loss(var: str, params: dict) -> list[str]:
    loss_type = params.get("type", "CrossEntropyLoss")
    return [f"{var} = nn.{loss_type}()"]


def _gen_training_loop(var: str, params: dict, inputs: dict[str, str]) -> list[str]:
    epochs = params.get("epochs", 5)
    device = params.get("device", "cpu")
    model_var = inputs.get("model", "model")
    loader_var = inputs.get("dataloader", "dataloader")
    opt_var = inputs.get("optimizer", "optimizer")
    loss_var = inputs.get("loss_fn", "loss_fn")
    return [
        "# Training loop",
        f"device = {device!r}",
        f"{model_var} = {model_var}.to(device)",
        f"{model_var}.train()",
        "epoch_losses = []",
        f"for epoch in range({epochs}):",
        "    running_loss = 0.0",
        "    batch_count = 0",
        f"    for batch_data in {loader_var}:",
        "        if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:",
        "            data, targets = batch_data",
        "            data, targets = data.to(device), targets.to(device)",
        "        else:",
        "            data = batch_data.to(device) if hasattr(batch_data, 'to') else batch_data",
        "            targets = None",
        f"        {opt_var}.zero_grad()",
        f"        outputs = {model_var}(data)",
        f"        loss = {loss_var}(outputs, targets) if targets is not None else {loss_var}(outputs)",
        "        loss.backward()",
        f"        {opt_var}.step()",
        "        running_loss += loss.item()",
        "        batch_count += 1",
        "    avg_loss = running_loss / max(batch_count, 1)",
        "    epoch_losses.append(avg_loss)",
        f'    print(f"Epoch {{epoch + 1}}/{epochs} — Loss: {{avg_loss:.4f}}")',
        f"{var}_losses = torch.tensor(epoch_losses)",
    ]


def _gen_model_saver(var: str, params: dict, inputs: dict[str, str]) -> list[str]:
    path = params.get("path", "model_weights.pt")
    mode = params.get("save_mode", "state_dict")
    model_var = inputs.get("model", "model")
    if mode == "state_dict":
        return [f"torch.save({model_var}.state_dict(), {path!r})", f"print('Model saved to', {path!r})"]
    return [f"torch.save({model_var}, {path!r})", f"print('Model saved to', {path!r})"]


def _gen_model_loader(var: str, params: dict, inputs: dict[str, str]) -> list[str]:
    path = params.get("path", "model_weights.pt")
    mode = params.get("load_mode", "state_dict")
    device = params.get("device", "cpu")
    model_var = inputs.get("model", "model")
    if mode == "state_dict":
        return [
            f"state_dict = torch.load({path!r}, map_location={device!r}, weights_only=True)",
            f"{model_var}.load_state_dict(state_dict)",
            f"{model_var} = {model_var}.to({device!r})",
        ]
    return [f"{model_var} = torch.load({path!r}, map_location={device!r}, weights_only=False)"]


def _gen_inference(var: str, params: dict, inputs: dict[str, str]) -> list[str]:
    device = params.get("device", "cpu")
    model_var = inputs.get("model", "model")
    input_var = inputs.get("input", "input_tensor")
    return [
        f"{model_var} = {model_var}.to({device!r})",
        f"{input_var} = {input_var}.to({device!r})",
        f"{model_var}.eval()",
        "with torch.no_grad():",
        f"    {var} = {model_var}({input_var})",
        f'print(f"Output shape: {{{var}.shape}}")',
    ]


def _gen_visualize(var: str, params: dict, inputs: dict[str, str]) -> list[str]:
    title = params.get("title", "Plot")
    plot_type = params.get("plot_type", "line")
    data_var = inputs.get("data", "data")
    lines = ["plt.figure(figsize=(8, 5))"]
    if plot_type == "line":
        lines.append(f"plt.plot({data_var}.cpu().numpy() if hasattr({data_var}, 'cpu') else {data_var})")
    else:
        lines.append(f"plt.bar(range(len({data_var})), {data_var}.cpu().numpy() if hasattr({data_var}, 'cpu') else {data_var})")
    lines += [
        f"plt.title({title!r})",
        "plt.tight_layout()",
        "plt.show()",
    ]
    return lines


def _gen_print(var: str, params: dict, inputs: dict[str, str]) -> list[str]:
    label = params.get("label", "")
    val_var = inputs.get("value", "value")
    if label:
        return [f'print(f"[{label}] {{{val_var}}}")']
    return [f"print({val_var})"]


# ── Generator dispatch ────────────────────────────────────────────


_GENERATORS: dict[str, Any] = {
    "SequentialModel": lambda v, p, i: _gen_sequential_model(v, p),
    "Dataset": lambda v, p, i: _gen_dataset(v, p),
    "DataLoader": _gen_dataloader,
    "Optimizer": _gen_optimizer,
    "Loss": lambda v, p, i: _gen_loss(v, p),
    "TrainingLoop": _gen_training_loop,
    "ModelSaver": _gen_model_saver,
    "ModelLoader": _gen_model_loader,
    "Inference": _gen_inference,
    "Visualize": _gen_visualize,
    "Print": _gen_print,
}

# Start is an execution-flow marker — it has no runtime representation in
# the exported script, so we skip it silently rather than dumping a
# placeholder comment.
_SKIP_TYPES = {"Start"}


# ── Main entry point ─────────────────────────────────────────────


def generate_python(
    nodes: list[dict],
    edges: list[dict],
    order: list[str],
    name: str = "Untitled",
) -> str:
    """Generate a runnable Python script from graph data."""

    node_map = {n["id"]: n for n in nodes}

    # input_mapping[node_id][target_handle] = source variable name
    input_mapping: dict[str, dict[str, str]] = {n["id"]: {} for n in nodes}

    for nid in order:
        node = node_map[nid]
        ntype = node["type"]
        var = _var(nid)

        out_map: dict[str, str] = {}
        if ntype == "SequentialModel":
            out_map["model"] = var
        elif ntype == "Dataset":
            out_map["dataset"] = var
        elif ntype == "DataLoader":
            out_map["dataloader"] = var
        elif ntype == "Optimizer":
            out_map["optimizer"] = var
        elif ntype == "Loss":
            out_map["loss_fn"] = var
        elif ntype == "TrainingLoop":
            model_in = input_mapping[nid].get("model", "model")
            out_map["model"] = model_in
            out_map["losses"] = f"{var}_losses"
        elif ntype == "ModelLoader":
            model_in = input_mapping[nid].get("model", "model")
            out_map["model"] = model_in
        elif ntype == "ModelSaver":
            model_in = input_mapping[nid].get("model", "model")
            out_map["model"] = model_in
            out_map["path"] = repr(node.get("data", {}).get("params", {}).get("path", "model_weights.pt"))
        elif ntype == "Inference":
            out_map["output"] = var
            out_map["model"] = input_mapping[nid].get("model", "model")
        elif ntype == "Visualize":
            out_map["image"] = f"{var}_img"
        elif ntype == "Print":
            pass
        else:
            out_map["output"] = var

        for edge in edges:
            if edge["source"] != nid:
                continue
            if edge.get("type", "data") == "trigger":
                continue
            target = edge["target"]
            src_handle = edge.get("sourceHandle", "output")
            tgt_handle = edge.get("targetHandle", "input")
            var_name = out_map.get(src_handle, var)
            if target in input_mapping:
                input_mapping[target][tgt_handle] = var_name

    # ── Header (imports) ─────────────────────────────────────────────
    needs_torch = False
    needs_nn = False
    needs_optim = False
    needs_dataloader = False
    needs_datasets = False
    needs_transforms = False
    needs_pyplot = False

    for nid in order:
        ntype = node_map[nid]["type"]
        if ntype in ("SequentialModel", "Loss", "TrainingLoop", "Inference"):
            needs_torch = True
            needs_nn = True
        if ntype == "Optimizer":
            needs_torch = True
            needs_optim = True
        if ntype == "Dataset":
            needs_datasets = True
            needs_transforms = True
        if ntype == "DataLoader":
            needs_dataloader = True
        if ntype in ("ModelSaver", "ModelLoader", "TrainingLoop"):
            needs_torch = True
        if ntype == "Visualize":
            needs_pyplot = True

    header: list[str] = [
        '"""',
        name,
        "Auto-generated by CodefyUI",
        '"""',
        "",
    ]
    if needs_torch:
        header.append("import torch")
    if needs_nn:
        header.append("import torch.nn as nn")
    if needs_optim:
        header.append("import torch.optim as optim")
    if needs_dataloader:
        header.append("from torch.utils.data import DataLoader")
    if needs_datasets:
        header.append("from torchvision import datasets")
    if needs_transforms:
        header.append("from torchvision import transforms")
    if needs_pyplot:
        header.append("import matplotlib.pyplot as plt")
    header.append("")

    # ── Body ────────────────────────────────────────────────────────
    body: list[str] = []
    for nid in order:
        node = node_map[nid]
        ntype = node["type"]
        if ntype in _SKIP_TYPES:
            continue
        params = node.get("data", {}).get("params", {})
        var = _var(nid)
        inputs = input_mapping.get(nid, {})

        gen = _GENERATORS.get(ntype)
        if gen:
            body.append("")
            body.extend(gen(var, params, inputs))
        else:
            body.append("")
            body.append(f"# TODO: {ntype} (id: {nid}) — no codegen template yet, please implement manually.")
            if params:
                body.append(f"# params: {params}")
            body.append(f"{var} = None")

    return "\n".join(header + body) + "\n"
