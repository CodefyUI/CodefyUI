---
sidebar_position: 7
title: PythonScript Node
description: Write Python directly on the canvas — the contract, the Tier-0 import policy, the honest security model, and statistics recipes.
---

# PythonScript Node

Every other node in CodefyUI is a Python file someone wrote and installed. **PythonScript** is the node whose body you type into the browser. It exists because statistics and research work always outrun the node library, and authoring a [custom node](./custom-nodes.md) or a [plugin pack](./plugins.md) is a lot of ceremony for four lines of numpy.

Drag it from the palette (category **Utility**), double-click the node to open its **Code** tab, and write:

```python
def run(inputs, params):
    x = inputs["in1"]
    return {"out1": x.mean()}
```

## The contract

```python
def run(inputs: dict, params: dict) -> dict
```

| | |
|---|---|
| `inputs` | One key per **connected** input port: `in1`, `in2`, ... An unwired port is simply absent, so use `inputs.get("in2")` when a port is optional in your design. |
| `params` | A copy of this node's parameters (including `code` itself). Mutating it does nothing — the node hands you a copy. |
| return | A dict keyed by output port: `{"out1": ..., "out2": ...}`. **A bare value becomes `out1`**, so `return x` is fine for the common one-output case. |
| extra keys | A returned key that names no declared port is dropped, and the drop is reported in the Execution Log rather than silently swallowed. |

The function must be defined at module level and be called exactly `run`. The editor warns as soon as it is missing, rather than waiting for the run to fail.

`async def run` is **rejected at the gate**: the node calls `run` on a worker thread with no event loop, so nothing would await it and the output port would carry a coroutine object.

Two more names are in scope besides the allowlisted libraries:

* `device` — the compute device this run resolved to, so `torch.zeros(3, device=device)` lands where the rest of the graph is.
* `should_stop()` — the run's cooperative stop flag. Nothing can interrupt a script from outside (see the security model below), so a long loop that wants to be stoppable has to ask:

  ```python
  def run(inputs, params):
      total = 0.0
      for row in inputs["in1"]:
          if should_stop():
              break
          total += float(row.sum())
      return total
  ```

## Ports

`input_ports` and `output_ports` (1–8 each) decide how many handles the node has. The **Code** tab's Ports section sets both the count and the `DataType` of each port; the same values live in the `input_types` / `output_types` params as a comma-separated list, so a graph JSON stays readable.

* Input ports are **optional** by construction: the script decides what it needs, so declaring four ports and wiring two is a valid graph.
* Output ports default to `ANY`, which connects to anything. Naming a real type (`TENSOR`, `SCALAR`, `STRING`, ...) turns the graph validator into a check on your own wiring — worth doing once the script works.
* Lowering a port count **removes** any edge that was attached to a port that no longer exists, and marks the affected downstream nodes for re-execution.

## Caching

`code` is an ordinary parameter, so the execution cache keys on it like any other: editing the script re-runs this node and everything downstream, and running the same script twice over the same inputs serves the second one from cache.

## Output and errors

`print()` from your script is captured (up to 64 KB per execution) and appears in the **Execution Log** as the node's log line. It is not a global stdout redirect — a library writing straight to `sys.stdout` goes to the server console instead — because nodes share a thread pool and hijacking the process's stdout would swallow what other nodes print at the same moment.

An exception is reported with **the line number in your script**:

```
PythonScript failed at line 4: ZeroDivisionError: division by zero
```

The line named is the deepest frame inside your code, so a failure inside `statistics.mean([])` points at the line that called it, and a failure inside your own helper points at the helper. Whatever the script printed before it died is appended to the message. The graph's error-handling mode (fail-fast / continue / retry) applies exactly as it does to any other node.

## The Tier-0 policy

Code is checked **on every edit**, before it is ever compiled, by the same AST walker that gates plugin packs (`backend/app/core/plugin_validator.py`) running in allowlist mode. A rejection is a red banner under the editor with the offending line marked, not a failed run ten minutes into a training graph.

**Importable modules** (`backend/app/core/script_policy.py`, `TIER0_MODULES`):

```
collections   itertools   json   math   numpy   re   statistics   torch
```

They are also **pre-bound** in the namespace under those exact names, so `math.floor(x)` works with no import line; `import numpy as np` works too, if you prefer the alias.

**Refused, with a message pointing at the escape hatches:**

* Any other import — `os`, `sys`, `pathlib`, `subprocess`, `socket`, `urllib`, `requests`, and equally `pandas` or `sklearn`, which are not dangerous, just not on the list. Relative imports are refused with them.
* `exec`, `eval`, `compile`, `__import__`, `open`, `input`, `globals`, `locals`, `vars`, `dir`, `breakpoint`, `exit`.
* Bare uses of the module machinery — `__loader__`, `__spec__`, `__builtins__`, `__package__` — and dunder attribute access (`__class__`, `__globals__`, `__subclasses__`, `__code__`, ...). These are the universal escape primitives: `__loader__.load_module('nt')` hands back the real `os` module without an import statement.
* Doors *inside* the allowed libraries that lead back out to the filesystem, the network, a compiler or another process: `torch.hub` (downloads and executes a remote `hubconf.py`), `torch.utils.cpp_extension` (compiles and runs C++), `torch.distributed`, `torch.multiprocessing`, `numpy.savetxt` / `loadtxt` / `fromfile` / `tofile` / `save` / `memmap`, `numpy.ctypeslib`, and the rest of `TIER0_DENIED_ATTRS` in `script_policy.py`. Importing one of those by name, or reaching it with a literal `getattr`, is refused the same way.
* `torch.load(...)` / `numpy.load(...)` without an explicit `weights_only=True`, and any `load(allow_pickle=True)`: those execute code from the file they read. The receiver is resolved through your import aliases, so `import torch as t; t.load(...)` is caught too.

`json.load` and `json.loads` are **not** caught by that last rule — `json` is a Tier-0 module and parsing JSON is exactly what it is there for.

Two of those rules have a second lock at run time: the namespace is built from an explicit allowlist of builtins (so `open`, `eval` and friends are not merely un-writable, they are absent), and `__import__` is replaced with one that re-checks the module list. The dunder and library-door rules are **AST-only** — nothing re-checks them once the code is running, which is precisely why the gate runs before the code is compiled.

### Need something off the list?

That is what the other two paths are for, and they are better tools for it:

* A [custom node](./custom-nodes.md) — a file in your own project, which you wrote and can review.
* A [plugin pack](./plugins.md) — installable, versioned, and able to declare extra modules in its manifest for the user to accept with `--trust-author`.

## Security model — a guardrail, not a sandbox

Read this before you enable CodefyUI on a network interface.

The gate blocks the *easy* escapes. It is **not** a sandbox, and it is not trying to be one:

* **It limits which libraries a script can reach, not what those libraries can do.** numpy and torch are big enough to contain file IO, downloads and a C++ compiler on their own. The denied-attribute list above closes the doors we know about; it is a blocklist over two enormous APIs, and it should be read as raising the cost of an escape, never as a guarantee that files and the network are out of reach.
* The script runs **in the CodefyUI server process**, with your user's permissions. Nothing containerises it.
* A determined attacker who can already type into your canvas can probably still find a way out. The gate raises the cost of drive-by code execution; it does not make the surface safe against a motivated adversary.
* Nothing limits CPU or memory, and a runaway is not contained to its own node. Nodes execute on the interpreter's **default thread pool**, so a `while True:` in one script starves *every* node execution in the process until the server is restarted. Stop is cooperative and checked *between* nodes, so it cannot interrupt a loop inside one — a long loop has to call `should_stop()` itself.
* **Module objects are shared with the host process.** The allowlisted modules are the real ones, not copies, so a script that assigns `torch.zeros = something_else` changes it for every other node, in every graph, until the server restarts. Nothing is rolled back between runs.
* The `code` param is saved in the graph JSON like any other parameter. **Opening a graph from an untrusted source and pressing Run executes that person's Python.** The policy check is the only thing between the two, which is exactly why it runs before the code is compiled rather than at import time.

The real boundary is *who can reach the editor*. CodefyUI binds to localhost by default; keep it that way unless you trust everyone on the network.

## Export

`Export as Python` writes the script into the generated file one source line per string literal, under a provenance comment:

```python
def n02_pythonscript(ctx):
    "PythonScript - node 'py1'."
    params = {
        # ---- 'code' of node 'py1': the script this node runs, verbatim ----
        'code': (
            'def run(inputs, params):\n'
            '    x = inputs["in1"]\n'
            '    return {"out1": x.mean(dim=0)}\n'
        ),
        'input_ports': 1,
        'output_ports': 1,
    }
    return _call('PythonScript', 'py1', params, ctx, inputs={'in1': in1})
```

It is readable, it is editable — change a line and the exported graph runs the change — and it is still a string literal, so a `'''` inside your code can never become program text.

## Statistics recipes

### Per-channel mean and standard deviation

One TENSOR in, two TENSOR out (`output_ports: 2`, `output_types: TENSOR,TENSOR`):

```python
def run(inputs, params):
    x = inputs["in1"]                     # (N, C, H, W)
    flat = x.reshape(x.shape[0], x.shape[1], -1)
    return {
        "out1": flat.mean(dim=(0, 2)),
        "out2": flat.std(dim=(0, 2)),
    }
```

### Class balance of a label batch

One TENSOR in, one STRING out:

```python
import collections

def run(inputs, params):
    labels = inputs["in1"].flatten().tolist()
    counts = collections.Counter(labels)
    total = sum(counts.values())
    lines = [
        f"class {int(k)}: {v} ({100 * v / total:.1f}%)"
        for k, v in sorted(counts.items())
    ]
    print("\n".join(lines))          # also lands in the Execution Log
    return {"out1": "\n".join(lines)}
```

### Robust summary (median, IQR, outlier count)

```python
import statistics

def run(inputs, params):
    values = sorted(float(v) for v in inputs["in1"].flatten().tolist())
    q1, _, q3 = statistics.quantiles(values, n=4)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return {
        "out1": {
            "median": statistics.median(values),
            "iqr": iqr,
            "outliers": sum(1 for v in values if v < low or v > high),
        }
    }
```

### Comparing two tensors

Two TENSOR in (`input_ports: 2`), one SCALAR out:

```python
def run(inputs, params):
    a, b = inputs["in1"], inputs["in2"]
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    return float((a - b).abs().max())     # bare value -> out1
```

### Creating a tensor on the run's device

`device` holds the compute device the run resolved to, so a script that creates tensors puts them where the rest of the graph is:

```python
def run(inputs, params):
    return torch.zeros(4, 4, device=device)
```

## Limits

| | |
|---|---|
| Script length | 100,000 characters. Past that, write a custom node. |
| Ports | 1–8 per side. |
| Captured output | 64,000 characters per execution, then truncated with a notice. |
| Async | `run` must be a plain `def`; `async def run` is rejected at the gate, and `asyncio` is not importable. |
| State | Each execution gets a fresh namespace, so your own module-level variables do not carry over. The imported **modules** are the host's, though — see the security model. Use the graph for state you mean to keep. |
