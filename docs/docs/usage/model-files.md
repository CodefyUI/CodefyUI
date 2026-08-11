---
sidebar_position: 3.7
title: Saving and Loading Models
description: What ModelSaver writes and what ModelLoader will read back — state_dict versus full_model, and why loading a model file is a trust decision.
---

# Saving and Loading Models

`ModelSaver` and `ModelLoader` (both in the palette's **IO** group) each offer two modes, and the pair you choose decides what the file *is*: a bag of numbers, or a Python object. The difference is not about convenience. Loading a saved Python object is the step that can run code, so it is the one place in CodefyUI where a file you open gets a say in what happens next.

## The two modes

| | `state_dict` (default) | `full_model` |
|---|---|---|
| What is in the file | Tensors, keyed by parameter name | The whole `nn.Module`, pickled |
| To load it you need | The same architecture, wired into `ModelLoader.model` | Nothing — the module rebuilds itself |
| `.safetensors` support | Yes | No (the format stores tensors only) |
| Survives a layer rename | No — the keys are the attribute names | Yes |
| Readable outside CodefyUI | Yes, by anything that reads a torch file | Only with the class definitions importable |

**`state_dict` is the default and the recommended path.** Every shipped example uses it. It is the mode with no conditions attached to the file: `torch.load` reads it under torch's restricted unpickler, so there is nothing in it that could execute, and any Python program with the same architecture can consume it.

**`full_model` exists for the case where you do not want to rebuild the architecture** — hand someone a file and let them run it, or reload a model whose graph you no longer have.

## Why `full_model` is restricted

A full-model file is a **pickle**, and unpickling is not reading data: a pickle can name a function and ask for it to be called. That is why `weights_only=True` is torch's default, and CodefyUI never turns it off.

Instead, `ModelLoader` widens the restricted unpickler by exactly two sets of names, for the duration of that one load:

1. **`torch.nn`'s own layer classes.** Derived by walking the loaded subclasses of `nn.Module` and keeping the ones torch defines, so it tracks whatever torch you installed rather than a list written down here.
2. **CodefyUI's own module classes** — `GraphModelModule` (which every layer-editor model is), the `SequentialModel` wrappers (`Reshape`, `SelectIndex`, the LSTM/GRU/attention/transformer blocks), `CausalLMModule` and the blocks it is built from, the diffusion U-Net, the VLA policy, and the rest. A curated list of exact classes, each audited against the rule below.

Everything else is refused with a message naming what it stopped on. A pickle that names `os.system` does not load, because `os.system` is on neither list.

### What "audited" means, exactly

Admitting a class by name lets a file do two things with it, and both have to be harmless:

- **Restore its attributes.** So an admitted class must not define `__reduce__`, `__setstate__` or `__getnewargs__` — anything that turns restoring an attribute into running something.
- **Call its constructor with arguments the file chose.** torch's restricted unpickler runs `func(*args)` for any allowed name, so `cls(...)` is reachable. An admitted constructor must therefore touch no files, no network and no global state (a local `torch.Generator` is fine; `torch.manual_seed` is not). Bad arguments raising an error is acceptable — that is a failed load, not a compromised one.

The second half has always been true of torch's own classes too: admitting `nn.Linear` admits `nn.Linear(...)` on file-chosen sizes. It is worth stating because the CodefyUI list is one a human maintains, and a test enforces the mechanically checkable parts of both halves on every run.

:::note What this means in practice
A `full_model` file CodefyUI wrote **loads back into CodefyUI**. A `full_model` file containing a class from a [custom node](/advanced/custom-nodes), a [plugin](/advanced/plugins), or somebody's own script **does not** — that code has not been through review, and admitting it is the line CodefyUI does not cross. `ModelSaver` tells you which of the two you just wrote, in its **Log** tab, at save time rather than one node later.
:::

### Two known edges

- **A `TransformerEncoder` or `TransformerDecoder` layer still does not come back.** Not because of the wrapper — that is on the list — but because `nn.TransformerEncoderLayer` stores its activation as `torch.nn.functional.relu`, a *function*, and no functions are on the list. Admitting functions means admitting callables the unpickler may be asked to invoke, which is a wider decision than the class one and has not been taken. Use `state_dict` for these.
- **A file CodefyUI wrote is not self-contained.** Reading it needs CodefyUI's own classes importable, so a CodefyUI older than the version that admitted them refuses it, and plain `torch.load` outside CodefyUI needs `weights_only=False` plus the backend package on `sys.path`. `state_dict` files have neither condition.

## If a load is refused and you trust the file

Convert it once, outside CodefyUI, and load the result as a `state_dict`:

```python
import torch
model = torch.load("PATH.pt", weights_only=False)   # the code-execution step
torch.save(model.state_dict(), "NEW_PATH.pt")
```

Do this only for a file you produced yourself or got from a source you trust. That `weights_only=False` is exactly the step CodefyUI will not take on your behalf, which is why it is a line you type rather than a checkbox in the app.

## Related

- [Reproducing Baselines](./reproducing-baselines) — `CheckpointSaver` / `CheckpointLoader`, which save training *state* (model, optimizer, schedule, epoch) rather than a model, and always as tensors.
- [Running Graphs](./running-graphs) — why neither of these nodes is ever served from the execution cache.
- [Python Script node](/advanced/python-script-node) — where `torch.load` is refused outright, and why.
