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

Instead, `ModelLoader` widens the restricted unpickler by exactly three sets of names, for the duration of that one load:

1. **`torch.nn`'s own layer classes.** Derived by walking the loaded subclasses of `nn.Module` and keeping the ones torch defines, so it tracks whatever torch you installed rather than a list written down here.
2. **CodefyUI's own module classes** — `GraphModelModule` (which every layer-editor model is), the `SequentialModel` wrappers (`Reshape`, `SelectIndex`, the LSTM/GRU/attention/transformer blocks), `CausalLMModule` and the blocks it is built from, the diffusion U-Net, the VLA policy, and the rest. A curated list of exact classes, each audited against the rules below.
3. **Two torch activation functions** — `torch.nn.functional.relu` and `torch._C._nn.gelu`. Torch's transformer layers store their activation as a *callable* attribute rather than a layer, so these two are what a `TransformerEncoder` or `TransformerDecoder` checkpoint needs in order to come back. Exact identities, not the `torch.nn.functional` namespace: `handle_torch_function` also lives there and dispatches to an arbitrary object's `__torch_function__`, so admitting the namespace would admit a general-purpose call gadget and whatever torch adds to it next.

Everything else is refused with a message naming what it stopped on. A pickle that names `os.system` does not load, because `os.system` is on none of the three lists — and neither is any function other than those two.

### What "audited" means, exactly

Admitting a **class** by name lets a file do two things with it, and both have to be harmless:

- **Restore its attributes.** So an admitted class must not define `__reduce__`, `__setstate__` or `__getnewargs__` — anything that turns restoring an attribute into running something.
- **Call its constructor with arguments the file chose.** torch's restricted unpickler runs `func(*args)` for any allowed name, so `cls(...)` is reachable. An admitted constructor must therefore touch no files, no network and no global state (a local `torch.Generator` is fine; `torch.manual_seed` is not). Bad arguments raising an error is acceptable — that is a failed load, not a compromised one.

The second half has always been true of torch's own classes too: admitting `nn.Linear` admits `nn.Linear(...)` on file-chosen sizes. It is worth stating because the CodefyUI list is one a human maintains, and a test enforces the mechanically checkable parts of both halves on every run.

Admitting a **function** is judged on four points, all four required: it is torch-owned, it is a pure tensor operation, it has no filesystem / network / process side effects, and it mutates no global state — so that calling it with arbitrary file-chosen arguments returns a tensor or raises. That is the same standard, and a *smaller* surface than the class case it sits next to: a file could already reach `nn.Linear(...)` through the same code path, and a function has no constructor and no attributes to restore.

Both lists are enumerated from the **save side** — from what the models CodefyUI can actually build store — and both have a test that re-runs that enumeration, so a class or a callable that starts appearing in saved models fails the suite instead of quietly becoming a checkpoint nobody can reopen.

:::note What this means in practice
A `full_model` file CodefyUI wrote **loads back into CodefyUI**. A `full_model` file containing a class from a [custom node](/advanced/custom-nodes), a [plugin](/advanced/plugins), or somebody's own script **does not** — that code has not been through review, and admitting it is the line CodefyUI does not cross. `ModelSaver` tells you which of the two you just wrote, in its **Log** tab, at save time rather than one node later.
:::

### One known edge

- **A file CodefyUI wrote is not self-contained.** Reading it needs CodefyUI's own classes importable, so a CodefyUI older than the version that admitted them refuses it, and plain `torch.load` outside CodefyUI needs `weights_only=False` plus the backend package on `sys.path`. `state_dict` files have neither condition. (A file made only of stock torch layers — transformers included — has neither condition either; the `Log` note tells you which kind you wrote.)

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
