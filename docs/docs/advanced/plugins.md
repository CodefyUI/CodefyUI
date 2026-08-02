---
sidebar_position: 3
title: Plugins
description: Install plugin packs of educational nodes, and learn how to write and publish your own.
---

# Plugin Packs

Educational ("Edu") nodes ship as installable **plugin packs**, organised **by direction** so each maps onto a hands-on textbook module and installs cumulatively as you progress.

```bash
cdui plugin install foundations deep rl   # full textbook companion
cdui plugin list
cdui plugin info deep                      # manifest, lessons covered, node names
cdui plugin search attention               # query the catalog
cdui plugin install foo/bar                # third-party pack from GitHub
cdui plugin uninstall deep
```

## What's available

| Pack | Hands-on modules | Nodes |
|------|------------------|-----------|
| `foundations` | I1 Data Representation · I2 Classical ML | Edu-ColumnStats, Edu-KNN, Edu-LinearRegression, Edu-LogisticRegression, Edu-TokenEmbedding, Edu-FFN |
| `deep` | I3 Vision · I4 Sequences | Edu-CrossAttention, Edu-ResBlock, Edu-SelfAttention, Edu-MultiHeadAttention, Edu-Patchify |
| `rl` | I5 Reinforcement Learning | Edu-PolicyGradient |
| `stats` | — (any dataset) | Stats-Describe, Stats-GroupByAggregate, Stats-Histogram, Stats-Percentile, Stats-Correlation, Stats-ConfusionMatrix, Stats-TableView, Stats-ChartView |

`stats` is the odd one out: not a textbook companion but a working reference for third-party pack authors. It is pure numpy + torch, installs at [Tier 0](#security--three-tiers) with **no `[security]` section at all**, and its [README](https://github.com/CodefyUI/CodefyUI/blob/main/plugins/stats/README.md) is the normative write-up of the two contracts a data pack needs — how a table travels between ports, and how a `chart` output is declared and drawn.

Each Edu node decomposes a single lesson concept into a chain of named steps that the [Teaching Inspector](/usage/teaching-inspector) renders one row at a time — `Edu-ColumnStats` shows the population-std formula as `sum → divide → deviations² → variance → sqrt`; `Edu-PolicyGradient` exposes `softmax → gather → log → baseline → loss`; `Edu-Patchify` makes `unfold → permute → flatten` visible. Switch on **Verbose mode** in the Settings popover to capture them.

## How packs are stored

- **Built-in direction packs** live in `plugins/<id>/` inside the repo and are activated in place (no copies).
- **Third-party packs** are downloaded as a pinned-SHA tarball into `<USER_DATA>/plugins/<id>/` and **AST-validated** before install (see [Security](#security--three-tiers)).
- A lockfile at `<USER_DATA>/plugins/installed.json` records every install — including which capabilities you granted — so `cdui start` rediscovers them on the next launch.

Plugin nodes are namespaced to avoid collisions and to self-document graphs — built-in nodes use a bare name like `Conv2d`, while plugin nodes are qualified like `foundations:Edu-KNN`.

## Security — three tiers

A plugin pack is Python that runs in the CodefyUI process. Before a third-party pack is installed, every `.py` file in it is walked by an AST gate that decides what it may import. The gate has three answers, and the middle one is the interesting one.

| Tier | How a plugin gets it | What it covers |
|------|----------------------|----------------|
| **0 — default** | nothing to declare | Pure computation: `math`, `statistics`, `collections`, `itertools`, `functools`, `json`, `re`, `dataclasses`, `typing`, `enum`, `decimal`, `random`, `numpy`, `torch`, `pandas` — plus the path helpers (below). Every first-party pack lives here. |
| **1 — declared capabilities** | `[security] capabilities = [...]` in the manifest, confirmed by the user at install | One named group of modules per capability. |
| **2 — trusted** | `[security] allowed_modules = [...]` **and** `cdui plugin install --trust-author` | Anything, including `subprocess`, `ctypes` and `importlib`. |

### The capabilities

| Capability | Unlocks | What you are agreeing to |
|------------|---------|--------------------------|
| `network` | `requests`, `urllib`, `http`, `socket` | The plugin can send and receive data from any host — **and write what it downloads to disk**, because `urllib.request.urlretrieve(url, dest)` is one call. |
| `filesystem` | `pathlib`, `tempfile`, `shutil`, `zipfile`, `tarfile`, `gzip`, `bz2`, `lzma`, `codecs`, `sqlite3`, `glob`, `fileinput` | The plugin can use the file **libraries**. This is not a write boundary: plain `open(p, "w")` is a builtin and needs no declaration at all (see [What this is not](#what-this-is-not)). |
| `process-env` | `os`, `ntpath`, `posixpath`, `genericpath` | The plugin gets **the whole `os` module**: read *and change* this process's environment (**including any API keys in it**), start other programs (`os.execv`, `os.spawnve`, `os.startfile`), and delete or rename files. The name is what people ask for it for; the grant is bigger than the name. |

Nothing else is a capability. `subprocess`, `sys`, `importlib`, `ctypes`, `pickle`, `marshal`, `dill`, `shelve`, `runpy`, `code`, `signal`, `atexit`, `webbrowser`, `threading`, `asyncio` and `multiprocessing` are Tier 2 only: **no capability hands over a module whose purpose is running code or reaching the interpreter.** Note the precise claim — `process-env` grants `os`, and `os` starts processes. What you do not get from any capability is a module built for executing code.

### Path helpers are Tier 0

`os.path.join` is string manipulation, so it needs no capability — but only in the **one** form that binds the helpers themselves, and only for the names that really are pure string functions:

```python
from os.path import join, basename   # fine, Tier 0
from os.path import expandvars       # needs "process-env" — reads os.environ
from os.path import exists, getsize  # needs "process-env" — real stat()
from os.path import genericpath      # needs "process-env" — a module
from os import path                  # needs "process-env" — binds ntpath
import os / import os.path           # needs "process-env"
import ntpath / posixpath            # needs "process-env"
```

The Tier-0 list is exactly: `join`, `basename`, `dirname`, `split`, `splitext`, `splitdrive`, `normpath`, `normcase`, `isabs`, `commonpath`, `commonprefix`, and the `sep` / `altsep` / `extsep` / `pathsep` / `curdir` / `pardir` / `defpath` constants.

The refused lines are not pedantry — `os.path` is a real module and most of its surface is not string manipulation:

- `os.path` **is** `ntpath` / `posixpath`, and those modules `import os` and `import sys` at module level, leaving both bound as ordinary attributes — so `path.os.remove(p)` deletes a file and `path.sys.modules['subprocess'].run([...])` runs a command.
- `expandvars("%WANDB_API_KEY%")` returns the value of the environment variable — the exact thing `process-env` exists to gate — and `expanduser("~")` returns your home directory.
- `exists`, `isfile`, `isdir`, `getsize`, `getmtime` and friends call `stat()` on any path you name; `abspath`, `realpath` and `relpath` resolve against the working directory and so disclose where CodefyUI is installed.

Each name on the Tier-0 list was checked by *calling* it, not by reading its source — on Windows `abspath` reaches `nt._getfullpathname`, which a source audit for `os.` usage does not see.

### Declaring, and being asked

```toml
[security]
capabilities = ["network"]
```

```console
$ cdui plugin install alice/metric-logger
  Source: https://github.com/alice/metric-logger
  Ref: default branch (a1b2c3d)

> This plugin requests the following capabilities
    network -> reach the network -- send and receive data from any host, and write what it downloads to disk (requests, urllib, http, socket)
  A capability is a declaration, not a sandbox: once granted, the plugin may
  use that group of modules and CodefyUI stops asking.
  Grant these? [y/N]:
```

- **Without a terminal** (a script, CI, a piped install) the answer is **no**, and the message names `--accept-capabilities`, which grants the declared set without the prompt. `-y` / `--no-confirm` does *not* imply it: that flag skips the "install from this URL?" question, and consenting to code that reaches the network is a different question.
- **What was granted is recorded** in `<USER_DATA>/plugins/installed.json` and shown by `cdui plugin list` and `cdui plugin info`.
- **`cdui plugin update` does not re-ask** when the new version's request is a subset of what you already granted — and **stops** when it grew a capability, which is the supply-chain shape an update can actually catch.

### What holds in every tier

`torch.load(...)` still requires an explicit `weights_only=True`; dunder access (`__class__`, `__globals__`, `__subclasses__`, …), frame walking (`f_globals`, `gi_frame`, …), and the **builtins** `eval` / `exec` / `compile` / `__import__` — as bare calls and through the `builtins` module — are refused whatever was declared. **A capability never buys reflection.**

The builtins, not the spelling: a *method* that shares one of those names is ordinary code and passes at every tier, so `torch.compile(model)` and `model.eval()` are allowed for plugins. That is deliberate — refusing them was a long-standing false positive — and it is why the rule asks whose `eval` this is rather than matching the word.

It does not follow that a capability never buys process execution. `os.system(...)` and `os.popen(...)` are refused *as calls* — but only as calls, so `f = os.system` then `f(cmd)` is one assignment past the rule — and `os.spawnve` / `os.execv` / `os.startfile` are not refused at all once `process-env` is granted. That is the same fact the `process-env` row states; it is repeated here because an earlier version of this paragraph claimed the opposite.

### What this is not

**This is a guardrail, not a sandbox** — the same framing the [in-canvas script policy](/advanced/python-script-node) carries, and worth repeating here because this is where a *stranger's* code runs.

- **The gate reads your plugin's own `import` statements.** It does not read the libraries you import, and it cannot tell what a permitted function does.
- **A capability gates an *import*, not an *action*.** Two consequences worth stating outright rather than leaving to be discovered:
  - **`filesystem` does not gate writing files.** `open(p, "w")` is a builtin, needs no import, and passes at Tier 0 with nothing declared. Gating it was considered and rejected: the mode is often computed (`open(p, "w" if overwrite else "r")`), so the check would be evaded by one variable while breaking honest plugins — a false positive with no matching security value.
  - **`network` implies a file write**, via `urllib.request.urlretrieve(url, dest)`.
- **A capability covers the blocklisted roots, not the category.** `requests` is gated; `httpx` was never on the blocklist, so a plugin that imports it reaches the network with nothing declared. Enumerating every HTTP client on PyPI is not a thing a list can do.
- **"No capability hands over a module whose purpose is running code" is a statement about the capability *map*, not a guarantee about what a granted plugin can reach.** Standard-library modules import each other and leave the results bound as ordinary attributes, so `import shutil` (under `filesystem`) puts `shutil.sys.modules['subprocess'].run(...)` one line away. The gate refuses the module names it knows about *as imports*; it does not walk the object graph of what it lets in. Note this escalates nothing — the identical line works with **zero** declarations on any CodefyUI before this feature, because `shutil` was equally importable there. It is a limit of the gate, not a cost of the tier.
- **Two paths skip the gate entirely, on purpose.** Built-in packs ship inside this repo and are reviewed by PR; `cdui plugin link` loads *your own* working tree and says so with a warning. `cdui project restore` also grants a project manifest's declared capabilities non-interactively — it already passes `--trust-author`, so this adds no exposure, but it means a project file is a consent decision too.
- **Anything that can write `installed.json` can pre-authorize the next update.** The lockfile is what makes `cdui plugin update` skip the prompt for an already-granted capability, so code that can edit it (including a plugin that already has `filesystem`, or any plugin at all via `open`) can add a capability to its own entry and have the next update accept it silently. This is post-compromise persistence, not a first-step escalation — but the lockfile is a trust store, and it is only as protected as your user account.
- **A declaration is a statement of intent by the author.** It raises the cost of a drive-by and gives you something to read before you consent. Treat "do I trust whoever wrote this?" as the real question.

### Upgrading from an older install

Nothing to do. A lockfile entry written before capabilities existed has no `capabilities` key, which reads as "none granted" — exactly the behaviour it had. Existing packs revalidate unchanged.

## Writing your own plugin

The fastest start is **`cdui plugin new`**, which scaffolds a ready-to-edit plugin in one command:

```bash
cdui plugin new my-plugin          # backend-only skeleton
cdui plugin new my-plugin --ui     # also a React frontend wired to the SDK
```

It generates a manifest, an example node, a test (with the `cdui_plugins.<id>` namespace shim so `pytest` works locally), and — with `--ui` — a Vite + React `ui/` whose `src/sdk/` is the typed plugin SDK. The plugin lands in `./my-plugin/`; link it with `cdui plugin dev` (below) and start editing.

For a richer reference, fork the **[Official Plugin Template](https://github.com/treeleaves30760/CodefyUI-Plugin-Official)** — a working, MIT-licensed plugin with two example nodes, a sample example graph, a test suite, and a fully-commented manifest. Its README walks through every field and the AST security gate.

```bash
# Install the template itself to see the pattern live
cdui plugin install treeleaves30760/CodefyUI-Plugin-Official

# After forking
cdui plugin install your-username/your-fork
```

A pack ships any of: a `nodes/` directory (auto-discovered), a `presets/` directory, an `examples/` directory, and an `assets/` directory (mounted at `/plugins/<id>/assets/<file>`). A `cdui.plugin.toml` manifest declares the id, version, `requires_codefyui`, content directories, lesson metadata, and — only if you need them — the `[security]` declarations described under [Security](#security--three-tiers). Delete that section if your nodes are pure computation; most are.

:::warning Breaking change (v0.3)
The chapter packs `c1`–`c6` were repackaged into three direction packs `foundations` / `deep` / `rl`, and every Edu node's type id gained a dash (`EduKNN` → `Edu-KNN`). Saved graphs referencing the old `cN:EduFoo` types must be updated to `<pack>:Edu-Foo` and the packs reinstalled with `cdui plugin install foundations deep rl`.
:::

## Local development

You don't need to push to GitHub between iterations while building a plugin. **Link** your working directory and CodefyUI loads it in place:

```bash
cdui plugin link ./my-plugin     # register the local dir in place (no copy)
# ...edit nodes/ or frontend/...
cdui plugin reload               # pick up changes in a running server
cdui plugin unlink my-plugin     # remove the link — your files are untouched
```

Even simpler, **`dev`** links and watches in one command, hot-reloading on every save:

```bash
cdui plugin dev ./my-plugin      # link + watch; reloads on every change
```

Run the server in another terminal (`cdui start` or `cdui dev`). `dev` polls the plugin's manifest, `nodes/`, `presets/`, and `frontend/`; `--once` links and reloads a single time (no watch), and `--interval` tunes the poll frequency. `link`, `dev`, and `reload` reach the server on its configured port (`CODEFYUI_PORT`, default `8000`), so running on a non-default port needs no extra flags.

`link` reads the id from your `cdui.plugin.toml` and records the directory's absolute path in the lockfile as `source_kind = "local"`, so discovery walks your working tree directly. The AST security gate is skipped for linked plugins (it's your own code, and a warning says so); `unlink` drops only the lockfile entry, never your files. After editing Python nodes, `cdui plugin reload` (or `cdui plugin dev`) reloads them. **Frontend edits to a linked plugin reload automatically too** — while a linked plugin is installed the editor watches for reloads and re-mounts the plugin's UI in place, no browser refresh needed.

:::tip Dev data isolation
Running plugin commands through `scripts/dev.py` — or setting `CODEFYUI_USER_DATA_DIR` — keeps a clone's lockfile inside the repo (`.codefyui_dev/`) instead of the machine-wide user-data dir, so multiple clones don't clobber each other.
:::

## REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/plugins` | GET | List installed plugin packs. |
| `/api/plugins/{id}` | GET | Get a plugin's manifest + README. |
| `/api/plugins/reload` | POST | Hot-reload all node and preset sources. |
