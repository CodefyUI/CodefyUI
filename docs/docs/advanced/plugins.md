---
sidebar_position: 3
title: Plugins
description: Install plugin packs of educational nodes, and learn how to write and publish your own.
---

# Plugin Packs

Educational ("Edu") nodes ship as installable **plugin packs**, organised **by direction** so each maps onto a hands-on textbook module and installs cumulatively as you progress.

```bash
cdui plugin sync                           # install every built-in pack you have not decided about
cdui plugin install foundations deep rl   # or pick them one by one
cdui plugin install edu stats              # hands-on labs, descriptive statistics
cdui plugin list
cdui plugin info deep                      # manifest, lessons covered, node names
cdui plugin search attention               # query the catalog
cdui plugin install foo/bar                # third-party pack from GitHub
cdui plugin uninstall deep                 # remembered: sync will not re-add it
```

## What's available

| Pack | Hands-on modules | Nodes |
|------|------------------|-----------|
| `foundations` | I1 Data Representation · I2 Classical ML | Edu-ColumnStats, Edu-KNN, Edu-LinearRegression, Edu-LogisticRegression, Edu-TokenEmbedding, Edu-FFN |
| `deep` | I3 Vision · I4 Sequences | Edu-CrossAttention, Edu-ResBlock, Edu-SelfAttention, Edu-MultiHeadAttention, Edu-Patchify |
| `rl` | I5 Reinforcement Learning | Edu-PolicyGradient |
| `edu` | I1 Data Representation · I2 Classical ML (hands-on labs) | FilterRows, SlidingWindow2D, SentenceEmbedding, Classifier, AdvancedClassifier, FFNLayer, ActivationLayer, TrainAndEvaluate |
| `stats` | — (any dataset) | Stats-Describe, Stats-GroupByAggregate, Stats-Histogram, Stats-Percentile, Stats-Correlation, Stats-ConfusionMatrix, Stats-TableView, Stats-ChartView |

`stats` is the odd one out: not a textbook companion but a working reference for third-party pack authors. It is pure numpy + torch, installs at [Tier 0](#security--three-tiers) with **no `[security]` section at all**, and its [README](https://github.com/CodefyUI/CodefyUI/blob/main/plugins/stats/README.md) is the normative write-up of the two contracts a data pack needs — how a table travels between ports, and how a `chart` output is declared and drawn.

Three more official plugins live in their own repositories. They install by the same name the catalog lists them under, and — unlike the packs above — each one is downloaded from GitHub, so installing it is a decision you are asked to confirm.

| Plugin | What it is | Install |
|------|------------------|-----------|
| `graph-copilot` | AI chat assistant that builds and edits node graphs by conversation, runs approved isolated experiments and parameter searches, and keeps portable evidence. Needs an LLM provider (Codex, Ollama, OpenAI or Anthropic). | `cdui plugin install graph-copilot` |
| `self-learning` | Turns a free-form machine-learning problem into a step-by-step lesson: an LLM builds a working graph and proves it runs, then the plugin screenshots each step and emits a Traditional-Chinese Markdown lesson, a printable page, the graph and a starter exercise. | `cdui plugin install self-learning` |
| `official-template` | Working starter template for plugin authors: two example nodes, a preset, an example workflow, an asset file, a sample test suite and a React tool panel. Install it to see what a plugin can do, fork it to write your own. | `cdui plugin install official-template` |

Each Edu node decomposes a single lesson concept into a chain of named steps that the [Teaching Inspector](/usage/teaching-inspector) renders one row at a time — `Edu-ColumnStats` shows the population-std formula as `sum → divide → deviations² → variance → sqrt`; `Edu-PolicyGradient` exposes `softmax → gather → log → baseline → loss`; `Edu-Patchify` makes `unfold → permute → flatten` visible. Switch on **Verbose mode** in the Settings popover to capture them.

## How packs are stored

- **Built-in direction packs** live in `plugins/<id>/` inside the repo and are activated in place (no copies).
- **Third-party packs** are downloaded as a pinned-SHA tarball into `<USER_DATA>/plugins/<id>/` and **AST-validated** before install (see [Security](#security--three-tiers)).
- A lockfile at `<USER_DATA>/plugins/installed.json` records every install — including which capabilities you granted — so `cdui start` rediscovers them on the next launch. It is also what "already installed" means: re-installing over a pack whose directory was deleted by hand, or over one you linked with `cdui plugin link`, asks for `--force` rather than overwriting silently, because replacing either is a decision for the person at the keyboard.

Plugin nodes are namespaced to avoid collisions and to self-document graphs — built-in nodes use a bare name like `Conv2d`, while plugin nodes are qualified like `foundations:Edu-KNN`.

### Catching up after an update — `cdui plugin sync`

The lockfile is what activates a pack, and an update does not write it. So when a release **adds** a built-in pack, its files land on your disk and nothing loads them: the nodes are installable and invisible at the same time. `cdui plugin sync` is the catch-up — it installs every built-in pack this install has not made a decision about, asks once before doing it, and reports per pack so one pack whose `python_deps` cannot be downloaded on a school network does not take the others down with it.

```bash
cdui plugin sync --dry-run   # just tell me what is pending
cdui plugin sync             # install it all (one confirmation)
cdui plugin sync --yes        # no prompt — scripts, CI, classroom images
cdui plugin sync --prune      # also drop lockfile entries for packs that no longer ship
```

Two things it deliberately does **not** do. It does not run at startup, and `cdui update` does not offer to run it: activating code you never asked for because a release shipped it is a consent decision, not an update detail. And it never re-adds a pack you uninstalled — `cdui plugin uninstall` records the removal in the lockfile (a `removed` map beside `plugins`), so "I have never seen this pack" and "I threw this pack away" stop being the same state. `cdui start` and `cdui plugin list` stop mentioning a removed pack for the same reason. To undo a removal, install the pack by name: `cdui plugin install stats` clears the record and sync counts it again.

## Security — three tiers

A plugin pack is Python that runs in the CodefyUI process. Before a third-party pack is installed, every `.py` file anywhere in it — `nodes/`, `examples/`, `tests/`, `docs/`, `assets/`, any other subdirectory — is walked by an AST gate that decides what it may import. Nothing is exempted by directory name: the plugin loader can import from anywhere in the pack (`from ..tests import helper` works from a node file), so the scan has to reach everywhere the loader does. The gate has three answers, and the middle one is the interesting one.

| Tier | How a plugin gets it | What it covers |
|------|----------------------|----------------|
| **0 — default** | nothing to declare | Pure computation: `math`, `statistics`, `collections`, `itertools`, `functools`, `json`, `re`, `dataclasses`, `typing`, `enum`, `decimal`, `random`, `numpy`, `torch`, `pandas` — plus the path helpers (below). Every first-party pack lives here. |
| **1 — declared capabilities** | `[security] capabilities = [...]` in the manifest, confirmed by the user at install | One named group of modules per capability. |
| **2 — trusted** | `[security] allowed_modules = [...]` **and** `cdui plugin install --trust-author` | Anything, including `subprocess`, `ctypes` and `importlib`. |

### The capabilities

| Capability | Unlocks | What you are agreeing to |
|------------|---------|--------------------------|
| `network` | `requests`, `urllib`, `http`, `socket`, `ssl`, and the raw C modules behind them (`_socket`, `_ssl`) | The plugin can send and receive data from any host — **and write what it downloads to disk**, because `urllib.request.urlretrieve(url, dest)` is one call. |
| `filesystem` | `pathlib`, `tempfile`, `shutil`, `zipfile`, `tarfile`, `gzip`, `bz2`, `lzma`, `codecs`, `sqlite3` (and `_sqlite3`), `glob`, `fileinput`, `readline` | The plugin can use the file **libraries**. This is not a write boundary: plain `open(p, "w")` is a builtin and needs no declaration at all (see [What this is not](#what-this-is-not)). |
| `process-env` | `os`, `ntpath`, `posixpath`, `genericpath`, `nt`, `posix` | The plugin gets **the whole `os` module**: read *and change* this process's environment (**including any API keys in it**), start other programs (`os.execv`, `os.spawnve`, `os.startfile`), and delete or rename files. The name is what people ask for it for; the grant is bigger than the name. |

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
import nt / posix                    # needs "process-env" — the raw module os.py builds itself from
```

The Tier-0 list is exactly: `join`, `basename`, `dirname`, `split`, `splitext`, `splitdrive`, `normpath`, `normcase`, `isabs`, `commonpath`, `commonprefix`, and the `sep` / `altsep` / `extsep` / `pathsep` / `curdir` / `pardir` / `defpath` constants.

The refused lines are not pedantry — `os.path` is a real module and most of its surface is not string manipulation:

- `os.path` **is** `ntpath` / `posixpath`, and those modules `import os` and `import sys` at module level, leaving both bound as ordinary attributes — so `path.os.remove(p)` deletes a file and `path.sys.modules['subprocess'].run([...])` runs a command.
- `os` itself **is** `nt` (Windows) or `posix` (POSIX) — CPython's own `os.py` does `from nt import *` / `from posix import *`, which is where `os.remove`, `os.environ` and `os.system` come from. Importing the raw module by name reaches the identical surface with nothing in between.
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
  Metric Logger 0.4.0
  Ships each run's metrics to a collector.
  Python packages: httpx>=0.27
  Proceed? [y/N]: y

> This plugin requests the following capabilities
    network -> reach the network -- send and receive data from any host, and write what it downloads to disk (requests, urllib, http, socket, ssl)
  A capability is a declaration, not a sandbox: once granted, the plugin may
  use that group of modules and CodefyUI stops asking.
  Grant these? [y/N]: y
  Resolving alice/metric-logger
  Downloading alice/metric-logger@a1b2c3d
    [##########] 100% 0.1/0.1 MB
  Unpacking metric-logger
  Scanning metric-logger for unsafe code
  Installing packages: httpx>=0.27
  Installing metric-logger
  Recording metric-logger
  + Hot-reloaded backend
  + Installed: metric-logger (a1b2c3d)
```

Everything above the first `y` is read from the manifest alone, at the one commit the install would use — what the plugin is, the packages it would add to your venv, the modules it asks to import outside the allowlist, and whether it ships JavaScript. So both questions are asked, and can be answered `no`, before a byte of the repository is fetched. What follows the second `y` is the install itself, step by step; those step lines are the shared install path's own, which is why they read the same here as they do in the [Plugin Center](#plugin-center).

- **Without a terminal** (a script, CI, a piped install) the answer is **no**, and the message names `--accept-capabilities`, which grants the declared set without the prompt. `-y` / `--no-confirm` does *not* imply it: that flag skips the "install from this URL?" question, and consenting to code that reaches the network is a different question.
- **What was granted is recorded** in `<USER_DATA>/plugins/installed.json` and shown by `cdui plugin list` and `cdui plugin info`.
- **`cdui plugin update` does not re-ask** when the new version's request is a subset of what you already granted — and **stops** when it grew a capability, which is the supply-chain shape an update can actually catch.

### What holds in every tier

`torch.load(...)` still requires an explicit `weights_only=True`; dunder access (`__class__`, `__globals__`, `__subclasses__`, …), frame walking (`f_globals`, `gi_frame`, …), and the **builtins** `eval` / `exec` / `compile` / `__import__` — as bare calls and through the `builtins` module — are refused whatever was declared. **A capability never buys reflection.**

The builtins, not the spelling: a *method* that shares one of those names is ordinary code and passes at every tier, so `torch.compile(model)` and `model.eval()` are allowed for plugins. That is deliberate — refusing them was a long-standing false positive — and it is why the rule asks whose `eval` this is rather than matching the word.

It does not follow that a capability never buys process execution. `os.system(...)` and `os.popen(...)` are refused *as calls* — but only as calls, so `f = os.system` then `f(cmd)` is one assignment past the rule — and `os.spawnve` / `os.execv` / `os.startfile` are not refused at all once `process-env` is granted. That is the same fact the `process-env` row states; it is repeated here because an earlier version of this paragraph claimed the opposite.

### Attribute names closed by default, lifted at Tier 2

Separately from every rule above — which holds whatever was declared, with no exception at any tier — a fixed list of attribute names on the Tier-0 libraries is refused at Tier 0 and Tier 1, and lifted at Tier 2. `numpy.zeros(3).dump(path)` pickles straight to any path with substantially attacker-chosen content; `torch.hub.load(...)` downloads and executes a remote `hubconf.py`; `.savetxt`, `.tofile`, `.load_state_dict_from_url`, `.tensorboard` and about a dozen more are the same shape — a *method* on a value a Tier-0 import hands back, not an import of its own, so the capability gate (which only ever looks at `import` statements) never sees it. No capability lifts these — the module they live on is already Tier 0, so no capability grants anything new by naming it — the same list the [in-canvas script policy](/advanced/python-script-node) already carries.

The rule is receiver-independent, which cuts both ways: it also refuses the plugin's *own* method if it happens to share one of these names — `self.save(...)` on your own class is blocked exactly like `numpy.array(...).save(...)`, the same cost the script policy already imposes on a script's own `obj.save()`. At Tier 0 or Tier 1 alone, that means a class cannot define a method called `save`, `dump`, `hub`, or any of the others on the list, full stop.

**`--trust-author` lifts this list entirely.** Once a plugin is installed with `--trust-author` and `[security] allowed_modules`, `.dump` / `.hub` / `.save` and the rest of it are ordinary attribute names again — a plugin trusted with `subprocess` and `ctypes` gains nothing from also being refused `arr.dump()`, and the refusal would otherwise have made it impossible to write a plugin with a method named `save` at all. This is unlike every rule in [What holds in every tier](#what-holds-in-every-tier), which stays refused without exception: those refuse *reflection*, which no capability or trust level ever buys; `.dump` and `.hub` are file writes and remote code fetches, and `--trust-author` already grants an equivalent or greater version of both by a shorter route.

### Ship source, not bytecode

Every file in a plugin tarball that Python's import system could load has to
be readable as **source**. The installer scans the whole directory, not just
`nodes/`, and it enumerates by what the loader accepts (`importlib.machinery.all_suffixes()`)
rather than by `*.py`: `.py` and `.pyw` are scanned, and `.pyc` / `.pyo` /
`.pyd` / `.so` / `.dylib` are **refused by name** at install time, on every
platform regardless of which one you install from.

The refusal is the honest answer rather than a policy preference. A `.pyc`
would have to be decompiled to be scanned and a compiled extension cannot be
scanned even in principle, so the alternative is importing code the gate
never opened — which is exactly what used to happen: a pack whose `nodes/`
held `helper.pyc` and no `helper.py` was imported at server boot, at full
trust, with no capability declared and without `--trust-author`, having never
been looked at.

Compilation artifacts are not affected. CPython writes its cache as
`__pycache__/<name>.cpython-311.pyc`, whose stem is not a valid identifier,
so no `import` statement can name it — those are skipped. An
attacker-supplied `__pycache__/payload.pyc` **can** be named, and is refused.

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

## Plugin Center

**Installing a plugin from inside the app is the same install the terminal runs.** `cdui plugin install` and the Plugin Center are two front ends over one function in `backend/app/core/plugins/`, so the order of an install, what counts as a failure and what a failure is called are decided once — the console and the panel cannot come to disagree about any of the three. The routes below are the whole of it: the editor's panel is one client, and `cdui plugin install` is another.

**It is a conversation with two turns.** `POST /api/plugins/inspect` reads one source — a catalog name, an `owner/repo`, a URL — and answers with everything a person needs in order to decide: what the plugin is, what it would add to your Python environment, which capabilities it declares, which modules it asks to have the security scan turned off for, whether it ships JavaScript that will run in your editor, and whether you already have it. Nothing is downloaded and nothing is installed; the manifest is read at ONE resolved commit, and the answer is kept under an `inspection_id`. `POST /api/plugins/install` then installs *that* inspection, by its id, with the answers only a person can give (`accept_capabilities`, `trust_author`, and `force` to replace what is already there). The server never takes a manifest, a commit or a capability list from a request body — so the manifest you agreed to is the manifest that is installed, and a tarball that grew a capability, changed its id or added a module to its allowlist between the two turns is refused rather than installed. The install runs as a job: `202` with a `job_id`, then `GET /api/plugins/jobs/{job_id}/events` replays that job's log from a cursor and long-polls for the rest, and `POST /api/plugins/jobs/{job_id}/cancel` stops it cleanly enough that nothing half-written is left behind.

**A review screen is the three tiers, one for one.** [Tier 0](#security--three-tiers) has nothing to ask. Tier 1 is the inspection's `capabilities` — one row each, in the same words the console prints — and Tier 2 is its `allowed_modules`, which is a separate decision about the *author* and travels as `trust_author`. Neither is a sandbox: once a capability is granted the plugin may import that group of modules and CodefyUI stops asking, which is why [What this is not](#what-this-is-not) is the paragraph to read before granting one.

**Installing is a local-only operation.** Every route that changes what code is on this machine — inspect, install, cancel, update, delete — needs the session token *and* the server to be bound to loopback: installing a plugin puts a stranger's code where this process will import it, inspecting reaches out to GitHub on the caller's word, and deleting takes somebody's plugin away. A classroom or lab server that deliberately serves the LAN opts back in with `CODEFYUI_ALLOW_REMOTE_PLUGIN_INSTALL=1`. Reload, enable and disable take the token but not the loopback gate — they act on code this machine already has and you already agreed to — and reads are open, including a job's events, which is what a second tab that opened mid-install follows.

**The steps and the failure messages are English.** `Resolving …`, `Downloading …`, `Unpacking …`, `Scanning … for unsafe code`, `Installing packages: …`, `Installing …`, `Recording …`, and every sentence a refused or failed install carries, come out of the shared install path, which has one set of words rather than one per front end. The interface around them is translated; these are not.

**An install can end in `needs_restart`, which is not a failure.** A plugin's `[python_deps]` install add-only, under a constraints file that pins every package the running server has already loaded, so nothing a plugin asks for can downgrade what your session is holding open. When the resolver says that cannot be done live, the job ends `needs_restart` and carries the exact `command` to run with the server stopped. Nothing is wrong with the plugin, and asking the same server again will end the same way. (`cdui plugin install` prints that command too, and exits `3`.)

**Uninstalling removes what this install downloaded, and nothing else.** `DELETE /api/plugins/{id}` deletes a downloaded pack's directory; a built-in pack keeps its files — they belong to the release — and is remembered as removed, so `cdui plugin sync` leaves it alone until you install it by name again; a directory you linked with `cdui plugin link` stays exactly where its author put it. What it never removes is the plugin's Python packages: uninstalling packages from inside the process that imported them is how you get a half-loaded interpreter serving requests. So the answer says so instead — `python_deps_left` names them and `uninstall_command` is the line to run once the server is stopped:

```bash
uv pip uninstall --python <the CodefyUI venv's python> httpx
```

If the directory cannot be deleted — something holding a file open, the ordinary cause on Windows — nothing is removed at all: the lockfile entry stays, the plugin stays installed, and the answer is `409` `files_locked`, carrying the operating system's own sentence and the directory that is still there. Close whatever is using it, or stop the server, and remove the plugin again.

**Updating asks the plugin's own repository, and answers one of three ways.** `POST /api/plugins/{id}/update` re-reads the manifest at whatever commit that repository has now. `200` `{"status": "up_to_date", "sha": …}` — the commit you have is the commit that is there. `202` with a `job_id` — the new version asks for nothing you have not already granted, so it is already being installed. `200` `{"status": "needs_consent", …}` — it asks for more, and the body carries the same inspection `/inspect` returns plus `capabilities_added` and `allowed_modules_added`, which are the whole content of an update's review screen. You finish that one by posting the inspection back to `POST /api/plugins/install` with `inspection_id`, `accept_capabilities` and `trust_author` — and no `force`: the server recorded that this inspection came from the update button, so it does not make you say "yes, replace it" about a plugin you asked it to replace.

Two things an update will not do. It never installs a *different* plugin: a repository whose manifest now declares another id is refused with `400` `not_updatable` rather than fetched, because updating `metric-logger` and ending up with `metric-logger-ng` — possibly on top of a plugin of that name you already had — is not an update. If a renamed repository is what you want, install it as the new plugin it now is. And an update keeps the plugin switched off if you had switched it off: coming back enabled is a decision, and re-installing the same plugin from the same repository is not where it belongs.

A built-in pack and a linked directory answer `400` `not_updatable` too, with the hint that says what to do about it: a pack that ships in this release updates with `cdui update`, and a linked directory is already whatever is on its author's disk.

**One install at a time, across both centers.** Starting an install — or an update, which is an install — while one is already running answers `409` with the `job_id` to follow, and so does uninstalling, enabling or disabling *that* plugin while its own install is in flight: a lockfile entry rewritten halfway through is how a plugin ends up on disk with nothing pointing at it. Another plugin's install blocks none of the three, because two plugins are two directories and two lockfile keys.

## Writing your own plugin

The fastest start is **`cdui plugin new`**, which scaffolds a ready-to-edit plugin in one command:

```bash
cdui plugin new my-plugin          # backend-only skeleton
cdui plugin new my-plugin --ui     # also a React frontend wired to the SDK
```

It generates a manifest, an example node, a test (with the `cdui_plugins.<id>` namespace shim so `pytest` works locally), and — with `--ui` — a Vite + React `ui/` whose `src/sdk/` is the typed plugin SDK. The plugin lands in `./my-plugin/`; link it with `cdui plugin dev` (below) and start editing.

For a richer reference, fork the **[Official Plugin Template](https://github.com/CodefyUI/CodefyUI-Plugin-Official)** — a working, MIT-licensed plugin with two example nodes, a sample example graph, a test suite, and a fully-commented manifest. Its README walks through every field and the AST security gate. The catalog lists it as `official-template`, so you can install it by name.

```bash
# Install the template itself to see the pattern live
cdui plugin install official-template

# After forking — any repository, by owner/repo or by URL
cdui plugin install your-username/your-fork
```

A pack ships any of: a `nodes/` directory (auto-discovered), a `presets/` directory, an `examples/` directory, and an `assets/` directory (served at `/plugins/<id>/assets/<file>`). A `cdui.plugin.toml` manifest declares the id, version, `requires_codefyui`, content directories, lesson metadata, and — only if you need them — the `[security]` declarations described under [Security](#security--three-tiers). Delete that section if your nodes are pure computation; most are.

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

A linked plugin's `[python_deps]` are installed by the same rules a downloaded pack's are: add-only, under the constraints file that pins every package the running server already loaded. So `cdui plugin link` has the install path's exit codes as well — `3` when a package it asks for cannot go into a live server (the command to run with the server stopped is printed), and `130` for a `Ctrl+C` — where it used to hand back whatever the package manager returned.

:::tip Dev data isolation
Running plugin commands through `scripts/dev.py` — or setting `CODEFYUI_USER_DATA_DIR` — keeps a clone's lockfile inside the repo (`.codefyui_dev/`) instead of the machine-wide user-data dir, so multiple clones don't clobber each other.
:::

## REST API

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/plugins` | GET | open | Every installed plugin pack, enabled or not. |
| `/api/plugins/catalog` | GET | open | What this build can install by name, merged with what you have installed — one row per plugin, each saying which state it is in. |
| `/api/plugins/generation` | GET | open | The reload counter the editor polls to notice the palette changed. |
| `/api/plugins/{id}` | GET | open | One plugin's manifest, nodes and README. |
| `/api/plugins/jobs/{job_id}/events` | GET | open | An install job's log and progress after `?cursor=`; `?wait=` long-polls for the rest. |
| `/api/plugins/reload` | POST | token | Re-discover nodes, presets and packs. |
| `/api/plugins/{id}/enable` | POST | token | Turn an installed plugin on. |
| `/api/plugins/{id}/disable` | POST | token | Turn it off without uninstalling it. |
| `/api/plugins/inspect` | POST | token + loopback | Read one source at one commit and say what installing it would cost. Installs nothing. |
| `/api/plugins/install` | POST | token + loopback | Install what an inspection described — `202` with a `job_id`. |
| `/api/plugins/jobs/{job_id}/cancel` | POST | token + loopback | Ask the running install to stop. |
| `/api/plugins/{id}/update` | POST | token + loopback | Fetch what the plugin's own repository has now. Three answers: `202` `{job_id}`, `200` `{status: "up_to_date", sha}`, or `200` `{status: "needs_consent", inspection, capabilities_added, allowed_modules_added}` — which the client finishes with `POST /install {inspection_id, accept_capabilities, trust_author}`, no `force`. |
| `/api/plugins/{id}` | DELETE | token + loopback | Uninstall it, and say what that left behind. |

**open** is a read the editor polls, like every other read in the app. **token** is the session header every mutating call takes. **token + loopback** additionally refuses unless the server is bound to loopback, unless `CODEFYUI_ALLOW_REMOTE_PLUGIN_INSTALL=1` says otherwise — see [Plugin Center](#plugin-center) for why the line falls where it does.
