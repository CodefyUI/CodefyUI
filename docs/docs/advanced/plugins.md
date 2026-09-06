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
cdui plugin disable deep                   # disable without deleting files
cdui plugin enable deep                    # re-enable without downloading
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

Each Edu node decomposes a single lesson concept into a chain of named steps that the [Teaching Inspector](/usage/teaching-inspector) renders one row at a time — `Edu-ColumnStats` shows the population-std formula as `sum → divide → deviations² → variance → sqrt`; `Edu-PolicyGradient` exposes `softmax → gather → log → baseline → loss`; `Edu-Patchify` makes `unfold → permute → flatten` visible. Switch on **Verbose internals** in the Settings popover to capture them.

## How packs are stored

- **Built-in direction packs** live in `plugins/<id>/` inside the repo and are activated in place (no copies).
- **Third-party packs** are downloaded as a pinned-SHA tarball into `<USER_DATA>/plugins/<id>/` and **AST-validated** before install (see [Security](#security--three-tiers)).
- A lockfile at `<USER_DATA>/plugins/installed.json` records every install, including granted capabilities, so `cdui start` can rediscover plugins on the next launch. `<USER_DATA>` is `<install dir>/.codefyui_dev/` for commands run through `cdui`: `cdui start`, `cdui dev`, and every `cdui plugin` command set `CODEFYUI_USER_DATA_DIR` to that directory unless it was already exported. A default install therefore uses `~/CodefyUI/.codefyui_dev/plugins/installed.json`. The platform user-data directory (`%LOCALAPPDATA%\codefyui`, `~/.local/share/codefyui`, or `~/Library/Application Support/codefyui`) applies only to `uvicorn app.main:app` launched directly without `CODEFYUI_USER_DATA_DIR`. The lockfile also determines whether a plugin is already installed. Reinstalling when its directory was deleted manually, or replacing a directory linked through `cdui plugin link`, requires `--force`.

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

> Installing plugin: alice/metric-logger
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

**The Plugin Center and `cdui plugin install` use the same backend implementation.** Both clients therefore use the same installation steps, failure criteria, and error codes.

### Using the Plugin Center

**Opening the Plugin Center.** Use either the **Plugin Center...** button in the **Plugins** section of the sidebar's **Custom & Plugins** tab or the **Open** button in **Settings → Plugins → Plugin Center**. The Settings row includes the text "Install teaching node packs and plugins from GitHub." and an *N installed, M available* summary. The Package Center and Plugin Center can remain open at the same time; **Escape** closes the topmost window.

**Plugin list.** The list contains every built-in or official plugin available by catalog name and every installed plugin. Filters are **All**, **Installed**, and **Available**. Each card shows the name, status, version, origin, repository and pin (`ref @ sha`), lessons, node count, and Python dependencies. Origin is **Built-in**, **Official**, or **Linked folder** for a directory registered with `cdui plugin link`; other third-party repositories have no origin chip. Available actions depend on the [install state](#install-states).

**Installing a plugin outside the catalog.** Enter `owner/repo`, `owner/repo@ref`, a GitHub URL, or a catalog name in **Install from GitHub**, then select **Review**. Other formats are rejected before submission with "Enter a catalog name, owner/repo[@ref] or a GitHub URL."

**Review and consent.** Selecting **Install** on a card or **Review** in the input reads the manifest at one resolved commit. When consent is required, a **Review before installing** card appears at the top of the list. It shows the name, version, description, author, newly registered nodes, Python dependencies, commit pin, and an HTTP or HTTPS **Homepage** link when provided. Each required decision has a checkbox. **This plugin asks for:** lists each declared capability and its access; **Grant these capabilities** records [Tier 1](#security--three-tiers) consent. **I trust this author. Allows: ...** lists `allowed_modules` and records [Tier 2](#security--three-tiers) consent. Browser code adds the warning "Ships JavaScript that runs in this editor with full access." **Install** remains disabled until all required checkboxes are selected. Built-in packs require no consent and install directly from their row. An installed plugin displays a replacement warning and a **Reinstall** button.

**Install progress.** The right pane shows the current step, progress bar, recent log entries, and **Cancel install**. Steps can include *Resolving the source*, *Downloading*, *Unpacking*, *Checking the code*, a pip step when `[python_deps]` is present, *Copying files*, *Recording the install*, and *Loading the nodes*. Final states are *installed*, *updated*, *failed* with a server hint, *cancelled*, *needs_restart*, or *lost*. For *needs_restart*, no plugin files were installed and the pane shows the `uv pip install` command to run after stopping the server; see [How an install runs](#how-an-install-runs). *Lost* means that the browser disconnected from the server; refresh to obtain the current state. Closing the Plugin Center does not cancel the job, and another tab can follow it. After a catalog install fails, the pane also shows `cdui plugin install <repo>[@ref]` so the same operation can be run from a terminal with the full log visible.

**Applying changes.** After an install, update, enable, disable, or uninstall, the panel reloads the catalog, node definitions, and plugin UIs. The palette and plugin panels reflect the new state without a page reload.

**Enable, disable, update, uninstall.** **Disable** removes an installed plugin's nodes from the palette and stops serving its bundle and assets without deleting files. **Enable** re-enables the plugin without downloading it again. `cdui plugin enable|disable <id>` provides the same operations in a terminal. **Update** is available only for plugins installed from GitHub; built-in packs update through `cdui update`, and linked directories use their current files. **Uninstall** first asks: "Uninstall *name*? Graphs that use its nodes will stop running. Its Python packages stay installed." Linked directories provide only Enable and Disable because `cdui plugin link` manages their registration.

**Using another computer.** When the server is not bound to a loopback address, the footer reads "Installing is only allowed from the computer that runs the server." **Review**, **Install**, **Update**, and **Uninstall** are disabled with the same tooltip; **Enable** and **Disable** remain available. Set `CODEFYUI_ALLOW_REMOTE_PLUGIN_INSTALL=1` to permit remote plugin operations. See [How an install runs](#how-an-install-runs) for the affected routes. A server older than the panel shows "This server has no Plugin Center. Update CodefyUI and restart it."

### Install states

`GET /api/plugins/catalog` assigns each plugin one of six states. The state determines its status pill and buttons. State precedence is an active job, then the lockfile entry, then a `removed` record when no lockfile entry exists.

| State | Pill | Meaning | Buttons |
|-------|------|---------|---------|
| `available` | Not installed | In the catalog, no lockfile entry. | Install |
| `removed` | Removed | No lockfile entry, but a `removed` record from an uninstall, so `cdui plugin sync` leaves it alone. Counted under **Available**. | Install (clears the record) |
| `installing` | Installing | A job for this plugin is running. Counted under **Installed**. | none |
| `installed` | Installed | Lockfile entry, files on disk, enabled. | Disable, Update (GitHub installs only), Uninstall |
| `disabled` | Disabled | Lockfile entry, files on disk, switched off: nodes unregistered, bundle and assets not served. | Enable, Uninstall |
| `missing_files` | Files missing | A lockfile entry exists, but its directory is missing, for example after moving a checkout or interrupting an uninstall. Counted under **Installed**. | Install; because the lockfile entry remains, the server answers `409` `already_installed` and the review card then offers **Reinstall**. Uninstall removes the entry. |

Linked directories (`source_kind` `local`) provide only Enable and Disable in every state. The sidebar and Settings counts include only `installed` and `disabled` plugins.

### How an install runs

**Inspection and installation are separate requests.** `POST /api/plugins/inspect` resolves a catalog name, `owner/repo`, or URL to one commit and returns the plugin description, Python dependencies, declared capabilities, `allowed_modules`, browser-code status, and current install status. It reads the manifest but does not download the plugin archive or install files. The result is stored under an `inspection_id`. `POST /api/plugins/install` accepts that `inspection_id` together with `accept_capabilities`, `trust_author`, and `force` when replacing an existing install. It does not accept a manifest, commit, or capability list. The server therefore rejects an archive whose manifest adds a capability, changes its id, or adds an allowed module after inspection. Installation runs as a job: the request returns `202` with a `job_id`; `GET /api/plugins/jobs/{job_id}/events` replays events after a cursor and long-polls for new events; and `POST /api/plugins/jobs/{job_id}/cancel` cancels the job and removes partial writes.

**The review fields correspond to the three security tiers.** [Tier 0](#security--three-tiers) requires no consent. Tier 1 lists each value from the inspection's `capabilities`. Tier 2 lists `allowed_modules` and requires the separate author-trust decision sent as `trust_author`. Neither tier is a sandbox. After a capability is granted, the plugin may import that module group without another prompt. Review [What this is not](#what-this-is-not) before granting access.

**Plugin installation is restricted to local clients by default.** The inspect, install, cancel, update, and delete routes require the session token and a server bound to a loopback address. These routes can retrieve external code, install it into the server process, or remove an installed plugin. A classroom or lab server on a LAN can allow remote access with `CODEFYUI_ALLOW_REMOTE_PLUGIN_INSTALL=1`. Reload, enable, and disable require the token but not a loopback binding because they operate on existing local files. Read routes remain open, including job events, so another tab can monitor an active installation.

**GitHub API rate limits.** Unauthenticated GitHub API access is limited to 60 requests per hour for each IP address. Computers behind a shared NAT use the same quota. When the quota is exhausted, the panel shows "GitHub's request limit was reached" (`502` `github_rate_limited`). Export `CODEFYUI_GITHUB_TOKEN` before `cdui start`, or in the shell that runs `cdui plugin install|info|update`; read access to public repositories is sufficient. The token is read from the environment for each request, so adding it does not require a server restart. It is sent only to GitHub as a bearer header, removed on redirects, and excluded from logs and error messages.

**Install steps and backend failure messages are in English.** The shared backend emits `Resolving …`, `Downloading …`, `Unpacking …`, `Scanning … for unsafe code`, `Installing packages: …`, `Installing …`, and `Recording …`. Refusal and failure messages from the same backend are also English. The surrounding interface is translated, but these messages are not.

**`needs_restart` is not a failure.** A plugin's `[python_deps]` are installed add-only under a constraints file that pins packages already loaded by the running server. If dependency resolution cannot satisfy those constraints during a live install, the job ends with `needs_restart` and returns the exact `command` to run after stopping the server. Repeating the installation while the same server is running produces the same result. `cdui plugin install` also prints the command and exits with code `3`.

**Uninstall behavior depends on the plugin source.** `DELETE /api/plugins/{id}` deletes the directory of a downloaded plugin. Built-in plugin files remain because they are part of the release; the server records the plugin as removed so `cdui plugin sync` does not restore it until it is installed by name. A directory registered with `cdui plugin link` also remains unchanged. Python dependencies are not removed because uninstalling modules already imported by the running server could leave the process in an inconsistent state. The response lists retained dependencies in `python_deps_left` and provides an `uninstall_command` to run after stopping the server:

```bash
uv pip uninstall --python <the CodefyUI venv's python> httpx
```

If the directory cannot be deleted, the operation makes no changes: the lockfile entry remains, the plugin stays installed, and the server returns `409` `files_locked` with the operating-system error and the remaining directory. This commonly occurs on Windows when another process has a file open. Close that process or stop the server, then retry the uninstall.

**An update returns one of three results.** `POST /api/plugins/{id}/update` reads the current manifest from the plugin's recorded repository. `200` `{"status": "up_to_date", "sha": …}` means the installed commit already matches. `202` with a `job_id` means installation has started because the update requires no additional consent. `200` `{"status": "needs_consent", …}` means the update requests additional access. That response contains the same inspection data as `/inspect`, plus `capabilities_added` and `allowed_modules_added` for the review screen. To continue, send its `inspection_id`, `accept_capabilities`, and `trust_author` to `POST /api/plugins/install`. Do not send `force`; the server records that the inspection was created for an update and permits replacement.

An update cannot change the plugin id. If the repository's current manifest declares a different id, the server returns `400` `not_updatable` without downloading the plugin. This prevents an update from replacing a different plugin, including one already installed under the new id. Install a renamed plugin separately under its new id. An update also preserves a disabled plugin's disabled state.

Built-in packs and linked directories also return `400` `not_updatable` with an alternative action in `hint`. Update a built-in pack with `cdui update`. A linked directory already uses its current files.

**Only one installation can run across the Plugin Center and Package Center.** Starting a plugin install or update while any installation is active returns `409` with its `job_id`. Uninstalling, enabling, or disabling a plugin while that same plugin is being installed also returns `409`; this prevents concurrent changes to its lockfile entry. An active installation for a different plugin does not block those three operations because the plugins use separate directories and lockfile keys.

### Refusal codes

Most install-route refusals use `{"detail": {"code": "...", ...}}`, with a code and any fields required by the client. The panel and `cdui` use the code for control flow and can localize the displayed message. Two exceptions return a plain-text `detail`: the loopback gate's `403` ("Installing plugins is only allowed from the computer that runs the server. Set CODEFYUI_ALLOW_REMOTE_PLUGIN_INSTALL=1 to override.") and the `404` returned by `enable` or `disable` when the plugin is not installed.

| Status | Code | From | Meaning |
|--------|------|------|---------|
| 400 | `unparseable_source` | inspect | The value is not a catalog name, `owner/repo[@ref]`, or GitHub URL. |
| 400 | `unknown_catalog_name` | inspect | The catalog has no matching bare name. `known` lists available names. |
| 400 | `reserved_id` | inspect, update | The manifest `id` conflicts with a built-in plugin or a reserved route name such as `catalog`. The response includes `id`. |
| 400 | `invalid_manifest` | inspect, update | The manifest has no `[plugin]` table, uses a `schema_version` other than `1`, has an invalid id or `[security]` value, contains invalid TOML, or is not text. |
| 400 | `consent_required` | install | One or more declared capabilities were not accepted. `missing_capabilities` lists them. |
| 400 | `trust_author_required` | install | The manifest has `allowed_modules`, but `trust_author` was not `true`. The response includes `allowed_modules`. |
| 400 | `not_updatable` | update | The plugin is built in, linked, has no recorded repository, or its repository now declares another id. `hint` gives the alternative action. |
| 404 | `not_found` | inspect, update | GitHub has no matching repository or ref. |
| 404 | `unknown_job` | events, cancel | The requested job is unavailable because only the most recent job is retained. The response includes `job_id`. |
| 404 | `inspection_expired` | install | The inspection has expired. Inspect the source again. The response includes `inspection_id`. |
| 404 | `not_installed` | update, DELETE | No plugin is installed under that id. |
| 409 | `already_installed` | install | The plugin is already installed. Retry with `force: true`, which is the panel's **Reinstall** action. The response includes `plugin_id`. |
| 409 | `busy` | install, update, DELETE, enable, disable | An installation is active: any plugin blocks `install` and `update`; a plugin blocks its own DELETE, enable, and disable operations. The response includes `job_id`. |
| 409 | `pack_install_running` | install, update | The Package Center is using the installation slot shared by both centers. The response includes `job_id`. |
| 409 | `inspect_busy` | inspect, update | Another inspection is active. Retry after it finishes. |
| 409 | `files_locked` | DELETE | The directory could not be deleted and no state changed. This commonly means a file is open on Windows. The response includes `error` and `hint`. |
| 502 | `github_rate_limited` | inspect, update | GitHub returned 403 or 429. Wait for the quota to reset or set `CODEFYUI_GITHUB_TOKEN`. |
| 502 | `github_unreachable` | inspect, update | Another error prevented the server from reaching GitHub. |
| 503 | `unavailable` | inspect, install, events, cancel, update | The plugin service did not start. `GET /catalog` remains available, and the panel shows "This server has no Plugin Center." |

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

Place plugin content in fixed directories next to the manifest: `nodes/` (auto-discovered), `presets/`, `examples/`, `assets/` (served at `/plugins/<id>/assets/<file>`), and `frontend/` (see [Plugin Frontend Extensions](/advanced/plugin-frontend-extensions)). These directory names are not configurable, and the scaffold's `[content]` table is ignored. The `cdui.plugin.toml` manifest declares the id, version, lesson metadata, and any `[security]` settings described under [Security](#security--three-tiers). Omit `[security]` when the nodes use only Tier-0 imports.

### Manifest reference

Only fields used by installation or loading are validated. Other fields may be displayed or stored without validation.

| Field | Validated | What it does |
|-------|-----------|--------------|
| `[plugin] id` | yes | Must match `^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$`. It is the install directory, the node namespace (`<id>:<NODE_NAME>`, hyphens kept) and the name you install and uninstall by; only the Python package it is imported as is snake_cased (`cdui_plugins.my_plugin`). |
| `[plugin] schema_version` | yes | Must be `1`; anything else is refused ("Unsupported plugin schema_version"). |
| `[plugin] name`, `version`, `description` | no | Shown on the Plugin Center card and review card, by `cdui plugin info`, and in `GET /api/plugins`. `name` falls back to the id. |
| `[plugin] homepage` | no | The **Homepage** link on the review card; http(s) URLs only. |
| `[plugin] authors` (list) or `author` (string) | no | Shown only on the review card ("Author: ..."). |
| `[plugin] requires_codefyui`, `license` | no | Stored but not enforced, checked, or printed. |
| `[security] capabilities` | yes | A list of strings out of `network`, `filesystem` and `process-env` — [Tier 1](#security--three-tiers). An unknown name refuses the whole manifest. |
| `[security] allowed_modules` | yes | A list of module names — [Tier 2](#security--three-tiers), installable only with `--trust-author` or the review card's "I trust this author". A bare string instead of a list is refused. |
| `[python_deps]` | at install | `name = "constraint"` pairs installed with `uv pip install` before the files are copied. A constraint that starts with an operator is used as written (`">=0.27"`), a bare version is pinned (`"1.2.0"` becomes `==1.2.0`), an empty string means any version. Extras, URLs and `git+` sources are refused. |
| `[frontend] entry` | at load | A relative POSIX path that must start with `frontend/` (`"frontend/index.js"`); anything else counts as "no frontend". |
| `[lessons] chapters`, `lessons` | no | Lists of strings: the card's **Lessons:** line and `cdui plugin info`. |
| `[content]` | no | Ignored — see above. |

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

Run the server in another terminal (`cdui start` or `cdui dev`). `dev` polls the plugin manifest, `nodes/`, `presets/`, and `frontend/`; `--once` links and reloads once without watching, and `--interval` sets the polling interval. Commands that request a server reload—`link`, `unlink`, `dev`, `reload`, `enable`, `disable`, `install`, and `uninstall`—send a POST request to `127.0.0.1:<port>`. They use `CODEFYUI_PORT` from the command's shell, or `8000` when unset. `cdui start --port <n>` sets the variable only for the server process it starts. For a non-default port, export `CODEFYUI_PORT=<n>` before running these commands or `cdui project publish`, which resolves the port the same way. Otherwise, the lockfile can be updated while the reload request reports `Server not running`.

`link` reads the id from your `cdui.plugin.toml` and records the directory's absolute path in the lockfile as `source_kind = "local"`, so discovery walks your working tree directly. The AST security gate is skipped for linked plugins (it's your own code, and a warning says so); `unlink` drops only the lockfile entry, never your files. After editing Python nodes, `cdui plugin reload` (or `cdui plugin dev`) reloads them. **Frontend edits to a linked plugin reload automatically too** — while a linked plugin is installed the editor watches for reloads and re-mounts the plugin's UI in place, no browser refresh needed.

A linked plugin's `[python_deps]` are installed by the same rules a downloaded pack's are: add-only, under the constraints file that pins every package the running server already loaded. So `cdui plugin link` has the install path's exit codes as well — `3` when a package it asks for cannot go into a live server (the command to run with the server stopped is printed), and `130` for a `Ctrl+C` — where it used to hand back whatever the package manager returned.

:::tip One lockfile per install
`cdui` commands are implemented by `scripts/dev.py`. Unless `CODEFYUI_USER_DATA_DIR` is already set, `cdui start`, `cdui dev`, `cdui run` and the `plugin` / `project` / `cache` / `packs` groups set it to the current installation's `<install dir>/.codefyui_dev/` (see [How packs are stored](#how-packs-are-stored)). Each clone therefore has a separate lockfile and separate linked-plugin registrations. A plugin linked in one clone is not available to a server started from another clone. Export `CODEFYUI_USER_DATA_DIR` to use another location; an existing value takes precedence.
:::

## REST API

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/plugins` | GET | open | List all installed plugins, including disabled plugins. |
| `/api/plugins/catalog` | GET | open | Merge catalog entries with installed plugins. Each plugin has one of the [six states](#install-states); the response also includes `active_job`, `remote_install_allowed`, and `generation`. |
| `/api/plugins/generation` | GET | open | Return the reload counter polled by the editor when refreshing the node palette. |
| `/api/plugins/{id}` | GET | open | Return one plugin's manifest, nodes, and README. |
| `/plugins/{id}/frontend/{path}` | GET | open | Serve a file from an enabled plugin's `frontend/` when its manifest declares `[frontend].entry`. This route is outside `/api`. The lockfile is read on each request, so a file becomes available after install or reload and returns `404` after disable or removal. Responses use `Cache-Control: no-cache`. |
| `/plugins/{id}/assets/{file}` | GET, HEAD | open | Serve a file from an enabled plugin's `assets/`. This uses the same enablement rules but does not require a frontend manifest entry. The media type follows the extension and defaults to `application/octet-stream`. |
| `/api/plugins/jobs/{job_id}/events` | GET | open | Return install-job events after `?cursor=`; `?wait=` long-polls for later events. |
| `/api/plugins/reload` | POST | token | Rediscover nodes, presets, and plugins. |
| `/api/plugins/{id}/enable` | POST | token | Enable an installed plugin. |
| `/api/plugins/{id}/disable` | POST | token | Disable an installed plugin without uninstalling it. |
| `/api/plugins/inspect` | POST | token + loopback | Inspect a source at one commit and return its installation and permission requirements without installing it. |
| `/api/plugins/install` | POST | token + loopback | Install the result identified by an `inspection_id`; returns `202` with a `job_id`. |
| `/api/plugins/jobs/{job_id}/cancel` | POST | token + loopback | Cancel an active install job. |
| `/api/plugins/{id}/update` | POST | token + loopback | Inspect the plugin's recorded repository for an update. Returns `202` `{job_id}`, `200` `{status: "up_to_date", sha}`, or `200` `{status: "needs_consent", inspection, capabilities_added, allowed_modules_added}`. For `needs_consent`, the client sends `POST /install {inspection_id, accept_capabilities, trust_author}` without `force`. |
| `/api/plugins/{id}` | DELETE | token + loopback | Uninstall a plugin and report retained files or dependencies. |

**open** identifies read routes that require no credential. **token** requires the session-token header. **token + loopback** also requires the server to be bound to a loopback address unless `CODEFYUI_ALLOW_REMOTE_PLUGIN_INSTALL=1` is set. See [Plugin Center](#plugin-center) for the affected operations.
