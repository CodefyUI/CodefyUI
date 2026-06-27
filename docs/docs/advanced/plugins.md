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

| Pack | Hands-on modules | Edu nodes |
|------|------------------|-----------|
| `foundations` | I1 Data Representation · I2 Classical ML | Edu-ColumnStats, Edu-KNN, Edu-LinearRegression, Edu-LogisticRegression, Edu-TokenEmbedding, Edu-FFN |
| `deep` | I3 Vision · I4 Sequences | Edu-CrossAttention, Edu-ResBlock, Edu-SelfAttention, Edu-MultiHeadAttention, Edu-Patchify |
| `rl` | I5 Reinforcement Learning | Edu-PolicyGradient |

Each Edu node decomposes a single lesson concept into a chain of named steps that the [Teaching Inspector](/usage/teaching-inspector) renders one row at a time — `Edu-ColumnStats` shows the population-std formula as `sum → divide → deviations² → variance → sqrt`; `Edu-PolicyGradient` exposes `softmax → gather → log → baseline → loss`; `Edu-Patchify` makes `unfold → permute → flatten` visible. Switch on **Verbose mode** in the Settings popover to capture them.

## How packs are stored

- **Built-in direction packs** live in `plugins/<id>/` inside the repo and are activated in place (no copies).
- **Third-party packs** are downloaded as a pinned-SHA tarball into `<USER_DATA>/plugins/<id>/` and **AST-validated** before install.
- A lockfile at `<USER_DATA>/plugins/installed.json` records every install, so `cdui start` rediscovers them on the next launch.

Plugin nodes are namespaced to avoid collisions and to self-document graphs — built-in nodes use a bare name like `Conv2d`, while plugin nodes are qualified like `foundations:Edu-KNN`.

## Writing your own plugin

Fork the **[Official Plugin Template](https://github.com/treeleaves30760/CodefyUI-Plugin-Official)** — a working, MIT-licensed plugin with two example nodes, a sample example graph, a test suite, and a fully-commented manifest. Its README walks through every field and the AST security gate.

```bash
# Install the template itself to see the pattern live
cdui plugin install treeleaves30760/CodefyUI-Plugin-Official

# After forking
cdui plugin install your-username/your-fork
```

A pack ships any of: a `nodes/` directory (auto-discovered), a `presets/` directory, an `examples/` directory, and an `assets/` directory (mounted at `/plugins/<id>/assets/<file>`). A `cdui.plugin.toml` manifest declares the id, version, `requires_codefyui`, content directories, and lesson metadata.

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

`link` reads the id from your `cdui.plugin.toml` and records the directory's absolute path in the lockfile as `source_kind = "local"`, so discovery walks your working tree directly. The AST security gate is skipped for linked plugins (it's your own code, and a warning says so); `unlink` drops only the lockfile entry, never your files. After editing Python nodes, `cdui plugin reload` (or a server restart) reloads them; a changed frontend bundle additionally needs a browser refresh.

:::tip Dev data isolation
Running plugin commands through `scripts/dev.py` — or setting `CODEFYUI_USER_DATA_DIR` — keeps a clone's lockfile inside the repo (`.codefyui_dev/`) instead of the machine-wide user-data dir, so multiple clones don't clobber each other.
:::

## REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/plugins` | GET | List installed plugin packs. |
| `/api/plugins/{id}` | GET | Get a plugin's manifest + README. |
| `/api/plugins/reload` | POST | Hot-reload all node and preset sources. |
