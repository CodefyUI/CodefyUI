---
sidebar_position: 7.7
title: Shared Instances
description: What one CodefyUI instance shares between everyone who can reach it -- ambient credentials, who gets billed, and what is stored per graph.
---

# Shared Instances

CodefyUI is a desktop tool that happens to speak HTTP. Everything below is
correct and unsurprising for a single-user install on your own laptop -- it is
the point of ambient credentials that they are ambient. On a box other people
can reach, the same behaviour means something different, and this page says
what.

Read [Publish](./publish) first if you have not; §6 there covers what a LAN
bind exposes. This page is the credentials half of the same story.

## An instance has ONE identity

There are no user accounts. The session token proves a request came from
something that could read a file on the machine -- not WHO sent it. So every
credential the server holds is held on behalf of the instance, not on behalf
of the person whose browser tab is open.

**On a shared box, whoever configured the credentials pays for everyone, and
nothing records who spent what.**

Five credentials work this way.

### ChatGPT sign-in

`POST /api/llm/codex/login` completes an OAuth flow and writes the access and
refresh tokens to `llm/codex_auth.json` under the user-data directory:
`<install dir>/.codefyui_dev/llm/` for a server started by `cdui start` or
`cdui dev` (default install dir `~/CodefyUI`), `<dir>/llm/` when
`CODEFYUI_USER_DATA_DIR` was exported first, and the platform directory
(`%LOCALAPPDATA%\codefyui\llm\` on Windows, `~/.local/share/codefyui/llm/` on
Linux, `~/Library/Application Support/codefyui/llm/` on macOS) only for a
hand-launched uvicorn -- see
[Graph as a Function](./graph-as-a-function#2-getting-the-token-for-external-scripts). The file is
chmod 0600 where that means anything -- on Windows it does not, and the
protection is the per-account ACL on the folder instead.

The proxy checks only that SOMEONE is signed in, not that it was you:

- Once one person signs in, **every** graph on that instance whose `LLMChat`
  uses the **Codex** provider bills to that person's ChatGPT account, and so
  does any plugin that calls the `openai-codex` provider of `/api/llm/chat`,
  such as [Graph Copilot](/advanced/graph-copilot). The **ChatGPT API**
  provider uses an OpenAI API key instead (next section).
- `POST /api/llm/codex/logout` takes no argument beyond the session token.
  Anyone who can reach the editor can sign you out.

### LLM API keys from the environment

When an `LLMChat` node's key param is empty, the node falls back to the
process environment, in order:

- OpenAI: `CODEFYUI_OPENAI_API_KEY`, then `OPENAI_API_KEY`
- Anthropic: `CODEFYUI_ANTHROPIC_API_KEY`, then `ANTHROPIC_API_KEY`

When the server was started with `--project`, `cdui` loads these from the
project `.env` at startup (see [Project Directories](./project-directories)),
so a key in that file is a key every graph on the instance can spend. The
fallback is silent by design -- a graph with an empty key param does not
announce that it used the instance's. **Assume any graph anyone can run is a
graph that can spend your org's LLM budget.**

The one thing the fallback deliberately does NOT do: the **Ollama** provider
never receives a key, so a graph that points `ollama_base_url` at an attacker's
server cannot carry yours off the box.

### Kaggle

The `KaggleDataset` node uses `KAGGLE_USERNAME` + `KAGGLE_KEY`, or the service
account's `~/.kaggle/kaggle.json`. Downloads are attributed to that Kaggle
account, including competition rules you accepted under it.

### Hugging Face

`HuggingFaceDataset`, `TextCorpusDataset` (with `source` set to `huggingface`)
and Package Center model downloads authenticate with `HF_TOKEN` from the
server's environment, or else with the token file `hf auth login` saved in the
service account's home (`~/.cache/huggingface/token`). Gated datasets and
models are then fetched under that Hugging Face account, including any terms
accepted with it.

### Git

On a server started with `--project`, the **Source Control** tab runs the
server account's own `git` in the project directory. Fetch, pull and push use
whatever credentials that account has (a credential helper, SSH keys), and
every commit is authored with the one git identity configured on the server, so
anyone who reaches the editor pushes to the project's remotes as that account.
See [Source Control](./source-control).

## What is per-graph instead

Params typed as SECRET -- an `LLMChat` node's `openai_api_key`, for instance --
belong to whoever typed them and are handled differently. They are blanked out
of every copy the server writes: saved graphs, exports, published app versions,
presets, generated Python, and the run history.

Three consequences:

- A SECRET param is NOT stored anywhere. Reloading the editor or re-importing
  an exported graph leaves the field blank; an `LLMChat` node then falls back
  to the instance's environment key, if one is set, or fails with its
  missing-key error. A queued run keeps the typed value in server memory only
  until the run ends; if the server stops before the run starts, the run is
  retired as `interrupted` and the value is gone. That is the intended trade,
  not a bug.
- **Anything you type from now on is fine.** The value never reaches the
  database, and deleted database pages are zeroed rather than recycled with
  their contents intact, so run history that ages out does not leave a
  readable copy behind. No rotation needed, no cleanup step.
- **If you ran a graph containing a SECRET param on an OLDER build, treat
  that key as disclosed and rotate it.** Those builds wrote the run's graph
  into the `exec_runs.graph_snapshot` column exactly as submitted, and run
  history is pruned by COUNT (the newest 200), not by age -- so on a quiet
  install the value never aged out. Upgrading sweeps the values out of the
  runs it can still see, and logs how many it removed. What it cannot reach
  is runs that were **already** pruned before you upgraded: their rows are
  gone, and on the older build the freed pages kept their contents, so a copy
  can remain in the database file until a `VACUUM`. Rotating the key is the
  only complete fix for that window. See the CHANGELOG entry for which
  release carries the change.

## If you need per-person attribution

There is no in-product answer today. Run one instance per person, and let each
person supply their own credentials:

- Give each instance its own install directory (the installer's
  `CODEFYUI_DIR`), environment file and port. Two servers never share a user
  data directory or a run database: the second one refuses to start, logs one
  line and exits non-zero. A second instance from the same install
  therefore needs its own `CODEFYUI_USER_DATA_DIR` (session token, ChatGPT
  sign-in, downloaded plugins and their lockfile, download cache, pack control
  files) and `CODEFYUI_DB_PATH` (run history, published apps, API keys), and it
  still shares the saved graphs, models, images, media and uploaded data files,
  and the Python environment that pack and plugin installs add packages to.
  `cdui start` runs one background server per install, and `cdui stop` stops
  every server started from it.
- Run each instance under its own OS account if anyone relies on a credential
  stored in the home directory: `~/.kaggle/kaggle.json`, the Hugging Face token
  file, git's credential helper and SSH keys.

Anything else shares an identity, and the sharing is not visible from inside
the editor.
