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

Three credentials work this way.

### ChatGPT sign-in

`POST /api/llm/codex/login` completes an OAuth flow and writes the access and
refresh tokens to `llm/codex_auth.json` under the user-data directory:
`<install dir>/.codefyui_dev/llm/` for a server started by `cdui start` or
`cdui dev` (default install dir `~/CodefyUI`), `<dir>/llm/` when
`CODEFYUI_USER_DATA_DIR` was exported first, and the platform directory
(`%LOCALAPPDATA%\codefyui\llm\` on Windows, `~/.local/share/codefyui/llm/` on
Linux, `~/Library/Application Support/codefyui/llm/` on macOS) only for a
hand-launched uvicorn -- see
[Project Directories](./project-directories#6-create-an-api-key-invoke-needs-one). The file is
chmod 0600 where that means anything -- on Windows it does not, and the
protection is the per-account ACL on the folder instead.

The proxy checks only that SOMEONE is signed in, not that it was you:

- Once one person signs in, **every** graph on that instance using the ChatGPT
  provider bills to that person's personal ChatGPT account.
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

The one thing the fallback deliberately does NOT do: the `custom` provider
never receives a key, so a graph pointing at an attacker's `base_url` cannot
carry yours off the box.

### Kaggle

The `KaggleDataset` node uses `KAGGLE_USERNAME` + `KAGGLE_KEY`, or the service
account's `~/.kaggle/kaggle.json`. Downloads are attributed to that Kaggle
account, including competition rules you accepted under it.

## What is per-graph instead

Params typed as SECRET -- an `LLMChat` node's `openai_api_key`, for instance --
belong to whoever typed them and are handled differently. They are blanked out
of every copy the server writes: saved graphs, exports, published app versions,
presets, generated Python, and the run history.

Two consequences worth stating plainly:

- A SECRET param is NOT stored anywhere. Reloading the editor, re-importing an
  exported graph, or promoting a queued run on a restarted server all leave the
  field blank, and the node fails with its "requires an api key" error. That is
  the intended trade, not a bug.
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
person supply their own credentials -- separate `.env` files, separate
`CODEFYUI_USER_DATA_DIR` values, separate ports. Anything else shares an
identity, and the sharing is not visible from inside the editor.
