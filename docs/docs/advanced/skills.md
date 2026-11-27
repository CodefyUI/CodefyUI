---
sidebar_position: 5.5
title: Agent Skills
description: Give an agent plugin such as Graph Copilot standard Agent Skills - where skills come from, how to install one, which one of a name is used, and why nothing in a skill is ever run.
---

# Agent Skills

An **Agent Skill** is a folder of instructions an AI agent reads when a task calls for it. CodefyUI serves skills to agent plugins such as [Graph Copilot](./graph-copilot) through `/api/skills`, so any plugin can use them and a skill written for another agent works here too. Skills follow the open [Agent Skills specification](https://agentskills.io/specification).

## What a skill is

A skill is a folder whose name is the skill's name, holding a `SKILL.md`:

```markdown
---
name: lr-finder
description: Find a learning rate for a new model with a short range test. Use when the user does not know what learning rate to start from.
license: MIT
---

# Learning-rate range test

1. ...
```

- The YAML frontmatter needs `name` (1 to 64 lower-case letters, digits and single hyphens, the same as the folder name) and `description` (up to 1024 characters, saying what the skill does and when to use it). `license`, `compatibility`, `metadata` and `allowed-tools` are optional.
- The Markdown body is the instructions. Keep it under 500 lines and put details in other files.
- Optional folders: `references/` (more documents), `assets/` (templates, data) and `scripts/`.

An agent sees skills in three steps, so a long skill costs nothing until it is needed: the `name` and `description` of every skill in use are always in its instructions; it loads a skill's body when a task matches the description; and it reads a file from `references/` only when the body points it there.

## Where skills come from

CodefyUI looks in five places. When two skills have the same name, the first one in this order is used and the others are marked as shadowed:

| Source | Folder |
| --- | --- |
| `project` | `<project>/skills/<name>/`, when a [project directory](/usage/project-directories) is open |
| `user` | `skills/<name>/` in CodefyUI's user data folder: skills you installed or placed there by hand |
| `plugin:<id>` | `skills/<name>/` inside an enabled plugin |
| `builtin` | `skills/<name>/` in the CodefyUI install |
| `extra` | Each folder listed in `CODEFYUI_SKILLS_EXTRA_DIRS` (separated by `:`, or `;` on Windows). Empty by default. |

A skill's id is `<source>/<name>`, for example `builtin/build-graph`. A skill that is invalid or turned off never shadows another one, so turning off a project's copy of a skill brings the built-in one back.

To let CodefyUI's agent use the skills you keep for other agents, list their folders when you start the server, for example `CODEFYUI_SKILLS_EXTRA_DIRS=~/.claude/skills:~/.agents/skills cdui start`. CodefyUI does not read them unless you do.

### The built-in skills

| Skill | Use |
| --- | --- |
| `build-graph` | Building or editing a graph: node and port types, the `Start` trigger, validation errors and their fixes, auto layout |
| `train-and-evaluate` | Training and measuring a model: the standard `TrainingLoop` graph, seeds, reading the Training tab, checkpoints, language models and PushWorld VLA |
| `run-experiments` | Parameter searches, ablations and multi-seed comparisons: design, repetitions, reading intervals, and what the evidence does not support |

## Installing a skill

Send the source to `POST /api/skills/install` with the session token (see the [API reference](./api-reference#agent-skills-and-context)). The source is one of:

- a folder on the server's disk, such as `/home/me/skills/lr-finder`;
- a `.zip`, `.tar.gz` or `.tgz` on the server's disk;
- an HTTPS git URL, optionally ending in `@<branch, tag or commit>` and `#<folder>`, such as `https://github.com/me/my-skills@v1#skills/lr-finder`.

A source can hold one skill or several. CodefyUI looks for them in this order, and does not follow symbolic links while it looks:

1. a `SKILL.md` at the source's root: that is the skill;
2. a single top-level folder with a `SKILL.md` in it: that folder is the skill;
3. every `skills/<name>/SKILL.md`, `.claude/skills/<name>/SKILL.md` and `.agents/skills/<name>/SKILL.md`: each is a skill. This is how skill repositories and Claude Code plugins are laid out. It is also looked for inside a single top-level folder, which is how a downloaded repository archive is shaped.

`#<folder>` on a git URL installs only what is in that folder of the repository.

Each skill found is copied, checked and installed on its own, under the `name` in its frontmatter, and the answer lists what happened: `{"installed": [...], "skipped": [{"name", "code", "message"}]}`. A skill with the same name already installed is skipped (`exists`) and never overwritten; a skill without a valid `SKILL.md` is skipped too, and neither stops the others. When every skill found was skipped as already installed the answer is 409; when the source holds no skill at all it is 400 `no_skill`. A source with a single skill answers with that skill's own refusal.

Limits: each skill may hold at most 500 files and folders and 20 MB; a whole archive or clone at most 5,000 and 100 MB. A skill may not contain symbolic links, and an archive may not hold a member that would land outside its folder, an absolute path, a link or a device file. A refused install leaves nothing behind. Only one install runs at a time; a second one answers 409 `busy`.

To remove one, `DELETE /api/skills/user/<name>`. Only installed skills can be deleted. A skill you linked into the user folder yourself loses only the link; the folder it points at is not touched. `PUT /api/skills/enabled` turns any skill off or on; the choice is kept in `skills/state.json` in the user data folder.

Installing and deleting are only allowed when the server listens on this computer alone (the default, `127.0.0.1`). A server bound to the network refuses both with 403 `remote_install_refused` unless it was started with `CODEFYUI_ALLOW_REMOTE_SKILL_INSTALL=1`, the same rule plugin installs follow: an install makes the server fetch a URL or read a folder on the caller's word, and the skill it installs steers the local user's agent. Turning skills on and off needs only the session token. Network shares (`\\host\share`, `//host/share`, `file://host/...`) are never accepted as a source.

## Project instructions: AGENTS.md

In project mode, an `AGENTS.md` at the root of the project directory holds standing instructions for any agent working in that project, such as "seed every run" or "the evaluation graph is `serve`". `GET /api/agent/context` hands it to agent plugins with the project's name and graph list and the skills in use. Up to 32 KB is read.

## Safety

A skill is text. CodefyUI never runs anything in it: not the files in `scripts/`, not an install step, not a dependency. Cloning a git source runs no hooks and no git-lfs filters. The server only lists, validates and hands back file contents. It refuses to read outside a skill's folder, whether by `..`, an absolute path or a symbolic link; it does not serve hidden files or folders such as `.git`, nor binary files. A frontmatter is capped at 16 KB and may not use YAML aliases.

Symbolic links are treated differently by source:

- **`project` and `plugin:<id>`** come from someone else's repository. A skill folder there (or the `skills/` folder itself) that is a link leading outside the project or plugin is listed as invalid with the issue `unsafe_link`, and nothing in it is read. Otherwise cloning a project that holds `skills/x -> ~/.ssh` would hand your files to an agent.
- **`user`, `builtin` and `extra`** are folders on your own machine, where linking a skill in (`ln -s ~/.claude/skills/foo`) is the point, so a linked skill folder is followed.
- In every source, `SKILL.md` must resolve inside its skill folder, and a file is served only when it does too.

The project's `AGENTS.md` is read only when it is a regular file inside the project directory; a link reads as no instructions.

What an agent does with a skill's instructions is up to the agent and to you: an agent plugin still asks before it changes or runs your graph. Install skills from sources you trust, as you would a plugin, because their text steers what the agent proposes.

## When a skill is not used

`GET /api/skills` lists every skill it found, including the ones not in use, and says why:

- `valid: false` with an `issues` entry of level `error`: no frontmatter, frontmatter that is not valid YAML (YAML aliases included) or is over 16 KB, or no `description`. The skill is not loaded.
- `valid: false` with the issue `unsafe_link`: a symbolic link leads out of where the skill may be read from (see [Safety](#safety)). Nothing in it is read.
- An `issues` entry of level `warning`: something the specification asks for is off, such as a `name` that does not match the folder. The skill is loaded, under its folder's name.
- `shadowed_by`: a skill of the same name from a higher source is used instead.
- `enabled: false`: it was turned off.
