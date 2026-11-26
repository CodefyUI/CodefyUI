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
- a `.zip`, `.tar.gz` or `.tgz` on the server's disk, holding the skill folder or its contents;
- an HTTPS git URL, optionally ending in `@<branch, tag or commit>`, such as `https://github.com/me/lr-finder@v1`. The repository's root must be the skill.

The skill is copied into the `user` source under the name in its frontmatter. The install is refused, and leaves nothing behind, when the source has no `SKILL.md` with a valid `name` and `description`, is larger than 20 MB or 500 files, contains a symbolic link, or has an archive member that would land outside the skill folder. A skill with the same name already installed is not overwritten (409); delete it first.

To remove one, `DELETE /api/skills/user/<name>`. Only installed skills can be deleted. `PUT /api/skills/enabled` turns any skill off or on; the choice is kept in `skills/state.json` in the user data folder.

## Project instructions: AGENTS.md

In project mode, an `AGENTS.md` at the root of the project directory holds standing instructions for any agent working in that project, such as "seed every run" or "the evaluation graph is `serve`". `GET /api/agent/context` hands it to agent plugins with the project's name and graph list and the skills in use. Up to 32 KB is read.

## Safety

A skill is text. CodefyUI never runs anything in it: not the files in `scripts/`, not an install step, not a dependency. Cloning a git source runs no hooks. The server only lists, validates and hands back file contents, and it refuses to read outside a skill's folder, whether by `..`, an absolute path or a symbolic link, and refuses binary files.

What an agent does with a skill's instructions is up to the agent and to you: an agent plugin still asks before it changes or runs your graph. Install skills from sources you trust, as you would a plugin, because their text steers what the agent proposes.

## When a skill is not used

`GET /api/skills` lists every skill it found, including the ones not in use, and says why:

- `valid: false` with an `issues` entry of level `error`: no frontmatter, frontmatter that is not valid YAML, or no `description`. The skill is not loaded.
- An `issues` entry of level `warning`: something the specification asks for is off, such as a `name` that does not match the folder. The skill is loaded, under its folder's name.
- `shadowed_by`: a skill of the same name from a higher source is used instead.
- `enabled: false`: it was turned off.
