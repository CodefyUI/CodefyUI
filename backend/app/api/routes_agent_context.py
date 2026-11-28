"""What an agent plugin is told about where it is: ``GET /api/agent/context``.

Not to be confused with ``app.agent``, the remote worker agent: this is
the editor-side context an in-browser agent (the Graph Copilot) reads once
per conversation.

- ``project``: the open project (``dir``, ``name``, ``graphs``, and since
  round 6 ``git``, ``graph_format`` and ``mods: {state, nodes}``), or null
  outside project mode.
- ``instructions``: ``<PROJECT_DIR>/AGENTS.md``, the project's standing
  instructions to any agent, cut at 32 KB; null when there is none.
- ``skills``: the skills the agent may load -- enabled, valid and not
  shadowed -- as summaries. The agent asks ``/api/skills/{id}`` for a body.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from fastapi import APIRouter

from ..config import settings
from ..core.graph_storage import GraphStore
from ..core.project import read_project_manifest
from ..core.skills import registry

router = APIRouter(prefix="/api/agent", tags=["agent"])

INSTRUCTIONS_FILE = "AGENTS.md"
MAX_INSTRUCTIONS_BYTES = 32 * 1024


def _project_name(project_dir: Path) -> str:
    table = read_project_manifest(project_dir).get("project")
    name = table.get("name") if isinstance(table, dict) else None
    return name if isinstance(name, str) and name.strip() else project_dir.name


def _graph_names() -> list[str]:
    # Lenient: when two files claim one name, list the name anyway, which is
    # all an agent needs to know here. Opening that graph reports the conflict.
    return GraphStore.current().stems(strict=False)


def _instructions(project_dir: Path) -> dict | None:
    """``AGENTS.md`` when it is a regular file inside the project, else None.

    Not a symbolic link, and resolving inside the project: this endpoint
    needs no token and its answer goes into an agent's prompt, so a cloned
    project holding ``AGENTS.md -> ~/.ssh/id_rsa`` must read as "no
    instructions", never as that file.
    """
    path = project_dir / INSTRUCTIONS_FILE
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            return None
        if not path.resolve().is_relative_to(project_dir.resolve()):
            return None
        # O_NOFOLLOW closes the gap between the check and the open where the
        # platform has it.
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as handle:
            raw = handle.read(MAX_INSTRUCTIONS_BYTES + 1)
    except (OSError, RuntimeError):
        return None
    truncated = len(raw) > MAX_INSTRUCTIONS_BYTES
    text = raw[:MAX_INSTRUCTIONS_BYTES].decode("utf-8", errors="replace")
    return {"path": INSTRUCTIONS_FILE, "content": text, "truncated": truncated}


def _project_facts(project_dir: Path) -> dict:
    """Round 6: whether the project is a git repository, the form new graphs
    are saved in, and the state of its mods with the node types they
    registered (``project:<NAME>``). Booleans and names, not file content."""
    from ..core import mods
    from ..core.graph_storage import manifest_graph_format

    status = mods.current_status(project_dir)
    return {
        "git": mods._git_repo(project_dir),
        "graph_format": "dir" if manifest_graph_format(project_dir) == "dir" else "file",
        "mods": {"state": status["state"], "nodes": list(status["nodes"])},
    }


@router.get("/context")
def agent_context() -> dict:
    project_dir = settings.PROJECT_DIR
    project = None
    instructions = None
    if project_dir is not None:
        project = {
            "dir": str(project_dir),
            "name": _project_name(project_dir),
            "graphs": _graph_names(),
            **_project_facts(project_dir),
        }
        instructions = _instructions(project_dir)
    skills = [
        {"id": s.id, "name": s.name, "description": s.parsed.description, "source": s.source}
        for s in registry.active_skills()
    ]
    return {"project": project, "instructions": instructions, "skills": skills}
