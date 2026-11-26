"""What an agent plugin is told about where it is: ``GET /api/agent/context``.

Not to be confused with ``app.agent``, the remote worker agent: this is
the editor-side context an in-browser agent (the Graph Copilot) reads once
per conversation.

- ``project``: the open project (``dir``, ``name``, ``graphs``), or null
  outside project mode.
- ``instructions``: ``<PROJECT_DIR>/AGENTS.md``, the project's standing
  instructions to any agent, cut at 32 KB; null when there is none.
- ``skills``: the skills the agent may load -- enabled, valid and not
  shadowed -- as summaries. The agent asks ``/api/skills/{id}`` for a body.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from ..config import settings
from ..core.project import GraphAmbiguityError, collect_graph_files, read_project_manifest
from ..core.skills import registry

router = APIRouter(prefix="/api/agent", tags=["agent"])

INSTRUCTIONS_FILE = "AGENTS.md"
MAX_INSTRUCTIONS_BYTES = 32 * 1024


def _project_name(project_dir: Path) -> str:
    table = read_project_manifest(project_dir).get("project")
    name = table.get("name") if isinstance(table, dict) else None
    return name if isinstance(name, str) and name.strip() else project_dir.name


def _graph_names(graphs_dir: Path) -> list[str]:
    try:
        return [base for base, _path in collect_graph_files(graphs_dir)]
    except GraphAmbiguityError:
        # Two files claim one name; list the names anyway, which is all an
        # agent needs to know here. Opening that graph reports the conflict.
        names = {p.name[:-len(".graph.json")] if p.name.endswith(".graph.json") else p.stem
                 for p in graphs_dir.glob("*.json") if not p.name.endswith(".layout.json")}
        return sorted(names)


def _instructions(project_dir: Path) -> dict | None:
    path = project_dir / INSTRUCTIONS_FILE
    try:
        if not path.is_file():
            return None
        with path.open("rb") as handle:
            raw = handle.read(MAX_INSTRUCTIONS_BYTES + 1)
    except OSError:
        return None
    truncated = len(raw) > MAX_INSTRUCTIONS_BYTES
    text = raw[:MAX_INSTRUCTIONS_BYTES].decode("utf-8", errors="replace")
    return {"path": INSTRUCTIONS_FILE, "content": text, "truncated": truncated}


@router.get("/context")
def agent_context() -> dict:
    project_dir = settings.PROJECT_DIR
    project = None
    instructions = None
    if project_dir is not None:
        project = {
            "dir": str(project_dir),
            "name": _project_name(project_dir),
            "graphs": _graph_names(settings.GRAPHS_DIR),
        }
        instructions = _instructions(project_dir)
    skills = [
        {"id": s.id, "name": s.name, "description": s.parsed.description, "source": s.source}
        for s in registry.active_skills()
    ]
    return {"project": project, "instructions": instructions, "skills": skills}
