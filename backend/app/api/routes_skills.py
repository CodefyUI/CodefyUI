"""Agent Skills over REST (``/api/skills``).

Any plugin can use these; the Graph Copilot reads them with
``api.http.fetch``. The logic is in :mod:`app.core.skills`; this module
turns its answers and refusals into HTTP.

The reads are open GETs, like the rest of the API. The writes (install,
enable, delete) go through ``auth_guard`` like every other mutating
``/api/`` route, so they need the session token. Skill content is text the
server hands back unread: nothing here runs a script, follows an
instruction in a skill, or installs a dependency.

Route order matters: ``/file`` and ``/enabled`` are declared before the
``{skill_id:path}`` routes, which would otherwise take them as ids.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..core.skills import install as skill_install
from ..core.skills import registry

router = APIRouter(prefix="/api/skills", tags=["skills"])


class EnabledBody(BaseModel):
    id: str
    enabled: bool


class InstallBody(BaseModel):
    source: str


def _skill_or_404(skill_id: str) -> registry.Skill:
    try:
        return registry.get_skill(skill_id)
    except registry.SkillNotFound:
        raise HTTPException(status_code=404, detail=f"No skill '{skill_id}'.") from None


@router.get("")
def list_skills() -> list[dict]:
    """Every skill in every source, shadowed and invalid ones included."""
    return [skill.summary() for skill in registry.discover()]


@router.get("/file")
def read_skill_file(id: str = Query(..., description="The skill's id"),
                    path: str = Query(..., description="A file path inside the skill")) -> dict:
    """One text file of a skill: ``{path, content, truncated}``.

    400 for a path that is absolute, has ``..`` or resolves outside the
    skill (a symbolic link included), 404 for no such file, 415 for a
    binary one. Past 256 KB the content is cut and ``truncated`` is true.
    """
    skill = _skill_or_404(id)
    try:
        return registry.read_file(skill, path)
    except registry.SkillPathRefused as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No file '{path}' in {id}.") from None
    except registry.SkillFileNotText:
        raise HTTPException(status_code=415,
                            detail=f"'{path}' is not a text file.") from None


@router.put("/enabled")
def set_skill_enabled(body: EnabledBody) -> dict:
    """Turn a skill on or off. Kept in ``<USER_DATA>/skills/state.json``."""
    _skill_or_404(body.id)
    registry.set_enabled(body.id, body.enabled)
    return _skill_or_404(body.id).summary()


@router.post("/install")
def install_skill(body: InstallBody) -> dict:
    """Install a folder, a ``.zip``/``.tar.gz`` or an HTTPS git URL into
    the ``user`` source. 409 when a skill of that name is installed."""
    try:
        return skill_install.install(body.source)
    except skill_install.SkillInstallError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.as_detail()) from None


@router.get("/{skill_id:path}")
def get_skill(skill_id: str) -> dict:
    """One skill with its Markdown ``body`` and its ``files``."""
    return _skill_or_404(skill_id).detail()


@router.delete("/{skill_id:path}")
def delete_skill(skill_id: str) -> dict:
    """Delete an installed skill. Only the ``user`` source can be deleted:
    the others belong to a project, a plugin or the install."""
    skill = _skill_or_404(skill_id)
    if skill.source != "user":
        raise HTTPException(
            status_code=400,
            detail=f"Only installed skills can be deleted; {skill_id} comes "
                   f"from '{skill.source}'.")
    skill_install.remove(skill.folder)
    registry.set_enabled(skill_id, True)  # drop it from the disabled list
    return {"deleted": skill_id}
