"""Agent Skills over REST (``/api/skills``).

Any plugin can use these; the Graph Copilot reads them with
``api.http.fetch``. The logic is in :mod:`app.core.skills`; this module
turns its answers and refusals into HTTP.

The reads are open GETs, like the rest of the API. The writes (install,
enable, delete) go through ``auth_guard`` like every other mutating
``/api/`` route, so they need the session token. Install and delete are
also limited to a loopback-bound server, the rule the plugin routes follow
(``ALLOW_REMOTE_SKILL_INSTALL`` opens them): an install makes the server
clone a URL or read a path on the caller's word, and a skill steers the
local user's agent. Enabling and disabling act on what is already here
and stay token-only, as for plugins. Skill content is text the
server hands back unread: nothing here runs a script, follows an
instruction in a skill, or installs a dependency.

Route order matters: ``/file`` and ``/enabled`` are declared before the
``{skill_id:path}`` routes, which would otherwise take them as ids.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..config import settings
from ..core.auth import bound_to_loopback
from ..core.skills import install as skill_install
from ..core.skills import registry

router = APIRouter(prefix="/api/skills", tags=["skills"])


class EnabledBody(BaseModel):
    id: str
    enabled: bool


class InstallBody(BaseModel):
    source: str


REMOTE_REFUSAL = {
    "code": "remote_install_refused",
    "message": "Installing and deleting skills is only allowed from the computer "
               "that runs the server. Set CODEFYUI_ALLOW_REMOTE_SKILL_INSTALL=1 "
               "to override.",
}


def remote_skill_install_allowed() -> bool:
    return bound_to_loopback() or bool(settings.ALLOW_REMOTE_SKILL_INSTALL)


async def _require_local() -> None:
    if not remote_skill_install_allowed():
        raise HTTPException(status_code=403, detail=REMOTE_REFUSAL)


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


@router.post("/install", dependencies=[Depends(_require_local)])
def install_skill(body: InstallBody) -> dict:
    """Install every skill in a folder, a ``.zip``/``.tar.gz`` or an HTTPS
    git URL (``@ref``, ``#sub/folder``) into the ``user`` source:
    ``{"installed": [...], "skipped": [{name, code, message}]}``. 409 when
    every skill found is already installed, or another install is running."""
    try:
        return skill_install.install(body.source)
    except skill_install.SkillInstallError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.as_detail()) from None


@router.get("/{skill_id:path}")
def get_skill(skill_id: str) -> dict:
    """One skill with its Markdown ``body`` and its ``files``."""
    return _skill_or_404(skill_id).detail()


@router.delete("/{skill_id:path}", dependencies=[Depends(_require_local)])
def delete_skill(skill_id: str) -> dict:
    """Delete an installed skill. Only the ``user`` source can be deleted:
    the others belong to a project, a plugin or the install."""
    skill = _skill_or_404(skill_id)
    if skill.source != "user":
        raise HTTPException(
            status_code=400,
            detail=f"Only installed skills can be deleted; {skill_id} comes "
                   f"from '{skill.source}'.")
    try:
        skill_install.remove(skill.folder)
    except skill_install.SkillInstallError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.as_detail()) from None
    except OSError as exc:
        raise HTTPException(status_code=500, detail={
            "code": "delete_failed", "message": f"{skill_id} could not be deleted: {exc.strerror}"}) from None
    registry.set_enabled(skill_id, True)  # drop it from the disabled list
    return {"deleted": skill_id}
