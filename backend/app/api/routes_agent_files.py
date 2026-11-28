"""Project files and data files for an in-editor agent (round 6, H5).

- ``GET /api/agent/files?path=`` -- a folder listing or a text file.
- ``PUT /api/agent/files`` -- write one text file (compare-and-swap).
- ``GET /api/agent/data-files`` -- the data files a node parameter can name.

Every rule is in ``core.agent_files``. These routes take the session token
themselves, GETs included: ``auth_guard`` lets GETs through, and a file
reader must not be one of those. A read also answers 409 ``project_changed``
for a window on another project, as every write does (``main.
_project_mismatch``).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..config import settings
from ..core import agent_files
from ..core.api_keys import require_session_token

router = APIRouter(prefix="/api/agent", tags=["agent"],
                   dependencies=[Depends(require_session_token)])


def _refuse(error: agent_files.AgentFileError) -> JSONResponse:
    return JSONResponse(status_code=error.status, content={"detail": error.detail()})


def _project_or_refusal(request: Request):
    from ..core.projects import project_id
    from ..main import PROJECT_HEADER

    sent = request.headers.get(PROJECT_HEADER)
    if sent is not None and sent != project_id(settings.PROJECT_DIR):
        return None, _refuse(agent_files.AgentFileError(
            409, "project_changed",
            "This window is on a project that is no longer open; reload the page "
            "to continue in the open project"))
    if settings.PROJECT_DIR is None:
        return None, _refuse(agent_files.AgentFileError(
            409, "no_project", "No project is open; project files need one."))
    return settings.PROJECT_DIR, None


@router.get("/files")
def read_files(request: Request, path: str = "") -> Any:
    project, refusal = _project_or_refusal(request)
    if refusal is not None:
        return refusal
    try:
        return agent_files.read_path(project, path)
    except agent_files.AgentFileError as error:
        return _refuse(error)


class WriteFileRequest(BaseModel):
    path: str
    content: str
    expected_sha256: str | None


@router.put("/files")
def write_files(request: Request, body: WriteFileRequest) -> Any:
    project, refusal = _project_or_refusal(request)
    if refusal is not None:
        return refusal
    try:
        return agent_files.write_file(project, body.path, body.content, body.expected_sha256)
    except agent_files.AgentFileError as error:
        return _refuse(error)


@router.get("/data-files")
def list_data_files(request: Request) -> Any:
    from ..core.projects import project_id
    from ..main import PROJECT_HEADER

    sent = request.headers.get(PROJECT_HEADER)
    if sent is not None and sent != project_id(settings.PROJECT_DIR):
        return _refuse(agent_files.AgentFileError(
            409, "project_changed",
            "This window is on a project that is no longer open; reload the page "
            "to continue in the open project"))
    return agent_files.data_files(settings.PROJECT_DIR)
