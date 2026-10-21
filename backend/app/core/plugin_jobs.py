"""Plugin-scoped durable jobs (#343 increment 6, core side).

The Graph Copilot plugin's "Durable backend agent contract" asks core for
three things before a plugin can hand a study to the server and walk away:
a job identity scoped to the plugin, fixed core-owned routes
(``/api/plugins/{plugin_id}/jobs...``), and a registry of job handlers. This
module is the registry and the plugin check; ``api.routes_plugin_jobs`` is
the routes.

What a plugin can submit
------------------------
A job *kind* is registered here by core. This release registers one,
``study`` (``spec_version`` 1): its spec is exactly a ``POST /api/studies``
body, and the job IS a study, driven by ``core.study_coordinator`` with the
plugin recorded as its ``owner``. That is everything Graph Copilot's
``run_graph_experiments`` / ``optimize_graph_parameters`` execute once the
plan is approved: N variants x R repetitions on one exact objective, under a
budget, ranked by the study rule. The planning (the LLM conversation and the
user's approval) stays in the plugin's page; nothing here runs a model.

Plugin-supplied Python handlers (the contract's ``[backend]`` manifest entry
with an ``AgentJobHandler``) are not loaded: a deterministic study needs no
plugin code on the server, and running a plugin's own long-lived coroutine
in the control plane needs a trust prompt and a sandbox this release does
not have. :func:`register_job_kind` is the seam a later kind attaches to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import plugin_loader

#: The version of the jobs API itself (routes and payload shape). Additive
#: changes keep it; a breaking one bumps it.
JOBS_API_VERSION = 1

#: The one built-in kind: a durable study.
KIND_STUDY = "study"


@dataclass(frozen=True)
class JobKind:
    """One kind of job a plugin may submit."""

    kind: str
    spec_version: int
    description: str

    def as_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "spec_version": self.spec_version,
                "description": self.description}


_KINDS: dict[str, JobKind] = {}


def register_job_kind(kind: JobKind) -> None:
    """Register a job kind. A second registration of one name is refused,
    so two subsystems cannot silently answer for the same kind."""
    if kind.kind in _KINDS:
        raise ValueError(f"job kind {kind.kind!r} is already registered")
    _KINDS[kind.kind] = kind


def get_job_kind(name: str) -> JobKind | None:
    return _KINDS.get(name)


def job_kinds() -> list[JobKind]:
    return [_KINDS[name] for name in sorted(_KINDS)]


register_job_kind(JobKind(
    kind=KIND_STUDY, spec_version=1,
    description="A durable study: variants x repetitions of queued child "
                "runs on one exact objective, with server-enforced budgets. "
                "The spec is a POST /api/studies body."))


class PluginNotFound(LookupError):
    """No installed plugin has this id. -> 404."""


class PluginDisabled(RuntimeError):
    """The plugin is installed but disabled. -> 409 for a submit."""


def require_enabled_plugin(plugin_id: str) -> None:
    """Raise unless *plugin_id* is an installed, enabled plugin.

    Checked on every submit and resume: a job is work a plugin starts on
    the user's machine, so only a plugin the user has enabled may start it.
    Reads, cancels and the event log stay available for a disabled plugin,
    so its studies can still be followed and stopped.
    """
    lockfile = plugin_loader.load_lockfile()
    entry = lockfile.get("plugins", {}).get(plugin_id)
    if entry is None:
        raise PluginNotFound(plugin_id)
    if not plugin_loader.is_enabled(entry):
        raise PluginDisabled(plugin_id)


__all__ = [
    "JOBS_API_VERSION",
    "KIND_STUDY",
    "JobKind",
    "PluginDisabled",
    "PluginNotFound",
    "get_job_kind",
    "job_kinds",
    "register_job_kind",
    "require_enabled_plugin",
]
