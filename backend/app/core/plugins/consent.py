"""Who has agreed to what, decided apart from who is doing the asking.

Two permissions travel with a plugin and neither is a sandbox. A capability
(``network``, ``filesystem``, ...) is a DECLARATION: once granted, the plugin
may use that whole family of modules and nothing intercepts it again.
``allowed_modules`` is narrower and blunter -- it turns the AST gate off for
the modules it names. Both are decisions only a person can make, which is
exactly why the arithmetic behind them has to be somewhere a person is not.

The CLI asks at a ``[y/N]`` prompt; a Plugin Center has to ask in a dialog
with a list and two buttons. What they must agree on is the ANSWER: which
capabilities are covered already, which are still outstanding, and -- the
one an update turns on -- which are new since the version the user actually
consented to. Capability creep across an update is the supply-chain shape a
plugin manager can realistically catch, and it is only visible by comparing
against the prior grant, so :class:`Decision` carries that comparison rather
than leaving each caller to redo it.

``prior=None`` and ``prior=()`` are different questions. ``None`` is "there
is no earlier install"; the empty tuple is "there is one and it was granted
nothing", which is what makes ``grew`` mean something on a first install
(nothing) and on an update (everything new). Nothing here prints, prompts or
raises for a capability -- :func:`decide_capabilities` describes, and the
caller decides what to do about it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .errors import ConsentRequired


@dataclass(frozen=True)
class Decision:
    """What a capability request comes to, given what is already agreed.

    ``granted`` and ``missing`` partition the request, both in the order the
    manifest asked -- a dialog lists them next to the manifest and a list
    that has been re-sorted reads as a different request.

    ``reused_prior`` is true when at least one capability is granted only
    because of an earlier install. It is the difference between "you already
    said yes to this" and "you are saying yes now", and the caller owes the
    user the first sentence when it is true.

    ``grew`` is what this version asks for beyond the last one. Empty when
    there was no last one.
    """

    granted: tuple[str, ...]
    missing: tuple[str, ...]
    reused_prior: bool
    grew: tuple[str, ...]


def decide_capabilities(
    requested: Iterable[str],
    *,
    accepted: Iterable[str] | None = None,
    prior: Iterable[str] | None = None,
) -> Decision:
    """Split *requested* into what is covered and what is still outstanding.

    *accepted* is what the user is agreeing to right now (a checked list in a
    dialog, or everything the manifest asked for behind
    ``--accept-capabilities``); *prior* is what an earlier install of the
    same plugin recorded. Either covers a capability, and a capability
    covered only by *prior* sets ``reused_prior`` -- the caller has to say so
    rather than let a re-grant pass unmentioned.

    Pass ``prior=None`` when there is no earlier install. With a prior, every
    capability not in it is ``grew``, whether or not it ends up granted:
    "this version asks for more than the one you agreed to" is true before
    anybody decides what to do about it.
    """
    asked = tuple(requested)
    accepted_set = set(accepted or ())
    prior_set = set(prior or ())
    granted: list[str] = []
    missing: list[str] = []
    reused_prior = False
    for capability in asked:
        if capability in accepted_set:
            granted.append(capability)
        elif capability in prior_set:
            granted.append(capability)
            reused_prior = True
        else:
            missing.append(capability)
    grew = (
        ()
        if prior is None
        else tuple(c for c in asked if c not in prior_set)
    )
    return Decision(
        granted=tuple(granted),
        missing=tuple(missing),
        reused_prior=reused_prior,
        grew=grew,
    )


def check_trust(
    allowed_modules: Iterable[str],
    *,
    trust_author: bool,
    prior_trusted: Iterable[str] | None = None,
) -> None:
    """Raise unless the user has agreed to this plugin's ``allowed_modules``.

    ``allowed_modules`` is the switch that stops the AST gate refusing the
    named imports, so there is no partial answer: either the author is
    trusted for all of them or the install does not happen. That is why this
    returns nothing and raises :class:`~.errors.ConsentRequired` carrying the
    whole list -- the caller's next move is to show the list and ask, and a
    boolean would leave it to re-derive what to show.

    A plugin that asks for nothing needs no trust, and an update whose module
    list is covered by what the user already trusted is not a second
    decision. Anything more than the prior list is.
    """
    modules = tuple(allowed_modules)
    if not modules or trust_author:
        return
    already = set(prior_trusted or ())
    if all(module in already for module in modules):
        return
    raise ConsentRequired(
        "Plugin requests modules outside the default allowlist: "
        f"{', '.join(modules)}.",
        missing_capabilities=(),
        allowed_modules=modules,
    )
