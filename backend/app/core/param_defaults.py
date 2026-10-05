"""A param the graph leaves out runs as its declared default.

The backend twin of the editor's ``frontend/src/utils/paramDefaults.ts``. The
editor fills in every param a node declares when it loads a graph (#570), but
a graph reaches the backend by other roads too: ``cdui run``,
``POST /api/graph/run`` and published apps, a preset's internal nodes (which
carry only the params their author set), the Map node's inner calls, and an
exported script whose ``params = {...}`` someone trimmed. Each node read a
missing param with its own ``params.get(name, fallback)``, and dozens of those
fallbacks disagreed with the declared ``default`` -- GridWorldEnv ran with no
traps instead of its default ``"1,1"`` -- so the canvas and the CLI could run
one file two different ways.

:func:`graph_engine.invoke_node`, which every node call goes through, and
:func:`graph_engine.prepare_executable_graph`, when a run asks for it, fill
the gaps with this one helper, so the declared default is the only answer.
"""

from __future__ import annotations

import copy
from typing import Any


def fill_missing_params(node_cls: Any, params: Any) -> Any:
    """*params* with every param *node_cls* declares and it leaves out, at the
    declared ``default``.

    Only a missing key is filled: a value the params carry, ``0``, ``""``,
    ``False`` and ``None`` included, is theirs and stays. *params* comes back
    as the same object when nothing is missing -- the common case, since the
    canvas sends every param -- and otherwise as a NEW dict: the argument is
    often the caller's own graph JSON and is never changed. Each default is
    deep-copied, so a list default one node mutates is not the next node's;
    one that cannot be copied is handed over as it is.

    Takes the class; a caller holding an instance passes ``type(instance)``.
    A ``define_params()`` that raises leaves *params* as they are: describing
    a node must not be what breaks its run (``codegen._code_param_names``
    takes the same stance). Params that are not a dict are returned as they
    are, for the node to report.
    """
    if not isinstance(params, dict):
        return params
    try:
        missing = [
            definition for definition in node_cls.define_params()
            if definition.name not in params
        ]
    except Exception:  # noqa: BLE001 - a node that cannot describe itself
        return params  # runs with the params it was given, as before
    if not missing:
        return params
    filled = dict(params)
    for definition in missing:
        try:
            value = copy.deepcopy(definition.default)
        except Exception:  # noqa: BLE001 - an uncopyable default is still
            value = definition.default  # the default; hand it over as it is
        filled[definition.name] = value
    return filled
