"""Validation findings that say what they are about.

``validate_graph`` returns a ``list[str]``, and a lot of code reads it as
text: the run routes put it in their 409 envelopes, Export Python joins it
into one refusal, the CLI prints it, and many tests match substrings of it.
The editor showed each line as it came -- in English, naming a node by its
raw id, or for an edge by its node TYPE alone.

A :class:`ValidationIssue` IS such a line, unchanged, so every one of those
readers keeps working. It also carries a ``code`` naming the check, the
``node_id`` the finding is about and the ``params`` its sentence was built
from, and :func:`issue_payload` hands those to a client, which can then say
the finding in its user's language, name the node by its title and jump to
it. A check that has not been given a code still produces a plain ``str``;
it arrives with ``code`` None.

Codes are attached where each message is built, never recovered from the
text afterwards: a client-side pattern over the English sentences is the
trap ``frontend/src/utils/errorMessages.ts`` documents, rules written for
strings the backend never sent.
"""

from __future__ import annotations

from typing import Any


class ValidationIssue(str):
    """A validation message (the ``str`` itself) plus what it is about.

    No ``__slots__``: a ``str`` subclass cannot have non-empty slots, and the
    instance dict is also what copies and pickles carry the fields in. Both
    rebuild a ``str`` subclass by calling it with the text alone, which is
    why everything after ``message`` defaults.
    """

    code: str | None
    node_id: str | None
    params: dict[str, Any]

    def __new__(
        cls,
        message: str,
        code: str | None = None,
        node_id: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> ValidationIssue:
        issue = super().__new__(cls, message)
        issue.code = code
        issue.node_id = node_id
        issue.params = dict(params or {})
        return issue


def validation_issue(
    code: str,
    message: str,
    /,
    *,
    node_id: str | None = None,
    **params: Any,
) -> ValidationIssue:
    """*message* as a :class:`ValidationIssue` with *code*, *node_id* and *params*.

    ``node_id`` is the node the finding is about: a canvas id, or an id that
    expansion made from one (``<block>/<inner>``, ``<card>__<inner>``), which
    a client maps back to the node it shows. ``params`` hold the values the
    sentence names -- ports, types, other node ids -- taken from the graph,
    so JSON can carry them. (A float JSON cannot write, which Python's JSON
    reader lets a request carry, goes out as null: pydantic writes the
    response that way.) ``code`` and ``message`` are positional-only, so a
    param may use either name.
    """
    return ValidationIssue(message, code, node_id, params)


def issue_payload(errors: list[str]) -> list[dict[str, Any]]:
    """One ``{message, code, node_id, params}`` per line of *errors*, in order.

    Takes the list ``validate_graph`` returned, before anything else touches
    it: a pydantic ``list[str]`` field turns each issue back into a plain
    ``str``. A plain line has no code, no node and no params.
    """
    return [
        {
            "message": str(error),
            "code": getattr(error, "code", None),
            "node_id": getattr(error, "node_id", None),
            "params": dict(getattr(error, "params", None) or {}),
        }
        for error in errors
    ]
