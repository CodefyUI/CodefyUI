"""Reading one ``SKILL.md`` against the Agent Skills specification.

https://agentskills.io/specification is the reference. A skill is a folder
holding ``SKILL.md``: YAML frontmatter, then a Markdown body. Two fields are
required (``name``, ``description``); ``license``, ``compatibility``,
``metadata`` and ``allowed-tools`` are optional.

Lenient to load, honest to report. Only three things keep a skill from
loading, because without them there is nothing an agent could be told about
it: no frontmatter, frontmatter that is not a YAML mapping, and no
``description``. Everything else the specification asks for -- a ``name``
that matches the folder, is lower case, is at most 64 characters -- is a
warning, and the folder name stands in for the name. A skill written for
another agent is usually a little off the specification, and refusing it
would help nobody; saying what is off does.

The content is untrusted text. Nothing here interprets the body, and the
YAML is read with ``safe_load``, which builds plain scalars, lists and
mappings and never a Python object.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import yaml

SKILL_FILE = "SKILL.md"

#: The longest ``SKILL.md`` read. The specification recommends a body under
#: 500 lines; a file a thousand times that is not a skill an agent can use,
#: and reading it would only cost memory.
MAX_SKILL_FILE_BYTES = 256 * 1024

NAME_MAX = 64
DESCRIPTION_MAX = 1024
COMPATIBILITY_MAX = 500

#: Lower-case letters and digits in hyphen-separated runs: no leading,
#: trailing or doubled hyphen.
_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass
class Issue:
    """One thing wrong with a skill. ``level`` is ``error`` (not loaded) or
    ``warning`` (loaded anyway)."""

    level: str
    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"level": self.level, "code": self.code, "message": self.message}


@dataclass
class ParsedSkill:
    """What a ``SKILL.md`` says, after validation."""

    name: str
    description: str = ""
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    allowed_tools: list[str] = field(default_factory=list)
    body: str = ""
    issues: list[Issue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)


def name_problem(name: str) -> str | None:
    """Why *name* breaks the specification's naming rule, or None."""
    if not name:
        return "it is empty"
    if len(name) > NAME_MAX:
        return f"it is {len(name)} characters long, over the limit of {NAME_MAX}"
    if not _NAME_RE.fullmatch(name):
        return ("it may hold only lower-case letters, digits and single hyphens, "
                "and may not start or end with a hyphen")
    return None


def split_frontmatter(text: str) -> tuple[str, str] | None:
    """``(frontmatter, body)`` when *text* opens with a ``---`` fenced block.

    The opening fence must be the first line; the closing one is the next
    line that is exactly ``---`` (trailing spaces allowed). ``...``, YAML's
    own document end, closes it too.
    """
    text = text.lstrip("﻿")
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip() != "---":
        return None
    for index in range(1, len(lines)):
        if lines[index].rstrip() in ("---", "..."):
            front = "".join(lines[1:index])
            body = "".join(lines[index + 1:])
            return front, body
    return None


def _error(issues: list[Issue], code: str, message: str) -> None:
    issues.append(Issue("error", code, message))


def _warn(issues: list[Issue], code: str, message: str) -> None:
    issues.append(Issue("warning", code, message))


def _string_field(data: dict, key: str, issues: list[Issue], *,
                  limit: int | None = None) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        _warn(issues, f"{key}_type", f"'{key}' should be a string; it was ignored.")
        return None
    text = str(value).strip()
    if not text:
        return None
    if limit is not None and len(text) > limit:
        _warn(issues, f"{key}_too_long",
              f"'{key}' is {len(text)} characters long, over the limit of {limit}.")
    return text


def _metadata(data: dict, issues: list[Issue]) -> dict[str, str]:
    value = data.get("metadata")
    if value is None:
        return {}
    if not isinstance(value, dict):
        _warn(issues, "metadata_type",
              "'metadata' should be a mapping of strings to strings; it was ignored.")
        return {}
    out: dict[str, str] = {}
    coerced = False
    for key, item in value.items():
        if isinstance(item, (dict, list)) or item is None:
            coerced = True
            continue
        if not isinstance(key, str) or not isinstance(item, str):
            coerced = True
        out[str(key)] = str(item).lower() if isinstance(item, bool) else str(item)
    if coerced:
        _warn(issues, "metadata_values",
              "'metadata' should map strings to strings; other values were "
              "converted to text or left out.")
    return out


def _allowed_tools(data: dict, issues: list[Issue]) -> list[str]:
    value = data.get("allowed-tools")
    if value is None:
        return []
    if isinstance(value, str):
        return value.split()
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        _warn(issues, "allowed_tools_list",
              "'allowed-tools' should be one space-separated string; a list "
              "was read the same way.")
        return [tool for item in value for tool in item.split()]
    _warn(issues, "allowed_tools_type",
          "'allowed-tools' should be a space-separated string; it was ignored.")
    return []


def parse_skill(text: str, folder_name: str) -> ParsedSkill:
    """Read the text of a ``SKILL.md`` that lives in *folder_name*.

    Never raises for bad content: every problem is an :class:`Issue` on
    the result, and ``valid`` says whether the skill can be loaded.
    """
    issues: list[Issue] = []
    skill = ParsedSkill(name=folder_name, issues=issues)

    folder_problem = name_problem(folder_name)
    split = split_frontmatter(text)
    if split is None:
        _error(issues, "no_frontmatter",
               "SKILL.md does not open with a YAML frontmatter block between "
               "'---' lines.")
        return skill
    front, body = split
    skill.body = body.lstrip("\n")
    try:
        data = yaml.safe_load(front) if front.strip() else None
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 2})" if mark is not None else ""
        _error(issues, "bad_yaml", f"The frontmatter is not valid YAML{where}.")
        return skill
    if not isinstance(data, dict):
        _error(issues, "frontmatter_not_mapping",
               "The frontmatter must be a mapping of fields such as 'name' "
               "and 'description'.")
        return skill

    description = data.get("description")
    if isinstance(description, str) and description.strip():
        skill.description = description.strip()
        if len(skill.description) > DESCRIPTION_MAX:
            _warn(issues, "description_too_long",
                  f"'description' is {len(skill.description)} characters long, "
                  f"over the limit of {DESCRIPTION_MAX}.")
    elif description is None:
        _error(issues, "description_missing",
               "'description' is required: it is what tells an agent when to "
               "use the skill.")
    else:
        _error(issues, "description_invalid",
               "'description' must be a non-empty string.")

    declared = data.get("name")
    if declared is None:
        _warn(issues, "name_missing",
              f"'name' is missing; the folder name '{folder_name}' is used.")
    elif not isinstance(declared, str):
        _warn(issues, "name_invalid",
              f"'name' must be a string; the folder name '{folder_name}' is used.")
    else:
        problem = name_problem(declared)
        if problem is not None:
            _warn(issues, "name_invalid",
                  f"'name' '{declared[:80]}' is not a valid skill name: {problem}. "
                  f"The folder name '{folder_name}' is used.")
        elif declared != folder_name:
            _warn(issues, "name_mismatch",
                  f"'name' is '{declared}' but the folder is '{folder_name}'; "
                  f"the folder name is used.")
    if folder_problem is not None:
        _warn(issues, "folder_name_invalid",
              f"The folder name '{folder_name[:80]}' is not a valid skill "
              f"name: {folder_problem}.")

    skill.license = _string_field(data, "license", issues)
    skill.compatibility = _string_field(data, "compatibility", issues,
                                        limit=COMPATIBILITY_MAX)
    skill.metadata = _metadata(data, issues)
    skill.allowed_tools = _allowed_tools(data, issues)
    return skill


def declared_name(text: str) -> Any:
    """The raw ``name`` field of a ``SKILL.md``, or None. For the installer,
    which names the installed folder after it."""
    split = split_frontmatter(text)
    if split is None:
        return None
    try:
        data = yaml.safe_load(split[0])
    except yaml.YAMLError:
        return None
    return data.get("name") if isinstance(data, dict) else None
