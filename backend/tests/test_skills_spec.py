"""Reading one SKILL.md against the Agent Skills specification.

Lenient to load, honest to report: only a missing frontmatter, YAML that is
not a mapping and a missing description keep a skill from loading; every
other departure from https://agentskills.io/specification is a warning and
the folder name stands in for the name.
"""

from __future__ import annotations

import pytest

from app.core.skills.spec import name_problem, parse_skill, split_frontmatter


def _codes(skill) -> set[str]:
    return {issue.code for issue in skill.issues}


def test_a_minimal_skill_is_valid_with_no_issues():
    skill = parse_skill("---\nname: demo\ndescription: Does a thing. Use when X.\n---\n# Body\n", "demo")
    assert skill.valid
    assert skill.issues == []
    assert skill.name == "demo"
    assert skill.description == "Does a thing. Use when X."
    assert skill.body == "# Body\n"


def test_every_optional_field_is_read():
    text = (
        "---\n"
        "name: pdf-tools\n"
        "description: PDFs.\n"
        "license: Apache-2.0\n"
        "compatibility: Needs network access\n"
        "metadata:\n  author: example-org\n  version: \"1.0\"\n"
        "allowed-tools: Bash(git:*) Read\n"
        "---\nbody"
    )
    skill = parse_skill(text, "pdf-tools")
    assert skill.valid and skill.issues == []
    assert skill.license == "Apache-2.0"
    assert skill.compatibility == "Needs network access"
    assert skill.metadata == {"author": "example-org", "version": "1.0"}
    assert skill.allowed_tools == ["Bash(git:*)", "Read"]


@pytest.mark.parametrize("text, code", [
    ("# no frontmatter\n", "no_frontmatter"),
    ("---\nname: x\n", "no_frontmatter"),  # never closed
    ("---\nname: [unclosed\n---\n", "bad_yaml"),
    ("---\n- a\n- b\n---\n", "frontmatter_not_mapping"),
    ("---\nname: demo\n---\nbody", "description_missing"),
    ("---\nname: demo\ndescription: '   '\n---\n", "description_invalid"),
    ("---\nname: demo\ndescription: [1, 2]\n---\n", "description_invalid"),
])
def test_what_keeps_a_skill_from_loading(text, code):
    skill = parse_skill(text, "demo")
    assert not skill.valid
    assert code in _codes(skill)
    assert all(i.level == "error" for i in skill.issues if i.code == code)


@pytest.mark.parametrize("declared, code", [
    ("other-name", "name_mismatch"),
    ("Demo", "name_invalid"),
    ("-demo", "name_invalid"),
    ("de--mo", "name_invalid"),
    ("x" * 65, "name_invalid"),
])
def test_a_bad_name_is_a_warning_and_the_folder_name_is_used(declared, code):
    skill = parse_skill(f"---\nname: {declared}\ndescription: d\n---\n", "demo")
    assert skill.valid
    assert skill.name == "demo"
    assert code in _codes(skill)
    assert all(i.level == "warning" for i in skill.issues)


def test_a_missing_name_is_a_warning():
    skill = parse_skill("---\ndescription: d\n---\n", "demo")
    assert skill.valid and "name_missing" in _codes(skill)


def test_an_off_spec_folder_name_is_reported():
    skill = parse_skill("---\nname: My Skill\ndescription: d\n---\n", "My Skill")
    assert skill.valid
    assert "folder_name_invalid" in _codes(skill)


def test_over_length_fields_are_warnings():
    text = f"---\nname: demo\ndescription: {'d' * 1025}\ncompatibility: {'c' * 501}\n---\n"
    skill = parse_skill(text, "demo")
    assert skill.valid
    assert {"description_too_long", "compatibility_too_long"} <= _codes(skill)


def test_odd_metadata_and_tools_are_coerced_with_a_warning():
    text = ("---\nname: demo\ndescription: d\nmetadata:\n  n: 3\n  flag: true\n  deep: {a: 1}\n"
            "allowed-tools: [Read, Grep]\n---\n")
    skill = parse_skill(text, "demo")
    assert skill.valid
    assert skill.metadata == {"n": "3", "flag": "true"}
    assert skill.allowed_tools == ["Read", "Grep"]
    assert {"metadata_values", "allowed_tools_list"} <= _codes(skill)


def test_yaml_is_read_safely():
    # A python/object tag is what an unsafe loader would execute.
    text = "---\nname: demo\ndescription: !!python/object/apply:os.system ['echo hi']\n---\n"
    skill = parse_skill(text, "demo")
    assert not skill.valid
    assert "bad_yaml" in _codes(skill)


def test_frontmatter_survives_a_bom_and_crlf():
    assert split_frontmatter("﻿---\r\nname: a\r\n---\r\nbody") == ("name: a\r\n", "body")


@pytest.mark.parametrize("name, ok", [
    ("a", True), ("pdf-processing", True), ("a1-b2", True),
    ("", False), ("A", False), ("a_b", False), ("a-", False), ("-a", False),
    ("a--b", False), ("x" * 64, True), ("x" * 65, False),
])
def test_name_rule(name, ok):
    assert (name_problem(name) is None) is ok
