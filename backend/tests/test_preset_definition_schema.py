"""The required fields of ``PresetDefinition`` are a ratchet (#541).

A graph's own ``presets[]`` entry that does not parse is unreadable, and an
unreadable entry costs data. ``RunService.scrub_stored_secrets`` re-reads every
finished run at each boot and withholds EVERY value of a card that uses such a
preset -- for good, because the sweep keeps no copy. What counts as unreadable
is decided by the CURRENT model, so a release that makes a field required,
here or in any schema the model nests, turns every stored entry that lacks it
unreadable: the first boot after the upgrade blanks those cards' settings
across all run history.

So the set of required fields is pinned here, and it only changes on purpose.
"""

from __future__ import annotations

import json
import typing
from typing import Any, Iterator

from pydantic import BaseModel

from app.schemas.models import PresetDefinition

#: Per model, the fields a stored ``presets[]`` entry must carry: every field
#: without a default, in ``PresetDefinition`` and each schema it nests.
REQUIRED_FIELDS: dict[str, set[str]] = {
    "PresetDefinition": {"preset_name", "nodes", "edges"},
    "InternalNodeSchema": {"id", "type"},
    "InternalEdgeSchema": {"source", "sourceHandle", "target", "targetHandle"},
    "ExposedPortSchema": {"name", "internal_node", "internal_port"},
    "ExposedParamSchema": {"internal_node", "param_name", "display_name"},
    "ParamDefinitionSchema": {"name", "param_type"},
}


def _nested_models(annotation: Any) -> Iterator[type[BaseModel]]:
    """Every pydantic model an annotation names, through lists and unions."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        yield annotation
        return
    for argument in typing.get_args(annotation):
        yield from _nested_models(argument)


def _required_fields() -> dict[str, set[str]]:
    """``REQUIRED_FIELDS`` as the models stand now."""
    found: dict[str, set[str]] = {}
    pending: list[type[BaseModel]] = [PresetDefinition]
    while pending:
        model = pending.pop()
        if model.__name__ in found:
            continue
        found[model.__name__] = {
            name for name, field in model.model_fields.items()
            if field.is_required()
        }
        for field in model.model_fields.values():
            pending.extend(_nested_models(field.annotation))
    return found


def test_no_field_of_a_stored_preset_definition_became_required():
    current = _required_fields()
    grown = {
        model: sorted(fields - REQUIRED_FIELDS.get(model, set()))
        for model, fields in current.items()
        if fields - REQUIRED_FIELDS.get(model, set())
    }
    assert not grown, (
        f"These fields became required: {grown}. A graph stored with an "
        "embedded preset that lacks one would no longer parse, and the "
        "startup sweep (RunService.scrub_stored_secrets) re-reads every "
        "finished run at each boot and withholds EVERY value of a card that "
        "uses an unreadable embedded preset (#541). The first boot after "
        "this release would blank those cards' settings across all run "
        "history, with no way back. Give the field a default, or migrate "
        "the stored run snapshots before releasing; only then add it to "
        "REQUIRED_FIELDS here."
    )


def test_the_pin_shrinks_with_the_model():
    current = _required_fields()
    shrunk = {
        model: sorted(fields - current.get(model, set()))
        for model, fields in REQUIRED_FIELDS.items()
        if fields - current.get(model, set())
    }
    assert not shrunk, (
        f"These fields are no longer required: {shrunk}. Remove them from "
        "REQUIRED_FIELDS, so that making one required again is caught by "
        "test_no_field_of_a_stored_preset_definition_became_required."
    )


def test_a_graph_owned_entry_reads_like_an_installed_preset_file(tmp_path):
    """One set of defaults, wherever a definition is read.

    The installed registry loads a preset file that names only
    ``preset_name``, ``nodes`` and ``edges``. The same entry in a graph's own
    ``presets[]`` used to be unreadable, so Save and Python export answered
    422 and Run refused a graph whose definition the palette accepted.
    """
    from app.core.graph_engine import build_preset_fallback
    from app.core.node_registry import registry
    from app.core.preset_registry import PresetRegistry

    minimal = {"preset_name": "Minimal",
               "nodes": [{"id": "n", "type": "Print"}], "edges": []}
    (tmp_path / "minimal.json").write_text(json.dumps(minimal),
                                           encoding="utf-8")
    installed = PresetRegistry()
    assert installed.discover(tmp_path, registry) == 1

    assert build_preset_fallback([minimal])["Minimal"] == installed.get("Minimal")
