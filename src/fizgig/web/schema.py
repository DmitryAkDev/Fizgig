"""What the web page is built from: every registered family's description and its form (fizgig.families.form).

Nothing here names a family or a setting. A family that registers, or an option, preset or model file added to a
description, reaches the page with no change here or in the page.
"""
import dataclasses
from typing import Any, Dict

from fizgig.families import form
from fizgig.families.registry import training_families


def _jsonable(obj: Any) -> Any:
    """Anything a FamilyDescription holds (nested dataclasses, tuples, dicts) as plain JSON data. Walks
    dataclasses.fields() instead of naming fields, so a field added to a description reaches the page by itself."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def describe_family(f) -> Dict[str, Any]:
    """One family for the page: its description, its form and the Preferences files a run reads."""
    data = _jsonable(f)
    data["presets"] = [{"name": name, "values": _jsonable(vals)} for name, vals in (f.presets or ())]
    data["form"] = form.form(f)
    data["model_keys"] = form.model_keys(f)
    return data


def get_schema() -> Dict[str, Any]:
    """Every training family, in the registry's order, read fresh on each call."""
    fams = [describe_family(f) for f in training_families()]
    return {"families": fams, "order": [f["key"] for f in fams]}
