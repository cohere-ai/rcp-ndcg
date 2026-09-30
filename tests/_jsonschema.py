"""A small JSON Schema checker for the exported schemas, so the tests need no validator dependency.

It checks the keywords the exported schemas use for structure: ``$ref`` into ``$defs``, ``anyOf``, ``oneOf``
(exactly one branch), ``allOf``, ``not``, ``type``, ``const``, ``enum``, ``properties``/``required``/
``additionalProperties``/``propertyNames``, ``items`` and ``prefixItems``. Value bounds (``minimum``,
``pattern``, ...) and ``format`` are not checked.
"""

from __future__ import annotations

from typing import Any

_TYPES: dict[str, Any] = {
    "string": str,
    "object": dict,
    "array": list,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
    "null": type(None),
}


def _is(value: Any, name: str) -> bool:
    if name in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, _TYPES[name])


def problems(schema: dict[str, Any], value: Any, *, root: dict[str, Any] | None = None, at: str = "$") -> list[str]:
    """Where ``value`` breaks ``schema``: one line per problem, empty when it conforms."""
    root = schema if root is None else root
    if "$ref" in schema:
        found = problems(root["$defs"][schema["$ref"].rsplit("/", 1)[-1]], value, root=root, at=at)
        if found:
            return found
    if "anyOf" in schema and all(problems(branch, value, root=root, at=at) for branch in schema["anyOf"]):
        return [f"{at}: matches no branch of anyOf"]
    if "oneOf" in schema:
        matched = sum(not problems(branch, value, root=root, at=at) for branch in schema["oneOf"])
        if matched != 1:
            return [f"{at}: matches {matched} branches of oneOf, not exactly one"]
    if "not" in schema and not problems(schema["not"], value, root=root, at=at):
        return [f"{at}: {value!r} matches what `not` excludes"]
    out = [line for branch in schema.get("allOf", ()) for line in problems(branch, value, root=root, at=at)]
    if "const" in schema and value != schema["const"]:
        out.append(f"{at}: {value!r} is not {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        out.append(f"{at}: {value!r} is not one of {schema['enum']}")
    kinds = schema.get("type")
    if kinds is not None and not any(_is(value, kind) for kind in ([kinds] if isinstance(kinds, str) else kinds)):
        return [*out, f"{at}: {type(value).__name__} is not {kinds}"]
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        out += [f"{at}: missing {name!r}" for name in schema.get("required", ()) if name not in value]
        extra = schema.get("additionalProperties", True)
        for key, item in value.items():
            if "propertyNames" in schema:
                out += problems(schema["propertyNames"], key, root=root, at=f"{at}.<key {key}>")
            if key in properties:
                out += problems(properties[key], item, root=root, at=f"{at}.{key}")
            elif extra is False:
                out.append(f"{at}: unexpected {key!r}")
            elif isinstance(extra, dict):
                out += problems(extra, item, root=root, at=f"{at}.{key}")
    if isinstance(value, list):
        prefix = schema.get("prefixItems", [])
        for index, item in enumerate(value):
            if index < len(prefix):
                out += problems(prefix[index], item, root=root, at=f"{at}[{index}]")
            elif isinstance(schema.get("items"), dict):
                out += problems(schema["items"], item, root=root, at=f"{at}[{index}]")
    return out
