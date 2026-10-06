"""Reading a YAML config: one file, optionally ``extends:`` another, plus ``key=value`` overrides.

The one config system of the package. A config is a YAML mapping; a top-level
``extends: <path>`` names a base config (relative to the extending file) whose
mapping is deep-merged under it, recursively; each override ``dotted.key=value``
then replaces one leaf, its value parsed as YAML (``k=10`` is an int,
``k=null`` is ``None``, ``k=[1, 2]`` a list). The result is a plain dict for a
pydantic model to validate, so a typo in a key fails there, not silently here: :func:`config_error` turns
that failure into a :class:`~rcp_ndcg.errors.ConfigError` whose ``details.errors`` name, per problem, the field
path, the given value, the expected type, the closest known key for an unknown one, and whether the value came
from the file or from a ``--set`` override.
"""

from __future__ import annotations

import copy
import difflib
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from rcp_ndcg.errors import ConfigError

EXTENDS_KEY = "extends"


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """``base`` with ``update`` merged in: mappings merge key by key, anything else is replaced.

    The result is a new mapping that shares no mutable value with ``base`` or ``update``.
    """
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def apply_overrides(config: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``dotted.key=value`` overrides; each value is parsed as YAML.

    Raises:
        ConfigError: an override without ``=``, or a path through a non-mapping value.
    """
    result: dict[str, Any] = deep_merge({}, config)
    for override in overrides:
        key, sep, raw = override.partition("=")
        if not sep or not key.strip():
            raise ConfigError(
                f"override {override!r} is not key=value",
                hint="write e.g. judge.concurrency=8",
                cli_hint="write e.g. --set judge.concurrency=8",
            )
        parts = key.strip().split(".")
        node: dict[str, Any] = result
        for depth, part in enumerate(parts[:-1]):
            child = node.get(part)
            if child is None:
                child = node[part] = {}
            elif not isinstance(child, dict):
                raise ConfigError(f"override {override!r}: {'.'.join(parts[: depth + 1])} is not a mapping")
            node = child
        node[parts[-1]] = yaml.safe_load(raw) if raw.strip() else ""
    return result


def load_config(
    path: str | Path,
    *,
    overrides: Iterable[str] = (),
    relative: Callable[[dict[str, Any], Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Read a YAML config, resolving ``extends:`` chains, then apply ``overrides``.

    Args:
        path: The config file.
        overrides: ``dotted.key=value`` overrides, applied last.
        relative: Applied to each file's own mapping with that file's directory, before it is merged: a config
            model's relative paths then resolve against the file that declares them, not the one that extends it.

    Raises:
        ConfigError: the file is not a mapping, or ``extends`` forms a cycle.
        FileNotFoundError: the file or a base it extends does not exist.
    """
    return apply_overrides(_resolve(Path(path), (), relative), overrides)


def _resolve(
    path: Path, chain: tuple[Path, ...], relative: Callable[[dict[str, Any], Path], dict[str, Any]] | None
) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in chain:
        raise ConfigError(f"config {path} extends itself through {' -> '.join(str(p) for p in chain)}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"config {path} must be a YAML mapping, got {type(data).__name__}")
    base_ref = data.pop(EXTENDS_KEY, None)
    if relative is not None:
        data = relative(data, path.parent)
    if base_ref is None:
        return data
    if not isinstance(base_ref, str):
        raise ConfigError(f"config {path}: `extends` must be one path, got {base_ref!r}")
    base = _resolve((path.parent / base_ref), (*chain, resolved), relative)
    return deep_merge(base, data)


def validation_problems(
    exc: Any,
    *,
    model: Any = None,
    source: str = "config",
    overrides: Sequence[str] = (),
    prefix: str = "",
) -> list[dict[str, Any]]:
    """One entry per problem of a pydantic ``ValidationError`` — the one shape of ``details.errors``.

    Each entry carries ``field`` (the dotted path), ``problem``, ``input`` (the given value; ``null`` for a
    missing field), ``expected`` (the type or the allowed values, when known), ``did_you_mean`` (the closest
    known key, for an unknown one) and ``source`` (``"--set"`` when an override set the field, else *source*).
    Config validation (:func:`config_error`) and the CLI's argument validation raise with this shape.
    """
    schema = _json_schema(model)
    set_keys = [override.partition("=")[0].strip() for override in overrides]
    problems = []
    for error in exc.errors(include_url=False):
        # pydantic names the member of a union it tried (``slurm``, ``function-after[..., JudgeConfig]``) in the
        # location; the user wrote no such key, so the field path leaves it out.
        loc, node = _walk(schema, [str(part) for part in error["loc"]])
        field = ".".join([prefix, *loc] if prefix else loc) or "<config>"
        problem: dict[str, Any] = {
            "field": field,
            "problem": str(error["msg"]).removeprefix("Value error, "),
            "input": None if error["type"] == "missing" else _plain(error.get("input")),
        }
        expected = _expected(error, node)
        if expected is not None:
            problem["expected"] = expected
        if error["type"] == "extra_forbidden":
            problem["problem"] = "unknown key"
            known = _known_keys(schema, [str(part) for part in error["loc"][:-1]])
            close = difflib.get_close_matches(loc[-1], known, n=1) if loc else []
            if close:
                problem["did_you_mean"] = ".".join([*field.split(".")[:-1], close[0]])
        by_set = any(field == key or field.startswith(f"{key}.") or key.startswith(f"{field}.") for key in set_keys)
        problem["source"] = "--set" if by_set else source
        problems.append(problem)
    return problems


def config_error(
    exc: Any,
    *,
    model: Any = None,
    source: str = "config",
    overrides: Sequence[str] = (),
    prefix: str = "",
    hint: str | None = None,
) -> ConfigError:
    """A pydantic ``ValidationError`` of a config as a :class:`~rcp_ndcg.errors.ConfigError` a caller can act on.

    ``details.errors`` holds one entry per problem, in the shape of :func:`validation_problems`. The message
    names the first problem and where it came from.

    Args:
        exc: The ``pydantic.ValidationError``.
        model: The model class (or ``TypeAdapter``) validated, for the expected types and the known keys.
        source: What was validated (a file path, ``"run config"``), blamed for problems no override explains.
        overrides: The ``dotted.key=value`` overrides applied before validating.
        prefix: The dotted path of the validated value inside the overrides' namespace (``"judge"`` when
            ``--set judge.x=...`` is validated as a judge config).
        hint: The hint, instead of the default one (fix the field in the file, or the ``--set`` value).
    """
    problems = validation_problems(exc, model=model, source=source, overrides=overrides, prefix=prefix)
    first = problems[0]
    where = f"--set {first['field']}" if first["source"] == "--set" else f"{source}: {first['field']}"
    guess = f"; did you mean {first['did_you_mean']!r}?" if "did_you_mean" in first else ""
    more = f" (and {len(problems) - 1} more)" if len(problems) > 1 else ""
    cli_hint = None
    if hint is not None:
        pass
    elif first["source"] == "--set":
        hint = "fix the override; values are YAML literals (k=10, k=null, 'k=[1, 2]')"
        cli_hint = "fix the --set value; values are YAML literals (--set k=10, --set k=null, --set 'k=[1, 2]')"
    elif first["field"] == "<config>":
        hint = f"fix {source}"
    else:
        hint = f"fix {first['field']} in {source}, or override it (overrides=['{first['field']}=...'])"
        cli_hint = f"fix {first['field']} in {source}, or override it with --set {first['field']}=..."
    return ConfigError(
        f"{where}: {first['problem']}{guess}{more}", hint=hint, cli_hint=cli_hint, details={"errors": problems}
    )


def _json_schema(model: Any) -> dict[str, Any]:
    if model is None:
        return {}
    try:
        if hasattr(model, "model_json_schema"):
            return model.model_json_schema()
        return model.json_schema()
    except Exception:  # noqa: BLE001 -- a schema that cannot be built only costs the hints
        return {}


def _walk(schema: dict[str, Any], loc: Sequence[str]) -> tuple[list[str], dict[str, Any] | None]:
    """A validation location as the user wrote it, and the sub-schema there.

    A location part that names no key of a union's members is the member pydantic tried (a discriminator tag, a
    class or type name): it is dropped from the path, and the walk continues in that member.
    """
    definitions = schema.get("$defs", {})

    def resolve(node: dict[str, Any]) -> dict[str, Any]:
        while "$ref" in node:
            node = definitions.get(node["$ref"].rsplit("/", 1)[-1], {})
        return node

    node: dict[str, Any] | None = resolve(schema) if schema else None
    path: list[str] = []
    for part in loc:
        if node is not None:
            node = resolve(node)
            members = [resolve(member) for member in node.get("anyOf", []) + node.get("oneOf", [])]
            members = [member for member in members if member.get("type") != "null"]
            if len(members) == 1:  # an optional value is no union: pydantic names no member
                node = members[0]
                members = [resolve(member) for member in node.get("anyOf", []) + node.get("oneOf", [])]
            if len(members) > 1 and not _has_key(node, part) and not any(_has_key(member, part) for member in members):
                node = _member(node, members, part, resolve)
                continue
        path.append(part)
        node = _step(node, part, resolve) if node is not None else None
    return path, resolve(node) if node is not None else None


def _has_key(node: dict[str, Any], part: str) -> bool:
    """Whether ``part`` can be a key (or an index) of the value ``node`` describes."""
    extra = node.get("additionalProperties")
    return part in node.get("properties", {}) or (part.isdigit() and "items" in node) or bool(extra)


#: Python type names pydantic puts in a location, and the JSON Schema type of each.
_JSON_TYPES = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "dict": "object",
    "list": "array",
}


def _member(node: dict[str, Any], members: list[dict[str, Any]], tag: str, resolve: Any) -> dict[str, Any] | None:
    """The member of a union a location's ``tag`` names (``None`` when no member is recognised)."""
    for union in (node, *members):  # a discriminated union, or one nested in an optional one
        mapping = union.get("discriminator", {}).get("mapping", {})
        if tag in mapping:
            return resolve({"$ref": mapping[tag]})
    for member in members:
        name = member.get("properties", {}).get("name", {})
        if name.get("const") == tag or tag in name.get("enum", ()):
            return member
    for member in members:
        title = str(member.get("title", ""))
        if title and (title == tag or title in tag or title.lower().startswith(tag.lower())):
            return member
    json_type = _JSON_TYPES.get(tag)
    return next((member for member in members if json_type is not None and member.get("type") == json_type), None)


def _known_keys(schema: dict[str, Any], loc: Sequence[str]) -> list[str]:
    """The keys a mapping at ``loc`` may hold (every member of a union counted)."""
    node = _walk(schema, loc)[1] or {}
    keys = set(node.get("properties", {}))
    for member in node.get("anyOf", []) + node.get("oneOf", []):
        while "$ref" in member:
            member = schema.get("$defs", {}).get(member["$ref"].rsplit("/", 1)[-1], {})
        keys |= set(member.get("properties", {}))
    return sorted(keys)


def _step(node: dict[str, Any], part: str, resolve: Any) -> dict[str, Any] | None:
    node = resolve(node)
    if part in node.get("properties", {}):
        return node["properties"][part]
    if part.isdigit() and "items" in node:
        return node["items"]
    if isinstance(node.get("additionalProperties"), dict):
        return node["additionalProperties"]
    for member in node.get("anyOf", []) + node.get("oneOf", []):
        found = _step(member, part, resolve)
        if found is not None:
            return found
    return None


def _expected(error: Mapping[str, Any], node: dict[str, Any] | None) -> Any:
    context = error.get("ctx") or {}
    if "expected" in context:
        return str(context["expected"])
    if node is None or error["type"] == "extra_forbidden":
        return None
    if "enum" in node:
        return node["enum"]
    if "const" in node:
        return [node["const"]]
    types = [member.get("type") for member in node.get("anyOf", []) if member.get("type")] or [node.get("type")]
    names = [name for name in types if name]
    return " | ".join(names) if names else None


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return repr(value)


__all__ = ["EXTENDS_KEY", "apply_overrides", "config_error", "deep_merge", "load_config", "validation_problems"]
