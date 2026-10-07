"""Collectors for the public surface: each returns a JSON-serialisable dict of *shape*, never prose.

Help strings and docstrings are excluded, so rewording documentation never touches a snapshot. Rendering is
``json.dumps(obj, indent=2, sort_keys=True) + "\\n"``.

* :func:`collect_python` -- S1: ``__all__`` of every module in :data:`PUBLIC_MODULES`, with kinds, signatures
  and model fields.
* :func:`collect_cli` -- S2: the CLI tree, from the package's collector (every lazy group loaded).
* :func:`collect_mcp` -- S4: the MCP tool manifest.
* :func:`collect_exit_codes` -- S5: exit codes and the exit code of every public error class.
* :func:`collect_packaging` -- S7: console scripts, entry-point groups, heavy imports at import time.
"""

from __future__ import annotations

import enum
import importlib
import inspect
import json
import pkgutil
import re
import subprocess
import sys
import tempfile
import tomllib
import types
import typing
from pathlib import Path, PurePath
from typing import Any

import pydantic
from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[2]
PACKAGES = ("rcp_ndcg_core", "rcp_ndcg")
HEAVY = (
    "torch",
    "transformers",
    "vllm",
    "sglang",
    "datasets",
    "pyarrow",
    "pandas",
    "scipy",
    "sentence_transformers",
    "bm25s",
    "mteb",
    "matplotlib",
)
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")
_SCRATCH = re.compile(r"<tmp>/rcp-ndcg-tests-[\w]+")


def render(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


# ----------------------------------------------------------------------------------------------------------------
# normalisation
# ----------------------------------------------------------------------------------------------------------------


def _path_markers() -> list[tuple[str, str]]:
    import rcp_ndcg_core

    import rcp_ndcg

    markers = [
        (tempfile.gettempdir(), "<tmp>"),
        (str(Path(rcp_ndcg.__file__).resolve().parent), "<rcp_ndcg>"),
        (str(Path(rcp_ndcg_core.__file__).resolve().parent), "<rcp_ndcg_core>"),
        (str(REPO), "<repo>"),
        (str(Path.cwd()), "<cwd>"),
        (str(Path.home()), "<home>"),
    ]
    return sorted(markers, key=lambda m: -len(m[0]))


def _norm_str(text: str) -> str:
    for prefix, marker in _path_markers():
        text = text.replace(prefix, marker)
    text = _SCRATCH.sub("<scratch>", text)
    return _ADDRESS.sub("", text)


def norm_value(value: Any) -> Any:
    """A default or constant as stable JSON: scalars as is, paths and reprs with machine paths masked."""
    if value is inspect.Parameter.empty:
        return "<required>"
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return _norm_str(value)
    if isinstance(value, PurePath):
        return f"Path({_norm_str(str(value))!r})"
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, tuple | list | frozenset | set):
        items = [norm_value(v) for v in value]
        return sorted(items, key=repr) if isinstance(value, frozenset | set) else items
    if isinstance(value, dict):
        return {str(k): norm_value(v) for k, v in value.items()}
    if callable(value):
        return f"<callable {getattr(value, '__qualname__', type(value).__name__)}>"
    return _norm_str(repr(value))


def fmt_type(tp: Any) -> str:
    """An annotation rendered the same way on every supported Python version (never ``repr`` of typing objects)."""
    if isinstance(tp, str):
        return " ".join(tp.split())
    if tp is None or tp is type(None):
        return "None"
    if tp is typing.Any:
        return "Any"
    if isinstance(tp, typing.ForwardRef):
        return tp.__forward_arg__
    if isinstance(tp, typing.TypeVar):
        return tp.__name__
    if isinstance(tp, typing.TypeAliasType):
        return f"{tp.__module__}.{tp.__name__}"
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin is typing.Literal:
        return "Literal[" + ", ".join(repr(a) for a in args) + "]"
    if origin is typing.Annotated:
        return fmt_type(args[0])
    if origin in (typing.Union, types.UnionType):
        return " | ".join(fmt_type(a) for a in args)
    if origin is not None:
        base = fmt_type(origin)
        return f"{base}[{', '.join(fmt_type(a) for a in args)}]" if args else base
    if isinstance(tp, type):
        if tp.__module__ == "builtins":
            return tp.__qualname__
        return f"{tp.__module__}.{tp.__qualname__}"
    return _norm_str(repr(tp))


def _signature(obj: Any) -> list[dict[str, Any]] | str:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return "<no signature>"
    out = []
    for p in sig.parameters.values():
        out.append(
            {
                "name": p.name,
                "kind": p.kind.name.lower(),
                "annotation": None if p.annotation is inspect.Parameter.empty else fmt_type(p.annotation),
                "default": norm_value(p.default),
            }
        )
    returns = sig.return_annotation
    out.append({"returns": None if returns is inspect.Signature.empty else fmt_type(returns)})
    return out


# ----------------------------------------------------------------------------------------------------------------
# S1 Python API
# ----------------------------------------------------------------------------------------------------------------


#: The public Python modules: the facade, the core with its documented modules, and the modules the docs present
#: as API. Only these are pinned; every other module is internal and may change without a CHANGELOG entry.
PUBLIC_MODULES: tuple[str, ...] = (
    "rcp_ndcg_core",
    "rcp_ndcg_core.gain",
    "rcp_ndcg_core.irt",
    "rcp_ndcg_core.metric",
    "rcp_ndcg_core.protocol",
    "rcp_ndcg",
    "rcp_ndcg.calibration",
    "rcp_ndcg.data",
    "rcp_ndcg.data.preprocess",
    "rcp_ndcg.data.revisions",
    "rcp_ndcg.errors",
    "rcp_ndcg.eval",
    "rcp_ndcg.eval.mteb",
    "rcp_ndcg.examples",
    "rcp_ndcg.inference",
    "rcp_ndcg.llm",
    "rcp_ndcg.retrieval",
    "rcp_ndcg.runners",
    "rcp_ndcg.runs",
    "rcp_ndcg.testing",
    "rcp_ndcg.testing.corpus",
    "rcp_ndcg.testing.engines",
)


def all_modules() -> list[str]:
    """Every module of both packages whose dotted name has no ``_``-prefixed part (the hygiene checks walk these)."""
    names = []
    for top in PACKAGES:
        pkg = importlib.import_module(top)
        names.append(top)
        for info in pkgutil.walk_packages(pkg.__path__, top + "."):
            if not any(part.startswith("_") for part in info.name.split(".")[1:]):
                names.append(info.name)
    return sorted(names)


def public_modules() -> list[str]:
    return sorted(PUBLIC_MODULES)


def _is_pydantic(obj: Any) -> bool:
    return isinstance(obj, type) and issubclass(obj, BaseModel)


def _kind(obj: Any) -> str:
    if isinstance(obj, types.ModuleType):
        return "module"
    if isinstance(obj, type):
        if _is_pydantic(obj):
            return "pydantic_model"
        if issubclass(obj, enum.Enum):
            return "enum"
        if issubclass(obj, BaseException):
            return "exception"
        if getattr(obj, "_is_protocol", False):
            return "protocol"
        return "class"
    if typing.get_origin(obj) is typing.Literal:
        return "literal_alias"
    if isinstance(obj, typing.TypeAliasType) or typing.get_origin(obj) is not None:
        return "type_alias"
    if inspect.isroutine(obj):
        return "function"
    return "constant"


def _describe_model(cls: type[BaseModel]) -> dict[str, Any]:
    fields = {}
    for name, field in cls.model_fields.items():
        if field.default_factory is not None:
            default: Any = f"<factory {getattr(field.default_factory, '__qualname__', 'callable')}>"
        else:
            default = "<required>" if field.is_required() else norm_value(field.default)
        fields[name] = {
            "type": fmt_type(field.annotation),
            "required": field.is_required(),
            "default": default,
            "alias": field.alias,
            "frozen": bool(field.frozen),
        }
    config = cls.model_config
    return {"fields": fields, "extra": config.get("extra"), "frozen": bool(config.get("frozen", False))}


def _describe_class(cls: type) -> dict[str, Any]:
    out: dict[str, Any] = {
        "bases": [f"{b.__module__}.{b.__qualname__}" for b in cls.__bases__ if b is not object],
    }
    if _is_pydantic(cls):
        out.update(_describe_model(cls))
    elif issubclass(cls, enum.Enum):
        out["values"] = [norm_value(m.value) for m in cls]
        return out
    else:
        out["init"] = _signature(cls)
    members: dict[str, Any] = {}
    for name, member in vars(cls).items():
        if name.startswith("_") and name != "__call__":
            continue
        if isinstance(member, property):
            members[name] = "property"
        elif isinstance(member, staticmethod | classmethod):
            members[name] = {"kind": type(member).__name__, "signature": _signature(member.__func__)}
        elif inspect.isfunction(member):
            members[name] = {"kind": "method", "signature": _signature(member)}
    if members:
        out["members"] = members
    return out


def _describe(obj: Any) -> dict[str, Any]:
    kind = _kind(obj)
    entry: dict[str, Any] = {"kind": kind}
    if kind == "module":
        entry["module"] = obj.__name__
        return entry
    if kind in ("function", "class", "pydantic_model", "enum", "exception", "protocol"):
        entry["defined_in"] = obj.__module__
    if kind == "function":
        entry["signature"] = _signature(obj)
    elif kind in ("class", "pydantic_model", "enum", "exception", "protocol"):
        entry.update(_describe_class(obj))
    elif kind == "literal_alias":
        entry["values"] = [norm_value(v) for v in typing.get_args(obj)]
    elif kind == "type_alias":
        entry["type"] = fmt_type(obj.__value__ if isinstance(obj, typing.TypeAliasType) else obj)
    else:
        scalar = obj is None or isinstance(obj, bool | int | float | str)
        flat = isinstance(obj, tuple | list) and all(v is None or isinstance(v, bool | int | float | str) for v in obj)
        mapping = isinstance(obj, dict) and all(
            v is None or isinstance(v, bool | int | float | str) for v in obj.values()
        )
        entry["type"] = fmt_type(type(obj))
        if scalar or flat or mapping:
            entry["value"] = norm_value(obj)
        elif isinstance(obj, dict):
            entry["keys"] = sorted(map(str, obj))
    return entry


def collect_python(modules: list[str] | None = None) -> dict[str, Any]:
    """S1 over ``modules`` (default: :data:`PUBLIC_MODULES`, what the snapshot pins)."""
    out: dict[str, Any] = {}
    for name in public_modules() if modules is None else modules:
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.split(".")[0] in {"rcp_ndcg", "rcp_ndcg_core"}:
                raise
            out[name] = {"requires_extra": exc.name.split(".")[0] if exc.name else "?"}
            continue
        except ImportError as exc:  # an optional extra that raises its own ImportError with an install hint
            match = re.search(r"`(\w+)` extra", str(exc))
            out[name] = {"requires_extra": match.group(1) if match else "?"}
            continue
        exported = getattr(module, "__all__", None)
        if exported is None:
            out[name] = {"missing_all": True}
            continue
        out[name] = {"all": {str(n): _describe(getattr(module, n)) for n in sorted(exported)}}
    return out


# ----------------------------------------------------------------------------------------------------------------
# S2 CLI
# ----------------------------------------------------------------------------------------------------------------


def collect_cli() -> dict[str, Any]:
    """The CLI tree from the package's own collector (what ``rcp-ndcg schema show commands`` prints)."""
    from rcp_ndcg.cli.introspect import describe_commands

    tree = describe_commands(include_help=False).model_dump(mode="json")["commands"]
    for command in tree.values():
        command.pop("help", None)
        for param in command["params"].values():
            param["default"] = norm_value(param["default"])
            if param["is_flag"] is None:
                del param["is_flag"]
    return tree


# ----------------------------------------------------------------------------------------------------------------
# S4 MCP, S5 exit codes
# ----------------------------------------------------------------------------------------------------------------


def collect_mcp() -> dict[str, Any]:
    from rcp_ndcg import mcp

    tools = {}
    for tool in mcp.tool_manifest().model_dump(mode="json", by_alias=True)["tools"]:
        tools[tool["name"]] = {
            "inputSchema": tool["inputSchema"],
            "outputSchema": (tool["outputSchema"] or {}).get("x-rcp-ndcg-schema"),
            "annotations": tool["annotations"],
            "mirrors": tool["_meta"]["x-cli-command"],
        }
    return tools


def collect_exit_codes() -> dict[str, Any]:
    from rcp_ndcg import errors

    codes = {member.name: int(member.value) for member in errors.ExitCode}
    classes = {}
    for name in sorted(errors.__all__):
        obj = getattr(errors, name)
        if isinstance(obj, type) and issubclass(obj, BaseException):
            code = getattr(obj, "exit_code", None)
            classes[name] = code.name if isinstance(code, enum.Enum) else None
    return {"exit_codes": codes, "error_classes": classes}


# ----------------------------------------------------------------------------------------------------------------
# S7 packaging
# ----------------------------------------------------------------------------------------------------------------

_WEIGHT_PROBE = "import json, sys, {mod}; print(json.dumps(sorted(m for m in {heavy!r} if m in sys.modules)))"


def _heavy_after(module: str) -> list[str]:
    code = _WEIGHT_PROBE.format(mod=module, heavy=HEAVY)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def collect_packaging() -> dict[str, Any]:
    projects = {}
    for pyproject in (REPO / "pyproject.toml", REPO / "packages/rcp-ndcg-core/pyproject.toml"):
        data = tomllib.loads(pyproject.read_text())["project"]
        projects[data["name"]] = {
            "scripts": data.get("scripts", {}),
            "entry_points": {group: sorted(eps) for group, eps in data.get("entry-points", {}).items()},
        }
    read_groups = set()
    for base in (REPO / "src", REPO / "packages/rcp-ndcg-core/src"):
        for path in base.rglob("*.py"):
            for match in re.finditer(r"entry_points\(\s*group\s*=\s*([A-Z_a-z.\"']+)", path.read_text()):
                token = match.group(1).strip("\"'")
                if "." not in token and token.isupper():  # a constant: resolve it in the same file
                    const = re.search(rf"^{token}\s*(?::[^=]+)?=\s*[\"']([^\"']+)[\"']", path.read_text(), re.M)
                    token = const.group(1) if const else token
                read_groups.add(token)
    return {
        "projects": projects,
        "entry_point_groups_read": sorted(read_groups),
        "heavy_imports": {mod: _heavy_after(mod) for mod in ("rcp_ndcg_core", "rcp_ndcg", "rcp_ndcg.cli.main")},
        "pydantic_version": pydantic.VERSION,
    }


COLLECTORS = {
    "python_api": collect_python,
    "cli": collect_cli,
    "mcp_tools": collect_mcp,
    "exit_codes": collect_exit_codes,
    "packaging": collect_packaging,
}
