"""The MCP server: the CLI's commands as MCP tools, over stdio.

Each tool mirrors one ``--json`` command and is generated from the same declaration
(:class:`rcp_ndcg.cli.command.CommandSpec`): its input schema is the command's request model, its output schema
the command's result model, and a call runs the same handler through :func:`rcp_ndcg.cli.command.execute`. A
successful call returns the result as MCP structured content (plus the same JSON as text); a failed call is an
MCP tool error (``isError: true``) whose structured content is the ``error`` object of the CLI envelope. Both
server implementations behave the same: the official ``mcp`` SDK when it is installed, and otherwise a minimal
JSON-RPC loop over stdio.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from rcp_ndcg.errors import RcpNdcgError, UsageError
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)

PROTOCOL_VERSION = "2025-06-18"


@dataclass(frozen=True)
class Tool:
    """An MCP tool over a CLI command.

    Attributes:
        name: The tool name.
        command: The CLI command it mirrors (``"data inspect"``).
        fixed: Arguments the tool always passes (``describe`` is ``schema show commands``); they leave the
            input schema.
        output_schema: The JSON Schema name of its structured content when it is not the command's result model.
        description: The tool description when it is not the command's.
        inputs: The request fields the tool takes (default: every field that is not fixed).
        read_only: Whether the tool leaves no artifacts behind (default: the command's declaration).
        destructive: Whether the tool undoes work (``destructiveHint``).
    """

    name: str
    command: str
    fixed: dict[str, Any] = field(default_factory=dict)
    output_schema: str | None = None
    description: str | None = None
    inputs: tuple[str, ...] | None = None
    read_only: bool | None = None
    destructive: bool = False


TOOLS: tuple[Tool, ...] = (
    Tool(
        "describe",
        "schema show",
        fixed={"name": "commands", "full": False},
        output_schema="command-index",
        description="Describe the CLI: every command with its flags (type, default, help) and output schema.",
    ),
    Tool("schema_show", "schema show"),
    Tool("data_inspect", "data inspect"),
    Tool(
        "eval_score",
        "eval score",
        read_only=False,  # it overwrites `out` with the full report when given
        inputs=(
            "rankings",
            "suite",
            "dataset",
            "subset",
            "revision",
            "calibration",
            "protocol",
            "k",
            "metrics",
            "system",
            "bootstrap",
            "seed",
            "per_query",
            "fields",
            "out",
        ),
    ),  # fmt: skip
    Tool("eval_compare", "eval compare"),
    Tool("eval_explain", "eval explain"),
    Tool("calibration_show", "calibration show"),
    Tool("run_list", "run list"),
    Tool("run_show", "run show"),
    Tool("run_status", "run status"),
    Tool(
        "estimate",
        "run start",
        fixed={"estimate": True},
        inputs=("config", "set", "judge", "judge_url", "judge_model", "only"),
        read_only=True,
        output_schema="run-start",
        description="Estimate a run config before running it: calls, tokens and wall time per judging step.",
    ),
    Tool(
        "run_start",
        "run start",
        fixed={"detach": True},
        description="Start a run config and return at once with its run directory; follow it with run_status.",
        inputs=(
            "config",
            "set",
            "runner",
            "judge",
            "judge_url",
            "judge_model",
            "only",
            "label",
            "runs_dir",
            "mirror",
        ),
    ),  # fmt: skip
    Tool("run_cancel", "run cancel", destructive=True),
)


class McpToolInfo(BaseModel):
    """One tool as ``tools/list`` serves it."""

    name: str
    description: str
    inputSchema: dict[str, Any]
    outputSchema: dict[str, Any] | None = None
    annotations: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict, alias="_meta")


class McpManifest(BaseModel):
    """The tools the server offers."""

    tools: list[McpToolInfo]


def _specs() -> dict[str, Any]:
    from rcp_ndcg.cli.introspect import command_specs

    return command_specs()


def _tool_spec(tool: Tool, specs: dict[str, Any]) -> Any:
    spec = specs.get(tool.command)
    if spec is None:
        raise RuntimeError(f"MCP tool {tool.name!r} mirrors {tool.command!r}, which is not a declared command")
    return spec


def _input_schema(tool: Tool, spec: Any) -> dict[str, Any]:
    schema = spec.request.model_json_schema()
    schema.pop("title", None)
    properties = schema.setdefault("properties", {})
    keep = [key for key in (tool.inputs or tuple(properties)) if key not in tool.fixed]
    schema["properties"] = {key: properties[key] for key in keep if key in properties}
    required = [key for key in schema.get("required", []) if key in schema["properties"]]
    if required:
        schema["required"] = required
    else:
        schema.pop("required", None)
    schema["type"] = "object"
    return schema


def _output_schema(tool: Tool, spec: Any) -> dict[str, Any] | None:
    from rcp_ndcg import schemas

    if tool.output_schema is not None:
        return schemas.show(tool.output_schema)
    if spec.result is None:
        return None
    return schemas.show(spec.output_schema)


def tool_manifest() -> McpManifest:
    """The tool list in MCP's ``tools/list`` shape, generated from the commands' request and result models."""
    specs = _specs()
    tools = []
    for tool in TOOLS:
        spec = _tool_spec(tool, specs)
        tools.append(
            McpToolInfo(
                name=tool.name,
                description=tool.description or spec.summary,
                inputSchema=_input_schema(tool, spec),
                outputSchema=_output_schema(tool, spec),
                annotations={
                    "readOnlyHint": spec.read_only if tool.read_only is None else tool.read_only,
                    "destructiveHint": tool.destructive,
                },
                _meta={"x-cli-command": tool.command},
            )
        )
    return McpManifest(tools=tools)


def call_tool(name: str, arguments: Any = None) -> dict[str, Any]:
    """Run one tool and return an MCP ``CallToolResult`` as a dict.

    Never raises for a failed call: the failure comes back as ``isError: true`` with the typed ``error`` object
    as structured content, so a client reads the reason instead of a transport error. That includes arguments
    that are not a JSON object (a string, a list, a number): they are refused as a ``USAGE`` tool error, never
    coerced to ``{}``.

    Args:
        name: The tool name.
        arguments: The tool's input as a mapping; ``None`` for no arguments.

    Returns:
        ``{"content": [text], "structuredContent": ..., "isError": bool}``.
    """
    from rcp_ndcg.cli.command import execute

    if arguments is None:
        arguments = {}
    elif not isinstance(arguments, Mapping):
        return _error_result(
            UsageError(
                f"arguments must be a JSON object, got {type(arguments).__name__}",
                hint="pass the tool's arguments as an object: {'run': 'runs/x'}",
            )
        )
    arguments = dict(arguments)
    specs = _specs()
    tool = next((tool for tool in TOOLS if tool.name == name), None)
    if tool is None:
        known = [tool.name for tool in TOOLS]
        return _error_result(
            UsageError(f"unknown tool {name!r}", hint=f"known tools: {', '.join(known)}", details={"known": known})
        )
    spec = _tool_spec(tool, specs)
    unknown = sorted(set(arguments) - set(_input_schema(tool, spec)["properties"]))
    if unknown:
        return _error_result(UsageError(f"{name} takes no argument {unknown[0]!r}", details={"unknown": unknown}))
    outcome = execute(spec, {**arguments, **tool.fixed})
    for warning in outcome.warnings:  # the MCP analogue of the CLI's stderr line; the envelope has no warnings
        logger.warning(f"{warning['code']}: {warning['message']}")
    if outcome.error is not None:
        return _error_result(outcome.error)
    data = outcome.data if isinstance(outcome.data, dict) else {"result": outcome.data}
    return {
        "content": [{"type": "text", "text": json.dumps(data, indent=2, default=str)}],
        "structuredContent": data,
        "isError": False,
    }


def _error_result(error: RcpNdcgError) -> dict[str, Any]:
    payload = error.to_dict()
    return {
        "content": [{"type": "text", "text": json.dumps(payload, indent=2, default=str)}],
        "structuredContent": payload,
        "isError": True,
    }


def _version() -> str:
    from rcp_ndcg import __version__

    return __version__


def serve() -> None:
    """Serve the tools over stdio until stdin closes.

    Uses the official ``mcp`` SDK (version 2) when it is installed and otherwise the built-in JSON-RPC loop, so
    agent access does not depend on an optional package. Both answer ``tools/call`` through :func:`call_tool`.
    """
    try:
        server = sdk_server()
    except (ImportError, TypeError) as exc:
        logger.info(f"serving the built-in stdio JSON-RPC loop (the mcp SDK 2 is not available: {exc})")
        _serve_stdio()
        return
    _serve_with_sdk(server)


def sdk_server() -> Any:
    """The server on the official ``mcp`` SDK (version 2: ``mcp.server.lowlevel.Server`` with ``on_*`` handlers).

    Its handlers answer through :func:`tool_manifest` and :func:`call_tool`, so both server implementations return
    the same results.

    Raises:
        ImportError: The ``mcp`` package is not installed.
        TypeError: The installed ``mcp`` predates the handler-per-constructor API.
    """
    import mcp.types as types
    from mcp.server.lowlevel import Server

    async def list_tools(_context: Any, _params: Any) -> Any:
        manifest = tool_manifest().model_dump(mode="json", by_alias=True, exclude_none=True)
        return types.ListToolsResult(tools=[types.Tool(**tool) for tool in manifest["tools"]])

    async def call(_context: Any, params: Any) -> Any:
        return types.CallToolResult(**call_tool(params.name, params.arguments))

    return Server("rcp-ndcg", version=_version(), on_list_tools=list_tools, on_call_tool=call)


def _serve_with_sdk(server: Any) -> None:
    import anyio
    from mcp.server.stdio import stdio_server

    async def main() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    anyio.run(main)


def _serve_stdio() -> None:
    """Minimal JSON-RPC 2.0 loop, one message per line.

    A malformed request is answered as an error and skipped, never fatal: a client that sends one bad line
    (or a handler that raises) must not end the session."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            _write({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
            continue
        if not isinstance(request, dict):
            _write({"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "invalid request"}})
            continue
        try:
            response = handle(request)
        except Exception:  # noqa: BLE001 -- the loop answers and goes on; a dead server answers nothing
            logger.exception("the request failed inside the server")
            _write({"jsonrpc": "2.0", "id": request.get("id"), "error": {"code": -32603, "message": "internal error"}})
            continue
        if response is not None:
            _write(response)


def handle(request: dict[str, Any]) -> dict[str, Any] | None:
    """Answer one JSON-RPC request of the built-in loop (``None`` for a notification).

    A ``tools/call`` whose ``params`` or ``arguments`` is not a JSON object is answered as invalid params
    (``-32602``), so a malformed request never reaches the tool, let alone ends the session."""
    method = request.get("method")
    request_id = request.get("id")
    if method == "initialize":
        return _result(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "rcp-ndcg", "version": _version()},
            },
        )
    if method is not None and method.startswith("notifications/"):
        return None
    if method == "ping":
        return _result(request_id, {})
    if method == "tools/list":
        manifest = tool_manifest().model_dump(mode="json", by_alias=True, exclude_none=True)
        return _result(request_id, manifest)
    if method == "tools/call":
        params = request.get("params")
        if not isinstance(params, dict):
            return _error(request_id, -32602, "invalid params: tools/call takes an object with a tool name")
        arguments = params.get("arguments")
        if arguments is not None and not isinstance(arguments, dict):
            return _error(request_id, -32602, "invalid params: arguments must be a JSON object")
        return _result(request_id, call_tool(params.get("name", ""), arguments))
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"unknown method {method!r}"}}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    """A JSON-RPC error response (``-32600``, ``-32602``, ``-32603``)."""
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _write(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, default=str) + "\n")
    sys.stdout.flush()


__all__ = [
    "PROTOCOL_VERSION",
    "TOOLS",
    "McpManifest",
    "McpToolInfo",
    "Tool",
    "call_tool",
    "handle",
    "sdk_server",
    "serve",
    "tool_manifest",
]
