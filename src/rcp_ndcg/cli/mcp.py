"""``rcp-ndcg mcp``: serve the commands as MCP tools over stdio (:mod:`rcp_ndcg.mcp`).

Register the server with an MCP client, e.g.::

    {"mcpServers": {"rcp-ndcg": {"command": "rcp-ndcg", "args": ["mcp", "serve"]}}}

Without an MCP client, ``rcp-ndcg mcp tools --call <tool> --args '<json>'`` calls one tool from the shell and prints
the MCP result (``structuredContent``, ``isError``) exactly as the server would answer it.
"""

from __future__ import annotations

import json
from typing import Any

import click
from pydantic import BaseModel, Field

from rcp_ndcg.cli.command import command
from rcp_ndcg.errors import UsageError
from rcp_ndcg.mcp import McpManifest, call_tool, serve, tool_manifest


class McpToolsRequest(BaseModel):
    call: str | None = Field(
        default=None, description="Call this tool (with --args) and print its MCP result instead of the manifest."
    )
    args: str = Field(
        default="{}", description='The tool\'s arguments for --call, as a JSON object: \'{"run": "runs/x"}\'.'
    )


@command("mcp tools", request=McpToolsRequest, result=None, output_schema="rcp-ndcg.mcp-manifest.v1")
def mcp_tools(request: McpToolsRequest) -> Any:
    """Print the MCP tool manifest (names, input and output schemas, hints), or call one tool with --call/--args.

    `mcp tools --call eval_score --args '{"rankings": "run.jsonl", "suite": "nanobeir"}'` runs the tool from the
    shell and prints its result exactly as `mcp serve` would answer an MCP client.
    """
    if request.call is None:
        return tool_manifest().model_dump(mode="json", by_alias=True)
    try:
        arguments = json.loads(request.args)
    except json.JSONDecodeError as exc:
        raise UsageError(f"--args is not JSON: {exc}", hint="pass a JSON object of the tool's arguments") from exc
    return call_tool(request.call, arguments)


@click.command(name="serve")
def mcp_serve() -> None:
    """Serve the commands as MCP tools over stdio."""
    serve()


@click.group(name="mcp", help="Serve the commands as MCP tools over stdio.")
def mcp_group() -> None:
    """``rcp-ndcg mcp``."""


mcp_group.add_command(mcp_serve)
mcp_group.add_command(mcp_tools)


__all__ = ["McpManifest", "McpToolsRequest", "mcp_group"]
