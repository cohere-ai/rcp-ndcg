"""``rcp mcp``: serve the commands as MCP tools over stdio (:mod:`rcp.mcp`).

Register the server with an MCP client, e.g.::

    {"mcpServers": {"rcp": {"command": "rcp", "args": ["mcp", "serve"]}}}

The tool list and a tool call are also plain Python: :func:`rcp.mcp.tool_manifest` and
:func:`rcp.mcp.call_tool`.
"""

from __future__ import annotations

import click

from rcp.mcp import serve


@click.command(name="serve")
def mcp_serve() -> None:
    """Serve the commands as MCP tools over stdio."""
    serve()


@click.group(name="mcp", help="Serve the commands as MCP tools over stdio.")
def mcp_group() -> None:
    """``rcp mcp``."""


mcp_group.add_command(mcp_serve)


__all__ = ["mcp_group"]
