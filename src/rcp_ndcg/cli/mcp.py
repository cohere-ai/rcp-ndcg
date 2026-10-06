"""``rcp-ndcg mcp``: serve the commands as MCP tools over stdio (:mod:`rcp_ndcg.mcp`).

Register the server with an MCP client, e.g.::

    {"mcpServers": {"rcp-ndcg": {"command": "rcp-ndcg", "args": ["mcp", "serve"]}}}

The tool list and a tool call are also plain Python: :func:`rcp_ndcg.mcp.tool_manifest` and
:func:`rcp_ndcg.mcp.call_tool`.
"""

from __future__ import annotations

import click

from rcp_ndcg.mcp import serve


@click.command(name="serve")
def mcp_serve() -> None:
    """Serve the commands as MCP tools over stdio."""
    serve()


@click.group(name="mcp", help="Serve the commands as MCP tools over stdio.")
def mcp_group() -> None:
    """``rcp-ndcg mcp``."""


mcp_group.add_command(mcp_serve)


__all__ = ["mcp_group"]
