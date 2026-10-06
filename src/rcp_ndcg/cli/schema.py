"""``rcp-ndcg schema``: list, show and export the JSON Schemas (:mod:`rcp_ndcg.schemas`).

``schema show commands`` prints the CLI itself: every command with its help, flags (type, default, help) and output
schema, the same document the MCP ``describe`` tool returns (``rcp-ndcg.command-index.v1``). ``--full`` prints the
whole click tree instead (``rcp-ndcg.commands.v1``: groups, hidden options, click's parameter details).
"""

from __future__ import annotations

from typing import Any

import click
from pydantic import BaseModel, Field

from rcp_ndcg import schemas
from rcp_ndcg.cli.command import command
from rcp_ndcg.errors import UsageError
from rcp_ndcg.schemas import SchemaKind


class SchemaListRequest(BaseModel):
    """No inputs."""


class SchemaInfo(BaseModel):
    """One exported schema."""

    name: str
    kind: SchemaKind
    id: str = Field(description="rcp-ndcg.<name>.v1, the value payloads carry in their 'schema' field.")
    url: str = Field(description="The $id: the committed copy under schemas/ in the repository.")


class SchemaList(BaseModel):
    """Every exported schema."""

    schemas: list[SchemaInfo]


class SchemaShowRequest(BaseModel):
    name: str = Field(description="A schema name from `schema list` (e.g. run-manifest), or 'commands' for the CLI.")
    full: bool = Field(
        default=False, description="With 'commands': the whole click tree instead of the compact command index."
    )


class SchemaExportRequest(BaseModel):
    out: str = Field(description="Directory to write <name>.v1.json files into (created).")


class SchemaExport(BaseModel):
    """What `schema export` wrote."""

    out: str
    written: list[str]


@command("schema list", request=SchemaListRequest, result=SchemaList)
def schema_list(request: SchemaListRequest) -> SchemaList:
    """List every exported JSON Schema with its kind (config, artifact, cli-output, mcp) and id."""
    return SchemaList(
        schemas=[
            SchemaInfo(name=entry.name, kind=entry.kind, id=entry.schema_id, url=entry.url)
            for entry in schemas.entries()
        ]
    )


def _show_text(result: dict[str, Any]) -> str:
    import json

    return json.dumps(result, indent=2)


@command(
    "schema show",
    request=SchemaShowRequest,
    result=None,
    output_schema="https://json-schema.org/draft/2020-12/schema",
    positional=("name",),
    text=_show_text,
)
def schema_show(request: SchemaShowRequest) -> dict[str, Any]:
    """Print one JSON Schema; `schema show commands` prints the CLI (commands, flags with help, output schemas)."""
    if request.name == "commands":
        from rcp_ndcg.cli.introspect import command_index, describe_commands

        if request.full:
            return {"schema": "rcp-ndcg.commands.v1", **describe_commands(include_help=True).model_dump(mode="json")}
        return {"schema": "rcp-ndcg.command-index.v1", **command_index().model_dump(mode="json")}
    if request.full:
        raise UsageError(
            "--full applies to `schema show commands` only",
            hint="show one schema plainly: schema show NAME (the tree is `schema show commands --full`)",
        )
    return schemas.show(request.name)


@command("schema export", request=SchemaExportRequest, result=SchemaExport, read_only=False)
def schema_export(request: SchemaExportRequest) -> SchemaExport:
    """Write every JSON Schema to a directory as <name>.v1.json (what the repository's schemas/ holds)."""
    written = schemas.export(request.out)
    return SchemaExport(out=request.out, written=[str(path) for path in written])


@click.group(name="schema", help="List, show and export the JSON Schemas of configs, artifacts and outputs.")
def schema_group() -> None:
    """``rcp-ndcg schema``."""


for _command in (schema_list, schema_show, schema_export):
    schema_group.add_command(_command)


__all__ = [
    "SchemaExport",
    "SchemaExportRequest",
    "SchemaInfo",
    "SchemaList",
    "SchemaListRequest",
    "SchemaShowRequest",
    "schema_group",
]
