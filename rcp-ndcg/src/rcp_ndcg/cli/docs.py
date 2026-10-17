"""``rcp-ndcg docs``: the reference documentation, generated from the pinned public surface.

``docs api`` writes one Markdown page per public module (the modules of
:data:`rcp_ndcg.support.api_docs.PUBLIC_MODULES`) from the pinned surface snapshot
(``tests/contract/snapshots/python_api.json``) and the modules' own docstrings: every public name with its kind,
signature and one-line role. The repository's ``docs/reference/api/`` is that output, committed; the contract
suite fails when the committed pages differ from a fresh generation.
"""

from __future__ import annotations

import click
from pydantic import BaseModel, Field

from rcp_ndcg.cli.command import command
from rcp_ndcg.support import api_docs


class DocsApiRequest(BaseModel):
    """The pages' target directory and the surface snapshot they are rendered from."""

    out: str = Field(
        default="docs/reference/api", description="Directory to write one <module>.md page per public module into."
    )
    snapshot: str = Field(
        default=str(api_docs.DEFAULT_SNAPSHOT),
        description="The pinned surface snapshot (the definition of the public surface).",
    )


class DocsApi(BaseModel):
    """What ``docs api`` wrote."""

    out: str
    written: list[str] = Field(description="The page paths written, module-sorted.")


@command("docs api", request=DocsApiRequest, result=DocsApi, read_only=False)
def docs_api(request: DocsApiRequest) -> DocsApi:
    """Write one Markdown page per public module from the pinned surface snapshot (what `docs/reference/api` holds)."""
    written = api_docs.write_pages(request.out, snapshot=request.snapshot)
    return DocsApi(out=request.out, written=[str(path) for path in written])


@click.group(name="docs", help="Generate the reference documentation from the pinned public surface.")
def docs_group() -> None:
    """``rcp-ndcg docs``."""


docs_group.add_command(docs_api)


__all__ = [
    "DocsApi",
    "DocsApiRequest",
    "docs_group",
]
