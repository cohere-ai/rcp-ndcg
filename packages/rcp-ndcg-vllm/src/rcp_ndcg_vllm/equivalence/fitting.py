"""The bridge to the product's text-budget mechanism: the recipe's tokenizer, its declared shapes, the pairs
file, and the reference subprocess's tokenizer spec.

The harness owns no render, cut or fold code: the recipe's ``client`` block constructs the product's endpoint
model at load, and the product's role clients (:mod:`rcp_ndcg.inference.clients`) make every content decision
-- the prompts, the text budget, the reranker's settle-once query -- through
:func:`rcp_ndcg.data.preprocess.fit` with the budget the config declares.  This module keeps only what the
harness itself needs: loading the recipe's tokenizer, reading the pairs file, naming the declared shapes and
the shapes' ``add_special_tokens`` flags, and resolving the tokenizer spec the reference subprocess loads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rcp_ndcg.data.preprocess import ChunkPolicy, TextBudget
from rcp_ndcg.data.templates import RequestShape, TemplateSpec
from rcp_ndcg.data.tokenizer import load_tokenizer

from ..errors import HarnessError
from ..recipe import Recipe

__all__ = [
    "budget_of",
    "cast_shape",
    "chunk_of",
    "declared_shapes",
    "default_shape",
    "load_pairs",
    "resolved_tokenizer_spec",
    "template_of",
    "tokenizer_of",
]


def tokenizer_of(recipe: Recipe) -> Any:
    """The recipe's loaded :class:`~rcp_ndcg.data.tokenizer.TextTokenizer`.

    The recipe's ``client.tokenizer`` names a Hub repository id or a tokenizer file; a relative path resolves
    against the recipe directory. The harness process holds no weights: the tokenizer is the one CPU-side
    thing it loads, through the product's own loader.
    """
    spec = recipe.client.tokenizer
    if spec is None:
        raise HarnessError(
            f"recipe {recipe.id}: the client config declares no tokenizer; stage 1 cannot fit or audit without "
            "one (a hosted vendor profile has no harness-side stage 1)"
        )
    resolved = resolved_tokenizer_spec(recipe)
    try:
        return load_tokenizer(resolved)
    except Exception as error:
        raise HarnessError(f"recipe {recipe.id}: loading the tokenizer {resolved!r} failed: {error}") from error


def template_of(recipe: Recipe) -> TemplateSpec | None:
    """The recipe's declared :class:`~rcp_ndcg.data.templates.TemplateSpec` (``None`` when it sets none)."""
    return recipe.client.template


def chunk_of(recipe: Recipe) -> ChunkPolicy | None:
    """The recipe's declared :class:`~rcp_ndcg.data.preprocess.ChunkPolicy` (``None`` when it sets none)."""
    return getattr(recipe.client, "chunk", None)


def declared_shapes(recipe: Recipe) -> list[str]:
    """The recipe's declared request shapes: the template's, else the role's default (a one-shape recipe)."""
    template = recipe.client.template
    if template is not None:
        return [str(shape) for shape in template.shapes()]
    return [default_shape(recipe)]


def cast_shape(shape: str) -> RequestShape:
    """A validated shape string as the product's :data:`~rcp_ndcg.data.templates.RequestShape` literal.

    The recipe's product template validated the shapes at load; the harness names declared shapes only, so
    the string is one of the product's literals."""
    return shape  # type: ignore[no-any-return]


def default_shape(recipe: Recipe) -> str:
    """The recipe's default request shape: ``pair`` for a reranker, ``document`` for an embedder."""
    return DEFAULT_SHAPE_OF_SHAPE[recipe.role]


DEFAULT_SHAPE_OF_SHAPE: dict[str, str] = {"embed": "document", "multi_vector": "document", "rerank": "pair"}
"""The shape a role's requests are fitted as, when the pairs file does not name one."""


def resolved_tokenizer_spec(recipe: Recipe) -> str:
    """The recipe's tokenizer spec, with a recipe-relative path resolved against the recipe directory.

    The resolved spec is what the reference subprocess loads and what the product's role clients load (a
    config whose tokenizer names a file resolves it the way :func:`load_tokenizer` does, against the working
    directory), so every reader sees one tokenizer.
    """
    spec = recipe.client.tokenizer
    if spec is None:  # pragma: no cover - the endpoint config requires it for self-hosted roles
        raise HarnessError(f"recipe {recipe.id}: the client config declares no tokenizer")
    directory = recipe._dir
    candidate = Path(spec)
    if not candidate.is_absolute() and directory is not None and (directory / candidate).exists():
        return str(directory / candidate)
    return spec


def budget_of(recipe: Recipe) -> TextBudget:
    """The product's :class:`~rcp_ndcg.data.preprocess.TextBudget` the recipe's client config declares."""
    client = recipe.client
    return TextBudget(
        tokenizer=client.tokenizer,
        max_tokens=client.max_tokens or 1,
        query_max_tokens=getattr(client, "query_max_tokens", None),
        template=client.template,
        on_overflow=client.on_overflow,
        chunk=client.chunk,
        aggregation=client.aggregation,
    )


def load_pairs(path: str | Path) -> list[dict[str, Any]]:
    """The pairs file: JSONL, one object per query, ``{"query": str, "documents": [str, ...]}``.

    Optional per-row fields: ``shape`` (the request shape to fit the row as; the recipe's default when unset)
    and ``instruction`` (the run-level task text, filled where the template declares an instruction span).
    """
    pairs: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise HarnessError(f"{path}:{number} is not a JSON object: {error}") from error
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("query"), str)
                or not isinstance(row.get("documents"), list)
            ):
                raise HarnessError(f'{path}:{number}: expected {{"query": str, "documents": [str, ...]}}')
            if not all(isinstance(document, str) for document in row["documents"]):
                raise HarnessError(f"{path}: documents must be strings")
            pairs.append(row)
    if not pairs:
        raise HarnessError(f"{path} holds no pairs")
    return pairs
