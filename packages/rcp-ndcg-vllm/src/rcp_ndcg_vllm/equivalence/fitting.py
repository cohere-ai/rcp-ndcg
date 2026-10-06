"""The bridge to the product's text-budget mechanism: everything stage 1 fits goes through
:func:`rcp_ndcg.data.preprocess.fit` with the recipe's product endpoint config.

The harness owns no assembly and no budget code: the recipe's ``client`` block constructs the product's
endpoint model at load, this module turns that endpoint into the product's
:class:`~rcp_ndcg.data.preprocess.TextBudget` and :class:`~rcp_ndcg.data.tokenizer.TextTokenizer`, and every
render, cut and token count below comes from :func:`fit`. The harness keeps only the audit (the anchor
assertions over ``fit``'s output and its census) and the comparisons (against the reference's subprocess
render, the served template file, and the engine's ``/tokenize``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rcp_ndcg.data.preprocess import ChunkPolicy, TextBudget, TextTruncationCensus, fit
from rcp_ndcg.data.templates import TemplateSpec
from rcp_ndcg.data.tokenizer import load_tokenizer

from ..errors import HarnessError
from ..recipe import Recipe

__all__ = [
    "budget_of",
    "chunk_of",
    "declared_shapes",
    "default_shape",
    "fit_rows",
    "load_pairs",
    "template_of",
    "tokenizer_of",
]

DEFAULT_SHAPE_OF_ROLE: dict[str, str] = {"embed": "document", "multi_vector": "document", "rerank": "pair"}
"""The shape a role's requests are fitted as, when the pairs file does not name one."""


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
    resolved = _resolve_spec(recipe, spec)
    try:
        return load_tokenizer(resolved)
    except Exception as error:
        raise HarnessError(f"recipe {recipe.id}: loading the tokenizer {resolved!r} failed: {error}") from error


def _resolve_spec(recipe: Recipe, spec: str) -> str:
    """A recipe-relative tokenizer path into an absolute one; repo ids and absolute paths pass through."""
    candidate = Path(spec)
    directory = recipe._dir
    if not candidate.is_absolute() and directory is not None:
        resolved = directory / candidate
        if resolved.exists():
            return str(resolved)
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
    The over-length samples stage 1 derives carry their own ``shape``.
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
                raise HarnessError(f"{path}:{number}: documents must be strings")
            pairs.append(row)
    if not pairs:
        raise HarnessError(f"{path} holds no pairs")
    return pairs


def fit_rows(
    recipe: Recipe,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    census: TextTruncationCensus | None = None,
) -> dict[str, Any]:
    """Fit every row with the product's :func:`fit`, grouped per declared shape.

    Inputs: the recipe (whose ``client`` block carries the product's template and budget), the rows (each with
    ``query``, ``documents`` and an optional ``shape``/``instruction``) and the loaded tokenizer.  Output: a
    mapping with, per shape, the fitted inputs in row order — the rendered strings (``fit``'s
    :attr:`FitResult.texts`), the cut contents, and the census rows the cuts recorded. This is the one place
    the harness turns pairs into prompts: everything downstream reads ``fit``'s output.
    """
    budget = budget_of(recipe)
    # fit checks the budget's declared tokenizer against the loaded one by their (normalized) specs; pass the
    # recipe's spec through unchanged so the product's own consistency check sees the same string.
    budget = budget.model_copy(update={"tokenizer": tokenizer.name})
    default_shape = DEFAULT_SHAPE_OF_SHAPE[recipe.role]
    by_shape: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        shape = row.get("shape") or default_shape
        by_shape.setdefault(str(shape), []).append((index, row))
    results: dict[str, Any] = {"per_shape": {}}
    for shape, items in by_shape.items():
        inputs: list[Any] = []
        for _, row in items:
            query = fold_query(recipe, row["query"], row.get("instruction"))
            if shape == "pair":
                inputs.append((query, row["documents"][0]))
            else:
                inputs.append(query if shape == "query" else row["documents"][0])
        try:
            result = fit(
                inputs,
                shape,  # type: ignore[arg-type]
                budget,
                tokenizer,
                ids=[f"{index}" for index, _ in items],
                instruction=items[0][1].get("instruction"),
                census=census,
            )
        except Exception as error:
            raise HarnessError(f"recipe {recipe.id}: fitting the {shape!r} shape failed: {error}") from error
        results["per_shape"][shape] = {
            "row_indexes": [index for index, _ in items],
            "texts": list(result.texts),
            "contents": [list(content) if isinstance(content, tuple) else content for content in result.contents],
            "overhead": result.overhead,
            "budget_source": result.budget_source,
            "cuts": len(result.cuts),
        }
    return results


def default_shape(recipe: Recipe) -> str:
    """The recipe's default request shape: ``pair`` for a reranker, ``document`` for an embedder."""
    return DEFAULT_SHAPE_OF_SHAPE[recipe.role]


DEFAULT_SHAPE_OF_SHAPE: dict[str, str] = {"embed": "document", "multi_vector": "document", "rerank": "pair"}
"""The default request shape per recipe role, when the pairs file does not name one."""


def declared_shapes(recipe: Recipe) -> list[str]:
    """The recipe's declared template shapes, in canonical order (the product's own list)."""
    template = recipe.client.template
    if template is None:
        return [DEFAULT_SHAPE_OF_SHAPE[recipe.role]]
    return [str(shape) for shape in template.shapes()]


def chunk_of(recipe: Recipe) -> ChunkPolicy | None:
    """The recipe's chunk geometry, when ``on_overflow: chunk`` declares one."""
    return getattr(recipe.client, "chunk", None)


def template_of(recipe: Recipe) -> TemplateSpec | None:
    """The recipe's product template."""
    return recipe.client.template


def fold_query(recipe: Recipe, query: str, instruction: str | None) -> str:
    """The product's fold render for ``instruction: fold`` (the client folds the raw query the same way).

    The product's :class:`~rcp_ndcg.inference.clients.RerankClient` applies this to the raw query for
    ``instruction: fold``, via ``rcp_ndcg.data.dataset.Query.format_query``.
    """
    from rcp_ndcg.inference.config import RerankEndpoint

    if not isinstance(recipe.client, RerankEndpoint) or recipe.client.instruction != "fold" or not instruction:
        return query
    from rcp_ndcg.data.dataset import Query

    return str(Query(query_id="", query=query, instruction=instruction).format_query())
