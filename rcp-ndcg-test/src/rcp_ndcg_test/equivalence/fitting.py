"""The bridge to the product's text-budget mechanism: the recipe's tokenizer, its declared shapes, the pairs
file, and the reference subprocess's tokenizer spec.

The harness owns no render, cut or fold code: the recipe's ``client`` block is plain data that ``rcp-ndcg``
validates when it resolves ``recipe: <id>``, and the product's role clients (:mod:`rcp_ndcg.inference.clients`)
make every content decision -- the prompts, the text budget, the reranker's settle-once query -- through
:func:`rcp_ndcg.data.preprocess.fit` with the budget the config declares.  This module keeps only what the
harness itself needs: loading the recipe's tokenizer, reading the pairs file, naming the declared shapes and
the shapes' ``add_special_tokens`` flags, building the client's own token ids for the harness's requests
(:func:`sent_ids`), and resolving the tokenizer spec the reference subprocess loads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rcp_ndcg_vllm.recipe import Recipe

from rcp_ndcg.data.templates import RequestShape, TemplateSpec
from rcp_ndcg.data.tokenizer import load_tokenizer
from rcp_ndcg_test.errors import HarnessError

__all__ = [
    "cast_shape",
    "client_template",
    "declared_shapes",
    "default_shape",
    "load_pairs",
    "query_split_special_tokens",
    "resolved_tokenizer_spec",
    "sent_ids",
    "split_special_tokens",
    "tokenizer_of",
]


def client_template(recipe: Recipe) -> TemplateSpec | None:
    """The recipe's client template as the product's :class:`~rcp_ndcg.data.templates.TemplateSpec`.

    The recipe's ``client.template`` is plain data (the lean package validates no endpoint model); the product's
    model validates it at the read, so the harness sees exactly what ``rcp-ndcg`` would construct — never a
    re-derived fit (R30).
    """
    data = recipe.client.get("template")
    return None if data is None else TemplateSpec.model_validate(data)


def tokenizer_of(recipe: Recipe) -> Any:
    """The recipe's loaded :class:`~rcp_ndcg.data.tokenizer.TextTokenizer`.

    The recipe's ``client.tokenizer`` names a Hub repository id or a tokenizer file; a relative path resolves
    against the recipe directory. The harness process holds no weights: the tokenizer is the one CPU-side
    thing it loads, through the product's own loader.
    """
    spec = recipe.client.get("tokenizer")
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


def split_special_tokens(recipe: Recipe, shape: str) -> bool:
    """Whether the client tokenises ``shape``'s render with the reference's split parse.

    The recipe's ``client.document_split_special_tokens`` (the pplx-embed-v2-context document contract:
    the plugin's role check reads the reference's id space, where an added SPECIAL token is textified):
    the document shape tokenises with ``split_special_tokens=True``, the query shape (see
    :func:`query_split_special_tokens`, which is not one parse of the whole render) and every recipe
    that declares nothing with the file's own parse.  One home for the flag: the product's client
    (``PoolingClient._sent_ids``) sends these ids, so every harness check that derives ids to compare
    against them reads the same declaration (R30).
    """
    return bool(shape == "document" and recipe.client.get("document_split_special_tokens"))


def query_split_special_tokens(recipe: Recipe) -> bool:
    """Whether the client sends the query side's reference construction (``client.query_split_special_tokens``).

    The reference tokenizes the query *body* with ``split_special_tokens=True`` and prepends its prefix id
    (pplx-embed-v2-context's remote ``prepare_inputs``), so the sent ids are the fixed head's added ids
    followed by the split-parsed body -- not one parse of the whole render.  A check that derives the
    query's ids therefore goes through the product's ``TemplateSpec.split_ids``, the one home of the rule
    the client sends (R30); this function is only the declaration's reader.
    """
    return bool(recipe.client.get("query_split_special_tokens"))


def declared_shapes(recipe: Recipe) -> list[str]:
    """The recipe's declared request shapes: the template's, else the role's default (a one-shape recipe)."""
    template = client_template(recipe)
    if template is not None:
        return [str(shape) for shape in template.shapes()]
    return [default_shape(recipe)]


def sent_ids(recipe: Recipe, shape: str, text: str) -> list[int]:
    """The token ids the recipe's client sends for the content text ``text`` rendered as ``shape``.

    The one construction for every harness caller that puts the client's own ids on a ``token_ids`` wire --
    the smoke request, the corpus's standing probes and its plan's bare rows -- and for every caller that
    compares ids against the client's (R30).  The render is the recipe's template's
    (:meth:`~rcp_ndcg.data.templates.TemplateSpec.render`, the same text the client's fit ships); the ids
    are that render under the shape's declared parse, exactly the rule the pooling client's ``_sent_ids``
    sends with: the query side through :meth:`~rcp_ndcg.data.templates.TemplateSpec.split_ids` when the
    recipe declares ``query_split_special_tokens`` (the reference's prefix-plus-body construction), the
    document side with ``split_special_tokens`` when it declares ``document_split_special_tokens``, the
    file's own parse otherwise -- under the shape's declared ``add_special_tokens`` flag throughout.

    Inputs: the recipe, the declared shape (``"query"`` or ``"document"``) and the content span's text.
    Output: the ids, in order.  Raises :class:`HarnessError` for a ``token_ids`` recipe whose client
    declares no template (the product's endpoint config requires one for that wire) and the product's
    ``ConfigError`` for a shape the template does not declare.
    """
    template = client_template(recipe)
    if template is None:
        raise HarnessError(f"recipe {recipe.id}: request_shape token_ids needs the client template")
    tokenizer = tokenizer_of(recipe)
    request_shape = cast_shape(shape)
    rendered = template.render(
        request_shape,
        tokenizer,
        query=text if shape == "query" else "",
        document=text if shape == "document" else "",
    )
    flag = bool(template.adds_special_tokens(request_shape))
    if shape == "query" and query_split_special_tokens(recipe):
        return list(template.split_ids("query", tokenizer, rendered, add_special_tokens=flag))
    return list(
        tokenizer.ids(rendered, add_special_tokens=flag, split_special_tokens=split_special_tokens(recipe, shape))
    )


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
    spec = recipe.client.get("tokenizer")
    if spec is None:  # pragma: no cover - the endpoint config requires it for self-hosted roles
        raise HarnessError(f"recipe {recipe.id}: the client config declares no tokenizer")
    directory = recipe._dir
    candidate = Path(spec)
    if not candidate.is_absolute() and directory is not None and (directory / candidate).exists():
        return str(directory / candidate)
    return spec


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
            if not row["documents"]:
                raise HarnessError(f"{path}:{number}: the row carries no documents")
            pairs.append(row)
    if not pairs:
        raise HarnessError(f"{path} holds no pairs")
    return pairs
