"""The stage functions: stage 1 (prompts, CPU, zero tolerance) and stage 2 (scores or vectors against the gates).

Both take the loaded recipe, the pairs file and the reference subprocess interpreter, and return plain
dictionaries for the report (every number with its referent).

- Stage 1 runs the product's :func:`rcp_ndcg.data.preprocess.fit` over every sampled input (the pairs file's
  rows plus, per declared shape, at least ``over_length_per_shape`` over-length inputs), audits the anchors on
  ``fit``'s output and census, compares the render against the reference subprocess's ``render``, proves a
  served ``chat_template`` renders to the declared shapes' ids, and — with an engine URL — requires the
  engine's ``/tokenize`` to agree with ``fit``'s ids and counts (R29: the engine is the tokenization truth;
  without an engine the check is reported ``not_run``, never passed).
- Stage 2 sends through the product's role clients
  (:class:`~rcp_ndcg.inference.clients.EmbeddingClient`, ``PoolingClient``, ``RerankClient``) built from
  :func:`~rcp_ndcg_vllm.recipe.client_config`, and compares against the reference subprocess's ``score``/
  ``embed`` output.  The product's clients on this branch refuse a budget (the wiring lands in
  ``clients-final``), so the harness pre-fits with the product's ``fit`` and sends the fitted contents through
  a client whose budget fields are cleared — the wire path, the adapter and the interpretation are the
  product's.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from rcp_ndcg_core.content import Content as WireContent

from rcp_ndcg.data.preprocess import TextTruncationCensus, fit

from ..errors import HarnessError
from ..recipe import Recipe, client_config
from .fitting import load_pairs
from .gates import kendall_tau_b, resolve_gates
from .reference import run_reference


def _fitting() -> ModuleType:
    """The fitting module, imported lazily (stage 1 runs long before stage 2 needs it)."""
    from . import fitting

    return fitting


__all__ = ["load_pairs", "stage1_prompts", "stage2_scores"]

_SNIPPET = 80
_OUTPUT_SNIPPET = 500


def stage1_prompts(
    recipe: Recipe,
    pairs_path: str | Path,
    reference_python: str | None,
    *,
    limit: int | None = None,
    over_length_per_shape: int = 20,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Stage 1: the product's ``fit`` renders every sampled prompt; the audit and the comparisons run on it.

    Inputs: the recipe, the pairs file, and (optionally) the reference interpreter and the engine's URL.
    The report carries:

    - ``fit`` — the product's fitted renders per shape, with the measured overhead and the census row count;
    - ``anchor_check`` — every declared shape sampled on purpose with over-length inputs (at least
      ``over_length_per_shape`` per shape, padded in that shape's own span): every anchor must survive the
      product's cut, asserted on ``fit``'s output and census;
    - ``render_check`` — the reference subprocess's ``render`` ids against ``fit``'s, zero tolerance (needs
      ``--reference-python``; reported ``not_run`` without one);
    - ``template_render_check`` — when ``serve.chat_template`` is set: the product's render of each declared
      shape against the template file's jinja2 render (the engine's settings);
    - ``engine_tokenize_check`` — with an engine URL: the engine's ``/tokenize`` of every sampled prompt must
      equal ``fit``'s ids and counts; reported ``not_run`` without an engine, never as passed.
    """
    tokenizer = _fitting().tokenizer_of(recipe)
    rows = load_pairs(pairs_path)
    shown = rows if limit is None else rows[:limit]
    sampled = _sampled_rows(recipe, shown, tokenizer, over_length_per_shape)
    census = TextTruncationCensus()
    fitted = _fitting().fit_rows(recipe, sampled, tokenizer, census=census)
    document: dict[str, Any] = {
        "pairs": len(rows),
        "sampled": len(sampled),
        "fit": {
            shape: {
                "texts_head": [text[:_SNIPPET] for text in body["texts"][:4]],
                "n_texts": len(body["texts"]),
                "overhead": body["overhead"],
                "budget_source": body["budget_source"],
                "cuts": body["cuts"],
            }
            for shape, body in fitted["per_shape"].items()
        },
        "anchor_check": _anchor_check(recipe, fitted, tokenizer),
        "render_check": _render_check(recipe, reference_python, sampled, tokenizer),
        "template_render_check": _template_check(recipe, fitted, rows),
        "engine_tokenize_check": _engine_tokenize_check(recipe, fitted, tokenizer, base_url),
        "passed": False,
    }

    def _gate_pass(check: dict[str, Any] | None) -> bool:
        """A check that ran must pass; ``not_run`` is neutral (it does not gate the stage)."""
        return check is None or check.get("passed") is not False

    document["passed"] = bool(
        document["anchor_check"]["passed"]
        and _gate_pass(document["render_check"])
        and _gate_pass(document["template_render_check"])
        and _gate_pass(document["engine_tokenize_check"])
    )
    return document


def _sampled_rows(
    recipe: Recipe, rows: list[dict[str, Any]], tokenizer: Any, over_length_per_shape: int
) -> list[dict[str, Any]]:
    """The sampled rows: the pairs file's rows, plus per declared shape the over-length inputs.

    Each over-length sample pads that shape's own content span beyond ``client.max_tokens`` (the query for the
    query shape, the document for the document shape, both for the pair) and carries its ``shape`` in the row,
    so the reference subprocess renders the same inputs.
    """
    sampled: list[dict[str, Any]] = [dict(row) for row in rows]
    seed: dict[str, Any] = (
        rows[0] if rows else {"query": "anchor check", "documents": ["anchor check document"], "instruction": None}
    )
    seed_query = str(seed["query"])
    seed_document = str(seed["documents"][0])
    template = recipe.client.template
    shapes = [str(shape) for shape in template.shapes()] if template is not None else [_fitting().default_shape(recipe)]
    for shape in shapes:
        for index in range(max(over_length_per_shape, 1)):
            padded_query = _over_length(seed_query, recipe.client.max_tokens, tokenizer, index)
            padded_document = _over_length(seed_document, recipe.client.max_tokens, tokenizer, index)
            sampled.append(
                {
                    "query": padded_query if shape in ("query", "pair") else seed["query"],
                    "documents": [padded_document if shape in ("document", "pair") else seed["documents"][0]],
                    "shape": shape,
                    "instruction": seed.get("instruction"),
                }
            )
    return sampled


def _over_length(seed: str, max_tokens: int | None, tokenizer: Any, index: int) -> str:
    """One over-length sample: the seed text repeated deterministically past the cap."""
    unit = f"{seed} part {index}"
    one = tokenizer.count(unit)
    repetitions = max(max_tokens // max(one, 1) + 2, 2)
    return " ".join([unit] * repetitions)


def _add_specials_flag(recipe: Recipe, shape: str) -> bool:
    """The shape's declared engine behaviour: whether the route adds the tokenizer's post-processor tokens."""
    template = recipe.client.template
    return bool(template.adds_special_tokens(cast_shape(shape))) if template is not None else False


def cast_shape(shape: str) -> Any:
    """A validated shape string as the product's :data:`~rcp_ndcg.data.templates.RequestShape` literal.

    The recipe's product template validated the shapes at load; the harness fits declared shapes only, so the
    string is one of the product's literals."""
    return shape


def _anchor_check(recipe: Recipe, fitted: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """The anchor audit over ``fit``'s output: every anchor survives the product's cut, per declared shape.

    ``anchor: last`` requires the rendered ids to end with the tail fixed segment's ids (plus the tokenizer's
    post-processor tokens when the shape pins ``add_special_tokens``); ``anchor: first`` to start with the
    head's; ``anchor: marker`` to contain the named markers; ``mean`` has no positional anchor.
    """
    template = recipe.client.template
    failures: list[dict[str, Any]] = []
    checked = 0
    for shape, body in fitted["per_shape"].items():
        edge = _anchor_edge_ids(recipe, tokenizer, shape)
        flag = _add_specials_flag(recipe, shape)
        at_start = template is not None and template.anchor == "first"
        for text in body["texts"]:
            ids = tokenizer.ids(text, add_special_tokens=flag)
            checked += 1
            if template is None or template.anchor == "mean":
                continue
            if template.anchor == "marker":
                missing = sorted(name for name in template.anchor_markers if tokenizer.special_text(name) not in text)
                if missing:
                    failures.append({"shape": shape, "check": "markers", "missing_names": missing})
                continue
            actual = ids[: len(edge)] if at_start else ids[-len(edge) :]
            if not edge or actual != edge:
                failures.append(
                    {
                        "shape": shape,
                        "check": "head" if at_start else "tail",
                        "expected_edge_ids": edge,
                        "actual_edge_ids": actual,
                        "text": text[:_SNIPPET],
                    }
                )
    return {
        "anchor": template.anchor if template is not None else None,
        "checked": checked,
        "passed": not failures,
        "failures": failures,
        "referent": "the anchor ids (the tail/head fixed segment plus the post-processor block, or the named "
        "markers) must sit at their declared positions in fit's rendered ids",
    }


def _post_processor_tail(tokenizer: Any, text: str) -> list[int]:
    """The tokens ``add_special_tokens: true`` adds after the content, for the shape's tokenizer.

    Measured, not assumed: the ids of ``text`` with the post-processor are aligned against its plain ids and
    the tail after the content's last occurrence is the post-processor's suffix — empty for a prepend-only
    post-processor, the end token for the common append-only one.  When the text's own render is empty or
    displaced (a template literal that happens to contain the specials), the tail is measured on a sentinel.
    """
    plain = list(tokenizer.ids(text, add_special_tokens=False))
    full = list(tokenizer.ids(text, add_special_tokens=True))
    index = _last_sublist(full, plain) if plain else None
    if index is None:
        probe_plain = list(tokenizer.ids("x", add_special_tokens=False))
        probe_full = list(tokenizer.ids("x", add_special_tokens=True))
        probe_index = _last_sublist(probe_full, probe_plain)
        if probe_index is None:  # pragma: no cover - a post-processor that both displaces and reshapes
            return full[len(plain) :]
        return probe_full[probe_index + len(probe_plain) :]
    return full[index + len(plain) :]


def _last_sublist(haystack: list[int], needle: list[int]) -> int | None:
    """The last index where ``needle`` occurs in ``haystack`` contiguously, or ``None`` (empty needle too)."""
    if not needle or len(needle) > len(haystack):
        return None
    for start in range(len(haystack) - len(needle), -1, -1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return None


def _anchor_edge_ids(recipe: Recipe, tokenizer: Any, shape: Any) -> list[int]:
    """The anchor ids of one shape: the tail (or, for ``anchor: first``, head) fixed segment's rendered ids,
    plus the post-processor tokens the shape's ``add_special_tokens`` flag appends after them."""
    template = recipe.client.template
    if template is None:
        return []
    segments = template.segments(shape)
    if template.anchor == "first":
        head = next((segment for segment in segments if segment.fixed is not None), None)
        text = head.render(tokenizer) if head is not None else ""
    else:
        fixed = [segment for segment in segments if segment.fixed is not None]
        text = fixed[-1].render(tokenizer) if fixed else ""
    ids = list(tokenizer.ids(text, add_special_tokens=False))
    if template.adds_special_tokens(shape):
        ids += _post_processor_tail(tokenizer, text)
    return ids


def _render_check(
    recipe: Recipe, reference_python: str | None, sampled: list[dict[str, Any]], tokenizer: Any
) -> dict[str, Any] | None:
    """The reference render comparison, via the reference subprocess (stage 1's reference side).

    Only the pairs file's rows are compared (the over-length samples are audited for anchor survival, not
    compared token for token: the reference subprocess's own cut is the ground truth for its shape).  The
    comparison is on the rendered TEXT (both sides use the product's fit, so the texts must be byte-identical).
    """
    if reference_python is None:
        return {"status": "not_run", "passed": None, "reason": "no --reference-python given"}
    recipe_dir = recipe._dir
    if recipe_dir is None:  # pragma: no cover - load_recipe sets it
        raise HarnessError(f"recipe {recipe.id} was not loaded from a directory; use load_recipe")
    entry = str(recipe_dir / recipe.reference.entry)
    in_budget = [row for row in sampled if "shape" not in row]
    served_by_key = _fit_texts_by_row(recipe, in_budget, tokenizer)
    with tempfile.TemporaryDirectory() as work:
        pairs_path = Path(work) / "pairs.jsonl"
        out_path = Path(work) / "reference.json"
        _write_rows([row for row in sampled if "shape" not in row], pairs_path)
        reference = run_reference(
            reference_python,
            entry,
            mode="render",
            pairs_path=pairs_path,
            out_path=out_path,
            tokenizer_spec=_resolved_tokenizer_spec(recipe),
        )
    failures: list[dict[str, Any]] = []
    for row in reference.get("rows", []):
        key = (int(row["index"]), str(row.get("shape", "")))
        served = served_by_key.get(key)
        if served is None:
            failures.append({"row": row, "note": "the reference rendered a row the harness did not sample"})
        elif row.get("text", "") != served:
            failures.append(
                {
                    "index": row["index"],
                    "shape": key[1],
                    "served_text_head": served[:_SNIPPET],
                    "reference_text_head": row.get("text", "")[:_SNIPPET],
                    "text": str(row.get("query", ""))[:_SNIPPET],
                }
            )
    return {"status": "run", "rows": len(reference.get("rows", [])), "passed": not failures, "failures": failures}


def _template_check(recipe: Recipe, fitted: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """When ``serve.chat_template`` is set: the product's render of each declared shape must equal the file's.

    The template file is rendered with the engine's own jinja2 settings, with the raw query, document and
    instruction of the first pairs row; ``fit``'s render of the same inputs must be byte-identical.
    """
    if recipe.serve.chat_template is None or recipe.client.template is None or not rows:
        return None
    directory = recipe._dir
    if directory is None:  # pragma: no cover - load_recipe sets it
        raise HarnessError(f"recipe {recipe.id} was not loaded from a directory")
    template_text = (directory / recipe.serve.chat_template).read_text(encoding="utf-8")
    row = rows[0]
    # The jinja file receives the same texts fit received: the query pre-folded by the client (fold mode).
    folded_query = _fitting().fold_query(recipe, row["query"], row.get("instruction"))
    jinja_text = (
        _jinja_environment()
        .from_string(template_text)
        .render(query=folded_query, document=row["documents"][0], instruction=row.get("instruction") or "")
    )
    failures: list[dict[str, Any]] = []
    for shape, body in fitted["per_shape"].items():
        if not body["texts"] or jinja_text != body["texts"][0]:
            failures.append(
                {
                    "shape": shape,
                    "jinja_head": jinja_text[:_SNIPPET],
                    "fit_head": (body["texts"][0][:_SNIPPET] if body["texts"] else ""),
                }
            )
    return {
        "template": recipe.serve.chat_template,
        "shapes": sorted(fitted["per_shape"]),
        "passed": not failures,
        "failures": failures,
    }


def _jinja_environment() -> Any:
    """The jinja2 environment the engine renders chat templates with (transformers' compile settings)."""
    try:
        from jinja2 import StrictUndefined
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError as error:  # pragma: no cover - exercised only without jinja2
        raise HarnessError(
            "the template-render check renders the served chat template with jinja2: install rcp-ndcg-vllm[test]"
        ) from error
    return ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False, undefined=StrictUndefined
    )


def _engine_tokenize_check(
    recipe: Recipe, fitted: dict[str, Any], tokenizer: Any, base_url: str | None
) -> dict[str, Any] | None:
    """The engine is the tokenization truth (R29): ``/tokenize`` of fit's renders must equal fit's ids.

    With an engine URL, every sampled rendered prompt (per shape) goes to the engine's ``/tokenize`` with the
    shape's ``add_special_tokens`` flag; the ids and the count must equal ``fit``'s.  Without an engine the
    check is reported ``not_run`` — never as passed.
    """
    if base_url is None:
        return {
            "status": "not_run",
            "passed": None,
            "reason": "no engine URL; the /tokenize check runs only against a live engine",
        }
    failures: list[dict[str, Any]] = []
    checked = 0
    for shape, body in fitted["per_shape"].items():
        add_flag = _add_specials_flag(recipe, shape)
        for text in body["texts"]:
            engine_ids = _engine_tokenize(recipe, base_url, text, add_special_tokens=add_flag)
            fit_ids = tokenizer.ids(text, add_special_tokens=add_flag)
            checked += 1
            if engine_ids != fit_ids:
                failures.append(
                    {
                        "shape": shape,
                        "fit_ids_head": fit_ids[:24],
                        "engine_ids_head": list(engine_ids)[:24],
                        "fit_len": len(fit_ids),
                        "engine_len": len(engine_ids),
                        "text": text[:_SNIPPET],
                    }
                )
    return {
        "status": "run",
        "checked": checked,
        "passed": not failures,
        "failures": failures,
        "referent": "the engine's /tokenize ids and count of fit's rendered prompt, per shape",
    }


def tokenize_url(base_url: str) -> str:
    """The engine's ``/tokenize`` URL: vLLM serves it at the root, next to the versioned APIs."""
    root = base_url.rstrip("/")
    if root.endswith(("/v1", "/v2")):
        root = root.rsplit("/", 1)[0]
    return f"{root}/tokenize"


def _engine_tokenize(recipe: Recipe, base_url: str, text: str, *, add_special_tokens: bool) -> list[int]:
    """POST the rendered prompt to the engine's ``/tokenize`` and return its ids."""
    import httpx

    response = httpx.post(
        tokenize_url(base_url),
        json={"model": recipe.client.model, "prompt": text, "add_special_tokens": add_special_tokens},
        timeout=60.0,
    )
    if response.status_code != 200:
        raise HarnessError(
            f"the engine's /tokenize returned HTTP {response.status_code}: {response.text[:_OUTPUT_SNIPPET]}"
        )
    tokens = response.json().get("tokens")
    if not isinstance(tokens, list):
        raise HarnessError("the engine's /tokenize reply carries no 'tokens' list")
    return [int(value) for value in tokens]


# ---------------------------------------------------------------------------
# Stage 2: the product's role clients, the reference subprocess's outputs, the gates.
# ---------------------------------------------------------------------------


def stage2_scores(
    recipe: Recipe,
    pairs_path: str | Path,
    reference_python: str,
    *,
    base_url: str,
    served_model_name: str | None = None,
    device: str = "cpu",
    recorder: Any | None = None,
) -> dict[str, Any]:
    """Stage 2: the served engine against the reference subprocess, under the recipe's gates.

    The engine is talked to through the product's transport and adapter (the product's wire path): the
    adapter builds the :class:`~rcp_ndcg.inference.types.Call` objects, the product's
    :class:`~rcp_ndcg.inference.transport.Transport` sends them, and the adapter interprets the replies.
    The product's clients on this branch refuse a budget (the wiring lands in ``clients-final``), so the
    harness pre-fits every input with the product's ``fit`` and sends the fitted contents through the product's
    adapter and transport directly; when the wiring lands, stage 2 uses the product's role clients with the
    config's own budget.  The reference runs as a subprocess in its own environment (``--reference-python``,
    required); the harness process imports no torch.
    """
    rows = load_pairs(pairs_path)
    reference = _reference_outputs(recipe, reference_python, rows, device=device)
    gates = resolve_gates(recipe)
    if recipe.role == "rerank":
        return _rerank_stage2(recipe, rows, reference, base_url, gates, served_model_name, recorder)
    return _vector_stage2(recipe, rows, reference, base_url, gates, served_model_name, recorder)


def _reference_outputs(
    recipe: Recipe, reference_python: str, rows: list[dict[str, Any]], *, device: str
) -> dict[str, Any]:
    """The reference subprocess's outputs for the rows, written to a temporary file and parsed."""
    if reference_python == "":
        raise HarnessError("stage 2 needs --reference-python: the reference runs in its own environment")
    mode = {"rerank": "score"}.get(recipe.role, "embed")
    recipe_dir = recipe._dir
    if recipe_dir is None:  # pragma: no cover - load_recipe sets it
        raise HarnessError(f"recipe {recipe.id} was not loaded from a directory")
    entry = str(recipe_dir / recipe.reference.entry)
    with tempfile.TemporaryDirectory() as work:
        pairs_path = Path(work) / "pairs.jsonl"
        out_path = Path(work) / "reference.json"
        _write_rows(rows, pairs_path)
        return run_reference(
            reference_python,
            entry,
            mode=mode,
            pairs_path=pairs_path,
            out_path=out_path,
            tokenizer_spec=_resolved_tokenizer_spec(recipe),
            device=device,
        )


def _resolved_tokenizer_spec(recipe: Recipe) -> str:
    """The tokenizer spec the reference subprocess tokenises with (the recipe's, resolved)."""
    spec = recipe.client.tokenizer
    if spec is None:  # pragma: no cover - the endpoint config requires it for self-hosted roles
        raise HarnessError(f"recipe {recipe.id}: the client config declares no tokenizer")
    directory = recipe._dir
    candidate = Path(spec)
    if not candidate.is_absolute() and directory is not None and (directory / candidate).exists():
        return str(directory / candidate)
    return spec


def _write_rows(rows: list[dict[str, Any]], path: str | Path) -> None:
    """The sampled rows as the pairs file the reference subprocess reads."""
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _recording_transport(recipe: Recipe, base_url: str, recorder: Any | None) -> Any:
    """The product's :class:`~rcp_ndcg.inference.transport.Transport` over a recording httpx transport.

    The recorder observes the product's own wire path through an ``httpx`` transport hook (the event-hook seam
    the transport accepts); there is no second request path.  Without a recorder the transport is the
    product's own.
    """
    from rcp_ndcg.inference.config import EmbeddingEndpoint, PoolingEndpoint, RerankEndpoint
    from rcp_ndcg.inference.transport import Transport

    classes: dict[str, type] = {"embed": EmbeddingEndpoint, "multi_vector": PoolingEndpoint, "rerank": RerankEndpoint}
    endpoint = classes[recipe.role](**client_config(recipe, base_url=base_url))
    if recorder is None:
        return Transport(endpoint)
    return Transport(endpoint, httpx_transport=recorder)


def _wire_vectors(
    recipe: Recipe, base_url: str, texts: list[str], role: str, served_model_name: str | None, recorder: Any | None
) -> list[list[list[float]]]:
    """The served vectors for the fitted texts, through the product's transport and adapter.

    Embedding: one vector per text; late interaction: one ragged matrix of per-token vectors per text.
    """
    from rcp_ndcg.inference.adapters.base import get_adapter
    from rcp_ndcg.inference.config import EmbeddingEndpoint, PoolingEndpoint
    from rcp_ndcg.inference.transport import Transport
    from rcp_ndcg.inference.types import EmbedRequest, EncodeRole, PoolRequest

    classes: dict[str, type] = {"embed": EmbeddingEndpoint, "multi_vector": PoolingEndpoint}
    endpoint = classes[recipe.role](**client_config(recipe, base_url=base_url))
    adapter = get_adapter(endpoint.api, role=recipe.role)()  # type: ignore[call-arg,arg-type]
    encode_role = EncodeRole.QUERY if role == "query" else EncodeRole.DOCUMENT
    wire_contents = tuple(WireContent.from_text(text) for text in texts)
    if recipe.role == "multi_vector":
        request = PoolRequest(
            contents=wire_contents, role=encode_role, embed_dtype=endpoint.embed_dtype, dim=endpoint.dim
        )  # type: ignore[union-attr]
    else:
        request = EmbedRequest(contents=wire_contents, role=encode_role)
    calls = adapter.calls(request, model=recipe.id)
    transport = Transport(endpoint, httpx_transport=recorder) if recorder else Transport(endpoint)
    replies = list(transport.run(transport.send(calls)))
    transport.aclose()
    embeddings = adapter.interpret(request, replies)
    if recipe.role == "embed" or embeddings.offsets is None:
        return [[[float(value) for value in vector] for vector in embeddings.vectors]]
    out = []
    for i in range(len(embeddings.offsets) - 1):
        item = embeddings.vectors[embeddings.offsets[i] : embeddings.offsets[i + 1]]
        out.append([[float(value) for value in vector] for vector in item])
    return out


def _wire_rerank(
    recipe: Recipe,
    base_url: str,
    query: str,
    documents: list[str],
    served_model_name: str | None,
    recorder: Any | None,
) -> Any:
    """One rerank wire call through the product's adapter and transport (the product's wire path)."""
    from rcp_ndcg.inference.adapters.base import get_adapter
    from rcp_ndcg.inference.config import RerankEndpoint
    from rcp_ndcg.inference.transport import Transport
    from rcp_ndcg.inference.types import RerankRequest

    endpoint = RerankEndpoint(**client_config(recipe, base_url=base_url))
    adapter = get_adapter(endpoint.api, role="rerank")(endpoint)  # type: ignore[call-arg]
    request = RerankRequest(
        query=WireContent.from_text(query),
        documents=tuple(WireContent.from_text(document) for document in documents),
    )
    calls = adapter.calls(request, model=served_model_name or recipe.id)
    transport = Transport(endpoint, httpx_transport=recorder) if recorder else Transport(endpoint)
    replies = list(transport.run(transport.send(calls)))
    transport.aclose()
    return adapter.interpret(request, replies)


def _rerank_stage2(
    recipe: Recipe,
    rows: list[dict[str, Any]],
    reference: dict[str, Any],
    base_url: str,
    gates: Any,
    served_model_name: str | None,
    recorder: Any | None,
) -> dict[str, Any]:
    """Rerank scores from the served engine through the product's transport and adapter.

    The inputs are fitted first (the product's ``fit``, the pair shape), then the fitted contents go in one
    request per query.  The client's instruction fold is applied to the raw query via
    :func:`~rcp_ndcg_vllm.equivalence.fitting.fold_query`, exactly the text fit rendered.  With the declared
    ``anchor_drop_over_cap`` deviation, over-cap pairs are excluded from the gates and reported separately; the
    Kendall tau covers the under-cap subset of every gated query.
    """
    deviation = "anchor_drop_over_cap" in recipe.reference.known_deviations
    tokenizer = _fitting().tokenizer_of(recipe)
    max_tokens = recipe.client.max_tokens or 1
    per_document: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    over_cap: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if row_index >= len(reference["rows"]):
            break
        reference_scores = [float(value) for value in reference["rows"][row_index]["scores"]]
        folded_query, fitted_documents = _fit_pair(recipe, row, tokenizer)
        result = _wire_rerank(recipe, base_url, folded_query, fitted_documents, served_model_name, recorder)
        flags = [_pair_tokens(recipe, row["query"], document, tokenizer) > max_tokens for document in row["documents"]]
        for document_index, (served_score, reference_score) in enumerate(
            zip(result.scores, reference_scores, strict=True)
        ):
            delta = abs(served_score - reference_score)
            bound = _score_bound(gates, recipe.reference.score_scale, reference_score)
            entry = {
                "query_index": row_index,
                "document_index": document_index,
                "query": row["query"],
                "document": row["documents"][document_index][:_SNIPPET],
                "served": served_score,
                "reference": reference_score,
                "abs_delta": delta,
                "bound": bound,
                "within": bool(delta <= bound),
                "over_cap": flags[document_index],
            }
            if flags[document_index]:
                over_cap.append(entry)
            if flags[document_index] and deviation:
                continue
            per_document.append(entry)
        under_cap = [index for index, over in enumerate(flags) if not over]
        if under_cap:
            tau = kendall_tau_b(
                [result.scores[index] for index in under_cap],
                [reference_scores[index] for index in under_cap],
            )
            per_query.append(
                {
                    "query_index": row_index,
                    "query": row["query"],
                    "documents": len(under_cap),
                    "kendall_tau": tau,
                    "within": bool(tau is not None and tau >= gates.tau_min),
                }
            )
    summary = _rerank_summary(per_document, per_query, gates, recipe.reference.score_scale)
    summary["over_cap"] = {
        "known_deviation": deviation,
        "n_pairs": len(over_cap),
        "gating": False,
        "pairs": over_cap,
        "passed": True,
        "referent": "pairs whose uncut prompt exceeds client.max_tokens; served and reference may differ by "
        "design when reference.known_deviations declares anchor_drop_over_cap",
    }
    return summary


def _fit_pair(recipe: Recipe, row: dict[str, Any], tokenizer: Any) -> tuple[str, list[str]]:
    """The pair fit of one row: the folded query cut to its share, each document to the remainder.

    The query is folded with the product's own render (``Task: ...\nQuery: ...``) before the fit, so the
    client's ``instruction: fold`` re-folds the raw query to exactly the text that was fitted.
    """
    instruction = row.get("instruction")
    folded = _fold(recipe, row["query"], instruction)
    budget = _fitting().budget_of(recipe).model_copy(update={"tokenizer": tokenizer.name})
    inputs = [(folded, document) for document in row["documents"]]
    result = fit(inputs, "pair", budget, tokenizer, ids=[str(index) for index in range(len(inputs))])
    contents = [list(content) for content in result.contents]
    return folded, [document for _, document in contents]


def _fold(recipe: Recipe, query: str, instruction: str | None) -> str:
    """The product's fold render for ``instruction: fold`` (the client folds the raw query the same way)."""
    from rcp_ndcg.inference.config import RerankEndpoint

    if not isinstance(recipe.client, RerankEndpoint) or recipe.client.instruction != "fold" or not instruction:
        return query
    from rcp_ndcg.data.dataset import Query

    return str(Query(query_id="", query=query, instruction=instruction).format_query())


def _pair_tokens(recipe: Recipe, query: str, document: str, tokenizer: Any) -> int:
    """The token count of one pair's prompt with NO client cut: fixed overhead plus full content."""
    template = recipe.client.template
    overhead = 0
    if template is not None:
        overhead = sum(
            len(tokenizer.ids(segment.render(tokenizer), add_special_tokens=False))
            for segment in template.segments("pair")
            if segment.fixed is not None
        )
        if template.adds_special_tokens("pair"):
            overhead += len(tokenizer.ids("", add_special_tokens=True))
    return (
        overhead
        + len(tokenizer.ids(query, add_special_tokens=False))
        + len(tokenizer.ids(document, add_special_tokens=False))
    )


def _rerank_summary(
    per_document: list[dict[str, Any]], per_query: list[dict[str, Any]], gates: Any, scale: str
) -> dict[str, Any]:
    """Aggregate the per-document deltas and per-query taus into the gate rows of the score's scale."""
    deltas = [entry["abs_delta"] for entry in per_document]
    worst = max(deltas) if deltas else 0.0
    taus = [entry["kendall_tau"] for entry in per_query if entry["kendall_tau"] is not None]
    median_tau = float(np.median(taus)) if taus else None
    within_p99 = sum(1 for entry in per_document if entry["abs_delta"] <= gates.prob_p99_abs) / max(
        len(per_document), 1
    )
    tau_row = {
        "gate": "kendall_tau_median",
        "passed": bool(median_tau is not None and median_tau >= gates.tau_min),
        "value": median_tau,
        "bound": gates.tau_min,
        "referent": "median per-query Kendall tau between served and reference scores (under-cap pairs)",
    }
    if scale == "probability":
        gate_rows = [
            {
                "gate": "p99_documents_within",
                "passed": bool(within_p99 >= 0.99),
                "value": within_p99,
                "bound": 0.99,
                "referent": f"fraction of documents with |served - reference| <= {gates.prob_p99_abs}",
            },
            {
                "gate": "max_abs_delta",
                "passed": bool(worst <= gates.prob_max_abs),
                "value": worst,
                "bound": gates.prob_max_abs,
                "referent": "max |served - reference| over all documents",
            },
            tau_row,
        ]
    elif scale == "logit":
        ratios = [entry["abs_delta"] / (1.0 + abs(entry["reference"])) for entry in per_document]
        worst_ratio = max(ratios) if ratios else 0.0
        gate_rows = [
            {
                "gate": "max_relative_delta",
                "passed": bool(worst_ratio <= gates.logit_rel_abs),
                "value": worst_ratio,
                "bound": gates.logit_rel_abs,
                "referent": "max |served - reference| / (1 + |reference|) over all documents",
            },
            tau_row,
        ]
    else:
        gate_rows = [
            {
                "gate": "max_abs_delta",
                "passed": bool(worst <= gates.cos_max_abs),
                "value": worst,
                "bound": gates.cos_max_abs,
                "referent": "max |served - reference| over all documents",
            },
            tau_row,
        ]
    return {
        "score_scale": scale,
        "n_queries": len(per_query),
        "n_documents": len(per_document),
        "per_document": per_document,
        "per_query": per_query,
        "abs_delta_max": worst,
        "within_p99_fraction": within_p99,
        "kendall_tau_median": median_tau,
        "gates": gate_rows,
        "passed": bool(all(row["passed"] for row in gate_rows)),
    }


def _score_bound(gates: Any, scale: str, reference_score: float) -> float:
    """The per-document |delta| bound on one score scale, at the reference's score."""
    if scale == "probability":
        return gates.prob_max_abs
    if scale == "logit":
        return gates.logit_rel_abs * (1.0 + abs(reference_score))
    return gates.cos_max_abs


def _vector_stage2(
    recipe: Recipe,
    rows: list[dict[str, Any]],
    reference: dict[str, Any],
    base_url: str,
    gates: Any,
    served_model_name: str | None,
    recorder: Any | None,
) -> dict[str, Any]:
    """Vectors from the served engine through the product's role client, against the reference's vectors.

    A dense embedder's vectors compare with a cosine floor per vector; a late-interaction model's ragged
    token vectors compare per token (in the transfer precision the product's client applied on the wire).
    The inputs are fitted first (the product's ``fit``, the row's shape), and the fitted strings are what the
    client sends.
    """
    per_vector: list[dict[str, Any]] = []
    budget = _fitting().budget_of(recipe).model_copy(update={"tokenizer": _fitting().tokenizer_of(recipe).name})
    for row_index, row in enumerate(rows):
        if row_index >= len(reference["rows"]):
            break
        reference_row = reference["rows"][row_index]
        template = recipe.client.template
        shapes = {str(shape) for shape in template.shapes()} if template is not None else {"document"}
        for role, served_key, inputs in (
            ("query", "query_vectors", [row["query"]]),
            ("document", "document_vectors", list(row["documents"])),
        ):
            shape = "query" if role == "query" else "document"
            if shape not in shapes:
                continue
            result = fit(
                inputs,
                cast_shape(shape),
                budget,
                _fitting().tokenizer_of(recipe),
                ids=[str(i) for i in range(len(inputs))],
            )
            texts_list: list[str] = list(result.texts)
            served_vectors = _wire_vectors(recipe, base_url, texts_list, role, None, recorder)
            expected = reference_row.get(served_key) or []
            _compare_shape(recipe, served_vectors, expected, row_index, role, per_vector, gates)
    return _vector_summary(recipe, per_vector, gates)


def _compare_shape(
    recipe: Recipe,
    served_vectors: list[list[list[float]]],
    expected: list[Any],
    row_index: int,
    role: str,
    per_vector: list[dict[str, Any]],
    gates: Any,
) -> None:
    """Pair served with reference vectors positionally and record the cosine rows (per vector, or per token)."""
    matrix = served_vectors[0] if served_vectors else []
    if len(matrix) != len(expected):
        per_vector.append(
            {
                "referent": f"row {row_index} {role} count",
                "cosine": None,
                "within": False,
                "note": f"the engine returned {len(matrix)} vector(s), the reference {len(expected)}",
            }
        )
        return
    for index, (served_vector, reference_vector) in enumerate(zip(matrix, expected, strict=True)):
        cosine = _cosine(np.asarray(served_vector, dtype=np.float64), np.asarray(reference_vector, dtype=np.float64))
        per_vector.append(
            {
                "referent": f"row {row_index} {role} {index}",
                "cosine": cosine,
                "within": bool(cosine >= gates.vec_min_cosine),
            }
        )


def _vector_summary(recipe: Recipe, per_vector: list[dict[str, Any]], gates: Any) -> dict[str, Any]:
    """Aggregate the per-vector cosines into the gate row; every non-within row fails the stage."""
    cosines = [entry["cosine"] for entry in per_vector if entry["cosine"] is not None]
    worst = min(cosines) if cosines else None
    multi = recipe.role == "multi_vector"
    gate_rows = [
        {
            "gate": "min_cosine",
            "passed": bool(worst is not None and worst >= gates.vec_min_cosine),
            "value": worst,
            "bound": gates.vec_min_cosine,
            "referent": f"cosine per {'token' if multi else 'vector'}",
        }
    ]
    return {
        "score_scale": recipe.reference.score_scale,
        "multi_vector": multi,
        "n_vectors": len(per_vector),
        "per_vector": per_vector,
        "cosine_min": worst,
        "gates": gate_rows,
        "passed": bool(all(row["passed"] for row in gate_rows) and all(entry["within"] for entry in per_vector)),
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """The cosine of two vectors, in float64; 0.0 when either is zero."""
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / norm) if norm else 0.0


def _fit_texts_by_row(recipe: Recipe, sampled: list[dict[str, Any]], tokenizer: Any) -> dict[tuple[int, str], str]:
    """``fit``'s rendered texts per sampled row, keyed by (row index, shape) — the engine's prompt.

    A pairs row (no ``shape``) is rendered under every declared shape: the reference contract emits one render
    per declared shape at the row's index, so a multi-shape recipe compares them all.
    """
    budget = _fitting().budget_of(recipe).model_copy(update={"tokenizer": tokenizer.name})
    declared = (
        [str(shape) for shape in recipe.client.template.shapes()]
        if recipe.client.template is not None
        else [_fitting().default_shape(recipe)]
    )
    out: dict[tuple[int, str], str] = {}
    for index, row in enumerate(sampled):
        shapes = [str(row["shape"])] if "shape" in row else declared
        for shape in shapes:
            query = _fitting().fold_query(recipe, row["query"], row.get("instruction"))
            inputs: list[Any] = (
                [(query, row["documents"][0])]
                if shape == "pair"
                else [query if shape == "query" else row["documents"][0]]
            )
            result = fit(inputs, cast_shape(shape), budget, tokenizer, ids=[str(index)])
            out[(index, shape)] = result.texts[0]
    return out
