"""The three-stage equivalence harness, driven through the product's role clients.

Stage 1 samples the pairs file, probes the recipe's role client for every sampled input through the product's
injection point (a capturing ``httpx`` transport), and audits what the client actually sends: the anchor audit
and the engine's ``/tokenize`` read the captured request bodies, the reference subprocess's ``render`` is
compared against them, and the served template file is rendered against them.  Over-cap inputs the client had
to cut are decided on the client's own census and -- under a declared over-cap deviation --
reported in a separate non-gating table.  Stage 2 sends the reference's pairs through the same clients and
gates the answers against the reference subprocess's outputs.  Stage 3 scores rankings with ``rcp-ndcg eval
score``.  The harness never re-derives a render, a cut or a settlement.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from rcp_ndcg.data.preprocess import TextTruncationCensus

from ..errors import HarnessError
from ..recipe import Recipe
from . import fitting
from .fitting import load_pairs
from .gates import kendall_tau_b, resolve_gates
from .reference import run_reference
from .wire import Capture, role_client

__all__ = ["load_pairs", "stage1_prompts", "stage2_scores"]

_SNIPPET = 240
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
    """Stage 1: the recipe's role client produces every sampled request; the audit and comparisons run on it.

    Inputs: the recipe, the pairs file, and (optionally) the reference interpreter and the engine's URL.  The
    client is probed through the product's injection point (the engine when ``--base-url`` is given, the
    product's offline fake otherwise), so the captured request bodies are what the client actually sends --
    for a reranker, the settled query span.  The report carries:

    - ``client`` — the captured request texts per shape (heads and counts), the product's wire;
    - ``anchor_check`` — every declared shape sampled on purpose with over-length inputs (at least
      ``over_length_per_shape`` per shape, padded in that shape's own span): every anchor must survive the
      client's cut, asserted on the captured requests and the client's census; for a reranker this is the
      settle-once query (one settled span per row, within its declared share, no cut on an in-budget pair);
    - ``render_check`` — the reference subprocess's ``render`` output against the captured texts, zero
      tolerance (needs ``--reference-python``; reported ``not_run`` without one).  Under a declared
      over-cap deviation, over-cap rows are reported separately and do not gate;
    - ``template_render_check`` — when ``serve.chat_template`` is set: the template file's jinja2 render (the
      engine's settings) of every declared shape against the client's render of the same inputs;
    - ``engine_tokenize_check`` — with an engine URL: the engine's ``/tokenize`` of every captured text must
      equal the recipe tokenizer's ids; reported ``not_run`` without an engine, never as passed.
    """
    tokenizer = fitting.tokenizer_of(recipe)
    rows = load_pairs(pairs_path)
    shown = rows if limit is None else rows[:limit]
    sampled = _sampled_rows(recipe, shown, tokenizer, over_length_per_shape)
    probe = _probe(recipe, sampled, base_url, tokenizer)
    document: dict[str, Any] = {
        "pairs": len(rows),
        "sampled": len(sampled),
        "checked": probe["checked"],
        "client": probe["client"],
        "anchor_check": _anchor_check(recipe, probe, tokenizer),
        "render_check": _render_check(recipe, reference_python, sampled, probe, tokenizer),
        "template_render_check": _template_check(recipe, rows, probe, tokenizer),
        "engine_tokenize_check": _engine_tokenize_check(recipe, probe, tokenizer, base_url),
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
    query shape, the document for the document shape, both for the pair) and carries its ``shape`` plus the
    private ``over_length`` marker, so the checks know what to audit (the cut) and what to compare (only the
    pairs file's rows).
    """
    sampled: list[dict[str, Any]] = [dict(row) for row in rows]
    seed: dict[str, Any] = (
        rows[0] if rows else {"query": "anchor check", "documents": ["anchor check document"], "instruction": None}
    )
    seed_query = str(seed["query"])
    seed_document = str(seed["documents"][0])
    for shape in fitting.declared_shapes(recipe):
        for index in range(max(over_length_per_shape, 1)):
            padded_query = _over_length(seed_query, recipe.client.max_tokens, tokenizer, index)
            padded_document = _over_length(seed_document, recipe.client.max_tokens, tokenizer, index)
            sampled.append(
                {
                    "query": padded_query if shape in ("query", "pair") else seed_query,
                    "documents": [padded_document if shape in ("document", "pair") else seed_document],
                    "shape": shape,
                    "over_length": True,
                    "instruction": seed.get("instruction"),
                }
            )
    return sampled


def _over_length(seed: str, max_tokens: int | None, tokenizer: Any, index: int) -> str:
    """A seed text padded to at least ``max_tokens`` tokens (plus one, per index), in whole words."""
    budget = max_tokens or 128
    words = seed.split() or ["anchor"]
    marker = f" pad{index}"
    text = seed
    while tokenizer.count(text) < budget * 2:
        text = text + marker * max(1, (budget * 2 - tokenizer.count(text)) // max(1, len(words) + 1))
        if text == seed:
            text = seed + marker
    return text


def _add_specials_flag(recipe: Recipe, shape: str) -> bool:
    """The shape's ``add_special_tokens`` flag (the engine's post-processor behaviour, declared)."""
    template = recipe.client.template
    return bool(template.adds_special_tokens(fitting.cast_shape(shape))) if template is not None else False


def cast_shape(shape: str) -> Any:
    """A validated shape string as the product's :data:`~rcp_ndcg.data.templates.RequestShape` literal.

    The recipe's product template validated the shapes at load; the harness fits declared shapes only, so the
    string is one of the product's literals."""
    return fitting.cast_shape(shape)


# ---------------------------------------------------------------------------
# The client probe: the recipe's role client produces every sampled request.
# ---------------------------------------------------------------------------


def _probe(recipe: Recipe, sampled: list[dict[str, Any]], base_url: str | None, tokenizer: Any) -> dict[str, Any]:
    """The client's own requests for the sampled inputs, captured through the product's injection point.

    Outputs: per row, the request texts the client produced (per declared shape: the rendered prompt for the
    embed roles; the settled query span and the document spans for the rerank wire), the client's census rows
    attributed per row (a cut the client recorded is an input it had to shorten), and the max-seq facts.  The
    probe talks to the engine when ``base_url`` is given, to the product's offline fake otherwise.
    """
    census = TextTruncationCensus()
    client, capture = role_client(recipe, base_url, census=census)
    per_row: list[dict[str, Any]] = []
    max_tokens = recipe.client.max_tokens or 0

    for row in sampled:
        start = len(census.cuts())
        shapes = [str(row["shape"])] if "shape" in row else fitting.declared_shapes(recipe)
        entry: dict[str, Any] = {"shapes": {}}
        if recipe.role == "rerank":
            _probe_rerank(client, capture, row, entry)
        else:
            _probe_vectors(client, capture, row, shapes, entry)
        cuts = census.cuts()[start:]
        entry["over_cap"] = any(cut.original_tokens > max_tokens for cut in cuts)
        entry["cuts"] = len(cuts)
        per_row.append(entry)
    heads = _captured_heads(capture, sampled)
    return {
        "rows": per_row,
        "exchanges": len(capture.exchanges),
        "checked": sum(len(_probe_texts(entry)) for entry in per_row),
        "client": heads,
        "tokenizer": tokenizer.name,
    }


def _probe_rerank(client: Any, capture: Capture, row: dict[str, Any], entry: dict[str, Any]) -> None:
    """One rerank call per row: the client settles the query once and fits every pair; the captured bodies
    carry the settled query span and the document spans it ships."""
    start = len(capture.exchanges)
    client.rerank(row["query"], row["documents"], instruction=row.get("instruction"))
    queries: list[str] = []
    documents: list[str] = []
    for exchange in capture.exchanges[start:]:
        texts = capture.texts(exchange)
        if texts.get("query") is not None:
            queries.append(texts["query"])
        documents.extend(texts.get("documents", []))
    settled = queries[0] if queries else ""
    entry["shapes"]["pair"] = {"query": settled, "queries": queries, "documents": documents}


def _probe_vectors(
    client: Any, capture: Capture, row: dict[str, Any], shapes: list[str], entry: dict[str, Any]
) -> None:
    """One encode call per declared side: the captured ``input`` texts are the client's rendered prompts."""
    from rcp_ndcg_core.content import Content

    from rcp_ndcg.inference.types import EncodeRole

    def _captured(start: int) -> list[str]:
        return [text for exchange in capture.exchanges[start:] for text in capture.texts(exchange)["input"]]

    for shape in shapes:
        if shape == "pair":
            continue  # the embed roles have no pair wire; a rerank recipe owns that shape
        if shape == "query":
            start = len(capture.exchanges)
            client.encode([Content.from_text(row["query"])], EncodeRole.QUERY)
            entry["shapes"]["query"] = {"texts": _captured(start)}
        elif shape == "document":
            # One call per text: the client's fan-out runs concurrently, so the captured exchange order is a
            # completion order, not an input order -- per-call captures keep the position attribution exact.
            texts: list[str] = []
            for document in row["documents"]:
                start = len(capture.exchanges)
                client.encode([Content.from_text(document)], EncodeRole.DOCUMENT)
                texts.extend(_captured(start))
            entry["shapes"]["document"] = {"texts": texts}


def _content_ids(tokenizer: Any, body: str | list[int], flag: bool) -> list[int]:
    """A captured body's ids without the post-processor's tokens: a text body tokenized without them, a
    ``token_ids`` body (sent with the shape's ``flag``) with the post-processor's prefix and tail stripped
    from its edges (measured on a sentinel, as :func:`_post_processor_prefix` measures them)."""
    if isinstance(body, str):
        return list(tokenizer.ids(body, add_special_tokens=False))
    ids = list(body)
    if flag:
        prefix = _post_processor_prefix(tokenizer, "x")
        tail = _post_processor_tail(tokenizer, "x")
        if prefix and ids[: len(prefix)] == prefix:
            ids = ids[len(prefix) :]
        if tail and ids[len(ids) - len(tail) :] == tail:
            ids = ids[: len(ids) - len(tail)]
    return ids


def _head_of(body: str | list[int]) -> str | list[int]:
    """A sent body's head for a report: the text's first characters, or a ``token_ids`` body's first ids."""
    return body[:_SNIPPET] if isinstance(body, str) else list(body[:24])


def _probe_texts(entry: dict[str, Any]) -> list[str]:
    """Every text one probe entry carries, in order (the audit and the /tokenize check read them)."""
    texts: list[str] = []
    for shape_body in entry["shapes"].values():
        if "texts" in shape_body:
            texts.extend(shape_body["texts"])
        else:
            if shape_body.get("query"):
                texts.append(shape_body["query"])
            texts.extend(shape_body.get("documents", []))
    return texts


def _captured_heads(capture: Capture, sampled: list[dict[str, Any]]) -> dict[str, Any]:
    """The captured requests, summarised: per exchange the route and the texts' heads (no full bodies)."""
    heads: list[dict[str, Any]] = []
    for exchange in capture.exchanges[:8]:
        texts = capture.texts(exchange)
        heads.append(
            {
                "url": exchange["url"],
                "status": exchange["status"],
                "texts_head": [text[:_SNIPPET] for text in _flatten(texts)],
                **({"media": texts["media"]} if any(texts.get("media") or ()) else {}),
            }
        )
    return {"exchanges": len(capture.exchanges), "first": heads, "sampled": len(sampled)}


def _flatten(texts: dict[str, Any]) -> list[str]:
    """The texts dict as one ordered list (the query first, then the documents, or the input texts)."""
    if "input" in texts:
        return [str(text) for text in texts["input"]]
    out = [str(texts["query"])] if texts.get("query") is not None else []
    return out + [str(document) for document in texts.get("documents", [])]


# ---------------------------------------------------------------------------
# Stage 1's checks: the audit, the comparisons -- all on the captured requests.
# ---------------------------------------------------------------------------


def _anchor_check(recipe: Recipe, probe: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """The anchor audit over the captured requests: every anchor survives the client's cut, per shape.

    The embed roles render the full prompt on the wire: the audit asserts the declared edge (the fixed
    segment at the anchor side, plus the post-processor tokens ``add_special_tokens`` puts there) sits at its
    declared position of every captured body's ids -- an ``anchor: first`` head as the engine reads it in the
    assembled render (:func:`_head_edge_ids`), an ``anchor: marker`` shape's markers in the sent content
    without the post-processor's tokens (:func:`_content_ids`).  The rerank wire ships spans -- the engine assembles the
    frame -- so its audit asserts the client's settle-once: one query span per row, identical across the row's
    pointwise requests, within its declared ``query_max_tokens``, and no cut on an in-budget pair (a cut
    recorded in the client's census for a pair under budget would mean the client shortened something the
    budget allowed whole).
    """
    template = recipe.client.template
    failures: list[dict[str, Any]] = []
    checked = 0
    max_tokens = recipe.client.max_tokens or 0
    share = getattr(recipe.client, "query_max_tokens", None)
    for index, entry in enumerate(probe["rows"]):
        for shape, shape_body in entry["shapes"].items():
            if recipe.role == "rerank":
                checked += _audit_rerank_span(index, shape, shape_body, share, max_tokens, entry, failures, tokenizer)
                continue
            flag = _add_specials_flag(recipe, shape)
            at_start = template is not None and template.anchor == "first"
            edge = [] if at_start else _anchor_edge_ids(recipe, tokenizer, shape)
            head, prefix = _head_parts(recipe, tokenizer, shape) if at_start else ("", [])
            stable = _stable_head_tokens(tokenizer, head) if at_start else []
            for text in shape_body["texts"]:
                # A ``token_ids`` body carries the ids as sent: the client tokenized its render with the
                # shape's flag, so they already hold the edge and the post-processor's tokens (G1).
                ids = list(text) if isinstance(text, list) else tokenizer.ids(text, add_special_tokens=flag)
                checked += 1
                if template is None or template.anchor == "mean":
                    continue
                if template.anchor == "marker":
                    # Counted in the sent content only: a post-processor that adds the same special (an
                    # appended end token) must not stand in for a marker the client dropped.
                    content_ids = _content_ids(tokenizer, text, flag)
                    missing = sorted(
                        name for name in template.anchor_markers if tokenizer.special_id(name) not in content_ids
                    )
                    if missing:
                        failures.append({"shape": shape, "check": "markers", "missing_names": missing, "row": index})
                    continue
                if at_start:
                    # The head edge is the head's own tokens in the ASSEMBLED render (G3): where the head meets
                    # the content a byte-level BPE re-tokenizes across the join (a trailing space reads
                    # ``Ġdocument``, a Qwen-style ``:`` reads ``:Paris``), as the fit's assembled-render count
                    # allows, so the head's standalone ids are not what the engine reads.
                    head_edge = _head_edge_ids(tokenizer, head, prefix, stable, text)
                    actual = ids[: len(head_edge)] if head_edge is not None else []
                    if head_edge is None or not head_edge or actual != head_edge:
                        failures.append(
                            {
                                "shape": shape,
                                "check": "head",
                                "row": index,
                                "expected_head_text": head[:_SNIPPET],
                                "expected_edge_ids": head_edge,
                                "actual_edge_ids": actual,
                                "text": _head_of(text),
                            }
                        )
                    continue
                actual = ids[-len(edge) :] if edge else []
                if not edge or actual != edge:
                    failures.append(
                        {
                            "shape": shape,
                            "check": "tail",
                            "row": index,
                            "expected_edge_ids": edge,
                            "actual_edge_ids": actual,
                            "text": _head_of(text),
                        }
                    )
    if not checked:
        # An audit that read no input proves nothing: an extraction gap (a wire shape the capture does not
        # read) must fail here, never pass vacuously.
        failures.append(
            {
                "check": "nothing_checked",
                "note": "the anchor audit read no captured input: the client's requests carried no text the "
                "capture could extract",
            }
        )
    return {
        "anchor": template.anchor if template is not None else None,
        "checked": checked,
        "passed": not failures,
        "failures": failures,
        "referent": "the anchor ids (the declared edge plus the post-processor's tokens on that side) must sit "
        "at their declared positions in the client's rendered request; a reranker's settled query must be one "
        "span per row, within its declared share, and no in-budget pair may be cut",
    }


def _audit_rerank_span(
    row_index: int,
    shape: str,
    shape_body: dict[str, Any],
    share: int | None,
    max_tokens: int,
    entry: dict[str, Any],
    failures: list[dict[str, Any]],
    tokenizer: Any,
) -> int:
    """The rerank side of the anchor audit, on the captured spans: the settle-once query and the budget."""
    checked = 0
    queries = shape_body.get("queries") or ([shape_body["query"]] if shape_body.get("query") else [])
    if len(set(queries)) > 1:
        failures.append(
            {
                "shape": shape,
                "check": "settle_once",
                "row": row_index,
                "note": f"the client shipped {len(set(queries))} different query span(s) for one query; the "
                "shared query span settles once per request",
            }
        )
    query = queries[0] if queries else ""
    if query:
        checked += 1
        if share is not None and tokenizer.count(query) > share:
            failures.append(
                {
                    "shape": shape,
                    "check": "query_share",
                    "row": row_index,
                    "query_tokens": tokenizer.count(query),
                    "bound": share,
                    "text": query[:_SNIPPET],
                }
            )
    for document in shape_body.get("documents", []):
        checked += 1
        if not entry.get("over_length") and entry.get("cuts", 0) == 0 and tokenizer.count(document) > max_tokens:
            failures.append(
                {
                    "shape": shape,
                    "check": "document_over_budget",
                    "row": row_index,
                    "document_tokens": tokenizer.count(document),
                    "bound": max_tokens,
                    "text": document[:_SNIPPET],
                }
            )
    return checked


def _render_check(
    recipe: Recipe,
    reference_python: str | None,
    sampled: list[dict[str, Any]],
    probe: dict[str, Any],
    tokenizer: Any,
) -> dict[str, Any] | None:
    """The reference render comparison, via the reference subprocess (stage 1's reference side).

    Only the pairs file's rows are compared (the injected over-length samples are audited for the cut, not
    compared: the reference cuts over-cap inputs its own way by declaration).  The comparison is on the
    client's captured texts: the rendered prompts the embed roles send, the settled query span and the
    document spans for the rerank wire.  A ``token_ids`` body is compared on ids: the ids it sent against
    the reference text's ids under the shape's ``add_special_tokens`` flag (the product tokenizer, the ids
    the client would have sent for that text).  Under a declared over-cap deviation, over-cap rows
    are reported separately and do not gate (the reference cuts them differently by declaration).
    """
    if reference_python is None:
        return {"status": "not_run", "passed": None, "reason": "no --reference-python given"}
    recipe_dir = recipe._dir
    if recipe_dir is None:  # pragma: no cover - load_recipe sets it
        raise HarnessError(f"recipe {recipe.id} was not loaded from a directory; use load_recipe")
    entry = str(recipe_dir / recipe.reference.entry)
    user_rows = [row for row in sampled if not row.get("over_length")]
    served_by_key = _served_texts_by_row(recipe, probe, sampled)
    with tempfile.TemporaryDirectory() as work:
        pairs_path = Path(work) / "pairs.jsonl"
        out_path = Path(work) / "reference.json"
        _write_rows(user_rows, pairs_path)
        reference = run_reference(
            reference_python,
            entry,
            mode="render",
            pairs_path=pairs_path,
            out_path=out_path,
            tokenizer_spec=fitting.resolved_tokenizer_spec(recipe),
        )
    deviation = recipe.reference.over_cap_deviation is not None
    failures: list[dict[str, Any]] = []
    over_cap: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for row in reference.get("rows", []):
        key = (int(row["index"]), str(row.get("shape", "")))
        seen.add(key)
        served = served_by_key.get(key)
        if served is None:
            failures.append({"row": row, "note": "the reference rendered a row the harness did not sample"})
            continue
        # Over-cap rows the client had to cut (its census says so) compare differently by declaration.
        over = bool(probe["rows"][key[0]]["over_cap"])
        if recipe.role == "rerank":
            mismatches = _span_mismatches(row, served)
        elif isinstance(served, list):
            reference_text = str(row.get("text", ""))
            reference_ids = list(tokenizer.ids(reference_text, add_special_tokens=_add_specials_flag(recipe, key[1])))
            mismatches = []
            if reference_ids != list(served):
                mismatches.append(
                    {
                        "index": row["index"],
                        "shape": key[1],
                        "served_ids_head": list(served[:24]),
                        "reference_ids_head": reference_ids[:24],
                        "reference_text_head": reference_text[:_SNIPPET],
                        "text": str(row.get("query", ""))[:_SNIPPET],
                    }
                )
        else:
            mismatches = []
            if row.get("text", "") != served:
                mismatches.append(
                    {
                        "index": row["index"],
                        "shape": key[1],
                        "served_text_head": served[:_SNIPPET],
                        "reference_text_head": row.get("text", "")[:_SNIPPET],
                        "text": str(row.get("query", ""))[:_SNIPPET],
                    }
                )
        if mismatches and over and deviation:
            over_cap.append({"index": key[0], "shape": key[1], "mismatches": mismatches})
        else:
            failures.extend(mismatches)
    for key in sorted(set(served_by_key) - seen):
        failures.append({"index": key[0], "shape": key[1], "note": "the reference did not render this declared shape"})
    summary: dict[str, Any] = {
        "status": "run",
        "rows": len(reference.get("rows", [])),
        "passed": not failures,
        "failures": failures,
    }
    if over_cap:
        summary["over_cap"] = {
            "known_deviation": deviation,
            "n_rows": len(over_cap),
            "gating": False,
            "rows": over_cap,
            "passed": True,
            "referent": "pairs-file rows whose uncut prompt exceeds client.max_tokens (what the client sent was "
            "shortened); under the declared over-cap deviation the reference cuts them its own way, "
            "so they are reported here instead of gated",
        }
    return summary


def _span_mismatches(row: dict[str, Any], served: dict[str, Any]) -> list[dict[str, Any]]:
    """The rerank render comparison: the reference's spans against the client's captured spans."""

    failures: list[dict[str, Any]] = []
    if row.get("query", "") != served.get("query"):
        failures.append(
            {
                "index": row["index"],
                "shape": "pair",
                "span": "query",
                "served_text_head": str(served.get("query", ""))[:_SNIPPET],
                "reference_text_head": str(row.get("query", ""))[:_SNIPPET],
                "text": str(row.get("query", ""))[:_SNIPPET],
            }
        )
    reference_documents = list(row.get("documents", []))
    served_documents = list(served.get("documents", []))
    if len(reference_documents) != len(served_documents):
        failures.append(
            {
                "index": row["index"],
                "shape": "pair",
                "span": "documents",
                "note": f"the client shipped {len(served_documents)} document span(s), the reference "
                f"rendered {len(reference_documents)}",
            }
        )
        return failures
    for position, (served_document, reference_document) in enumerate(
        zip(served_documents, reference_documents, strict=True)
    ):
        if served_document != reference_document:
            failures.append(
                {
                    "index": row["index"],
                    "shape": "pair",
                    "span": f"document {position}",
                    "served_text_head": served_document[:_SNIPPET],
                    "reference_text_head": str(reference_document)[:_SNIPPET],
                    "text": str(row.get("query", ""))[:_SNIPPET],
                }
            )
    return failures


def _served_texts_by_row(
    recipe: Recipe, probe: dict[str, Any], sampled: list[dict[str, Any]]
) -> dict[tuple[int, str], Any]:
    """The client's captured texts per pairs row, keyed by (row index, shape).

    A pairs row without a ``shape`` is captured under every declared shape (the reference contract renders
    one row per declared shape at the row's index); a row that declares its shape is captured under that one.
    A rerank row's value is ``{"query": str, "documents": [...]}`` (the spans); an embed row's is the rendered
    prompt string, or a ``token_ids`` body's sent ids (a list of ints).
    """
    out: dict[tuple[int, str], Any] = {}
    for index, entry in enumerate(probe["rows"]):
        if sampled[index].get("over_length"):
            continue
        for shape, shape_body in entry["shapes"].items():
            if recipe.role == "rerank":
                out[(index, shape)] = {"query": shape_body.get("query"), "documents": shape_body.get("documents", [])}
            else:
                # The contract renders one text per (row, shape): the shape's first input (the row's query for
                # the query shape, its first document for the document shape).
                texts = shape_body.get("texts", [])
                out[(index, shape)] = texts[0] if texts else ""
    return out


def _template_check(
    recipe: Recipe, rows: list[dict[str, Any]], probe: dict[str, Any], tokenizer: Any
) -> dict[str, Any] | None:
    """When ``serve.chat_template`` is set: the template file's render must equal the declared template's.

    Every declared shape is checked (the engine renders each of them): the file is rendered with the engine's
    own jinja2 settings over the first pairs row's inputs, and the declared template's render
    (:meth:`~rcp_ndcg.data.templates.TemplateSpec.render`, the string the client sends) must be byte-identical.
    """
    if recipe.serve.chat_template is None or recipe.client.template is None or not rows:
        return None
    directory = recipe._dir
    if directory is None:  # pragma: no cover - load_recipe sets it
        raise HarnessError(f"recipe {recipe.id} was not loaded from a directory")
    template_text = (directory / recipe.serve.chat_template).read_text(encoding="utf-8")
    row = rows[0]
    failures: list[dict[str, Any]] = []
    for shape in fitting.declared_shapes(recipe):
        jinja_text = (
            _jinja_environment()
            .from_string(template_text)
            .render(
                query=str(row.get("query", "")),
                document=str(row["documents"][0]) if row.get("documents") else "",
                instruction=row.get("instruction") or "",
            )
        )
        declared_text = recipe.client.template.render(
            cast_shape(shape),
            tokenizer,
            query=str(row.get("query", "")),
            document=row["documents"][0] if row.get("documents") else "",
            instruction=row.get("instruction") or "",
        )
        if jinja_text != declared_text:
            failures.append(
                {
                    "shape": shape,
                    "jinja_head": jinja_text[:_SNIPPET],
                    "declared_head": declared_text[:_SNIPPET],
                }
            )
    return {
        "template": recipe.serve.chat_template,
        "checked": len(failures) and len(fitting.declared_shapes(recipe)) or len(fitting.declared_shapes(recipe)),
        "passed": not failures,
        "failures": failures,
        "referent": "the served template file must render exactly what the recipe's declared template renders "
        "(the client's budget arithmetic assumes that frame)",
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
    return ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, undefined=StrictUndefined)


def _engine_tokenize_check(
    recipe: Recipe, probe: dict[str, Any], tokenizer: Any, base_url: str | None
) -> dict[str, Any] | None:
    """The engine is the tokenization truth (R29): ``/tokenize`` of the client's texts must equal the ids.

    With an engine URL, every captured text (the rendered prompt for the embed roles, the settled query span
    and the document spans for the rerank wire) goes to the engine's ``/tokenize``; the ids and the count must
    equal the recipe tokenizer's for the same text and flag.  Without an engine the check is reported
    ``not_run`` — never as passed.
    """
    if base_url is None:
        return {
            "status": "not_run",
            "passed": None,
            "reason": "no engine URL; the /tokenize check runs only against a live engine",
        }
    failures: list[dict[str, Any]] = []
    checked = 0
    sent_as_ids = 0
    for shape, body in _captured_texts_per_shape(recipe, probe).items():
        add_flag = _add_specials_flag(recipe, shape) if recipe.role != "rerank" else False
        for text in body:
            if isinstance(text, list):
                sent_as_ids += 1  # a token_ids body: the engine reads these ids as sent and tokenizes nothing
                continue
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
    if sent_as_ids and not checked:
        return {
            "status": "not_run",
            "passed": None,
            "reason": "request_shape token_ids: the client sends ids, so the engine tokenizes no text "
            "(the anchor audit reads the sent ids)",
        }
    return {
        "status": "run",
        "checked": checked,
        "passed": not failures,
        "failures": failures,
        "referent": "the engine's /tokenize ids and count of the client's captured request texts, per shape",
    }


def _captured_texts_per_shape(recipe: Recipe, probe: dict[str, Any]) -> dict[str, list[str]]:
    """The captured texts grouped per shape (the audit and the /tokenize check read them in one place)."""
    per_shape: dict[str, list[str]] = {}
    for entry in probe["rows"]:
        for shape, shape_body in entry["shapes"].items():
            texts = shape_body.get("texts")
            if texts is None:
                texts = ([shape_body["query"]] if shape_body.get("query") else []) + list(
                    shape_body.get("documents", [])
                )
            per_shape.setdefault(shape, []).extend(texts)
    return per_shape


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
# The anchor edge: the declared template's fixed positions, measured, not assumed.
# ---------------------------------------------------------------------------


def _post_processor_tail(tokenizer: Any, text: str) -> list[int]:
    """The tokens ``add_special_tokens: true`` adds after the content, for the shape's tokenizer.

    Measured, not assumed: the ids of ``text`` with the post-processor are aligned against its plain ids and
    the tail after the content's last occurrence is the post-processor's suffix — empty for a prepend-only
    post-processor, the end token for the common append-only one.  When the text's own render is empty or
    displaced (a template literal that happens to contain the specials), the tail is measured on a sentinel.
    """
    plain, full, index = _aligned_ids(tokenizer, text)
    if index is None:
        probe_plain, probe_full, probe_index = _aligned_ids(tokenizer, "x")
        if probe_index is None:  # pragma: no cover - a post-processor that both displaces and reshapes
            return full[len(plain) :]
        return probe_full[probe_index + len(probe_plain) :]
    return full[index + len(plain) :]


def _post_processor_prefix(tokenizer: Any, text: str) -> list[int]:
    """The tokens ``add_special_tokens: true`` adds before the content, for the shape's tokenizer.

    The mirror of :func:`_post_processor_tail`: the ids before the content's last occurrence — the leading
    special (for example a CLS head's or the processor's bos) for a prepend-style post-processor, empty for an
    append-only one.  Measured on a sentinel when the text's own render is empty or displaced.
    """
    plain, full, index = _aligned_ids(tokenizer, text)
    if index is None:
        probe_plain, probe_full, probe_index = _aligned_ids(tokenizer, "x")
        if probe_index is None:  # pragma: no cover - a post-processor that both displaces and reshapes
            return full[: len(full) - len(plain)]
        return probe_full[:probe_index]
    return full[:index]


def _aligned_ids(tokenizer: Any, text: str) -> tuple[list[int], list[int], int | None]:
    """The text's ids without and with the post-processor, plus where the plain ids end in the full render."""
    plain = list(tokenizer.ids(text, add_special_tokens=False))
    full = list(tokenizer.ids(text, add_special_tokens=True))
    return plain, full, _last_sublist(full, plain) if plain else None


def _last_sublist(haystack: list[int], needle: list[int]) -> int | None:
    """The last index where ``needle`` occurs in ``haystack`` contiguously, or ``None`` (empty needle too)."""
    if not needle or len(needle) > len(haystack):
        return None
    for start in range(len(haystack) - len(needle), -1, -1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return None


def _anchor_edge_ids(recipe: Recipe, tokenizer: Any, shape: Any) -> list[int]:
    """The anchor ids of one shape: the fixed segment's rendered ids at the anchor edge, plus the
    post-processor tokens the shape's ``add_special_tokens`` flag puts on that edge.

    The edge is whichever segment sits at it: the tail (or head) segment when it is fixed, else — the escape
    hatch the product validator endorses, where the anchor IS the post-processor's own special token — the
    post-processor's tail (or prefix) alone.
    """
    template = recipe.client.template
    if template is None:
        return []
    segments = template.segments(shape)
    first = template.anchor == "first"
    edge_segment = segments[0] if first else segments[-1] if segments else None
    edge_fixed = edge_segment is not None and edge_segment.fixed is not None
    if first:
        head = next((segment for segment in segments if segment.fixed is not None), None)
        text = head.render(tokenizer) if head is not None else ""
    else:
        fixed = [segment for segment in segments if segment.fixed is not None]
        text = fixed[-1].render(tokenizer) if fixed else ""
    ids = list(tokenizer.ids(text, add_special_tokens=False)) if edge_fixed else []
    if template.adds_special_tokens(shape):
        # The edge id sits on the edge the post-processor decorates: a fixed segment plus the processor's
        # tokens on that side, or the processor's tokens alone when the fixed segment is not the edge.
        if first:
            ids = (*_post_processor_prefix(tokenizer, text), *ids)
        else:
            ids = (*ids, *_post_processor_tail(tokenizer, text))
    return list(ids)


_JOIN_PROBES = ("a", "Z", "é", "中", "0", ".", ":", "'s", "(", "-", "_", " a", "  a", " ", "\n", "\t", "🙂")
"""The continuations a ``token_ids`` body's head edge is measured against (letters, digits, punctuation,
contractions, spaces, newlines, non-Latin scripts, emoji): its text is not on the wire, so the edge is the head
tokens that lie wholly inside the head in the assembled render with every one of them."""


def _head_parts(recipe: Recipe, tokenizer: Any, shape: Any) -> tuple[str, list[int]]:
    """An ``anchor: first`` shape's head: its fixed head segment's render ("" when the shape opens with
    content: the post-processor's prefix alone is the edge) and the ids the shape's ``add_special_tokens``
    flag puts before it (measured as :func:`_anchor_edge_ids` measures them)."""
    template = recipe.client.template
    if template is None:
        return "", []
    segments = template.segments(shape)
    fixed = next((segment for segment in segments if segment.fixed is not None), None)
    rendered = fixed.render(tokenizer) if fixed is not None else ""
    head = rendered if segments and segments[0].fixed is not None else ""
    prefix = _post_processor_prefix(tokenizer, rendered) if template.adds_special_tokens(shape) else []
    return head, list(prefix)


def _head_tokens_in(tokenizer: Any, render: str, head: str) -> list[int]:
    """The ids of ``render``'s leading tokens that lie wholly inside its first ``len(head)`` characters
    (the product tokenizer's offsets, no post-processor): the head's own tokens in that assembled render."""
    kept: list[int] = []
    for token_id, (_, end) in zip(tokenizer.ids(render), tokenizer.offsets(render), strict=True):
        if end > len(head):
            break
        kept.append(token_id)
    return kept


def _head_edge_ids(
    tokenizer: Any, head: str, prefix: list[int], stable: list[int], body: str | list[int]
) -> list[int] | None:
    """The head edge one captured body must open with: the post-processor's prefix, then the head's own tokens.

    A text body IS the assembled render: it must start with the head's characters (``None`` when it does
    not: the head was cut or changed), and the edge is its tokens lying wholly inside them.  A ``token_ids``
    body carries no text, so its edge is ``stable`` (:func:`_stable_head_tokens`): the head tokens that lie
    wholly inside the head in the assembled render with every :data:`_JOIN_PROBES` continuation.
    """
    if not head:
        return list(prefix)
    if isinstance(body, str):
        if not body.startswith(head):
            return None
        return [*prefix, *_head_tokens_in(tokenizer, body, head)]
    return [*prefix, *stable]


def _stable_head_tokens(tokenizer: Any, head: str) -> list[int]:
    """The head tokens that lie wholly inside the head in its assembled render with every probe continuation
    (their longest common prefix): the edge of a ``token_ids`` body, whose text is not on the wire."""
    if not head:
        return []
    stable = _head_tokens_in(tokenizer, head + _JOIN_PROBES[0], head)
    for probe in _JOIN_PROBES[1:]:
        run = _head_tokens_in(tokenizer, head + probe, head)
        common = 0
        while common < min(len(stable), len(run)) and stable[common] == run[common]:
            common += 1
        stable = stable[:common]
    return stable


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

    The engine is talked to through the product's role clients
    (:class:`~rcp_ndcg.inference.clients.EmbeddingClient`, ``PoolingClient``, ``RerankClient``), built from
    :func:`~rcp_ndcg_vllm.recipe.client_config` with the recipe's real budget: the client prompts, fits and
    settles every request exactly as the served path does -- for a reranker, the shared query span settles
    once per call.  The harness pre-fits nothing and clears no budget field; a recorder's raw request and
    reply bodies come from the capturing ``httpx`` transport handed to the client's transport (the product's
    injection point), never from a second request path.  The reference runs as a subprocess in its own
    environment (``--reference-python``, required); the harness process imports no torch.
    """
    rows = load_pairs(pairs_path)
    reference = _reference_outputs(recipe, reference_python, rows, device=device)
    gates = resolve_gates(recipe)
    if recipe.role == "rerank":
        return _rerank_stage2(recipe, rows, reference, base_url, gates, recorder)
    return _vector_stage2(recipe, rows, reference, base_url, gates, recorder)


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
            tokenizer_spec=fitting.resolved_tokenizer_spec(recipe),
            device=device,
        )


def _write_rows(rows: list[dict[str, Any]], path: str | Path) -> None:
    """The sampled rows as the pairs file the reference subprocess reads (the private markers dropped)."""
    public = [
        {key: value for key, value in row.items() if not key.startswith("_") and key != "over_length"} for row in rows
    ]
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in public), encoding="utf-8")


def _rerank_stage2(
    recipe: Recipe,
    rows: list[dict[str, Any]],
    reference: dict[str, Any],
    base_url: str,
    gates: Any,
    recorder: Any | None,
) -> dict[str, Any]:
    """Rerank scores from the served engine through the product's :class:`RerankClient`.

    One client call per row: the client folds the query per the config's instruction mode, settles the shared
    query span once, and fits every pair into the declared budget -- the wire carries exactly what the served
    path ships.  Over-cap pairs (the client recorded a cut in its census: the uncut prompt exceeds
    ``client.max_tokens``) are excluded from the gates and reported separately under a declared over-cap
    deviation; the Kendall tau covers the under-cap subset of every gated query.
    """
    deviation = recipe.reference.over_cap_deviation is not None
    census = TextTruncationCensus()
    client, capture = role_client(recipe, base_url, census=census)
    max_tokens = recipe.client.max_tokens or 0
    if len(rows) > len(reference.get("rows", [])):
        raise HarnessError(
            f"the reference emitted {len(reference.get('rows', []))} score row(s) for {len(rows)} pairs "
            "row(s): the comparison would silently drop the later rows -- fix the reference or the pairs file"
        )
    per_document: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    over_cap: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        reference_row = _reference_row(reference, row_index)
        reference_scores = [float(value) for value in reference_row.get("scores", [])]
        if len(reference_scores) != len(row["documents"]):
            raise HarnessError(
                f"the reference emitted {len(reference_scores)} score(s) for pairs row {row_index} with "
                f"{len(row['documents'])} document(s): scores align to the documents as given"
            )
        start = len(census.cuts())
        result = client.rerank(row["query"], row["documents"], instruction=row.get("instruction"))
        flags = _census_over_cap(census, start, max_tokens, len(row["documents"]))
        if len(result.scores) != len(reference_scores):
            raise HarnessError(
                f"the engine scored {len(result.scores)} document(s) for pairs row {row_index} whose "
                f"reference row carries {len(reference_scores)}: the request set must agree"
            )
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
                "over_cap": flags[document_index] if document_index < len(flags) else False,
            }
            if entry["over_cap"]:
                over_cap.append(entry)
            if entry["over_cap"] and deviation:
                continue
            per_document.append(entry)
        under_cap = [index for index, over in enumerate(flags) if not over]
        under_cap = [index for index in under_cap if index < len(result.scores)]
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
    if recorder is not None:
        recorder.extend(capture.exchanges)
    summary = _rerank_summary(per_document, per_query, gates, recipe.reference.score_scale)
    summary["over_cap"] = {
        "known_deviation": deviation,
        "n_pairs": len(over_cap),
        "gating": False,
        "pairs": over_cap,
        "passed": True,
        "referent": "pairs whose uncut prompt exceeds client.max_tokens, decided on the client's own census; "
        "served and reference may differ by design when reference.known_deviations declares an over-cap "
        "deviation (anchor_drop_over_cap or over_cap_cut_differs)",
    }
    return summary


def _reference_row(reference: dict[str, Any], row_index: int) -> dict[str, Any]:
    """The reference's output for the row at ``row_index``, matched by the contract's ``index`` field."""
    rows = reference.get("rows", [])
    for row in rows:
        if int(row.get("index", -1)) == row_index:
            return row
    return rows[row_index] if row_index < len(rows) else {}


def _census_over_cap(census: Any, start: int, max_tokens: int, n_documents: int) -> list[bool]:
    """Per document of one client call, whether the client recorded a cut over the budget (the client's own
    accounting: a ``text_budget`` cut whose whole input exceeds ``max_tokens`` means the pair was over cap)."""
    flags = [False] * n_documents
    for cut in census.cuts()[start:]:
        origin = cut.doc_id.split("#", 1)[0]  # a chunked document's rows carry <position>#<chunk>
        try:
            position = int(origin)
        except ValueError:
            continue  # the reranker's settled-query row (its own doc id), not a document's
        if 0 <= position < n_documents and cut.original_tokens > max_tokens:
            flags[position] = True
    return flags


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
    recorder: Any | None,
) -> dict[str, Any]:
    """Vectors from the served engine through the product's role clients, against the reference's vectors.

    A dense embedder's vectors compare with a cosine floor per vector; a late-interaction model's ragged
    token vectors compare per token (in the transfer precision the product's client applied on the wire).
    The client prompts and fits every text exactly as the served path does -- the harness pre-fits nothing.
    Under a declared over-cap deviation, the texts the client had to cut (a census cut over
    ``client.max_tokens``) are reported separately and do not gate: the reference renders them its own way by
    declaration.
    """
    deviation = recipe.reference.over_cap_deviation is not None
    per_vector: list[dict[str, Any]] = []
    over_cap: list[dict[str, Any]] = []
    census = TextTruncationCensus()
    client, capture = role_client(recipe, base_url, census=census)
    from rcp_ndcg.inference.types import EncodeRole

    for row_index, row in enumerate(rows):
        reference_row = _reference_row(reference, row_index)
        for role, served_key, texts in (
            ("query", "query_vectors", [row["query"]]),
            ("document", "document_vectors", list(row["documents"])),
        ):
            shape = "query" if role == "query" else "document"
            if shape not in fitting.declared_shapes(recipe):
                continue
            encode_role = EncodeRole.QUERY if role == "query" else EncodeRole.DOCUMENT
            from rcp_ndcg_core.content import Content

            served_matrices: list[list[list[float]]] = []
            cut_flags: list[bool] = []
            for text in texts:
                # One call per text: the client's fan-out runs concurrently, so per-call census windows keep
                # the position attribution exact.
                start = len(census.cuts())
                embeddings = client.encode([Content.from_text(text)], encode_role)
                served_matrices.extend(_embeddings_to_matrices(recipe, embeddings, 1))
                cut_flags.extend(_census_over_cap(census, start, recipe.client.max_tokens or 0, 1))
            expected = reference_row.get(served_key) or []
            _compare_shape(
                recipe,
                served_matrices,
                expected,
                row_index,
                role,
                per_vector,
                gates,
                cut_flags=cut_flags,
                deviation=deviation,
                over_cap=over_cap,
                row=row,
            )
    if recorder is not None:
        recorder.extend(capture.exchanges)
    summary = _vector_summary(recipe, per_vector, gates)
    summary["over_cap"] = {
        "known_deviation": deviation,
        "n_pairs": len(over_cap),
        "gating": False,
        "pairs": over_cap,
        "passed": True,
        "referent": "inputs whose uncut prompt exceeds client.max_tokens (the client shortened them, recorded "
        "in its census); under the declared over-cap deviation the reference renders them its own "
        "way, so they are reported here instead of gated",
    }
    return summary


def _embeddings_to_matrices(recipe: Recipe, embeddings: Any, n_texts: int) -> list[list[list[float]]]:
    """One matrix of vectors per text (dense or ragged), from the client's embeddings result."""
    matrix = np.asarray(embeddings.vectors)
    if embeddings.offsets is None:
        # Dense: one vector per input text, in the input's order (zeros where omit_zero omitted one).
        return [
            [[float(value) for value in row] for row in matrix[index : index + 1]]
            for index in range(min(n_texts, len(matrix)))
        ]
    out = []
    for index in range(min(n_texts, len(embeddings.offsets) - 1)):
        start, end = int(embeddings.offsets[index]), int(embeddings.offsets[index + 1])
        out.append([[float(value) for value in vector] for vector in matrix[start:end]])
    return out


def _compare_shape(
    recipe: Recipe,
    served_vectors: list[list[list[float]]],
    expected: list[Any],
    row_index: int,
    role: str,
    per_vector: list[dict[str, Any]],
    gates: Any,
    *,
    cut_flags: list[bool] | None = None,
    deviation: bool = False,
    over_cap: list[dict[str, Any]] | None = None,
    row: dict[str, Any] | None = None,
) -> None:
    """Pair served with reference vectors positionally, every text's rows (per vector, or per token).

    ``served_vectors`` carries one matrix per text (the client sent one request per text); ``expected`` one
    entry per text (a dense vector, or one ragged matrix per text for a late-interaction model).  Inputs the
    client cut (over cap) are recorded in ``over_cap`` and -- under the declared deviation -- reported
    separately instead of gated.
    """
    if len(served_vectors) != len(expected):
        per_vector.append(
            {
                "referent": f"row {row_index} {role} count",
                "cosine": None,
                "within": False,
                "note": f"the engine returned {len(served_vectors)} matrix/matri[ces], the reference {len(expected)}",
            }
        )
        return
    for index, (served_matrix, reference_entry) in enumerate(zip(served_vectors, expected, strict=True)):
        over = bool(cut_flags[index]) if cut_flags and index < len(cut_flags) else False
        # A reference entry is one vector (wrap it) or one ragged matrix per text (use it as given).
        expected_matrix = (
            reference_entry
            if isinstance(reference_entry, list) and (not reference_entry or isinstance(reference_entry[0], list))
            else [reference_entry]
        )
        if len(served_matrix) != len(expected_matrix):
            per_vector.append(
                {
                    "referent": f"row {row_index} {role} {index} count",
                    "cosine": None,
                    "within": False,
                    "over_cap": over,
                    "note": f"the engine returned {len(served_matrix)} vector(s), the reference {len(expected_matrix)}",
                }
            )
            continue
        rows = []
        for position, (served_vector, reference_vector) in enumerate(zip(served_matrix, expected_matrix, strict=True)):
            cosine = _cosine(
                np.asarray(served_vector, dtype=np.float64), np.asarray(reference_vector, dtype=np.float64)
            )
            rows.append(
                {
                    "referent": f"row {row_index} {role} {index} vector {position}",
                    "cosine": cosine,
                    "within": bool(cosine >= gates.vec_min_cosine),
                    "over_cap": over,
                }
            )
        per_vector.extend([] if (over and deviation) else rows)
        if over and deviation and over_cap is not None:
            over_cap.append(
                {
                    "referent": f"row {row_index} {role} {index}",
                    "query": str((row or {}).get("query", ""))[:_SNIPPET],
                    "n_vectors": len(rows),
                }
            )
        elif over and over_cap is not None:
            # Not declared: the over-cap input stays in the gates AND is named in the table.
            over_cap.append(
                {
                    "referent": f"row {row_index} {role} {index}",
                    "query": str((row or {}).get("query", "")),
                    "n_vectors": len(rows),
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
