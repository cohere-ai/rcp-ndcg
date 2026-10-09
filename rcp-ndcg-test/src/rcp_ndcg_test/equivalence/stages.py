"""The three-stage equivalence harness, driven through the product's role clients.

Stage 1 samples the pairs file, probes the recipe's role client for every sampled input through the product's
injection point (a capturing ``httpx`` transport), and audits what the client actually sends: the anchor audit
and the engine's ``/tokenize`` read the captured request bodies, the reference subprocess's ``render`` is
compared against them, and the served template file is rendered against them.  The inputs the client changed
(a cut of any cause: the budget counted with the frame, the reranker's query share, a declared per-shape cap)
are read from the client's own processing records (:attr:`RoleClient.processing`, one per changed row, each
change named by its mechanism) and, under a declared over-cap deviation,
reported in a separate non-gating table; an input the client sent uncut gates exactly.  Stage 2 sends the
reference's pairs through the same clients and gates the answers against the reference subprocess's outputs.
Stage 3 scores rankings with ``rcp-ndcg eval score``.  The harness never re-derives a render, a cut or a
settlement.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from rcp_ndcg_vllm.recipe import Recipe

from rcp_ndcg_test.errors import HarnessError

from . import fitting
from .fitting import load_pairs
from .gates import kendall_tau_b, resolve_gates
from .reference import run_reference
from .wire import Capture, role_client

__all__ = [
    "CHECKPOINT_TEMPLATE_FILES",
    "checkpoint_chat_template",
    "engine_conversation",
    "load_pairs",
    "render_chat",
    "served_chat_template",
    "stage1_prompts",
    "stage2_scores",
]

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
      client's cut, asserted on the captured requests and the client's processing records; for a reranker this
      is the settle-once query (one settled span per row, within its declared share, every document span within its
      declared ``document_max_tokens``, no cut on an in-budget pair);
    - ``render_check`` — the reference subprocess's ``render`` output against the captured texts, zero
      tolerance (needs ``--reference-python``; reported ``not_run`` without one).  Under a declared
      over-cap deviation, over-cap rows are reported separately and do not gate;
    - ``template_render_check`` — when ``serve.chat_template`` is set: the template file's jinja2 render (the
      engine's settings) of every declared shape against the client's render of the same inputs; on the
      ``messages`` route the served chat template (the checkpoint's own at the pinned revision when the recipe
      serves none) rendered over every captured conversation against the declared frame;
    - ``engine_tokenize_check`` — with an engine URL: the engine's ``/tokenize`` of every captured text must
      equal the recipe tokenizer's ids; reported ``not_run`` without an engine, never as passed.
    """
    from .media import text_rows

    tokenizer = fitting.tokenizer_of(recipe)
    loaded = load_pairs(pairs_path)
    rows = text_rows(loaded)  # the media rows are the media stage's (rcp_ndcg_test.equivalence.media)
    shown = rows if limit is None else rows[:limit]
    sampled = _sampled_rows(recipe, shown, tokenizer, over_length_per_shape)
    probe = _probe(recipe, sampled, base_url, tokenizer)
    document: dict[str, Any] = {
        "pairs": len(rows),
        "media_rows": len(loaded) - len(rows),
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
            padded_query = _over_length(seed_query, recipe.client.get("max_tokens"), tokenizer, index)
            padded_document = _over_length(seed_document, recipe.client.get("max_tokens"), tokenizer, index)
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
    """A seed text padded to at least twice ``max_tokens`` tokens (128 without a budget), in whole words.

    The padding appends `` pad<index>`` words, so each sample index pads differently.  Raises
    ``HarnessError`` when the recipe tokenizer's count never reaches the target within the bounded passes
    (a count that saturates at an embedded truncation ceiling): an over-length sample that was never
    measured over the target would audit an uncut input as if the client had cut it.
    """
    budget = max_tokens or 128
    marker = f" pad{index}"
    # One probe measures the marker's token rate, and each step sizes the append from the measured
    # deficit, so the loop converges in at most a few passes. The bound is what keeps the sampler
    # linear in the padded length: the old per-step re-count of the GROWING text (and its
    # word-count heuristic) re-tokenized a 2x-budget string O(steps) times -- a token-count storm
    # at 32768-token budgets (jina-embeddings-v5-text-small's stage-1 sample sat in tokenizer.count).
    unit_tokens = max(1, tokenizer.count(marker * 8))
    text = seed or "anchor"
    for _ in range(8):  # declarative bound: at most 8 measured passes, each sized from the measured deficit
        deficit = budget * 2 - tokenizer.count(text)
        if deficit <= 0:
            return text
        text = text + marker * max(2, (deficit * 8) // unit_tokens + 2)
    counted = tokenizer.count(text)
    if counted < budget * 2:
        raise HarnessError(
            f"cannot build an over-length sample of {budget * 2} tokens: the recipe tokenizer counts {counted} "
            "tokens after the bounded padding passes (does the tokenizer file carry a truncation ceiling?)"
        )
    return text


def _add_specials_flag(recipe: Recipe, shape: str) -> bool:
    """The shape's ``add_special_tokens`` flag (the engine's post-processor behaviour, declared)."""
    template = fitting.client_template(recipe)
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
    embed roles; the settled query span and the document spans for the rerank wire) and the client's
    processing records of the row (:attr:`~rcp_ndcg.inference.clients._base.RoleClient.processing`, one per
    input the client changed): ``changes`` those records (each change's mechanism, the uncut and kept request
    totals, the budget), ``over_cap`` whether the row holds any.  Gating is per TEXT, never per row: each shape
    body carries which of its texts the client changed -- ``changed`` per text for the embed roles;
    ``query_changed`` and ``documents_changed`` for a reranker, whose shared-query settlement changes the
    query span of every pair of its row and whose document records change only their document.  The probe
    talks to the engine when ``base_url`` is given, to the product's offline fake otherwise.
    """
    client, capture = role_client(recipe, base_url)
    per_row: list[dict[str, Any]] = []

    for row in sampled:
        start = len(client.processing)
        shapes = [str(row["shape"])] if "shape" in row else fitting.declared_shapes(recipe)
        entry: dict[str, Any] = {"shapes": {}}
        if recipe.role == "rerank":
            _probe_rerank(client, capture, row, entry)
        else:
            _probe_vectors(client, capture, row, shapes, entry, tokenizer)
        records = [record for record in client.processing[start:] if record.changed]
        entry["over_cap"] = bool(records)
        entry["cuts"] = len(records)
        entry["changes"] = [record.as_row() for record in records]
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
    carry the settled query span and the document spans it ships, and the call's processing records say
    which of them the client changed (the settlement: the query span; a document's record: that document)."""
    from rcp_ndcg.inference.clients.rerank import QUERY_DOC_ID

    start = len(capture.exchanges)
    records_start = len(client.processing)
    client.rerank(row["query"], row["documents"], instruction=row.get("instruction"))
    changed_ids = {record.input_id for record in client.processing[records_start:] if record.changed}
    unattributed = any(not (input_id.isdigit() or input_id == QUERY_DOC_ID) for input_id in changed_ids)
    queries: list[str] = []
    documents: list[str] = []
    for exchange in capture.exchanges[start:]:
        texts = capture.texts(exchange)
        if texts.get("query") is not None:
            queries.append(texts["query"])
        documents.extend(texts.get("documents", []))
    settled = queries[0] if queries else ""
    entry["shapes"]["pair"] = {
        "query": settled,
        "queries": queries,
        "documents": documents,
        "query_changed": QUERY_DOC_ID in changed_ids or unattributed,
        "documents_changed": [
            unattributed or str(position) in changed_ids for position in range(len(row["documents"]))
        ],
    }


def _probe_vectors(
    client: Any, capture: Capture, row: dict[str, Any], shapes: list[str], entry: dict[str, Any], tokenizer: Any
) -> None:
    """One encode call per declared side: the captured ``input`` texts are the client's rendered prompts.

    On the ``messages`` route the client sends the content and the engine's chat template frames it, so the
    rendered prompt is the declared frame around the captured content -- the product's one render
    (:func:`rcp_ndcg.data.preprocess.rendered_request`, under the client's own budget), which the template
    check holds the served chat template to -- and the captured conversations are kept for that check.
    """
    from rcp_ndcg_core.content import Content

    from rcp_ndcg.data.preprocess import rendered_request
    from rcp_ndcg.inference.types import EncodeRole

    budget = getattr(client, "text_budget", None)

    def _captured(start: int, shape: str, conversations: list[Any], generation: list[bool]) -> list[Any]:
        texts: list[Any] = []
        for exchange in capture.exchanges[start:]:
            captured = capture.texts(exchange)
            if "conversations" not in captured:
                texts.extend(captured["input"])
                continue
            conversations.extend(captured["conversations"])
            generation.extend([captured["add_generation_prompt"]] * len(captured["conversations"]))
            for content in captured["input"]:
                texts.append(
                    rendered_request(budget, tokenizer, fitting.cast_shape(shape), query=content, document=content)
                    if budget is not None
                    else content
                )
        return texts

    for shape in shapes:
        if shape == "pair":
            continue  # the embed roles have no pair wire; a rerank recipe owns that shape
        conversations: list[Any] = []
        generation: list[bool] = []
        inputs = [row["query"]] if shape == "query" else list(row["documents"])
        role = EncodeRole.QUERY if shape == "query" else EncodeRole.DOCUMENT
        texts: list[Any] = []
        changed: list[bool] = []
        # One call per text: the client's fan-out runs concurrently, so the captured exchange order is a
        # completion order, not an input order -- per-call captures keep the position attribution exact, and
        # the call's processing records say whether the client changed that one text.
        for text in inputs:
            start, records_start = len(capture.exchanges), len(client.processing)
            client.encode([Content.from_text(text)], role)
            sent = _captured(start, shape, conversations, generation)
            texts.extend(sent)
            changed.extend([any(record.changed for record in client.processing[records_start:])] * len(sent))
        entry["shapes"][shape] = {
            "texts": texts,
            "changed": changed,
            **({"conversations": conversations, "add_generation_prompt": generation} if conversations else {}),
        }


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
    without the post-processor's tokens (:func:`_content_ids`), an ``anchor: last_content`` shape's head
    marker and its last content token (:func:`_audit_last_content`).  The rerank wire ships spans -- the
    engine assembles the frame -- so its audit asserts the client's settle-once: one query span per row,
    identical across the row's pointwise requests, within its declared ``query_max_tokens``, every document
    span within its declared ``document_max_tokens``, and no cut on an
    in-budget pair (a change the client recorded for a pair under budget would mean the client
    shortened something the budget allowed whole).
    """
    template = fitting.client_template(recipe)
    failures: list[dict[str, Any]] = []
    checked = 0
    max_tokens = recipe.client.get("max_tokens") or 0
    share = recipe.client.get("query_max_tokens")
    document_cap = recipe.client.get("document_max_tokens")
    for index, entry in enumerate(probe["rows"]):
        for shape, shape_body in entry["shapes"].items():
            if recipe.role == "rerank":
                checked += _audit_rerank_span(
                    index, shape, shape_body, (share, document_cap), max_tokens, entry, failures, tokenizer
                )
                continue
            flag = _add_specials_flag(recipe, shape)
            if template is not None and template.anchor == "last_content":
                checked += _audit_last_content(recipe, tokenizer, shape, shape_body["texts"], index, failures)
                continue
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
        "span per row, within its declared share, every document span within its declared cap, and no in-budget "
        "pair may be cut",
    }


def _audit_last_content(
    recipe: Recipe,
    tokenizer: Any,
    shape: str,
    bodies: list[Any],
    row_index: int,
    failures: list[dict[str, Any]],
) -> int:
    """The ``anchor: last_content`` audit of one shape's captured bodies, per the product's definition
    (:data:`rcp_ndcg.data.templates.AnchorKind`): the model reads the last kept *content* token, and the fixed
    segments (a head marker) are still reserved and audited.

    Per body: the shape must end on its content span (a fixed tail would be the last token, not content); the
    head -- the fixed segments before the content, as the engine reads them in the assembled render, after the
    post-processor's prefix (:func:`_head_edge_ids`) -- must open the ids; the post-processor's tail (when the
    shape declares ``add_special_tokens``) must close them; and at least one token must sit between the two,
    which on a content-final shape is the last kept content token (the token a head marker merges into across
    the join counts as content: it carries the content's first characters).  Returns the bodies checked.
    """
    template = fitting.client_template(recipe)
    assert template is not None  # the caller branches on the template's anchor
    segments = template.segments(fitting.cast_shape(shape))
    flag = _add_specials_flag(recipe, shape)
    if segments[-1].content is None:
        failures.append(
            {
                "shape": shape,
                "check": "content_final",
                "row": row_index,
                "note": "anchor: last_content reads the last kept content token, but the declared shape ends "
                "with a fixed segment: its last token is the frame's",
            }
        )
        return len(bodies)
    leading = segments[: next(position for position, segment in enumerate(segments) if segment.content is not None)]
    head = "".join(segment.render(tokenizer) for segment in leading)
    prefix = _post_processor_prefix(tokenizer, "x") if flag else []
    tail = _post_processor_tail(tokenizer, "x") if flag else []
    stable = _stable_head_tokens(tokenizer, head)
    for body in bodies:
        ids = list(body) if isinstance(body, list) else list(tokenizer.ids(body, add_special_tokens=flag))
        head_edge = _head_edge_ids(tokenizer, head, list(prefix), stable, body)
        problem = None
        if head_edge is None or ids[: len(head_edge)] != head_edge:
            problem = "head"
        elif tail and ids[len(ids) - len(tail) :] != tail:
            problem = "tail"
        elif len(ids) - len(tail) <= len(head_edge):
            problem = "no_content_token"
        if problem is not None:
            failures.append(
                {
                    "shape": shape,
                    "check": problem,
                    "row": row_index,
                    "expected_head_text": head[:_SNIPPET],
                    "expected_head_ids": head_edge,
                    "expected_tail_ids": list(tail),
                    "actual_ids_head": ids[:24],
                    "actual_ids_tail": ids[-8:],
                    "text": _head_of(body),
                }
            )
    return len(bodies)


def _audit_rerank_span(
    row_index: int,
    shape: str,
    shape_body: dict[str, Any],
    caps: tuple[int | None, int | None],
    max_tokens: int,
    entry: dict[str, Any],
    failures: list[dict[str, Any]],
    tokenizer: Any,
) -> int:
    """The rerank side of the anchor audit, on the captured spans: the settle-once query, the declared caps
    (``caps``: the query's share and the per-document cap) and the budget."""
    share, document_cap = caps
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
        if document_cap is not None and tokenizer.count(document) > document_cap:
            failures.append(
                {
                    "shape": shape,
                    "check": "document_share",
                    "row": row_index,
                    "document_tokens": tokenizer.count(document),
                    "bound": document_cap,
                    "text": document[:_SNIPPET],
                }
            )
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
        entry = probe["rows"][key[0]]
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
        # Per text, never per row: only a text the client changed (its processing records say so) compares
        # differently by declaration; every other text of the row gates exactly.
        reported = [mismatch for mismatch in mismatches if deviation and _text_changed(entry, key[1], mismatch)]
        failures.extend(mismatch for mismatch in mismatches if mismatch not in reported)
        if reported:
            over_cap.append(
                {"index": key[0], "shape": key[1], "mismatches": reported, "changes": entry.get("changes", [])}
            )
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
            "referent": "pairs-file rows the client changed relative to the uncut input (its processing records "
            "name each change: a budget cut counted with the frame, the query share, a declared per-document cap, "
            "an empty-document substitution, a media resize or drop); under the "
            "declared over-cap deviation the reference cuts them its own way, so they are reported here instead "
            "of gated",
        }
    return summary


def _text_changed(entry: dict[str, Any], shape: str, mismatch: dict[str, Any]) -> bool:
    """Whether the client changed the one text a render mismatch is about (the probe's per-text flags).

    A rerank mismatch names its span, and the unit the client changes is the pair: the row's shared-query
    settlement changes every pair of the row -- its query span and, with it, each pair the reference renders
    its own way by declaration (a whole-prompt right cut that drops the document of an over-cap query, say),
    exactly as stage 2 counts every pair of a settled row changed -- while a document's own record changes only
    its pair (``document <i>``); a document-count mismatch (a chunked or omitted document) counts when any
    document or the query changed.  An embed mismatch is about the shape's first input (the one the reference
    contract renders)."""
    body = entry["shapes"].get(shape, {})
    span = str(mismatch.get("span", ""))
    if "query_changed" in body:
        settled = bool(body["query_changed"])
        documents = list(body.get("documents_changed", []))
        if span == "query":
            return settled
        if span.startswith("document "):
            position = int(span.removeprefix("document "))
            return settled or (bool(documents[position]) if position < len(documents) else any(documents))
        return settled or any(documents)
    changed = list(body.get("changed", []))
    return bool(changed[0]) if changed else False


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
    A ``messages`` recipe is checked on its captured conversations instead (:func:`_messages_template_check`).
    """
    if recipe.client.get("request_shape", "text") == "messages" and fitting.client_template(recipe) is not None:
        return _messages_template_check(recipe, probe)
    if recipe.serve.chat_template is None or fitting.client_template(recipe) is None or not rows:
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
        declared_template = fitting.client_template(recipe)
        assert declared_template is not None  # the guard above returned on the None case
        declared_text = declared_template.render(
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


def _messages_template_check(recipe: Recipe, probe: dict[str, Any]) -> dict[str, Any]:
    """The ``messages`` route's frame check: the engine frames each sent conversation exactly once.

    vLLM v0.31.0 renders every chat-shaped ``/embeddings`` request through the served chat template
    (vllm/entrypoints/pooling/embed/io_processor.py:302-355, with the request's ``add_generation_prompt``,
    false by default, vllm/entrypoints/pooling/base/protocol.py:230-237), and the client sends the content
    only.  So the served template file, rendered with transformers' jinja2 settings over every captured
    conversation (its content parts as the engine hands them to the template, :func:`engine_conversation`)
    and with the flag that request carried, must equal the declared template's render of the same content --
    the frame the client's budget reserved, once.  Without ``serve.chat_template`` the engine renders the
    checkpoint's own template, which the check reads at the pinned revision
    (:func:`checkpoint_chat_template`); one that cannot be read fails the check (``unresolved``), never passes.
    """
    try:
        source, template_text = served_chat_template(recipe)
    except HarnessError as error:
        return {
            "status": "unresolved",
            "passed": False,
            "failures": [{"check": "checkpoint_chat_template", "note": str(error)}],
            "reason": "the engine frames the content with the checkpoint's own chat template, and it could "
            "not be read: the frame is unchecked, which never passes",
        }
    template = _jinja_environment(strict=False).from_string(template_text)
    failures: list[dict[str, Any]] = []
    checked = 0
    for index, entry in enumerate(probe["rows"]):
        for shape, shape_body in entry["shapes"].items():
            flags = shape_body.get("add_generation_prompt", [])
            for position, (conversation, declared) in enumerate(
                zip(shape_body.get("conversations", []), shape_body.get("texts", []), strict=False)
            ):
                checked += 1
                engine = template.render(
                    messages=engine_conversation(conversation),
                    add_generation_prompt=bool(flags[position]) if position < len(flags) else False,
                    tools=None,
                )
                if engine != declared:
                    failures.append(
                        {
                            "shape": shape,
                            "row": index,
                            "engine_head": engine[:_SNIPPET],
                            "declared_head": str(declared)[:_SNIPPET],
                        }
                    )
    if not checked:
        failures.append({"check": "nothing_checked", "note": "no captured conversation to render"})
    return {
        "template": source,
        "template_sha256": hashlib.sha256(template_text.encode("utf-8")).hexdigest(),
        "status": "run",
        "checked": checked,
        "passed": not failures,
        "failures": failures,
        "referent": "the served chat template (serve.chat_template, else the checkpoint's own at the pinned "
        "revision), rendered over every conversation the client sent (its content), must render exactly the "
        "declared template's frame around it -- framed once",
    }


def served_chat_template(recipe: Recipe) -> tuple[str, str]:
    """The chat template the engine frames a chat-shaped request with: ``serve.chat_template``'s file, else the
    checkpoint's own at the pinned revision (:func:`checkpoint_chat_template`).  Output: ``(source, text)``."""
    if recipe.serve.chat_template is None:
        return checkpoint_chat_template(recipe)
    directory = recipe._dir
    if directory is None:  # pragma: no cover - load_recipe sets it
        raise HarnessError(f"recipe {recipe.id} was not loaded from a directory")
    return recipe.serve.chat_template, (directory / recipe.serve.chat_template).read_text(encoding="utf-8")


def render_chat(template_text: str, conversation: list[Any], *, add_generation_prompt: bool) -> str:
    """One sent conversation rendered as the engine renders it: the template under transformers' jinja2
    settings, the parts as vLLM hands them (:func:`engine_conversation`), the request's generation flag."""
    template = _jinja_environment(strict=False).from_string(template_text)
    return template.render(
        messages=engine_conversation(conversation), add_generation_prompt=add_generation_prompt, tools=None
    )


CHECKPOINT_TEMPLATE_FILES = ("chat_template.jinja", "chat_template.json", "tokenizer_config.json")
"""Where a checkpoint carries its chat template, in the order the engine's resolution reads them: vLLM v0.31.0
takes the AutoProcessor's template, then the AutoTokenizer's (vllm/renderers/hf.py:263-300), and transformers
reads a processor's from ``chat_template.jinja`` (else ``chat_template.json``) and a tokenizer's from
``chat_template.jinja`` (else ``tokenizer_config.json``'s ``chat_template``)."""


def checkpoint_chat_template(recipe: Recipe) -> tuple[str, str]:
    """The checkpoint's own chat template at the recipe's pinned revision: the frame the engine renders a
    chat-shaped request with when the recipe serves no template file.

    Inputs: the recipe (``model`` and its 40-hex ``revision``).  Output: ``(source, text)`` -- the source names
    ``<model>@<revision>:<file>`` -- read from the first of :data:`CHECKPOINT_TEMPLATE_FILES` the checkpoint
    carries (:func:`~rcp_ndcg_test.equivalence.checkpoint.checkpoint_file`: the local Hub cache answers a
    pinned revision without a request; the Hub otherwise, unless offline).  Only an absent file falls through
    to the next source.  A ``tokenizer_config.json`` or ``chat_template.json`` template is its ``chat_template``
    value (the ``default`` entry of a named list).  Raises :class:`HarnessError` naming every file tried when
    none resolves, or the file whose read failed.
    """
    from .checkpoint import checkpoint_file

    tried: list[str] = []
    for name in CHECKPOINT_TEMPLATE_FILES:
        path = checkpoint_file(recipe, name)  # None only when the file is absent; any other failure raises
        if path is None:
            tried.append(f"{name}: absent")
            continue
        text = path.read_text(encoding="utf-8")
        if name.endswith(".json"):
            value = json.loads(text).get("chat_template")
            if isinstance(value, list):
                value = next((entry.get("template") for entry in value if entry.get("name") == "default"), None)
            if not isinstance(value, str):
                tried.append(f"{name}: no chat_template")
                continue
            text = value
        return f"{recipe.model}@{recipe.revision}:{name}", text
    raise HarnessError(
        f"recipe {recipe.id}: no chat template resolves for {recipe.model} at {recipe.revision} ("
        + "; ".join(tried)
        + "): populate the Hub cache, or serve the template as serve.chat_template"
    )


def _jinja_environment(*, strict: bool = True) -> Any:
    """The jinja2 environment the engine renders chat templates with (transformers' compile settings).

    ``strict`` makes an undefined variable an error (the query/document score templates); a chat template is
    rendered as transformers renders it, where an unset variable (``tools``, ``add_vision_id``) is undefined."""
    try:
        from jinja2 import StrictUndefined
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError as error:  # pragma: no cover - exercised only without jinja2
        raise HarnessError(
            "the template-render check renders the served chat template with jinja2: install rcp-ndcg-vllm[test]"
        ) from error
    if not strict:
        return ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    return ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, undefined=StrictUndefined)


def engine_conversation(conversation: list[Any]) -> list[dict[str, Any]]:
    """A sent conversation as vLLM v0.31.0 hands it to a chat template (the ``openai`` content format).

    vllm/entrypoints/chat_utils.py:1875-1999 (``_parse_chat_message_content_part`` with ``wrap_dicts``): a
    string part becomes ``{"type": "text", "text": ...}``, a text part keeps its text, an ``image_url`` part
    becomes ``{"type": "image"}`` and a ``video_url`` part ``{"type": "video"}`` -- the template sees the
    modality, never the URL.  A message whose content is a string stays as it is.
    """
    engine: list[dict[str, Any]] = []
    for message in conversation:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            engine.append(dict(message))
            continue
        parts: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, str):
                parts.append({"type": "text", "text": part})
            elif part.get("type") in ("text", "input_text"):
                parts.append({"type": "text", "text": str(part.get("text", ""))})
            elif part.get("type") in ("image_url", "input_image", "image"):
                parts.append({"type": "image"})
            elif part.get("type") in ("video_url", "video"):
                parts.append({"type": "video"})
            else:
                parts.append(dict(part))
        engine.append({**message, "content": parts})
    return engine


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
        json={"model": recipe.client["model"], "prompt": text, "add_special_tokens": add_special_tokens},
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
    template = fitting.client_template(recipe)
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
    template = fitting.client_template(recipe)
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
    from .media import text_rows

    rows = text_rows(load_pairs(pairs_path))  # the media rows are the media stage's
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
    path ships.  The pairs the client changed (a processing record: a change to the document's row, or the
    shared query's settlement, which changes every pair of its call) are excluded from the gates and reported
    separately under a declared over-cap deviation; the Kendall tau covers the uncut subset of every gated
    query.
    """
    deviation = recipe.reference.over_cap_deviation is not None
    client, capture = role_client(recipe, base_url)
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
        start = len(client.processing)
        result = client.rerank(row["query"], row["documents"], instruction=row.get("instruction"))
        flags = _changed_rows(client, start, len(row["documents"]))
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
        "referent": "pairs the client changed relative to the uncut input, decided on the client's own "
        "processing records; served and reference may differ by design when reference.known_deviations "
        "declares an over-cap deviation (anchor_drop_over_cap or over_cap_cut_differs)",
    }
    return summary


def _reference_row(reference: dict[str, Any], row_index: int) -> dict[str, Any]:
    """The reference's output for the row at ``row_index``, matched by the contract's ``index`` field."""
    rows = reference.get("rows", [])
    for row in rows:
        if int(row.get("index", -1)) == row_index:
            return row
    return rows[row_index] if row_index < len(rows) else {}


def _changed_rows(client: Any, start: int, n_documents: int) -> list[bool]:
    """Per input of one client call, whether the client changed what it sends for it -- read from its own
    processing records since ``start`` (:attr:`~rcp_ndcg.inference.clients._base.RoleClient.processing`): a
    record of the input's position, or of the reranker's shared query
    (:data:`~rcp_ndcg.inference.clients.rerank.QUERY_DOC_ID`), which changes every pair of the call."""
    from rcp_ndcg.inference.clients.rerank import QUERY_DOC_ID

    flags = [False] * n_documents
    for record in client.processing[start:]:
        if not record.changed:
            continue  # pragma: no cover - a client emits records for changed rows only
        if record.input_id == QUERY_DOC_ID or not record.input_id.isdigit():
            # The shared query's settlement changes every pair; a record under no position (the media fit's
            # owner fallback) cannot be attributed to one input, so every input of the call counts as changed.
            return [True] * n_documents
        position = int(record.input_id)
        if 0 <= position < n_documents:
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
    Under a declared over-cap deviation, the texts the client changed (a processing record: a budget cut
    counted with the frame, a shape's own cap, an empty substitution, a media change) are reported separately
    and do not gate: the reference renders them its own way by declaration.
    """
    deviation = recipe.reference.over_cap_deviation is not None
    per_vector: list[dict[str, Any]] = []
    over_cap: list[dict[str, Any]] = []
    client, capture = role_client(recipe, base_url)
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
                # One call per text: the client's fan-out runs concurrently, so per-call record windows keep
                # the position attribution exact.
                start = len(client.processing)
                embeddings = client.encode([Content.from_text(text)], encode_role)
                served_matrices.extend(_embeddings_to_matrices(recipe, embeddings, 1))
                cut_flags.extend(_changed_rows(client, start, 1))
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
        "referent": "inputs the client changed relative to the uncut input (its processing records); under "
        "the declared over-cap deviation the reference renders them its own way, so they are reported here "
        "instead of gated",
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
