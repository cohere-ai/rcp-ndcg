"""The request generator: the deterministic request set every recording asks (OBSERVATIONS-SPEC section 1).

One versioned generator -- :data:`GENERATOR_VERSION`, seeded (:data:`SEED`), over the suites at their
:data:`PINNED_DATASET_COMMITS` and the synthetic adversarial set stored as text
(:mod:`rcp_ndcg_test.observe.adversarial`) -- plans one request set per recipe and writes it in the
harness's pairs format (:func:`write_pairs_file`): one JSONL row per planned request
``{"query": str, "documents": [str, ...]}`` with the documented optional ``instruction`` and ``media``
fields, plus ``_strata`` / ``_source`` provenance keys the reference subprocess never sees (the
harness's ``_write_rows`` strips ``_``-prefixed keys).  ``pairs/<recipe>.jsonl`` (this package)
is what the wave runner feeds the equivalence stages; the same plan drives the observation corpus.
Rows whose stage-1 render comparison goes red are pruned deterministically (``--reference-python``),
recorded in ``pairs/manifest.json``, and the file is re-validated before it is written.

Strata, per OBSERVATIONS-SPEC section 1: shapes and instruction modes, the length ladder in the
recipe's own tokens (tiny, short, median of the source data, at 90-100% of the budget, exactly at the
budget; over-cap rows for rerankers that declare ``anchor_drop_over_cap``, where stage 2 reports them
instead of gating), every content kind, real items sampled by id from the suites and, for the media
recipes, the synthetic media request set (:mod:`rcp_ndcg_test.observe.media_set`: one image per size bucket
and a captioned page, a batch mixing a text-only and an image document, a query image where the recipe
allows query media, interleaved and several-image documents where its ``max_images`` admits them, and video
clips for the recipes that accept them, after the text rows) and the ViDoRe pages.  Every stratum is
recorded present or absent -- absent only when inapplicable, said why -- in ``pairs/manifest.json``.

The CLI (``python -m rcp_ndcg_test.observe.requests``) needs the Hub (or a populated cache) for the
tokenizers and the source datasets at generation time only; the committed pairs files and the manifest
are the artifact every later step reads offline.

Public surface:

- :data:`GENERATOR_VERSION`, :data:`SEED`, :data:`PINNED_DATASET_COMMITS` — the generator's identity.
- :class:`PlannedRow`, :class:`RecipePlan` — the plan records.
- :func:`plan_recipe` — one recipe's deterministic request plan.
- :func:`pairs_jsonl`, :func:`write_pairs_file` — the harness pairs format.
- :func:`write_manifest` — ``pairs/manifest.json`` with provenance and hashes.
- :data:`CORPUS_PLAN_VERSION`, :class:`CorpusPlan`, :func:`corpus_plan` — what a recording sends beyond the
  pairs rows: the over-length ladder and the long content kinds through the client and uncut, the wire
  variants and the protocol edges.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import random
import re
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adversarial import CONTENT_KINDS, synthetic_text
from .sources import SUITE_SUBSETS, SourceCorpus, SourceDoc, SourceQuery

__all__ = [
    "CORPUS_PLAN_VERSION",
    "GENERATOR_VERSION",
    "PINNED_DATASET_COMMITS",
    "SEED",
    "CorpusPlan",
    "PlannedRow",
    "RecipePlan",
    "corpus_plan",
    "pairs_jsonl",
    "plan_recipe",
    "write_manifest",
    "write_pairs_file",
]

GENERATOR_VERSION = 1
"""The generator's version: any change to what it generates (strata, selection, pads) bumps it."""

CORPUS_PLAN_VERSION = 1
"""The version of the corpus request plan beyond the pairs rows (:func:`corpus_plan`: the over-length ladder,
the uncut content kinds, the wire variants and the protocol edges).  Versioned apart from
:data:`GENERATOR_VERSION` so the pairs files (and their seeded sampling) stay as generated."""

SEED = "rcp-observe-v1"
"""The generator's seed: the deterministic stream every sampling draws from.  Change only with
:data:`GENERATOR_VERSION`."""

PINNED_DATASET_COMMITS: dict[str, str] = {
    "fabianschmidt-cohere/rcp-ndcg-nanobeir": "4517f2cb9e342479725bf0931a329998b4d35038",
    "fabianschmidt-cohere/rcp-ndcg-bright": "1bee160c00bf8ea7576a41879a62cb2d4fbd9319",
    "fabianschmidt-cohere/rcp-ndcg-vidore-v3": "4e5ca46e2f9c050b18a7f170c23e3ade66f78a4e",
    "fabianschmidt-cohere/rcp-ndcg-trecdl": "4f1f664491d50419bc24a7c24bd5aa366992b4a5",
}
"""The commit every source suite is read at (OBSERVATIONS-SPEC section 1: dataset URIs at pinned
commits).  A re-pinning changes where every row comes from, so it is a ``GENERATOR_VERSION`` bump."""

NANOBEIR_SUBSETS = 3
NANOBEIR_QUERIES = 2
NANOBEIR_DOCS = 4
BRIGHT_SUBSETS = 1
BRIGHT_QUERIES = 2
BRIGHT_DOCS = 2
TRECDL_SUBSETS = 1
TRECDL_QUERIES = 2
TRECDL_DOCS = 3
VIDORE_SUBSETS = 3
VIDORE_QUERIES = 4
VIDORE_DOCS = 2

DOCUMENT_CHAR_CAP = 65_536
"""The per-document character cap for real source items (a long-document row at ~16k tokens).  A pooled
document over the cap is excluded WHOLE and recorded with its id (nothing is ever silently cut in a
pairs file); the corpus request set keeps the over-cap ladder separately."""

GENERATION_SUITES: tuple[str, ...] = ("nanobeir", "bright")
"""The source suites the pairs generator materializes at generation time (GPU-VALIDATION.md T2's pairs:
NanoBEIR and BRIGHT, plus ViDoRe for the media recipes once loadable).  TREC DL's full MS MARCO-passage
corpus and ViDoRe's per-page images exceed a generation-time load (the diagnosis of the operator's
stall); the corpus request set samples those two on the node, and ``--suites`` names them for a run
with the memory available there."""

_LENGTH_STRATA = ("tiny", "short", "median", "at_90", "at_budget")
_GUARD_TOKENS = 4
"""Tokens kept in reserve between the planned content and ``client.max_tokens``: span joins at the
template's boundaries can land a token or two either side of the exact target, and a planned
"exactly at the budget" row must fit without a cut."""

_CHAR_CAP = 500_000
"""The per-row padding cap (characters): the pairs files stay inside GPU-VALIDATION.md item 7's 2 MB
per recipe even for a recipe whose budget alone would exceed it.  Rows the cap shortens are recorded
stratum-by-stratum in the manifest as ``at_90: absent`` (the corpus request set carries the full
ladder)."""


@dataclass(frozen=True)
class PlannedRow:
    """One planned request: the pairs row plus the stratum and source provenance it stands for.

    Attributes:
        query: The query span (the request's ``query``; the embed roles' query-side text).
        documents: The document spans.
        shape: The row's request shape, when the plan pins one (``None``: every declared shape).
        instruction: The run-level instruction, where the recipe's instruction mode fills one.
        media: ``{"query": [...], "documents": [[...], ...]}`` of :class:`SourceMedia` JSON, or ``None``.
        strata: The spec's stratum labels this row covers.
        source: Where the row's real items come from (suite, subset, ids, commit), or the synthetic id.
    """

    query: str
    documents: tuple[str, ...]
    shape: str | None = None
    instruction: str | None = None
    media: dict[str, Any] | None = None
    strata: tuple[str, ...] = ()
    source: dict[str, Any] | None = None

    def to_pairs_row(self) -> dict[str, Any]:
        """This row as one harness pairs row (``_write_rows`` drops the ``_`` keys for the reference)."""
        row: dict[str, Any] = {"query": self.query, "documents": list(self.documents)}
        if self.shape is not None:
            row["shape"] = self.shape
        if self.instruction is not None:
            row["instruction"] = self.instruction
        if self.media is not None:
            row["media"] = self.media
        row["_strata"] = list(self.strata)
        row["_source"] = self.source or {"suite": "synthetic"}
        return row

    def provenance(self) -> dict[str, Any]:
        """The manifest's provenance entry for this row (its source identities)."""
        return {"source": self.source or {"suite": "synthetic"}, "strata": list(self.strata)}


@dataclass(frozen=True)
class RecipePlan:
    """One recipe's request plan: its rows and every stratum's presence record.

    Attributes:
        recipe_id: The recipe the plan was made for.
        rows: The planned rows, in deterministic order.
        strata: ``{stratum: {"present": bool} | {"present": False, "reason": str}}`` — every stratum
            of OBSERVATIONS-SPEC section 1, present or absent with a reason.
        skipped_sources: The source ids excluded from the rows (e.g. over the document cap), with the
            reason — the selection manifest lists every source id, selected and excluded.
        validation: What the stage-1 validation ran and found (``render_check`` is ``passed`` or a
            recorded blocker such as a reference-contract drift -- never a silent pass).
    """

    recipe_id: str
    rows: list[PlannedRow] = field(default_factory=list)
    strata: dict[str, dict[str, Any]] = field(default_factory=dict)
    skipped_sources: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)


def _rng(recipe_id: str, stream: str) -> random.Random:
    """The deterministic stream one sampling step draws from (stable across Python versions: the
    string seeding is SHA-512 based)."""
    return random.Random(f"{GENERATOR_VERSION}/{SEED}/{recipe_id}/{stream}")


def _media_json(doc: SourceDoc, documents_open: list[list[dict[str, Any]]]) -> None:
    """Append one document's media entries (when it carries any)."""
    documents_open.append([entry.to_json() for entry in doc.media])


def _source_ref(corpus: SourceCorpus, query: SourceQuery, docs: list[SourceDoc]) -> dict[str, Any]:
    """The source identity of one sampled row (every source id, per the spec's selection manifest)."""
    return {
        "suite": corpus.suite,
        "subset": corpus.subset,
        "commit": corpus.commit,
        "query_id": query.query_id,
        "doc_ids": [doc.doc_id for doc in docs],
    }


def _source_row(
    corpus: SourceCorpus,
    query: SourceQuery,
    docs: list[SourceDoc],
    *,
    recipe: Any,
    shape_label: str,
) -> PlannedRow:
    """One real-item row from a suite's pool (the pooled candidates in pool order)."""
    instruction = query.instruction if recipe.client.get("instruction", "none") != "none" else None
    media: dict[str, Any] | None = None
    if any(doc.media for doc in docs):
        documents_media: list[list[dict[str, Any]]] = []
        for doc in docs:
            _media_json(doc, documents_media)
        media = {"query": [], "documents": documents_media}
    strata = [f"source:{corpus.suite}", f"source:{corpus.suite}/{corpus.subset}", f"shapes:{shape_label}"]
    if query.instruction is not None:
        kept = instruction is not None
        strata.append("instruction:present" if kept else "instruction:dropped-the-recipes-mode-is-none")
    if media is not None:
        strata.append("media:page_image")
    return PlannedRow(
        query=query.text,
        documents=tuple(doc.text for doc in docs),
        instruction=instruction,
        media=media,
        strata=tuple(strata),
        source=_source_ref(corpus, query, docs),
    )


def _pick(
    corpus: SourceCorpus, rng: random.Random, queries: int, docs: int
) -> tuple[list[tuple[SourceQuery, list[SourceDoc]]], list[dict[str, Any]]]:
    """Deterministically sample ``queries`` queries (and up to ``docs`` pooled documents each).

    A pooled document whose text exceeds :data:`DOCUMENT_CHAR_CAP` is excluded whole (a pairs file
    never carries a silently cut source item) and recorded in the second return value with its id and
    the reason -- the selection manifest lists every source id, selected and excluded.
    """
    query_ids = sorted(corpus.queries)
    chosen = sorted(rng.sample(query_ids, min(queries, len(query_ids))))
    out: list[tuple[SourceQuery, list[SourceDoc]]] = []
    skipped: list[dict[str, Any]] = []
    for query_id in chosen:
        query = corpus.queries[query_id]
        picked: list[SourceDoc] = []
        for doc_id in query.candidates:
            doc = corpus.docs.get(doc_id)
            if doc is None:
                continue
            if len(doc.text) > DOCUMENT_CHAR_CAP:
                skipped.append(
                    {
                        "suite": corpus.suite,
                        "subset": corpus.subset,
                        "query_id": query_id,
                        "doc_id": doc_id,
                        "reason": f"document text exceeds the {DOCUMENT_CHAR_CAP}-character cap (a pairs file "
                        "never carries a silently cut source item)",
                    }
                )
                continue
            picked.append(doc)
            if len(picked) >= docs:
                break
        if picked:
            out.append((query, picked))
        else:
            skipped.append(
                {
                    "suite": corpus.suite,
                    "subset": corpus.subset,
                    "query_id": query_id,
                    "doc_id": None,
                    "reason": "no pooled document fits the character cap",
                }
            )
    return out, skipped


def _source_rows(
    recipe: Any, corpora: dict[str, list[SourceCorpus]]
) -> tuple[list[PlannedRow], set[str], list[dict[str, Any]]]:
    """The real-item rows: NanoBEIR, BRIGHT (long documents), TREC DL and — for the media recipes —
    ViDoRe v3 pages.  Also returns the suites actually sampled and the excluded source ids (the
    manifest's selection record)."""
    rows: list[PlannedRow] = []
    used: set[str] = set()
    skipped: list[dict[str, Any]] = []
    media_recipe = "image" in recipe.input
    plan = [
        ("nanobeir", NANOBEIR_SUBSETS, NANOBEIR_QUERIES, NANOBEIR_DOCS),
        ("bright", BRIGHT_SUBSETS, BRIGHT_QUERIES, BRIGHT_DOCS),
        ("trecdl", TRECDL_SUBSETS, TRECDL_QUERIES, TRECDL_DOCS),
    ]
    if media_recipe:
        plan.append(("vidore", VIDORE_SUBSETS, VIDORE_QUERIES, VIDORE_DOCS))
    for suite, n_subsets, n_queries, n_docs in plan:
        available = [c for c in corpora.get(suite, []) if c.queries and c.docs]
        if not available:
            continue
        rng = _rng(recipe.id, f"sampling/{suite}")
        by_key = {c.subset: c for c in available}
        chosen = sorted(rng.sample(sorted(by_key), min(n_subsets, len(by_key))))
        for key in chosen:
            corpus = by_key[key]
            picked, excluded = _pick(corpus, rng, n_queries, n_docs)
            skipped.extend(excluded)
            for query, docs in picked:
                rows.append(_source_row(corpus, query, docs, recipe=recipe, shape_label=_shape_label(recipe)))
                used.add(suite)
    return rows, used, skipped


def _shape_label(recipe: Any) -> str:
    """The shape the row exercises for the spec's stratum record: ``pair`` for rerankers, else the
    recipe's declared shapes joined (the harness fits an un-shaped row under every declared shape)."""
    from ..equivalence import fitting

    shapes = fitting.declared_shapes(recipe)
    return "pair" if recipe.role == "rerank" else "+".join(sorted(shapes))


def _declared_sides(recipe: Any) -> tuple[bool, bool]:
    """Whether the query side / the document side goes on the wire for this recipe's rows."""
    from ..equivalence import fitting

    if recipe.role == "rerank":
        return True, True
    shapes = set(fitting.declared_shapes(recipe))
    return "query" in shapes, bool(shapes & {"document", "pair"})


def _synthetic_rows(recipe: Any, tokenizer: Any) -> list[PlannedRow]:
    """One row per content kind (the synthetic adversarial set as text).

    The kind's text goes on the sides whose budget it fits (the recipe's own tokens); a side it does
    not fit carries a short honest anchor instead (``on_overflow: cut`` would shorten it silently --
    never in a pairs file).  A kind that fits neither side is planned absent and recorded so in the
    manifest (``absent only when inapplicable, said why``); the corpus request set sends the full text
    uncut on purpose.
    """
    anchor = "synthetic adversarial anchor text about retrieval"
    pair = recipe.role == "rerank"
    declares_query, declares_document = _declared_sides(recipe)
    share = recipe.client.get("query_max_tokens", None) or 0
    budget = recipe.client.get("max_tokens") or 0
    overhead_doc = _overhead(recipe, tokenizer, "pair" if pair else "document")
    overhead_query = _overhead(recipe, tokenizer, "pair" if pair else "query")
    doc_room = budget - overhead_doc - _GUARD_TOKENS - (tokenizer.count(anchor) if pair else 0)
    query_room = min(share or budget, budget) - overhead_query - _GUARD_TOKENS
    empty_query_ok = recipe.client.get("empty_query", "refuse") == "send"
    empty_doc_ok = recipe.client.get("empty_doc", "") in ("send", "send_text")
    rows: list[PlannedRow] = []
    for kind in CONTENT_KINDS:
        text = synthetic_text(kind, tokenizer)
        count = tokenizer.count(text) if text else 0
        on_query = declares_query and count <= query_room
        on_document = declares_document and count <= doc_room
        if kind == "empty":
            # The empty string travels only where the declared empty policy lets it (the product's
            # ``empty_query: refuse`` refuses an empty query before the engine would see one).
            on_query = on_query and empty_query_ok
            on_document = on_document and empty_doc_ok
            if not (on_query or on_document):
                continue
        strata = [f"content:{kind}"]
        if on_document:
            strata.append(f"content:{kind}@document")
        if on_query:
            strata.append(f"content:{kind}@query")
        strata.append(f"shapes:{_shape_label(recipe)}")
        rows.append(
            PlannedRow(
                query=text if on_query else anchor,
                documents=(text if on_document else anchor,),
                strata=tuple(strata),
                source={"suite": "synthetic", "content_kind": kind},
            )
        )
    return rows


def _pad_to_tokens(tokenizer: Any, seed: str, target: int) -> str:
    """Grow ``seed`` until it counts ``target`` tokens of the recipe's tokenizer (within 2 tokens).

    The pad is located by the tokenizer's offset mapping over ONE tokenization (the product's
    :func:`rcp_ndcg.data.preprocess.token_prefix`), never by re-tokenizing whole candidate strings --
    a long seed and a long target cost one pass, bounded by a test on a long synthetic document.

    Inputs: the recipe's tokenizer, a seed text and the target token count (tokens of the padded
    content span, specials off).  Output: the padded text (whole ``padNNNNN`` words appended to the
    seed).  Units: tokens of ``tokenizer.count(..., add_special_tokens=False)`` (the recipe's own
    tokens).
    """
    from rcp_ndcg.data.preprocess import token_prefix

    seed_tokens = tokenizer.count(seed)
    if target <= seed_tokens:
        return seed
    gap = target - seed_tokens
    pool = seed + " " + " ".join(f"pad{index:05d}" for index in range(gap + 16))
    text = token_prefix(pool, target, tokenizer)
    for _ in range(8):
        count = tokenizer.count(text)
        if abs(count - target) <= 2:
            break
        if count < target:
            text += " pad"
        else:
            text = text.rsplit(" ", 1)[0] if " " in text else text[:-1]
    return text


def _overhead(recipe: Any, tokenizer: Any, shape: str) -> int:
    """The rendered prompt's fixed overhead for one shape (tokens), as the product's fit measures it:
    the template's fixed segments plus the shape's post-processor specials, content spans empty.  0
    for a shape the recipe does not declare (nothing of it goes on the wire)."""
    from ..equivalence import fitting

    template = fitting.client_template(recipe)
    if template is None:
        return 0
    if shape not in fitting.declared_shapes(recipe):
        return 0
    from ..equivalence.stages import cast_shape

    rendered = template.render(cast_shape(shape), tokenizer, query="", document="", instruction="")
    flag = bool(template.adds_special_tokens(cast_shape(shape)))
    return tokenizer.count(rendered, add_special_tokens=flag)


def _length_rows(
    recipe: Any, tokenizer: Any, medians: dict[str, int]
) -> tuple[list[PlannedRow], dict[str, dict[str, Any]]]:
    """The length ladder in the recipe's own tokens (OBSERVATIONS-SPEC section 1, lengths).

    Rows: ``tiny`` (4-token query, 8-token document), ``short`` (~32 tokens), ``median`` (the sampled
    source data's median document length), ``at_90`` and ``at_budget`` (the content span grows to
    90% of / exactly the remaining budget once the fixed overhead and the query are reserved).  Rows
    are planned so the product's fit cuts nothing: over-cap requests are the corpus request set's
    (sent uncut on purpose), except for rerankers that declare ``anchor_drop_over_cap``, which get one
    ``over_cap`` row stage 2 reports instead of gating.  A row the :data:`_CHAR_CAP` shortens is not
    written and its stratum is recorded absent with the reason.
    """
    budget = recipe.client.get("max_tokens") or 0
    rerank = recipe.role == "rerank"
    rows: list[PlannedRow] = []
    strata: dict[str, dict[str, Any]] = {}

    tiny_query = _pad_to_tokens(tokenizer, "tiny query", 4)
    tiny_doc = _pad_to_tokens(tokenizer, "tiny document", 8)
    rows.append(
        PlannedRow(
            query=tiny_query,
            documents=(tiny_doc,),
            strata=("length:tiny", f"shapes:{_shape_label(recipe)}"),
            source={"suite": "synthetic", "content_kind": "length:tiny"},
        )
    )
    strata["length:tiny"] = {"present": True}

    short_query = _pad_to_tokens(tokenizer, "short query", 32)
    short_doc = _pad_to_tokens(tokenizer, "short document", 32)
    rows.append(
        PlannedRow(
            query=short_query,
            documents=(short_doc,),
            strata=("length:short", f"shapes:{_shape_label(recipe)}"),
            source={"suite": "synthetic", "content_kind": "length:short"},
        )
    )
    strata["length:short"] = {"present": True}

    if medians:
        median = int(statistics.median(medians.values()))
        rows.append(
            PlannedRow(
                query=_pad_to_tokens(tokenizer, "median query", min(32, median)),
                documents=(_pad_to_tokens(tokenizer, "median document", max(8, median)),),
                strata=("length:median", f"shapes:{_shape_label(recipe)}"),
                source={"suite": "synthetic", "content_kind": "length:median"},
            )
        )
        strata["length:median"] = {"present": True}
    else:
        strata["length:median"] = {"present": False, "reason": "no source rows were sampled for this recipe"}

    overhead = _overhead(recipe, tokenizer, "pair" if rerank else "document")
    query_tokens = 64 if rerank else 32
    query_text = _pad_to_tokens(tokenizer, "budget query", query_tokens)
    # The query span shares a pair's budget (rerank); an embed row's at-budget side is the document.
    reserved = query_tokens + _GUARD_TOKENS if rerank else _GUARD_TOKENS
    usable = budget - overhead - reserved
    for stratum, target in (("at_90", usable * 9 // 10), ("at_budget", usable)):
        row_target = max(8, target)
        if row_target * 10 > _CHAR_CAP:
            # Never build the pad strings first: the row could not be committed whole anyway.
            strata[f"length:{stratum}"] = {
                "present": False,
                "reason": f"the padded row (~{row_target * 10} characters) would exceed the {_CHAR_CAP}-character "
                "file-size cap (GPU-VALIDATION.md item 7); the corpus request set carries the full ladder",
            }
            continue
        text = _pad_to_tokens(tokenizer, f"{stratum} document", row_target)
        if len(text) > _CHAR_CAP:
            strata[f"length:{stratum}"] = {
                "present": False,
                "reason": f"the padded row would exceed the {_CHAR_CAP}-character file-size cap (GPU-VALIDATION.md "
                "item 7); the corpus request set carries the full ladder",
            }
            continue
        rows.append(
            PlannedRow(
                query=query_text,
                documents=(text,),
                strata=(f"length:{stratum}", f"shapes:{_shape_label(recipe)}"),
                source={"suite": "synthetic", "content_kind": f"length:{stratum}"},
            )
        )
        strata[f"length:{stratum}"] = {"present": True, "content_tokens": row_target, "overhead_tokens": overhead}

    if rerank and rerank_over_cap(recipe):
        over = _pad_to_tokens(tokenizer, "over-cap document", usable * 2)
        rows.append(
            PlannedRow(
                query=query_text,
                documents=(over,),
                strata=("length:over_cap", "shapes:pair"),
                source={"suite": "synthetic", "content_kind": "length:over_cap"},
            )
        )
        strata["length:over_cap"] = {
            "present": True,
            "reason": "the recipe declares anchor_drop_over_cap: stage 2 reports this row in its non-gating table",
        }
    else:
        reason = (
            "stage-2 vector gates have no over-cap exclusion (the recipe's own reference notes say the pairs must "
            "sit under the budget)"
            if not rerank
            else "the recipe declares no anchor_drop_over_cap deviation: an over-cap row would gate on two different "
            "cuts; the corpus request set sends over-cap requests uncut on purpose instead"
        )
        strata["length:over_cap"] = {"present": False, "reason": reason}
    return rows, strata


def rerank_over_cap(recipe: Any) -> bool:
    """Whether stage 2 reports (rather than gates) over-cap pairs: the ``anchor_drop_over_cap`` deviation."""
    return recipe.role == "rerank" and "anchor_drop_over_cap" in recipe.reference.known_deviations


def plan_recipe(recipe: Any, tokenizer: Any, corpora: dict[str, list[SourceCorpus]]) -> RecipePlan:
    """One recipe's deterministic request plan (OBSERVATIONS-SPEC section 1).

    Inputs: the loaded :class:`~rcp_ndcg_vllm.recipe.Recipe`, the recipe's tokenizer and the source
    corpora by suite name (:func:`rcp_ndcg_test.observe.sources.load_corpora` at
    :data:`PINNED_DATASET_COMMITS`).  Output: the :class:`RecipePlan` -- real-item rows (NanoBEIR,
    BRIGHT, TREC DL, ViDoRe pages for the media recipes), one row per content kind and the length
    ladder -- with every stratum recorded present or absent.  Deterministic in
    (:data:`GENERATOR_VERSION`, :data:`SEED`, the pinned commits, the recipe's role and shapes): the
    same inputs plan the same rows.
    """
    from ..equivalence import fitting

    source_rows, used_suites, skipped_sources = _source_rows(recipe, corpora)
    synthetic = _synthetic_rows(recipe, tokenizer)
    synthetic_kinds = {str(row.source.get("content_kind")) for row in synthetic if row.source}
    length_rows, length_strata = _length_rows(recipe, tokenizer, _token_medians(source_rows, tokenizer))

    from .media_set import planned_media_rows

    planned_media, media_strata = planned_media_rows(recipe)
    media = [
        PlannedRow(
            query=row["query"],
            documents=tuple(row["documents"]),
            media=row["media"],
            strata=tuple(row["strata"]),
            source=row["source"],
        )
        for row in planned_media
    ]
    plan = RecipePlan(
        recipe_id=recipe.id,
        # The media rows come last: the text rows keep their positions, which stage 1 reports failures by.
        rows=[*source_rows, *synthetic, *length_rows, *media],
        skipped_sources=skipped_sources,
    )
    plan.strata.update(media_strata)
    plan.strata["shapes:" + _shape_label(recipe)] = {"present": True}
    for shape in fitting.declared_shapes(recipe):
        plan.strata.setdefault(f"shapes:{shape}", {"present": True})
    mode = recipe.client.get("instruction", "none")
    plan.strata[f"instruction:{mode}"] = {"present": True}
    for kind in CONTENT_KINDS:
        plan.strata[f"content:{kind}"] = {
            "present": kind in synthetic_kinds,
            **({} if kind in synthetic_kinds else {"reason": _kind_absent_reason(kind, recipe)}),
        }
    for suite in SUITE_SUBSETS:
        plan.strata[f"source:{suite}"] = {
            "present": suite in used_suites,
            **({} if suite in used_suites else {"reason": _suite_absent_reason(suite, recipe, corpora)}),
        }
    if "image" in recipe.input:
        plan.strata["media:page_image"] = {"present": any(row.media for row in source_rows)}
    else:
        plan.strata["media:page_image"] = {
            "present": False,
            "reason": "the recipe is text-only (recipe.input declares no image)",
        }
    plan.strata.update(length_strata)
    return plan


_LADDER_OVER: tuple[tuple[str, int, int], ...] = (("over_by_1", 1, 1), ("over_2x", 2, 0), ("over_10x", 10, 0))
"""The over-length rungs of OBSERVATIONS-SPEC section 1, as ``(name, factor, plus)``: the content span grows to
``factor * usable + plus`` of the recipe's own tokens, where ``usable`` is the budget left once the template's
fixed overhead and the guard are reserved -- one token over, twice and ten times the budget."""

_ROLE_ROUTES = {"embed": "/v1/embeddings", "multi_vector": "/pooling", "rerank": "/rerank"}


@dataclass(frozen=True)
class CorpusPlan:
    """The request set one recording sends (OBSERVATIONS-SPEC section 1), beyond and including the pairs rows.

    Attributes:
        rows: The model-layer rows the product's role client sends (it fits and, over the budget, cuts them):
            the pairs rows (``pairs:<n>``), the over-length ladder (``ladder:<rung>``) and the content kinds too
            long for a pairs row (``content:<kind>``).
        bare: The requests sent as they are, never through the client: the ladder and the long content kinds
            UNCUT (the engine's own refusal), the wire variants and the protocol edges.  Each is
            ``{"request_id", "stratum", "probe", "layer", "method", "path", "body"}`` (or ``"raw"`` bytes).
        strata: Every section-1 stratum of the corpus plan, present or absent with the reason.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    bare: list[dict[str, Any]] = field(default_factory=list)
    strata: dict[str, dict[str, Any]] = field(default_factory=dict)

    def request_ids(self) -> list[str]:
        """Every planned request id (the completeness check's list)."""
        return [str(row["request_id"]) for row in self.rows] + [str(row["request_id"]) for row in self.bare]


def _bare_body(recipe: Any, query: str, documents: list[str], **fields: Any) -> dict[str, Any]:
    """One request body on the recipe's role route, as the engine reads it uncut (no client render)."""
    if recipe.role == "rerank":
        return {"model": recipe.id, "query": query, "documents": documents, **fields}
    body: dict[str, Any] = {"model": recipe.id, "input": documents or [query], "encoding_format": "float"}
    if recipe.role == "multi_vector":
        body["task"] = "token_embed"
    return {**body, **fields}


def corpus_plan(recipe: Any, tokenizer: Any, pairs_rows: list[dict[str, Any]]) -> CorpusPlan:
    """The corpus request plan of one recipe (OBSERVATIONS-SPEC section 1, :data:`CORPUS_PLAN_VERSION`).

    Inputs: the loaded recipe, its tokenizer and the recipe's pairs rows (the generator's own plan, as the
    pairs file carries it).  Output: the :class:`CorpusPlan`.  Beyond the pairs rows it plans, deterministically:

    - the **length ladder** past the budget (one token over, 2x and 10x, in the recipe's own tokens), sent
      through the client (its cut) and bare (uncut, so the engine's own refusal is observed);
    - every **content kind** too long for a pairs row, the same two ways;
    - the **wire variants** of the role's route: ``encoding_format`` (``float``/``base64`` for
      ``/v1/embeddings``; ``float``/``base64``/``bytes`` x ``embed_dtype`` ``float16``/``float32`` for
      ``/pooling``), ``dimensions`` on, and ``top_n``, ``use_activation`` and ``instruction`` on and off for
      ``/rerank``;
    - the **protocol edges** the request set can send alone: an invalid ``embed_dtype``, ``top_n`` larger
      than the documents (the unknown field, malformed JSON, wrong model, empty input, over-length,
      ``/v1/models`` and health probes are the collector's standing set; a request while the engine loads is
      recorded by the wave's restart).
    """
    rows = [{**row, "request_id": str(row.get("request_id", f"pairs:{index}"))} for index, row in enumerate(pairs_rows)]
    plan = CorpusPlan(rows=rows)
    rerank = recipe.role == "rerank"
    budget = recipe.client.get("max_tokens") or 0
    overhead = _overhead(recipe, tokenizer, "pair" if rerank else "document")
    query_tokens = 64 if rerank else 32
    query = _pad_to_tokens(tokenizer, "ladder query", query_tokens)
    usable = max(8, budget - overhead - (query_tokens if rerank else 0) - _GUARD_TOKENS)
    for name, factor, plus in _LADDER_OVER:
        text = _pad_to_tokens(tokenizer, f"{name} document", factor * usable + plus)
        stratum = f"length:{name}"
        plan.rows.append(
            {
                "request_id": f"ladder:{name}",
                "stratum": stratum,
                "query": query,
                "documents": [text],
                "_strata": [stratum],
            }
        )
        plan.bare.append(
            {
                "request_id": f"ladder:{name}:uncut",
                "stratum": stratum,
                "probe": f"uncut:{name}",
                "layer": "model",
                "method": "POST",
                "path": _ROLE_ROUTES[recipe.role],
                "body": _bare_body(recipe, query, [text]),
            }
        )
        plan.strata[stratum] = {"present": True, "content_tokens": factor * usable + plus}
    # A kind is carried by the pairs file only when one of its rows put the kind's text on a side (its strata
    # say ``content:<kind>@query`` or ``@document``); a kind too long for either side rode as an anchor only.
    pairs_kinds = {
        label.split(":", 1)[1].split("@", 1)[0]
        for row in rows
        for label in row.get("_strata") or []
        if label.startswith("content:") and "@" in label
    }
    for kind in CONTENT_KINDS:
        text = synthetic_text(kind, tokenizer)
        stratum = f"content:{kind}"
        if kind in pairs_kinds:
            plan.strata[f"{stratum}:uncut"] = {
                "present": False,
                "reason": "the kind fits the budget: its pairs row carries it",
            }
            continue
        if not text:
            plan.strata[f"{stratum}:uncut"] = {
                "present": False,
                "reason": "the empty string under a refusing empty policy: the empty_input probe sends it bare",
            }
            continue
        plan.rows.append(
            {
                "request_id": f"content:{kind}",
                "stratum": stratum,
                "query": query,
                "documents": [text],
                "_strata": [stratum],
            }
        )
        plan.bare.append(
            {
                "request_id": f"content:{kind}:uncut",
                "stratum": stratum,
                "probe": f"uncut:{kind}",
                "layer": "model",
                "method": "POST",
                "path": _ROLE_ROUTES[recipe.role],
                "body": _bare_body(recipe, query, [text]),
            }
        )
        plan.strata[f"{stratum}:uncut"] = {"present": True}
    first = rows[0] if rows else {"query": query, "documents": ["a document"]}
    _wire_variants(recipe, plan, str(first["query"]), [str(document) for document in first["documents"]])
    return plan


def _wire_variants(recipe: Any, plan: CorpusPlan, query: str, documents: list[str]) -> None:
    """The role route's wire variants and the protocol edges the request set sends bare."""
    route = _ROLE_ROUTES[recipe.role]

    def add(name: str, body: dict[str, Any], *, edge: bool = False) -> None:
        stratum = f"{'edge' if edge else 'wire'}:{name}"
        plan.bare.append(
            {
                "request_id": stratum,
                "stratum": stratum,
                "probe": stratum,
                "layer": "protocol",
                "method": "POST",
                "path": route,
                "body": body,
            }
        )
        plan.strata[stratum] = {"present": True}

    if recipe.role == "rerank":
        add("top_n=1", _bare_body(recipe, query, documents, top_n=1))
        add("top_n=all", _bare_body(recipe, query, documents, top_n=len(documents)))
        add("use_activation=true", _bare_body(recipe, query, documents, use_activation=True))
        add("use_activation=false", _bare_body(recipe, query, documents, use_activation=False))
        add("instruction=on", _bare_body(recipe, query, documents, instruction="Judge whether the document answers."))
        add("top_n_over_documents", _bare_body(recipe, query, documents, top_n=len(documents) + 5), edge=True)
        plan.strata["wire:encoding_format"] = {"present": False, "reason": "the /rerank route has no encodings"}
    elif recipe.role == "embed":
        add("encoding_format=base64", _bare_body(recipe, query, documents[:1], encoding_format="base64"))
        dim = recipe.client.get("dimensions", None) or 32
        add(f"dimensions={dim}", _bare_body(recipe, query, documents[:1], dimensions=dim))
        plan.strata["edge:invalid_embed_dtype"] = {
            "present": False,
            "reason": "/v1/embeddings takes no embed_dtype (the /pooling route's field)",
        }
    else:
        for encoding in ("float", "base64", "bytes"):
            for dtype in ("float16", "float32"):
                add(
                    f"encoding_format={encoding},embed_dtype={dtype}",
                    _bare_body(recipe, query, documents[:1], encoding_format=encoding, embed_dtype=dtype),
                )
        add("invalid_embed_dtype", _bare_body(recipe, query, documents[:1], embed_dtype="float64x"), edge=True)
    from .media_set import media_edges

    # The media request set (media_set): its size buckets ride the pairs rows, its edges go bare.
    edges, edge_strata = media_edges(recipe)
    plan.bare.extend({**edge, "path": route} for edge in edges)
    plan.strata.update(edge_strata)


def _kind_absent_reason(kind: str, recipe: Any) -> str:
    """Why one content kind is absent from a recipe's rows (absent only when inapplicable, said why)."""
    if kind == "empty":
        empty_query = recipe.client.get("empty_query", "refuse")
        empty_doc = recipe.client.get("empty_doc", "")
        return (
            f"the client's empty policy refuses the empty string on every side (empty_query: {empty_query}, "
            f"empty_doc: {empty_doc or 'unknown'}); the corpus request set probes the refusal itself"
        )
    return (
        "the kind's adversarial text exceeds the recipe's content budget on every side; nothing is cut "
        "silently in a pairs file (the corpus request set sends it uncut on purpose)"
    )


def _token_medians(source_rows: list[PlannedRow], tokenizer: Any) -> dict[str, int]:
    """The sampled source rows' document lengths, tokens of the recipe's tokenizer (for the median row)."""
    out: dict[str, int] = {}
    for row in source_rows:
        key = str((row.source or {}).get("query_id") or id(row))
        out[key] = tokenizer.count(row.documents[0]) if row.documents else 1
    return out


def _suite_absent_reason(suite: str, recipe: Any, corpora: dict[str, list[SourceCorpus]]) -> str:
    """Why one suite is absent from a recipe's plan."""
    if suite == "vidore" and "image" not in recipe.input:
        return "the recipe is text-only (recipe.input declares no image)"
    if not corpora.get(suite):
        return (
            f"the {suite} corpora were not requested at generation time (--suites {','.join(GENERATION_SUITES)} "
            "by default): its corpus is node-scale (TREC DL's full MS MARCO-passage corpus, ViDoRe's page "
            "images); the corpus request set samples it on the node"
        )
    return f"the deterministic sampling selected no usable rows from {suite}"


def pairs_jsonl(plan: RecipePlan) -> str:
    """The plan as the harness's pairs JSONL text (one :meth:`PlannedRow.to_pairs_row` per line)."""
    return "".join(json.dumps(row.to_pairs_row(), ensure_ascii=False) + "\n" for row in plan.rows)


def write_pairs_file(plan: RecipePlan, out_dir: str | Path) -> Path:
    """Write ``<out>/<recipe>.jsonl`` and return its path (UTF-8, no BOM)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{plan.recipe_id}.jsonl"
    path.write_text(pairs_jsonl(plan), encoding="utf-8")
    return path


def write_manifest(
    plans: list[RecipePlan],
    out_dir: str | Path,
    *,
    pruned: list[dict[str, Any]] | None = None,
    skipped_recipes: list[dict[str, Any]] | None = None,
) -> Path:
    """Write ``pairs/manifest.json``: the generator's identity, the pinned commits, every file's
    SHA-256 and row count, every stratum's presence record, the excluded source ids, the recipes that
    could not load (with their error), and any rows the stage-1 validation pruned.

    Runs merge: one bounded invocation per recipe updates its own entries and keeps the rest (the
    runbook generates under ``timeout`` per invocation), keyed by recipe id; pruned rows append and
    de-duplicate.
    """
    out = Path(out_dir)
    entries: list[dict[str, Any]] = []
    for plan in sorted(plans, key=lambda p: p.recipe_id):
        path = out / f"{plan.recipe_id}.jsonl"
        data = path.read_bytes() if path.is_file() else b""
        rows = [row.to_pairs_row() for row in plan.rows]
        entries.append(
            {
                "recipe": plan.recipe_id,
                "path": path.name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "rows": len(rows),
                "strata": plan.strata,
                "provenance": [row.provenance() for row in plan.rows],
                "skipped_sources": plan.skipped_sources,
                "validation": plan.validation,
            }
        )
    path = out / "manifest.json"
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            existing = loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            existing = {}
    # A recipe this run touched (generated or skipped) replaces ALL its entries: its file record, its
    # skipped record and its pruned rows; the recipes the run did not touch keep theirs.
    touched = {entry["recipe"] for entry in entries} | {str(entry["recipe"]) for entry in skipped_recipes or []}
    files = {str(entry.get("recipe")): entry for entry in existing.get("files", [])}
    files = {key: entry for key, entry in files.items() if key not in touched}
    files.update({entry["recipe"]: entry for entry in entries})
    skipped = {str(entry.get("recipe")): entry for entry in existing.get("skipped_recipes", [])}
    skipped = {key: entry for key, entry in skipped.items() if key not in touched}
    skipped.update({entry["recipe"]: entry for entry in skipped_recipes or []})
    pruned_all = [entry for entry in existing.get("pruned", []) if str(entry.get("recipe")) not in touched]
    seen = {json.dumps(entry, sort_keys=True, default=str) for entry in pruned_all}
    for entry in pruned or []:
        key = json.dumps(entry, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            pruned_all.append(entry)
    document = {
        "schema": "rcp-ndcg-vllm.pairs-manifest.v1",
        "schema_version": 1,
        "generator": {
            "module": "rcp_ndcg_test.observe.requests",
            "GENERATOR_VERSION": GENERATOR_VERSION,
            "SEED": SEED,
        },
        "dataset_commits": dict(PINNED_DATASET_COMMITS),
        "files": [files[key] for key in sorted(files)],
        "skipped_recipes": [skipped[key] for key in sorted(skipped)],
        "pruned": pruned_all,
    }
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


_PROBE_DIM = 8
"""The vector width of the offline fake's ``/pooling`` replies in stage-1 validation (:func:`_offline_probe`)."""


def _offline_probe(recipe: Any) -> Any:
    """The recipe as stage 1 probes it on the product's offline fake: a ``/pooling`` reply's width bounded.

    Stage 1 audits what the role client SENDS; the fake's reply is scaffolding.  A ``vllm_pooling`` recipe's
    ``dim`` shapes only the reply (the adapter decodes the reply by it; no request carries it), and the fake
    answers one ``dim``-wide vector per token, so the shipped width at twice a long budget is a multi-GiB
    reply per probed text: the probe copy drops it to 8, unless a Matryoshka ``mrl_dim`` is declared.  Input:
    a loaded recipe.  Output: the recipe itself (nothing to bound), or its copy; every field a request is
    built from is unchanged.
    """
    client = recipe.client
    if client.get("api") != "vllm_pooling":
        return recipe
    update: dict[str, Any] = {}
    dim = client.get("dim")
    if dim is not None and dim > _PROBE_DIM and client.get("mrl_dim") is None:
        update["dim"] = _PROBE_DIM
    if not update:
        return recipe
    return recipe.model_copy(update={"client": {**client, **update}})


_PROBE_MAX_PER_TOKEN_SAMPLE = 32768
"""The longest over-length sample (tokens, twice the budget) the offline fake answers per token on CPU."""


def _probe_infeasible(recipe: Any) -> str | None:
    """Why stage 1 cannot probe ``recipe`` on the offline fake, or ``None`` when it can.

    The fake answers ``/pooling`` with one vector per token, each seeded by the item's whole body, so a probe
    costs tokens x body bytes: stage 1's over-length samples, padded to twice the declared budget, make that
    quadratic for a long-context budget (half a million draws over a megabyte-long body per item at 262k
    tokens).  Such a recipe's render check is a recorded blocker, never a silent pass; the full-budget stage 1
    runs against the engine on the GPU wave.
    """
    client = recipe.client
    if client.get("api") != "vllm_pooling":  # the fake's /pooling reply is per token, whatever outputs
        return None
    budget = client.get("max_tokens") or 0
    if 2 * budget <= _PROBE_MAX_PER_TOKEN_SAMPLE:
        return None
    return (
        f"blocked: stage 1's over-length samples ({2 * budget} tokens, twice the {budget}-token budget) are "
        "infeasible on the offline fake's per-token /pooling replies (one draw per token, seeded by the whole "
        f"body; the CPU bound is {_PROBE_MAX_PER_TOKEN_SAMPLE} tokens); the rows are unpruned and the "
        "full-budget stage 1 runs against the engine on the GPU wave"
    )


def _validate_and_prune(
    recipe: Any, plan: RecipePlan, reference_python: str
) -> tuple[RecipePlan, list[dict[str, Any]]]:
    """Run the harness's stage 1 over the plan's rows and drop every row a check blames (deterministic).

    Inputs: the recipe, the plan and the reference interpreter (its render mode needs only the
    tokenizer libraries).  Outputs: the validated plan (its ``validation`` records what ran) and the
    pruned rows' provenance with the reason.  A failure that cannot be attributed to a row stops the
    pruning loudly -- never a silent drop.  One failure class is recorded as a per-recipe blocker
    instead of pruned: a reference that emits the pre-R30 ``{text}`` render rows where stage 1 reads
    ``{query, documents}`` spans is a contract drift of the whole recipe family (lane ``recipe-common``
    reconciles it on its side), not a row problem.
    """
    from rcp_ndcg_test.errors import HarnessError

    from ..equivalence.stages import stage1_prompts

    recipe = _offline_probe(recipe)
    infeasible = _probe_infeasible(recipe)
    if infeasible is not None:
        validation = {**plan.validation, "render_check": infeasible, "pruned_rows": 0}
        return (
            RecipePlan(
                recipe_id=plan.recipe_id,
                rows=plan.rows,
                strata=plan.strata,
                skipped_sources=plan.skipped_sources,
                validation=validation,
            ),
            [],
        )
    pruned: list[dict[str, Any]] = []
    blocked: dict[str, Any] = {}
    rounds = 0
    while plan.rows and rounds < 4:
        rounds += 1
        # ignore_cleanup_errors: the scratch dir's files are written, read by a subprocess and closed;
        # on a network-backed tempdir an entry can still turn visible after the cleanup's scan, and a
        # scratch cleanup race must never fail a recipe's validation.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as work:
            pairs = Path(work) / "pairs.jsonl"
            pairs.write_text(pairs_jsonl(plan), encoding="utf-8")
            try:
                document = stage1_prompts(recipe, pairs, reference_python, over_length_per_shape=2)
            except HarnessError as error:
                # A reference subprocess that cannot run (its environment, its tokenizer loader) is a
                # recorded blocker, never a silent pass and never a row problem: audit what stage 1
                # audits without the reference and say so.
                document = stage1_prompts(recipe, pairs, None, over_length_per_shape=2)
                blocked["render_check"] = f"blocked: the reference subprocess could not run: {error}"
        red, drift = _red_row_indexes(document)
        if drift:
            blocked["render_check"] = (
                "blocked: the recipe's reference emits the pre-R30 {index, shape, text} render rows while "
                "stage 1 reads {query, documents} spans (a contract drift of the recipe family, reconciled "
                "by lane recipe-common); the rows were audited by anchor_check and template_render_check only"
            )
        unattributable = [entry for index, entry in red if index is None]
        if unattributable:
            raise SystemExit(f"stage 1 failed with rows the checker cannot attribute: {unattributable[:3]}")
        if not red:
            break
        # Stage 1 reports the text rows' positions among the text rows (the media rows are the media stage's).
        text_positions = [index for index, row in enumerate(plan.rows) if not _inline_media(row)]
        red = [(text_positions[index] if index is not None else None, entry) for index, entry in red]
        drop = {index for index, _ in red if index is not None}
        kept: list[PlannedRow] = []
        for index, row in enumerate(plan.rows):
            if index in drop:
                pruned.append({**row.provenance(), "reason": next(e for i, e in red if i == index)})
            else:
                kept.append(row)
        plan = RecipePlan(
            recipe_id=plan.recipe_id,
            rows=kept,
            strata=plan.strata,
            skipped_sources=plan.skipped_sources,
            validation=blocked,
        )
    media_check, refused = _media_check(recipe, plan, reference_python)
    if refused:
        # A media row the recipe's reference refuses (an input its card does not define) is a row problem:
        # pruned with the reason, and the remaining media rows checked again.
        kept = []
        for index, row in enumerate(plan.rows):
            if index in refused:
                pruned.append({**row.provenance(), "reason": refused[index]})
            else:
                kept.append(row)
        plan = RecipePlan(
            recipe_id=plan.recipe_id,
            rows=kept,
            strata=plan.strata,
            skipped_sources=plan.skipped_sources,
            validation=plan.validation,
        )
        media_check, _ = _media_check(recipe, plan, reference_python)
    validation = {**blocked, **plan.validation}
    validation.setdefault("render_check", "passed")
    if media_check is not None:
        validation["media_check"] = media_check
    validation["pruned_rows"] = len(pruned)
    validation["over_length_per_shape"] = 2
    validation["over_length_note"] = (
        "the pairs-file validation samples 2 over-length inputs per shape; the wave's stage-1 gate on the "
        "node runs >= 20 per shape (GPU-VALIDATION.md T2)"
    )
    return (
        RecipePlan(
            recipe_id=plan.recipe_id,
            rows=plan.rows,
            strata=plan.strata,
            skipped_sources=plan.skipped_sources,
            validation=validation,
        ),
        pruned,
    )


def _inline_media(row: PlannedRow) -> bool:
    """Whether a planned row carries inline media (the media stage's row; stages 1 and 2 skip it)."""
    from ..equivalence.media import media_rows

    return bool(media_rows([row.to_pairs_row()]))


def _media_check(recipe: Any, plan: RecipePlan, reference_python: str) -> tuple[Any, dict[int, str]]:
    """The media stage offline over the plan's media rows: ``("passed" | the failures | why it could not run,
    the rows the reference refuses)``.  A recipe's media disagreeing with its reference is never a row to
    prune -- it is recorded, and the wave's gate fails on it; a row whose input the reference's card does
    not define (``reference_refused``) is returned for pruning.  ``(None, {})`` for a recipe without media."""
    from rcp_ndcg_test.errors import HarnessError

    from ..equivalence.media import stage_media

    # ignore_cleanup_errors: see the stage-1 loop above -- a scratch cleanup race on a network-backed
    # tempdir is the environment's, never a validation result.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as work:
        pairs = Path(work) / "pairs.jsonl"
        pairs.write_text(pairs_jsonl(plan), encoding="utf-8")
        try:
            document = stage_media(recipe, pairs, reference_python)
        except HarnessError as error:
            return f"blocked: the media stage could not run: {error}", {}
    if document is None:
        return None, {}
    failures = list(document.get("failures") or [])
    refused = {
        int(failure["row"]): f"media_check: the reference refuses the {failure['side']}: {failure['reason']}"
        for failure in failures
        if failure.get("check") == "reference_refused"
    }
    if document.get("passed"):
        return "passed", {}
    return (
        {
            "status": document.get("status"),
            "failures": failures[:8],
            "refusals": (document.get("refusals") or [])[:8],
            "reason": document.get("reason"),
        },
        refused,
    )


def _is_contract_drift(failure: dict[str, Any]) -> bool:
    """A render failure that says the reference emitted no comparable text: the pre-R30 ``{text}``
    contract drift (its rows carry neither ``query``/``documents`` spans), never a row problem."""
    if "span" not in failure:
        return False
    if failure.get("reference_text_head") == "":
        return True
    return "the reference rendered 0" in str(failure.get("note", ""))


def _red_row_indexes(document: dict[str, Any]) -> tuple[list[tuple[int | None, str]], bool]:
    """Map the stage-1 document's failures to pairs-row indexes (``None`` for an unattributable one),
    and whether the render comparison hit a reference-contract drift (never a row problem)."""
    red: list[tuple[int | None, str]] = []
    drift = False
    for section in ("anchor_check", "render_check", "template_render_check", "engine_tokenize_check"):
        body = document.get(section)
        if not isinstance(body, dict) or body.get("passed") is not False:
            continue
        failures = body.get("failures") or []
        for failure in failures:
            if _is_contract_drift(failure):
                drift = True
                continue
            index = failure.get("row", failure.get("index"))
            index = index.get("index") if isinstance(index, dict) else index
            if index is None and isinstance(failure.get("referent"), str):
                match = re.match(r"row (\d+)", failure["referent"])
                index = int(match.group(1)) if match else None
            red.append((int(index) if isinstance(index, int) and index >= 0 else None, _snippet(failure)))
    return red, drift


def _snippet(failure: dict[str, Any]) -> str:
    """One failure as the pruning reason (its key names, compact)."""
    keys = sorted(set(failure) - {"text", "rows"})
    return json.dumps({key: failure[key] for key in keys}, ensure_ascii=False, default=str)[:400]


def main(argv: list[str] | None = None) -> int:
    """The CLI: plan and write ``pairs/<recipe>.jsonl`` for every recipe, and ``manifest.json``.

    ``--reference-python`` runs the harness's stage-1 validation per recipe and prunes red rows first;
    without it the files are written unvalidated (the manifest records the rows as written).  A stall
    watchdog (``faulthandler.dump_traceback_later(60, repeat=True)``) writes every live stack to stderr
    every 60 s: wrap a run in ``timeout 300`` and redirect stderr to a scratch file to see where any
    stall sits (the operator's runbook).
    """
    faulthandler.dump_traceback_later(60, repeat=True)
    parser = argparse.ArgumentParser(
        prog="python -m rcp_ndcg_test.observe.requests",
        description="Generate the deterministic request set for every recipe (pairs files + manifest).",
    )
    parser.add_argument("--recipes-root", default=None, help="root of recipe directories (default: the package's)")
    parser.add_argument("--out", required=True, help="output directory (the pairs/ directory)")
    parser.add_argument("--recipes", default="", help="comma-separated recipe ids (default: every recipe)")
    parser.add_argument(
        "--reference-python",
        default=None,
        help="the python that runs each recipe's reference (render mode): stage-1 validation prunes red rows",
    )
    parser.add_argument(
        "--suites",
        default=",".join(GENERATION_SUITES),
        help="source suites to sample (default: nanobeir,bright -- the compact ones; trecdl and vidore are "
        "node-scale corpora and need the memory the node has)",
    )
    args = parser.parse_args(argv)
    from rcp_ndcg_vllm.recipe import RecipeError, load_recipe

    from .sources import load_corpora

    root = Path(args.recipes_root) if args.recipes_root else default_root()
    ids = [item.strip() for item in args.recipes.split(",") if item.strip()]
    recipes = []
    skipped_recipes: list[dict[str, Any]] = []
    if ids:
        for recipe_id in ids:
            try:
                recipes.append(load_recipe(recipe_id, root=root))
            except RecipeError as error:
                # One failing recipe never stops the wave (or the generator): record the error and go on.
                skipped_recipes.append({"recipe": recipe_id, "error": str(error)})
                print(f"{recipe_id}: cannot load -- {error}")
    else:
        # every family's every variant, in family/id order (one pairs file per variant id)
        from rcp_ndcg_vllm.recipe import load_family, load_recipes_of

        for directory in sorted(p for p in root.iterdir() if p.is_dir() and (p / "family.yaml").is_file()):
            try:
                recipes.extend(load_recipes_of(load_family(directory), directory))
            except RecipeError as error:
                skipped_recipes.append({"recipe": directory.name, "error": str(error)})
                print(f"{directory.name}: cannot load -- {error}")
    if not recipes and not skipped_recipes:
        print(f"error: no recipes under {root}")
        return 2
    corpora: dict[str, list[SourceCorpus]] = {}
    for suite in [s.strip() for s in args.suites.split(",") if s.strip()]:
        corpora[suite] = load_corpora(suite, SUITE_SUBSETS[suite], PINNED_DATASET_COMMITS)
    from ..equivalence.fitting import tokenizer_of

    out = Path(args.out)
    plans: list[RecipePlan] = []
    pruned: list[dict[str, Any]] = []
    for recipe in recipes:
        try:
            plan = plan_recipe(recipe, tokenizer_of(recipe), corpora)
            if args.reference_python:
                plan, pruned_rows = _validate_and_prune(recipe, plan, args.reference_python)
                if not plan.rows and pruned_rows:
                    # An empty pairs file would let stage 2 pass on nothing: the recipe is skipped, its
                    # previous file removed, and the first failure named.
                    (out / f"{recipe.id}.jsonl").unlink(missing_ok=True)
                    raise ValueError(
                        f"stage-1 validation pruned every row ({len(pruned_rows)}); the first: "
                        f"{pruned_rows[0].get('reason')}"
                    )
                pruned.extend({"recipe": recipe.id, **entry} for entry in pruned_rows)
            else:
                plan = RecipePlan(
                    recipe_id=plan.recipe_id,
                    rows=plan.rows,
                    strata=plan.strata,
                    skipped_sources=plan.skipped_sources,
                    validation={"render_check": "not_run: generation ran without --reference-python"},
                )
            write_pairs_file(plan, out)
        except Exception as error:  # noqa: BLE001 - one recipe's generation never stops the batch
            skipped_recipes.append(
                {"recipe": recipe.id, "error": f"generation failed: {type(error).__name__}: {error}"}
            )
            print(f"{recipe.id}: generation failed -- {type(error).__name__}: {error}")
            continue
        plans.append(plan)
        print(f"{recipe.id}: {len(plan.rows)} rows")
    write_manifest(plans, out, pruned=pruned, skipped_recipes=skipped_recipes)
    return 1 if skipped_recipes else 0


def default_root() -> Path:
    """The package's recipes directory (the recipe lanes' one home)."""
    from rcp_ndcg_vllm.recipe import default_recipes_root

    return default_recipes_root()


if __name__ == "__main__":
    raise SystemExit(main())
