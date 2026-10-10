"""The text budget: one fitting mechanism that fits every served role's requests into a model's input.

One home for the declared :class:`TextBudget`, the fit (:func:`fit`) that every served role's client
runs -- the fixed overhead measured once per (template, shape), only the content spans cut with the
token-boundary :func:`~rcp_ndcg.data.text_policy.token_prefix`, the template re-attached, chunking
that gives every chunk the full template, and every cut recorded -- and the per-row
:class:`ProcessingRecord` the pipeline emits for what it changed.

The judge's side of the machinery (the per-document text policy, the chunk geometry, the token-boundary
cut) is :mod:`rcp_ndcg.data.text_policy`; the census types are :mod:`rcp_ndcg.data.census`; where chunk
scores pool back onto their document is :mod:`rcp_ndcg.data.postprocess`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rcp_ndcg_core.content import TEXT_JOIN, split_text_across_parts

from rcp_ndcg.data.census import CUT_CAUSES, CutCause, TextCutRecord, TextTruncationCensus
from rcp_ndcg.data.mrl import MrlKind
from rcp_ndcg.data.postprocess import document_id_for_chunk
from rcp_ndcg.data.templates import RequestShape, TemplateSpec
from rcp_ndcg.data.text_policy import CHUNK_ID_SEPARATOR, ChunkPolicy, split_into_chunks, token_prefix
from rcp_ndcg.errors import ConfigError, DataError
from rcp_ndcg.support.identity import FieldRole, identity_payload
from rcp_ndcg.support.logging import get_logger

if TYPE_CHECKING:
    from rcp_ndcg.data.tokenizer import TextTokenizer

logger = get_logger(__name__)

BUDGET_DOC_ID = "<budget>"
"""The ``doc_id`` of the one per-corpus census row a hosted vendor profile records: its documented
limit as the effective budget (``budget_source: vendor``). No tokenizer means nothing is measured,
so there is no per-document row."""


class TextBudgetExceededError(DataError):
    """An input cannot be sent within the declared :class:`TextBudget`, and the declared policy refuses to
    shorten it -- ``on_overflow: fail``, or ``on_overflow: chunk`` applied to a query (queries are never
    chunked), or a pair whose query fills the budget and leaves the document nothing."""


ContentParts = str | tuple[str, str]
"""The cut content of one fitted output: the text itself for the ``query`` and ``document`` shapes, a
``(query, document)`` tuple for ``pair``."""


@dataclass(frozen=True)
class FitResult:
    """What :func:`fit` decided for one call's inputs, in input order (chunks flattened).

    Attributes:
        shape: The request shape the inputs were fitted as.
        texts: The rendered request strings, one per output, in ``ids`` order: the full template with the
            cut content spans re-attached. Empty only for an unrendered ``pair`` -- a hosted vendor profile
            (no tokenizer), or a pair whose budget declares no template (the engine renders it; the caller
            sends :attr:`contents`) -- a hosted profile's ``query`` and ``document`` shapes return their raw
            content strings, which is what those wires take. This is what a text or ``token_ids`` wire route
            sends; the strings are returned as text, not token ids, because every wire route accepts text,
            the engine's own tokenisation (with its ``add_special_tokens`` flag) stays authoritative, and the
            client-side tokenisation this mechanism needs to measure and cut is the same one either way.
        contents: The cut content per output: the span text (a str), or the ``(query, document)`` parts of
            a pair -- what a wire route the engine renders the template for receives.
        ids: The output id per rendered text: the input's id, or ``<id>#<k>`` for its chunks (as
            :func:`chunk_ranking_example` names them).
        chunk_mapping: For chunked inputs, chunk id -> input id (an unsplit input maps to itself), ready
            for :func:`max_pool_scores_by_document`. ``None`` when nothing was chunked.
        overhead: The measured fixed overhead of (template, shape), in tokens -- the empty render plus the
            shape's post-processor tokens. ``None`` when no tokenizer was declared (vendor mode: nothing is
            measured).
        budget_source: ``"tokenizer"`` when the budget was measured with the declared tokenizer (the
            content was cut to fit it); ``"vendor"`` when no tokenizer is declared and the budget is the
            hosted vendor's documented limit (content uncut).
        aggregation: How a chunked document's scores pool back onto it: ``"max"`` -- a document scores its
            best chunk, the same rule as :func:`max_pool_scores_by_document` -- recorded on every chunked
            census row. ``None`` when nothing was chunked.
        cuts: The recorded cuts (also recorded into the census passed to :func:`fit`, when one was).
    """

    shape: RequestShape
    texts: tuple[str, ...]
    contents: tuple[ContentParts, ...]
    ids: tuple[str, ...]
    chunk_mapping: dict[str, str] | None = None
    overhead: int | None = None
    budget_source: Literal["tokenizer", "vendor"] = "tokenizer"
    aggregation: Literal["max"] | None = None
    cuts: tuple[TextCutRecord, ...] = ()


ChangeMechanism = Literal[
    "budget_cut",
    "query_share",
    "document_share",
    "empty_doc",
    "media_resize",
    "media_drop",
    "skip_unapplied",
    "mrl_cut",
]
"""How a role client's preparation or postprocess changed an input row (a :class:`ProcessingRecord`'s
``mechanisms``)."""

CHANGE_MECHANISMS: tuple[ChangeMechanism, ...] = (
    "empty_doc",
    "media_resize",
    "media_drop",
    "document_share",
    "query_share",
    "budget_cut",
    "skip_unapplied",
    "mrl_cut",
)
"""Every :data:`ChangeMechanism`, in the order a client applies them (a record lists its mechanisms so;
both postprocess mechanisms -- the skip exemption and the Matryoshka head -- come last, after the reply)."""


@dataclass(frozen=True)
class ProcessingRecord:
    """What a role client's preparation changed in one input row, relative to the row as given.

    The one per-row record of the role clients (:attr:`~rcp_ndcg.inference.clients._base.RoleClient.processing`
    collects them): a client emits one for every input row it changed before sending it, and none for a row it
    sent as given -- so a consumer decides "changed or not" from the record alone (the equivalence harness
    reports a changed row under a declared over-cap deviation and gates an unchanged one exactly). Each change
    is named by its mechanism (:data:`CHANGE_MECHANISMS`):

    * ``empty_doc`` -- an empty document substituted (``empty_doc: send_text``) or never sent (``omit_zero``);
      an empty query is refused or sent as given (``empty_query``), so it never changes a row;
    * ``media_resize`` / ``media_drop`` -- the request's text budget shrank a media item below its prepared
      size, or dropped it (:func:`~rcp_ndcg.data.prepare.fit_media_to_budget`; the media census holds the
      items). The declared image policy's own resize is the instrument, applied alike by the reference
      (R20), not a change to the row;
    * ``document_share`` -- a document cut to the declared ``document_max_tokens``;
    * ``query_share`` -- the reranker's shared query settled at its declared share (row ``<query>``,
      :data:`~rcp_ndcg.inference.clients.rerank.QUERY_DOC_ID`; it rides every pair of its call);
    * ``budget_cut`` -- the content cut so that the request -- frame, specials, content and media -- fits its
      shape's budget (also when the content alone fits it);
    * ``skip_unapplied`` -- the declared ``document_skip_token_ids`` was not applied to this row: a media
      request's vector positions are the engine's chat-template render, which the client cannot tokenise,
      so the image positions are exempt and the client keeps every returned vector (the postprocess, after
      the reply).
    * ``mrl_cut`` -- the Matryoshka head applied the declared ``mrl_dim`` to this row's vectors (the
      truncation cut or the checkpoint's learned projection, after the reply). ``mrl_kind``, ``mrl_dim``
      and ``full_width`` record what was applied and to what.

    The text mechanisms read the census rows the cut wrote (:class:`TextCutRecord`): nothing is measured twice.

    Attributes:
        corpus: The role that prepared the row (the census rows' ``corpus``).
        input_id: The row's id within its call: its position among the call's inputs (``"0"``, ``"1"``, ...),
            or the reranker's shared query (``"<query>"``). A chunked document is one row.
        shape: The request shape the row was prepared as.
        mechanisms: Every change, by mechanism, in :data:`CHANGE_MECHANISMS` order; never empty.
        original_request_tokens: The uncut request's whole size as the engine would read it (frame, specials,
            content, reserved media), when a text mechanism measured it; ``None`` otherwise.
        kept_request_tokens: The sent request's whole size (the largest chunk's, for a chunked document), when
            measured; ``None`` otherwise.
        budget_tokens: The shape's budget the text mechanism measured against, when one did.
        mrl_kind: The declared Matryoshka kind applied to this row (``mrl_cut`` only); ``None`` otherwise.
        mrl_dim: The selected output dimension applied to this row (``mrl_cut`` only); ``None`` otherwise.
        full_width: The width of the row's vectors before the head ran (``mrl_cut`` only); ``None``
            otherwise -- the number that proves the cut narrowed full-width vectors.
    """

    corpus: str
    input_id: str
    shape: RequestShape
    mechanisms: tuple[ChangeMechanism, ...]
    original_request_tokens: int | None = None
    kept_request_tokens: int | None = None
    budget_tokens: int | None = None
    mrl_kind: MrlKind | None = None
    mrl_dim: int | None = None
    full_width: int | None = None

    @property
    def changed(self) -> bool:
        """Whether the client changed the row (always, for an emitted record; the harness's gating test)."""
        return bool(self.mechanisms)

    def as_row(self) -> dict[str, Any]:
        """The record as a JSON-ready row (a report's)."""
        return {
            "corpus": self.corpus,
            "input_id": self.input_id,
            "shape": self.shape,
            "mechanisms": list(self.mechanisms),
            "original_request_tokens": self.original_request_tokens,
            "kept_request_tokens": self.kept_request_tokens,
            "budget_tokens": self.budget_tokens,
            "mrl_kind": self.mrl_kind,
            "mrl_dim": self.mrl_dim,
            "full_width": self.full_width,
        }


def processing_records(
    corpus: str,
    shape: RequestShape,
    *,
    cuts: Sequence[TextCutRecord] = (),
    changes: Mapping[str, Sequence[ChangeMechanism]] | None = None,
    chunk_mapping: Mapping[str, str] | None = None,
) -> list[ProcessingRecord]:
    """The :class:`ProcessingRecord` of every row one preparation changed: its text cuts (read from the census
    rows ``fit`` or the rerank settlement wrote -- a chunk's ``<id>#<k>`` row counts for its input ``<id>``,
    through the fit's own ``chunk_mapping``) and the other mechanisms the client noted per input id (media,
    empty substitutions).

    Args:
        corpus: The role's name.
        shape: The shape the rows were prepared as.
        cuts: The text-budget cut rows of the preparation (vendor budget rows carry no cause and are skipped).
        changes: Per input id, the other mechanisms applied.
        chunk_mapping: The fit's chunk id -> input id mapping (:attr:`FitResult.chunk_mapping`), the
            authority for which input a chunk row names: a row the mapping does not name (the reranker's
            settlement under ``<query>``, a per-part row) is its own id, and an input id that itself
            contains the chunk separator is never mis-split.

    Returns:
        One record per changed input id, in the order the ids first appear (``changes`` first, then the cuts).
    """
    mechanisms: dict[str, set[ChangeMechanism]] = {}
    totals: dict[str, list[TextCutRecord]] = {}
    for input_id, applied in (changes or {}).items():
        if applied:
            mechanisms.setdefault(input_id, set()).update(applied)
    for cut in cuts:
        if cut.cause is None:
            continue
        input_id = (
            document_id_for_chunk(cut.doc_id, chunk_mapping)
            if chunk_mapping and cut.doc_id in chunk_mapping
            else cut.doc_id
        )
        mechanisms.setdefault(input_id, set()).add(cut.cause)
        totals.setdefault(input_id, []).append(cut)
    records = []
    for input_id, applied in mechanisms.items():
        rows = totals.get(input_id, [])
        original = [row.original_request_tokens for row in rows if row.original_request_tokens is not None]
        kept = [row.kept_request_tokens for row in rows if row.kept_request_tokens is not None]
        budgets = [row.budget_tokens for row in rows if row.budget_tokens is not None]
        records.append(
            ProcessingRecord(
                corpus=corpus,
                input_id=input_id,
                shape=shape,
                mechanisms=tuple(mechanism for mechanism in CHANGE_MECHANISMS if mechanism in applied),
                original_request_tokens=max(original) if original else None,
                kept_request_tokens=max(kept) if kept else None,
                budget_tokens=max(budgets) if budgets else None,
            )
        )
    return records


class TextBudget(BaseModel):
    """The declared text budget of a served role: one mechanism that fits every request into a model's input.

    The budget counts the model's whole input sequence -- the template's fixed segments, their special
    tokens, the instruction and the content together -- in tokens of the declared tokenizer. The fixed
    overhead is measured by rendering the template once with every content span empty (per request shape,
    the instruction filled: it is fixed for the run), only the content spans are cut, with the offset-based
    :func:`token_prefix`, and the template is re-attached after the cut. The engine never cuts: no request
    asks for engine-side truncation, so the declared budget is the only budget.

    Frozen and part of the content identity: :meth:`identity` returns the payload -- the template's
    canonical JSON, the budgets, the overflow policy, the chunk geometry, and the tokenizer *file's*
    SHA-256 (its name is runtime, as the judge's is).

    Attributes:
        tokenizer: The tokenizer the budget counts in: a Hugging Face repository id with an optional
            ``@revision``, or a local path to a ``tokenizer.json``. ``None`` only for a hosted vendor
            profile without one: content is then sent uncut, the declared ``max_tokens`` is the vendor's
            documented limit, and nothing is measured or cut client-side. Runtime by name; the file's
            SHA-256 is content (:meth:`identity`).
        max_tokens: The budget: the largest total input sequence, in the declared tokenizer's tokens
            (``_tokens``). Required: a budget without a number is not a budget.
        query_max_tokens: The query's budget. On a ``pair`` budget it is the query's share: when a pair
            overflows, the query is cut to it first and the document gets what remains (an input under
            budget is sent byte-identical to the uncut render, so the share binds on overflow only). On the
            ``query`` shape (an embedding role's per-shape budget) it is that shape's WHOLE budget --
            ``max_tokens`` then caps the document shape only. ``None`` (the default) declares no split: on
            a pair, a query that does not fit the budget is then refused rather than cut undeclared (declare
            the split instead). A share above ``max_tokens`` is refused here; EQUALITY is legal (both shapes
            capped the same) -- a pair share at or over the budget is refused one layer up, by the rerank
            config, and ``fit``'s pair cut refuses a query whose settled render would leave the document
            nothing.
        document_max_tokens: The document's own cap on a ``pair`` budget, in content tokens -- for a checkpoint
            that cuts each document itself (jina-reranker-v3 reads 2048 document tokens): a document over it is
            cut to it, also in a pair the budget would take whole (the model never reads past it), the frame
            re-attached and the cut recorded (``cause: document_share``), before the pair is fitted to
            ``max_tokens``. The pair shape only (an embedding role's document shape is capped by
            ``max_tokens``). ``None`` (the default) declares no cap. Above ``max_tokens`` it is refused (the
            rerank config refuses one at or over it: it could never bind), and beside ``on_overflow: chunk``
            too (the cap and the chunks would decide the same document two ways).
        template: The request template (:class:`~rcp_ndcg.data.templates.TemplateSpec`), whose fixed
            segments are measured once per (template, shape) and whose specials are resolved from the
            tokenizer. ``None`` fits raw text: the overhead is then the tokenizer post-processor's tokens
            under ``add_special_tokens=True`` -- the pooling routes' engine default -- so the appended
            anchor is reserved without a frame.
        on_overflow: What an input that does not fit does: ``cut`` (the default: the content is cut,
            every cut recorded in the census under ``text_budget``), ``chunk`` (the document span is split
            into :class:`ChunkPolicy` chunks, each carrying the full template), or ``fail`` (the input is
            refused with :class:`TextBudgetExceededError`). A query is never chunked.
        chunk: The chunk geometry, ``on_overflow: chunk`` only -- the judge's :class:`ChunkPolicy` reused,
            not copied: chunks of at most ``chunk.max_tokens`` content tokens, each carrying the full
            template, ids ``<id>#<k>``, scores pooled back by ``max``.
        aggregation: How a chunked document's scores pool back onto it: ``max``, its best chunk's -- the
            same rule as :func:`max_pool_scores_by_document`, which the caller applies to the returned
            chunk mapping. The only value for now; every census row of a chunked input names it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {
        "tokenizer": FieldRole.RUNTIME,
        "max_tokens": FieldRole.CONTENT,
        "query_max_tokens": FieldRole.CONTENT,
        "document_max_tokens": FieldRole.CONTENT,
        "template": FieldRole.CONTENT,
        "on_overflow": FieldRole.CONTENT,
        "chunk": FieldRole.CONTENT,
        "aggregation": FieldRole.CONTENT,
    }

    tokenizer: str | None = Field(default=None, min_length=1)
    max_tokens: int = Field(ge=1)
    query_max_tokens: int | None = Field(default=None, ge=1)
    document_max_tokens: int | None = Field(default=None, ge=1, exclude_if=lambda value: value is None)
    template: TemplateSpec | None = None
    on_overflow: Literal["cut", "chunk", "fail"] = "cut"
    chunk: ChunkPolicy | None = Field(default=None, exclude_if=lambda value: value is None)
    aggregation: Literal["max"] = "max"

    @model_validator(mode="after")
    def _declared_overflow_has_its_geometry(self) -> TextBudget:
        """The same rule as ``TextPolicy``: a declared chunk policy needs its geometry, and nothing else
        carries one."""
        if self.on_overflow == "chunk" and self.chunk is None:
            raise ValueError(
                "on_overflow 'chunk' needs a chunk geometry: {on_overflow: chunk, chunk: {max_tokens: ..., "
                "overlap_tokens: ...}}"
            )
        if self.chunk is not None and self.on_overflow != "chunk":
            raise ValueError(f"a chunk geometry applies to on_overflow 'chunk' only, not {self.on_overflow!r}")
        return self

    @model_validator(mode="after")
    def _a_share_above_the_budget_is_meaningless(self) -> TextBudget:
        """A query share above the whole budget leaves the document nothing and the query over the served
        context. Equal is legal: on the embedding roles ``query_max_tokens`` is the query shape's WHOLE
        budget, and both shapes may be capped the same; on a pair budget an equal share is refused one layer
        up (the rerank config), and ``fit``'s pair cut refuses a query that would leave the document nothing
        at runtime."""
        if self.query_max_tokens is not None and self.query_max_tokens > self.max_tokens:
            raise ValueError(
                f"query_max_tokens ({self.query_max_tokens}) must not exceed max_tokens ({self.max_tokens}): "
                "the query budget would be over the model's whole input budget"
            )
        if self.document_max_tokens is not None and self.document_max_tokens > self.max_tokens:
            raise ValueError(
                f"document_max_tokens ({self.document_max_tokens}) must not exceed max_tokens ({self.max_tokens}): "
                "the document cap would be over the model's whole input budget"
            )
        if self.document_max_tokens is not None and self.on_overflow == "chunk":
            raise ValueError(
                "document_max_tokens cuts every document to its cap, and on_overflow 'chunk' splits an over-budget "
                "document into chunks instead: the two would decide the same document two ways; declare one"
            )
        return self

    @model_validator(mode="after")
    def _a_budget_without_a_tokenizer_cuts_nothing(self) -> TextBudget:
        """A hosted vendor profile without a tokenizer sends content uncut: a template, chunk/fail and a
        query split would be silently inert, so they are refused instead of ignored."""
        if self.tokenizer is None:
            inert = [
                name
                for name, value in (
                    ("on_overflow", self.on_overflow),
                    ("query_max_tokens", self.query_max_tokens),
                    ("document_max_tokens", self.document_max_tokens),
                    ("chunk", self.chunk),
                    ("template", self.template),
                )
                if value is not None and value != "cut"
            ]
            if inert:
                raise ConfigError(
                    f"this budget declares no tokenizer, so its content is sent uncut (a hosted vendor "
                    f"profile) and {inert} would be inert",
                    hint="declare tokenizer (the profile then cuts like a self-hosted one), or drop the "
                    "inert fields (on_overflow, query_max_tokens, document_max_tokens, chunk, template)",
                )
        return self

    def shape_max_tokens(self, shape: RequestShape) -> int:
        """The token budget one request of ``shape`` is measured against: on the embedding roles
        ``query_max_tokens`` caps the query shape whole (its whole budget there); on a pair it stays the
        query's share inside ``max_tokens``, and on the document shape (and without a declared share)
        ``max_tokens`` caps. :func:`fit` measures every cut against it, and a role client's media allowance
        counts from it, so the two thresholds never disagree.

        Args:
            shape: The request shape.

        Returns:
            The shape's budget, in the declared tokenizer's tokens.
        """
        if self.query_max_tokens is not None and shape == "query":
            return self.query_max_tokens
        return self.max_tokens

    def identity(self, tokenizer: TextTokenizer | None = None) -> dict[str, Any]:
        """The content identity payload of the budget, with the tokenizer file's SHA-256 -- the one field
        the method exists to carry, so a budget that declares a tokenizer is never identified without it
        (two tokenizers' files are not told apart by name, which is RUNTIME): a loaded tokenizer is
        required, and it must be the budget's. A hosted vendor profile (a budget that declares no tokenizer)
        carries no hash: nothing is measured against it.

        Args:
            tokenizer: The budget's loaded tokenizer, when the budget declares one.

        Returns:
            The identity payload (with ``tokenizer_sha256`` when the budget declares a tokenizer).

        Raises:
            ConfigError: the budget declares a tokenizer and none was loaded (the identity would collide
                with a different tokenizer file's), or the loaded tokenizer's name is not the budget's (its
                hash beside content-true fields would mis-describe the budget).
        """
        payload = identity_payload(self)
        if self.tokenizer is not None:
            if tokenizer is None:
                raise ConfigError(
                    f"the budget declares tokenizer {self.tokenizer!r} and identity() was called without the "
                    "loaded tokenizer: the SHA-256 is the one field this payload exists to carry, and it "
                    "cannot be omitted on demand",
                    hint="load the budget's tokenizer and pass it: the identity is taken where the "
                    "tokenizer is already loaded",
                )
            if tokenizer.name != self.tokenizer:
                raise ConfigError(
                    f"the loaded tokenizer {tokenizer.name!r} is not the budget's {self.tokenizer!r}: its "
                    "hash would be recorded beside content-true fields and mis-describe the budget",
                    hint="load the tokenizer the budget declares",
                )
            payload["tokenizer_sha256"] = tokenizer.sha256
        return payload


_VENDOR_WARNED: set[tuple[str, str]] = set()


def _fit_vendor(
    items: list[Any],
    shape: RequestShape,
    budget: TextBudget,
    names: Sequence[str],
    *,
    corpus: str,
    census: TextTruncationCensus | None,
) -> FitResult:
    """The hosted-vendor path: no tokenizer, so nothing is measured, framed or cut; the declared
    ``max_tokens`` (the vendor's documented limit) is recorded as the effective budget -- one row per
    (corpus, limit) per census, and one warning per (corpus) per process."""
    key = (TextTruncationCensus.TEXT_BUDGET, corpus)
    if key not in _VENDOR_WARNED:
        _VENDOR_WARNED.add(key)
        logger.warning(
            "text budget (%s) on corpus %s: no tokenizer declared for this hosted profile; content is sent "
            "uncut and the vendor's documented limit (%d tokens) is the effective budget (budget_source: "
            "vendor); nothing client-side is measured or cut",
            TextTruncationCensus.TEXT_BUDGET,
            corpus or "<unnamed>",
            budget.max_tokens,
        )
    if census is not None and census.claim_budget_row(corpus, budget.max_tokens):
        census.record(
            corpus=corpus,
            doc_id=BUDGET_DOC_ID,
            original_tokens=budget.max_tokens,
            kept_tokens=budget.max_tokens,
            original_chars=0,
            kept_chars=0,
            mechanism=TextTruncationCensus.TEXT_BUDGET,
            budget_source="vendor",
            shape=shape,
            budget_tokens=budget.max_tokens,
        )
    contents: list[ContentParts] = [item if shape != "pair" else (item[0], item[1]) for item in items]
    texts: tuple[str, ...] = () if shape == "pair" else tuple(part for part in contents if isinstance(part, str))
    return FitResult(
        shape=shape,
        # A hosted profile sends the content itself (or the pair's parts): nothing is rendered.
        texts=texts,
        contents=tuple(contents),
        ids=tuple(names),
        overhead=None,
        budget_source="vendor",
    )


def fixed_overhead(
    budget: TextBudget, tokenizer: TextTokenizer | None, shape: RequestShape, *, instruction: str = ""
) -> int:
    """The fixed token cost of one request's frame, measured as :func:`fit` measures it: the template
    rendered once with every content span empty (the instruction filled -- it is fixed for the run), counted
    as the engine reads the route (the shape's ``add_special_tokens`` flag, its post-processor tokens
    included). Without a template the overhead is the post-processor's tokens on the raw text (the routes'
    default), so an appended anchor is reserved even with no frame.

    The one home of the overhead: :func:`fit` reserves it before cutting, and a role client that bounds its
    media against what the text will actually have left calls this first -- a media allowance computed from
    ``max_tokens`` alone lands in the dead zone where the media alone fit the budget but the template's
    fixed tokens no longer leave room for any.

    Args:
        budget: The declared text budget.
        tokenizer: The loaded tokenizer the budget declares (``None``: the hosted-vendor path, where nothing
            is measured -- the overhead is 0 because nothing client-side is known).
        shape: The request shape the overhead is measured for.
        instruction: The run-level instruction, where the template declares an ``instruction`` span.

    Returns:
        The overhead in tokens (``0`` without a tokenizer: the hosted-vendor path measures nothing).
    """
    if tokenizer is None:
        return 0
    template = budget.template
    if template is not None:
        return template.overhead(shape, tokenizer, instruction=instruction or "")
    return tokenizer.count("", add_special_tokens=True)


def rendered_request(
    budget: TextBudget,
    tokenizer: TextTokenizer,
    shape: RequestShape,
    *,
    query: str,
    document: str,
    instruction: str = "",
) -> str:
    """The full rendered request -- the template's fixed segments re-attached around the content spans,
    the instruction filled where the template declares a span -- as :func:`fit` assembles and verifies it.
    Without a template: the spans concatenated in shape order (``query + document`` for a pair).

    The one home of the render: :func:`fit` cuts against it, and a caller that checks a shipped request
    against the budget (the rerank pair fit) counts the same render, so the two can never disagree about
    what the engine reads.
    """
    if budget.template is None:
        return query + document if shape == "pair" else (query if shape == "query" else document)
    return budget.template.render(shape, tokenizer, query=query, document=document, instruction=instruction)


def rendered_pair_tokens(
    budget: TextBudget, tokenizer: TextTokenizer, *, query: str, document: str, instruction: str = ""
) -> int:
    """The token count of one pair's assembled render, exactly as :func:`fit` verifies a fitted pair (the
    shape's ``add_special_tokens`` flag applied). A rerank client checks every shipped pair against the
    budget with this -- the same measure the fit cut to, so a pair the fit verified passes here."""
    rendered = rendered_request(budget, tokenizer, "pair", query=query, document=document, instruction=instruction)
    flag = budget.template.adds_special_tokens("pair") if budget.template is not None else True
    return tokenizer.count(rendered, add_special_tokens=flag)


def fit(
    inputs: Sequence[str] | Sequence[tuple[str, str]],
    shape: RequestShape,
    budget: TextBudget,
    tokenizer: TextTokenizer | None = None,
    *,
    ids: Sequence[str] | None = None,
    instruction: str | None = None,
    media_tokens: Sequence[int] | None = None,
    corpus: str = "",
    census: TextTruncationCensus | None = None,
    parts: Sequence[Sequence[str] | tuple[Sequence[str], Sequence[str]]] | None = None,
) -> FitResult:
    """Fit every input into the model's input budget: the one call every served role's client makes.

    The algorithm, per request shape:

    1. **Measure the fixed overhead once** per (template, shape, tokenizer, instruction): the template
       rendered with every content span empty, counted as the engine reads it (the shape's
       ``add_special_tokens`` flag -- its post-processor tokens included). Without a template the
       overhead is the post-processor's tokens on the raw text (the routes' default), so an appended
       anchor is reserved even with no frame.
    2. **Budget the content spans**: ``remaining = max_tokens - overhead - media``. For a ``pair``, the
       query is cut to ``query_max_tokens`` first (or, with none declared, only when it alone fills the
       budget) and the document gets the rest.
    3. **Cut only the content spans**, with the offset-based :func:`token_prefix`, verified against the
       *assembled* render so a byte-level merge across a span join cannot push the request over the
       budget; the template is re-attached after the cut. Inputs under budget come back byte-identical
       to the uncut render.
    4. **Chunk** on ``on_overflow: chunk``: the document content is split into verbatim
       :class:`ChunkPolicy` slices and every chunk is rendered with the full template (engine-side
       chunking of a framed render loses the frame and the anchor -- chunk on the client, never on the
       engine). Output ids are ``<id>#<k>`` and ``FitResult.chunk_mapping`` carries each chunk back to
       its input, ready for :func:`max_pool_scores_by_document` (``aggregation: max``).
    5. **Record** every cut in the census under the ``text_budget`` mechanism, with the request shape,
       the budget source, and -- on chunked inputs -- the ``max`` aggregation.

    Media (a vision block, a video segment) are never cut: declare each input's media token count in
    ``media_tokens`` and it is reserved whole out of the budget before the content is cut. A declared
    media count that leaves no room for the frame is a :class:`ConfigError`.

    Args:
        inputs: The texts (``query`` and ``document`` shapes) or ``(query, document)`` pairs (``pair``
            shape), in order.
        shape: Which request shape the inputs are.
        budget: The declared :class:`TextBudget`.
        tokenizer: The loaded tokenizer, in whose tokens ``max_tokens`` is counted. ``None`` is the
            hosted-vendor path: content is sent uncut, the budget is the vendor's documented limit, and
            it is recorded with ``budget_source: vendor`` -- nothing is measured, so nothing is framed or
            cut (and a ``pair`` returns its parts, not rendered text).
        ids: Each input's id, for the census and the chunk ids; positional ``"<index>"`` when unset.
        instruction: The run-level instruction, where the template declares an ``instruction`` span; part
            of the fixed overhead (never cut).
        media_tokens: Each input's media token count, indivisible and reserved whole (the hook the media
            lane builds on: a vision block is counted, never cut).
        corpus: The corpus or role name, for the census and the vendor warning.
        census: Where the cuts are recorded.
        parts: Each input's text parts, in order, for the census rows of an item with several text parts
            (a served role client's interleaved content): one row per part, each with the part's own
            original and kept counts, so a cut is recorded per part where the parts stand. The joined
            parts must be the input's own text (for a ``pair``, the document side exactly and the query
            side as a prefix -- the reranker's settled query is one); the entries are
            ``(query_parts, document_parts)`` for the ``pair`` shape. A declared template normalisation
            disables the per-part rows (the parts as given are not the normalised spans the cut applies
            to), and a chunked input keeps its per-chunk rows (the chunks, not the parts, are the units).

    Returns:
        The :class:`FitResult`: the rendered strings (or, on ``text``/``token_ids`` routes without a
        frame, the contents), the cut contents per output, the output ids, the chunk mapping, the
        measured overhead, the budget source, and the cut records.

    Raises:
        ConfigError: the template or the media alone fill the budget, the declared chunk geometry cannot
            fit it, or a template special does not resolve in the tokenizer.
        TextBudgetExceededError: ``on_overflow: fail`` and an input over budget; a query under
            ``on_overflow: chunk``; a pair whose query leaves the document nothing.
    """
    items = list(inputs)
    if shape == "pair":
        for index, item in enumerate(items):
            if not (isinstance(item, (tuple, list)) and len(item) == 2 and all(isinstance(part, str) for part in item)):
                raise DataError(
                    f"inputs[{index}] must be a (query, document) pair of strings for the 'pair' shape",
                    hint="pass the pair's parts as strings (the client prepares them); this fit call was "
                    "given something else",
                )
    else:
        for index, item in enumerate(items):
            if not isinstance(item, str):
                raise DataError(
                    f"inputs[{index}] must be a string for the {shape!r} shape",
                    hint="pass the content strings (the client's preparation materialised them)",
                )
    names = [str(index) for index in range(len(items))] if ids is None else [str(name) for name in ids]
    if len(names) != len(items):
        raise DataError(
            f"ids ({len(names)}) must name every input ({len(items)})",
            hint="one id per input: the census and the chunk ids are built from them",
        )
    media = [0] * len(items) if media_tokens is None else list(media_tokens)
    if len(media) != len(items):
        raise DataError(
            f"media_tokens ({len(media)}) must be declared for every input ({len(items)})",
            hint="one media token count per input (0 where the input carries none)",
        )
    if any(not isinstance(count, int) or count < 0 for count in media):
        raise DataError(
            "media_tokens must be non-negative token counts",
            hint="the counts are the media's exact vision-block cost per input, as content_media_tokens counts them",
        )
    # The text parts per input, when the caller declares them: validated against the input's own text
    # (the join is the text the cut applies to) before anything is measured, so a mismatched declaration
    # is refused instead of recording rows against the wrong pieces. A declared template normalisation
    # disables the per-part rows entirely: the parts as given are not the normalised spans the cut applies
    # to, so the input keeps its one row (the row's original side is the raw text either way).
    normalising = budget.template is not None and bool(budget.template.normalisers(shape))
    part_lists: list[tuple[tuple[str, ...], ...]] | None = None
    if parts is not None and not normalising:
        if len(parts) != len(items):
            raise DataError(
                f"parts ({len(parts)}) must name every input ({len(items)})",
                hint="one entry per input: its text parts in order (the join of the parts is the input's text)",
            )
        part_lists = []
        for index, entry in enumerate(parts):
            if shape == "pair":
                if not (
                    isinstance(entry, (tuple, list))
                    and len(entry) == 2
                    and all(isinstance(side, (tuple, list)) for side in entry)
                    and all(all(isinstance(text, str) for text in side) for side in entry)
                ):
                    raise DataError(
                        f"parts[{index}] must be a (query_parts, document_parts) pair of text-part lists",
                        hint="pass the pair's two sides' parts in order; each side is a sequence of strings",
                    )
                sides = tuple(tuple(str(text) for text in side) for side in entry)
                query, document = items[index]  # type: ignore[misc]
                if not TEXT_JOIN.join(sides[0]).startswith(str(query)):
                    raise DataError(
                        f"the query parts of input {index} do not join to a text the input's query starts",
                        hint="the parts are the query's own text parts, in order; the fit's query is a prefix "
                        "of their join (the reranker's settled query is one)",
                    )
                if TEXT_JOIN.join(sides[1]) != str(document):
                    raise DataError(
                        f"the document parts of input {index} do not join to the input's document text",
                        hint="the parts are the document's own text parts, in order",
                    )
            else:
                if not isinstance(entry, (tuple, list)) or any(not isinstance(text, str) for text in entry):
                    raise DataError(
                        f"parts[{index}] must be the input's text parts (a sequence of strings)",
                        hint="pass the content's text parts in order; their join is the input's text",
                    )
                sides = (tuple(str(text) for text in entry),)
                if TEXT_JOIN.join(sides[0]) != str(items[index]):
                    raise DataError(
                        f"the parts of input {index} do not join to the input's text",
                        hint="pass the content's text parts in order; their join is the input's text",
                    )
            part_lists.append(sides)
    if budget.tokenizer is not None and (tokenizer is None or tokenizer.name != budget.tokenizer):
        raise ConfigError(
            f"fit was given {'no tokenizer' if tokenizer is None else f'the tokenizer {tokenizer.name!r}'} but "
            f"the budget declares {budget.tokenizer!r}: the budget's numbers are counted in the declared "
            "tokenizer's tokens",
            hint="load the budget's tokenizer and pass it; the hosted-vendor path (a tokenizer of none) is for "
            "budgets that declare none",
            cli_hint="set the same tokenizer the budget declares (judge-style: --set <role>.tokenizer=...), or "
            "drop the tokenizer field for a hosted profile",
        )
    # The shape's budget (:meth:`TextBudget.shape_max_tokens`): every cap and every message below counts
    # against the shape's own budget.
    shape_budget = budget.shape_max_tokens(shape)
    if tokenizer is None:
        if media_tokens is not None and any(media):
            raise ConfigError(
                "media_tokens need a tokenizer to reserve against, and this budget declares none (a hosted "
                "vendor profile sends content uncut): the media reservation cannot be honoured",
                hint="declare tokenizer on the budget, or drop media_tokens for this profile",
            )
        return _fit_vendor(items, shape, budget, names, corpus=corpus, census=census)

    template = budget.template
    instr = instruction or ""
    # The declared content normalisation: the template's per-shape strip/lowercase (the one call,
    # :meth:`TemplateSpec.normalize_text`), applied to the content spans before anything is measured (the
    # reference and the engine see the same text). Without a template there is no declaration, so nothing
    # is normalised.
    if template is not None and template.normalisers(shape):
        raw_items = list(items)  # the inputs as given, for the census rows' original side
        if shape == "pair":
            pairs = [(query, document) for query, document in items]
            items = [
                (template.normalize_text(shape, query), template.normalize_text(shape, document))
                for query, document in pairs
            ]
        else:
            assert all(isinstance(item, str) for item in items)  # validated at the top, for the type
            items = [template.normalize_text(shape, str(item)) for item in items]
    else:
        raw_items = items
    # The engine's behaviour for the route: declared on the template; a raw-text request gets the pooling
    # routes' default (the post-processor's tokens are appended), so its anchor is reserved either way.
    flag = template.adds_special_tokens(shape) if template is not None else True
    overhead = fixed_overhead(budget, tokenizer, shape, instruction=instr)

    def _budget_hint(verb: str, rest: str) -> str:
        """The raise hint that names the knob that binds: on a query shape budgeted by a declared
        ``query_max_tokens`` raising ``max_tokens`` moves nothing."""
        knob = "query_max_tokens" if shape == "query" and budget.query_max_tokens is not None else "max_tokens"
        return f"{verb} {knob}" + (f", {rest}" if rest else "")

    if overhead > shape_budget:
        raise ConfigError(
            f"the template's fixed overhead alone is {overhead} tokens, over the budget of {shape_budget}",
            hint=_budget_hint("raise", "or simplify the template"),
        )

    def assemble(query: str, document: str) -> str:
        """The full rendered request, the frame re-attached around whatever the spans now hold."""
        return rendered_request(budget, tokenizer, shape, query=query, document=document, instruction=instr)

    def _cut_span(text: str, *, span: Literal["query", "document"], other: str = "", cap: int) -> str:
        """The longest prefix of a content span whose assembled render fits ``cap`` (the budget minus the
        media, which ride beside the rendered string and are never cut). The piece is rendered into its OWN
        span, with the other span held at ``other``, so the verified count is the render the engine reads --
        the two joins of an asymmetric frame tokenize differently, and measuring a piece in the wrong span
        would ship an over-budget render."""
        if span == "query":
            rendered = lambda piece: assemble(piece, other)  # noqa: E731  (shape 'query' ignores ``other``; pair holds the document there)
        else:
            rendered = lambda piece: assemble(other, piece)  # noqa: E731  (shape 'document' ignores ``other``; pair holds the query there)
        return token_prefix(text, cap, tokenizer, rendered=rendered, add_special_tokens=flag)

    texts: list[str] = []
    contents: list[ContentParts] = []
    entries: list[tuple[str, str]] = []
    cuts: list[TextCutRecord] = []
    chunked_any = False

    def _part_pairs(index: int, side: int, kept_span: str) -> list[tuple[str, str]] | None:
        """The ``(original, kept)`` text of every text part of one side of input ``index``, the kept span
        distributed where the parts stand; ``None`` when the caller declared no parts."""
        if part_lists is None:
            return None
        side_parts = part_lists[index][side]
        return [
            (part, piece)
            for part, (piece, _kept) in zip(side_parts, split_text_across_parts(side_parts, kept_span), strict=True)
        ]

    def _pair_part_pairs(index: int, query: str, document: str) -> list[tuple[str, str]] | None:
        """The per-part ``(original, kept)`` rows of one pair: the query's parts then the document's,
        or ``None`` when the caller declared no parts."""
        query_pairs = _part_pairs(index, 0, query)
        document_pairs = _part_pairs(index, 1, document)
        if query_pairs is None and document_pairs is None:
            return None
        return [*(query_pairs or []), *(document_pairs or [])]

    def _record(
        *,
        doc_id: str,
        original: ContentParts,
        kept: ContentParts,
        aggregation: str | None,
        request_tokens: int,
        raw: ContentParts | None = None,
        cause: CutCause = "budget_cut",
        part_pairs: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        """One cut row per unit (also appended to the census when the caller passed one).

        ``raw`` is the input as given, when a declared normalisation changed the spans before the cut: the
        row's original side is then the raw text (the input), never the normalised one (declared policy).
        ``request_tokens`` is the uncut request's whole size as the engine would read it (the frame, its
        specials, the content and the reserved media); ``cause`` why the content changed. ``part_pairs``
        (the caller declared the input's text parts) records ONE ROW PER PART THAT CHANGED instead of the
        input's one row: each row's original and kept counts are the part's own -- the cut lands on the
        parts where they stand, so a part's own truncation is what the census shows -- while the request's
        own totals repeat on every row, exactly as a chunked input's rows repeat them. A part the cut kept
        whole records nothing, and an input whose no part changed keeps its own row (the frame-only
        overflow is still an event).
        """
        if tokenizer is None:  # a cut is recorded only on the tokenizer path (fit's guard)
            raise DataError(
                "a cut is recorded with no tokenizer to measure it",
                hint="fit's vendor path records no cuts; this is a bug in the text-budget mechanism",
            )
        if part_pairs is not None:
            rows = [(part, piece) for part, piece in part_pairs if part != piece]
        else:
            rows = []
        if not rows:  # no parts declared, or no part changed: the input's own row
            source = original if raw is None else raw
            original_text = source if isinstance(source, str) else source[0] + source[1]
            kept_text = kept if isinstance(kept, str) else kept[0] + kept[1]
            rows = [(original_text, kept_text)]
        if isinstance(kept, str):
            kept_render = assemble(kept, "") if shape == "query" else assemble("", kept)
        else:
            kept_render = assemble(kept[0], kept[1])
        kept_request_tokens = tokenizer.count(kept_render, add_special_tokens=flag) + spent
        for original_text, kept_text in rows:
            cut = TextCutRecord(
                corpus=corpus,
                doc_id=doc_id,
                original_chars=len(original_text),
                kept_chars=len(kept_text),
                original_tokens=tokenizer.count(original_text),
                kept_tokens=tokenizer.count(kept_text),
                mechanism=TextTruncationCensus.TEXT_BUDGET,
                budget_source="tokenizer",
                aggregation=aggregation,
                shape=shape,
                budget_tokens=shape_budget,
                cause=cause,
                original_request_tokens=request_tokens,
                kept_request_tokens=kept_request_tokens,
            )
            if census is not None:
                census.record(
                    corpus=cut.corpus,
                    doc_id=cut.doc_id,
                    original_chars=cut.original_chars,
                    kept_chars=cut.kept_chars,
                    original_tokens=cut.original_tokens,
                    kept_tokens=cut.kept_tokens,
                    mechanism=cut.mechanism,
                    budget_source=cut.budget_source,
                    aggregation=cut.aggregation,
                    shape=cut.shape,
                    budget_tokens=cut.budget_tokens,
                    cause=cut.cause,
                    original_request_tokens=cut.original_request_tokens,
                    kept_request_tokens=cut.kept_request_tokens,
                )
            cuts.append(cut)

    def _chunks(content: str, room: int, query: str, cap: int) -> list[str]:
        """The document's chunks under the declared geometry, each guaranteed to render under the budget:
        the whole template is re-attached per chunk, so the frame and its anchors survive every one."""
        assert budget.chunk is not None  # validated with on_overflow
        if budget.chunk.max_tokens > room:
            raise ConfigError(
                f"chunk.max_tokens ({budget.chunk.max_tokens} tokens of content) cannot fit the {room} tokens "
                f"left under the budget of {shape_budget}: every chunk carries the full template",
                hint="lower chunk.max_tokens, raise max_tokens, or (for a pair) set query_max_tokens",
            )
        pieces = split_into_chunks(content, budget.chunk, tokenizer)
        for index, piece in enumerate(pieces):
            # A slice can re-tokenize longer on its own than it did in place (a byte-level join can inflate
            # where the frame meets it); trim it until the render fits. The trimmed tail is counted: the
            # chunk's own census row records the trimmed kept size against the whole input.
            if tokenizer.count(assemble(query, piece), add_special_tokens=flag) > cap:
                pieces[index] = _cut_span(piece, span="document", other=query, cap=cap)
        return pieces

    for index, item in enumerate(items):
        input_id = names[index]
        spent = media[index]
        # The item's total: the budget minus the media, which ride beside the rendered string and are never cut.
        cap = shape_budget - spent
        if overhead + spent > shape_budget:
            raise ConfigError(
                f"the fixed template overhead ({overhead} tokens) plus the declared media ({spent}) already "
                f"fill the budget of {shape_budget}; the media are never cut",
                hint=_budget_hint("raise", "or shrink the declared media (a media block is indivisible)"),
            )
        # The census rows compare the input AS GIVEN with what ships: normalisation is declared policy,
        # not a cut, so the row's original side stays the raw text even when the spans were normalised.
        raw = raw_items[index]
        if shape == "pair":
            assert isinstance(item, (tuple, list))
            assert isinstance(raw, (tuple, list))
            query, document = item
            original: ContentParts = (query, document)
        else:
            assert isinstance(item, str) and isinstance(raw, str)
            query, document = (item, "") if shape == "query" else ("", item)
            original = item
        uncut_tokens = tokenizer.count(assemble(query, document), add_special_tokens=flag)
        # The uncut request's whole size as the engine would read it: every census row of this input names it.
        request_tokens = uncut_tokens + spent
        # A declared per-document cap binds first, whatever the budget says: the checkpoint never reads past it
        # (the content span only, the frame re-attached by the render below).
        document_cap = budget.document_max_tokens if shape == "pair" else None
        if document_cap is not None and tokenizer.count(document) > document_cap:
            document = token_prefix(document, document_cap, tokenizer)
            uncut_tokens = tokenizer.count(assemble(query, document), add_special_tokens=flag)
            if uncut_tokens <= cap:
                if template is not None:
                    texts.append(assemble(query, document))
                contents.append((query, document))
                entries.append((input_id, input_id))
                _record(
                    doc_id=input_id,
                    original=original,
                    kept=(query, document),
                    aggregation=None,
                    request_tokens=request_tokens,
                    raw=raw,
                    cause="document_share",
                    part_pairs=_pair_part_pairs(index, query, document),
                )
                continue
        if uncut_tokens <= cap:
            if not (template is None and shape == "pair"):
                texts.append(assemble(query, document))
            contents.append(original)
            entries.append((input_id, input_id))
            continue
        # Over budget: fail, cut, or chunk -- the declared policy and nothing else.
        if budget.on_overflow == "fail":
            media_note = f", the declared media {spent}" if spent else ""
            raise TextBudgetExceededError(
                f"input {input_id!r} is {tokenizer.count(query + document)} tokens of content, over the "
                f"declared text budget of {shape_budget} (the fixed template takes {overhead}{media_note})",
                hint="set on_overflow: 'cut' (or 'chunk' for documents) to shorten it, or " + _budget_hint("raise", ""),
            )
        if shape == "pair":
            # The query's span is settled first: to its declared share, else only when it fits the budget whole.
            if budget.query_max_tokens is None:
                if tokenizer.count(query) > cap - overhead:
                    raise TextBudgetExceededError(
                        f"the query of input {input_id!r} does not fit the pair budget of {shape_budget} "
                        "tokens, and no split is declared (query_max_tokens): cutting it undeclared would "
                        "silently eat the document's share",
                        hint="set query_max_tokens to the query's share, so the document keeps the rest",
                    )
                q_final = query
            else:
                share = budget.query_max_tokens
                q_final = query if tokenizer.count(query) <= share else token_prefix(query, share, tokenizer)
            # The query must leave room for the frame (and the post-processor's anchor) even with an empty document.
            q_final = _cut_span(q_final, span="query", other="", cap=cap)
            q_min = tokenizer.count(assemble(q_final, ""), add_special_tokens=flag)
            if q_min >= cap and tokenizer.count(document) > 0:
                raise TextBudgetExceededError(
                    f"the query of input {input_id!r} fills the pair budget of {shape_budget} tokens and "
                    "leaves the document nothing",
                    hint="lower query_max_tokens (or raise max_tokens), so the document keeps a share",
                )
            room = cap - q_min
            if budget.on_overflow == "cut":
                d_final = _cut_span(document, span="document", other=q_final, cap=cap)
                if (
                    tokenizer.count(assemble(q_final, d_final), add_special_tokens=flag) > cap
                ):  # pragma: no cover - guarded by construction
                    raise DataError(
                        f"the assembled render of input {input_id!r} exceeds the budget of {shape_budget} "
                        "tokens after both spans were verified: an internal invariant broke; report this",
                        hint="this is a bug in the text-budget mechanism: report it with the inputs",
                    )
                if not (template is None and shape == "pair"):
                    texts.append(assemble(q_final, d_final))
                contents.append((q_final, d_final))
                entries.append((input_id, input_id))
                _record(
                    doc_id=input_id,
                    original=original,
                    kept=(q_final, d_final),
                    aggregation=None,
                    request_tokens=request_tokens,
                    raw=raw,
                    part_pairs=_pair_part_pairs(index, q_final, d_final),
                )
            else:
                pieces = _chunks(document, room, q_final, cap)
                if len(pieces) == 1:
                    # One piece is the whole document: one request, its own id, no chunking (as
                    # chunk_ranking_example keeps an unsplit document).
                    if not (template is None and shape == "pair"):
                        texts.append(assemble(q_final, pieces[0]))
                    contents.append((q_final, pieces[0]))
                    entries.append((input_id, input_id))
                    _record(
                        doc_id=input_id,
                        original=original,
                        kept=(q_final, pieces[0]),
                        aggregation=None,
                        request_tokens=request_tokens,
                        raw=raw,
                        part_pairs=_pair_part_pairs(index, q_final, pieces[0]),
                    )
                    continue
                for k, piece in enumerate(pieces):
                    chunk_id = f"{input_id}{CHUNK_ID_SEPARATOR}{k}"
                    if not (template is None and shape == "pair"):
                        texts.append(assemble(q_final, piece))
                    contents.append((q_final, piece))
                    entries.append((chunk_id, input_id))
                    _record(
                        doc_id=chunk_id,
                        original=original,
                        kept=(q_final, piece),
                        aggregation=budget.aggregation,
                        request_tokens=request_tokens,
                        raw=raw,
                    )
                chunked_any = True
        elif budget.on_overflow == "chunk":
            if shape == "query":
                raise TextBudgetExceededError(
                    f"input {input_id!r} does not fit the budget of {shape_budget} tokens, and a query is "
                    "never chunked: queries are cut or refused, never split",
                    hint=_budget_hint("raise", "or shorten the query"),
                )
            assert isinstance(item, str)  # a pair chunked above; this branch is single-text only
            pieces = _chunks(item, cap - overhead, "", cap)
            if len(pieces) == 1:
                if (
                    tokenizer.count(assemble("", pieces[0]), add_special_tokens=flag) > cap
                ):  # pragma: no cover - guarded by construction
                    raise DataError(
                        f"the assembled render of input {input_id!r} exceeds the budget of {shape_budget} "
                        "tokens after the span was verified: an internal invariant broke; report this",
                        hint="this is a bug in the text-budget mechanism: report it with the inputs",
                    )
                texts.append(assemble("", pieces[0]))
                contents.append(pieces[0])
                entries.append((input_id, input_id))
                _record(
                    doc_id=input_id,
                    original=item,
                    kept=pieces[0],
                    aggregation=None,
                    request_tokens=request_tokens,
                    raw=raw,
                    part_pairs=_part_pairs(index, 0, pieces[0]),
                )
                continue
            for k, piece in enumerate(pieces):
                chunk_id = f"{input_id}{CHUNK_ID_SEPARATOR}{k}"
                texts.append(assemble("", piece))
                contents.append(piece)
                entries.append((chunk_id, input_id))
                _record(
                    doc_id=chunk_id,
                    original=item,
                    kept=piece,
                    aggregation=budget.aggregation,
                    request_tokens=request_tokens,
                    raw=raw,
                )
            chunked_any = True
        else:  # cut
            assert isinstance(item, str)  # the pair's cut is handled above
            kept = _cut_span(item, span="query" if shape == "query" else "document", cap=cap)
            if not (template is None and shape == "pair"):
                rendered = assemble(kept, "") if shape == "query" else assemble("", kept)
                if (
                    tokenizer.count(rendered, add_special_tokens=flag) > cap
                ):  # pragma: no cover - guarded by construction
                    raise DataError(
                        f"the assembled render of input {input_id!r} exceeds the budget of {shape_budget} "
                        "tokens after the span was verified: an internal invariant broke; report this",
                        hint="this is a bug in the text-budget mechanism: report it with the inputs",
                    )
                texts.append(rendered)
            contents.append(kept)
            entries.append((input_id, input_id))
            _record(
                doc_id=input_id,
                original=item,
                kept=kept,
                aggregation=None,
                request_tokens=request_tokens,
                raw=raw,
                part_pairs=_part_pairs(index, 0, kept),
            )

    out = [entry[0] for entry in entries]
    if len(set(out)) != len(out):
        duplicates = sorted({name for name in out if out.count(name) > 1})
        raise DataError(
            f"two outputs share an id ({duplicates}): a chunk of one input collides with another input's id, "
            "and one score would be pooled over the other",
            hint="pass ids that do not collide with any input id plus its '<id>#<k>' chunks",
        )
    return FitResult(
        shape=shape,
        texts=tuple(texts),
        contents=tuple(contents),
        ids=tuple(out),
        chunk_mapping=dict(entries) if chunked_any else None,
        overhead=overhead,
        budget_source="tokenizer",
        aggregation=budget.aggregation if chunked_any else None,
        cuts=tuple(cuts),
    )


__all__ = [
    "BUDGET_DOC_ID",
    "fixed_overhead",
    "rendered_pair_tokens",
    "rendered_request",
    "CHANGE_MECHANISMS",
    "CUT_CAUSES",
    "ChangeMechanism",
    "ContentParts",
    "CutCause",
    "FitResult",
    "ProcessingRecord",
    "TextBudget",
    "TextBudgetExceededError",
    "TextCutRecord",
    "TextTruncationCensus",
    "fit",
    "processing_records",
]
