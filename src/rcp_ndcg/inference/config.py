"""The role endpoint configs: an :class:`~rcp_ndcg.inference.endpoint.Endpoint` per role, with its wire default.

Each role sets the fields its wire protocol needs; every field is declared CONTENT or RUNTIME, so
:func:`rcp_ndcg.support.identity.check_declarations` passes and the roles' configs can feed identities. These
configs are not wired into :mod:`rcp_ndcg.retrieval.config` yet (the retrieval-config wiring does
that); they are the frozen shapes the later lanes build against.

The text budget is explicit: a self-hosted role names its tokenizer and its ``max_tokens`` -- without
them the config is refused, with the two fields in the hint. A hosted vendor profile without a
tokenizer may declare only ``max_tokens`` (the vendor's documented limit): the content is sent uncut
and the limit is recorded as the effective budget (``budget_source: vendor``). With a tokenizer, a
vendor profile follows the same rule as self-hosted.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field, model_validator

from rcp_ndcg.data.preprocess import ChunkPolicy
from rcp_ndcg.data.templates import TemplateSpec
from rcp_ndcg.errors import ConfigError
from rcp_ndcg.inference.endpoint import Endpoint
from rcp_ndcg.support.identity import FieldRole

SELF_HOSTED_APIS = frozenset({"openai_embeddings", "vllm_pooling", "rerank"})
"""The wire adapters a self-hosted engine speaks. A role config with one of these ``api`` values must declare
its text budget explicitly -- ``tokenizer`` and ``max_tokens`` -- because the package does the cutting itself;
any other ``api`` is a hosted vendor profile, which may declare only the vendor's documented ``max_tokens``
limit (no tokenizer, nothing client-side is measured or cut)."""


def _require_explicit_budget(config: EmbeddingEndpoint | RerankEndpoint) -> None:
    """Decision 1 of the budget lane: no implicit default budget. A self-hosted role must name both fields;
    a tokenizer without a number counts nothing, hosted or not. A hosted vendor profile may name only the
    limit (its documented budget); with neither, nothing is declared and nothing is cut or measured."""
    if config.tokenizer is not None and config.max_tokens is None:
        raise ConfigError(
            f"{type(config).__name__} declares a tokenizer ({config.tokenizer!r}) without max_tokens: a budget "
            "without a number counts nothing",
            hint="declare both fields: tokenizer: <repo id@revision or tokenizer.json path> and "
            "max_tokens: <the model's whole input budget, in the tokenizer's tokens>",
        )
    if config.api in SELF_HOSTED_APIS and (config.tokenizer is None or config.max_tokens is None):
        raise ConfigError(
            f"{type(config).__name__} with api {config.api!r} is self-hosted and must declare its text budget "
            "explicitly (the fields tokenizer and max_tokens): the package cuts the content itself, so it must "
            "know the tokenizer and the budget",
            hint="declare both fields: tokenizer: <repo id@revision or tokenizer.json path> and max_tokens: "
            "<the model's whole input budget, in that tokenizer's tokens> (a hosted vendor profile may declare "
            "only max_tokens, its documented limit)",
        )


def _chunk_geometry_matches_overflow(config: EmbeddingEndpoint | RerankEndpoint) -> None:
    """A declared chunk geometry belongs to ``on_overflow: chunk`` and it needs one: the same rule the judge's
    ``TextPolicy`` applies, refused (never ignored) here too."""
    matches = (config.on_overflow == "chunk") == (config.chunk is not None)
    if not matches:
        if config.on_overflow == "chunk":
            raise ValueError(
                "on_overflow 'chunk' needs a chunk geometry: "
                "{on_overflow: chunk, chunk: {max_tokens: ..., overlap_tokens: ...}}"
            )
        raise ValueError(
            f"on_overflow {config.on_overflow!r} declares a chunk geometry, which applies to on_overflow 'chunk' only"
        )


def _empty_doc_pairing(config: EmbeddingEndpoint | RerankEndpoint) -> None:
    """``send_text`` names its placeholder text, and nothing else carries one."""
    if config.empty_doc == "send_text" and config.empty_doc_text is None:
        raise ValueError(
            "empty_doc 'send_text' needs empty_doc_text: the literal placeholder text the empty document is sent as"
        )
    if config.empty_doc != "send_text" and config.empty_doc_text is not None:
        raise ValueError(f"empty_doc_text applies to empty_doc 'send_text' only, not {config.empty_doc!r}")


def _no_inert_overflow_policies(config: EmbeddingEndpoint | RerankEndpoint) -> None:
    """Without a tokenizer the content is sent uncut (a hosted vendor profile): an overflow policy that needs
    one would be silently inert, so it is refused instead of ignored."""
    if config.tokenizer is not None:
        return
    inert: list[str] = []
    if config.on_overflow != "cut":
        inert.append("on_overflow")
    if getattr(config, "query_max_tokens", None) is not None:
        inert.append("query_max_tokens")
    if config.chunk is not None:
        inert.append("chunk")
    if inert:
        raise ConfigError(
            f"{type(config).__name__} declares no tokenizer, so its content is sent uncut (a hosted vendor "
            f"profile) and {inert} would be inert",
            hint="declare tokenizer (the profile then cuts like a self-hosted one), or drop the inert fields",
        )


class EmbeddingEndpoint(Endpoint):
    """A dense-embedding endpoint speaking OpenAI ``POST {base_url}/embeddings``.

    The package owns every content decision itself: it applies the prompts in the text, cuts at token
    boundaries with the declared tokenizer, sends ``dimensions`` only when set, and L2-normalises the result.

    Attributes:
        api: The wire adapter; ``"openai_embeddings"`` by default (a hosted profile overrides it in its own
            config).
        recipe: The server-side settings the package cannot read (the pooling, the template, the overrides), as
            a free string chosen from the engine's docs; ``None`` records none. Content: two recipes never
            share a cache.
        tokenizer: The model's tokenizer, in whose tokens ``max_tokens`` is counted: a Hugging Face repository
            id with an optional ``@revision``, or a local path to a ``tokenizer.json``. Runtime by its name;
            the file's SHA-256 enters the identity, as the judge's already does.
        max_tokens: What the budget counts is the model's whole input sequence as the engine sees it -- the
            rendered template, its special tokens, the instruction (the prompts) and the content together,
            in the declared tokenizer's tokens. The content is cut on the client, in a budget computed after
            reserving every fixed template token (the anchors a model reads its output from: for a last-token
            pooler, the trailing end-of-turn marker), and the template is re-attached after the cut, so the
            anchors always survive. The cut is never left to the engine: an engine-side truncation of the
            rendered prompt drops anchors from one end or the other. ``None`` sends every item whole.
            Content.
        template: The request template as data
            (:class:`~rcp_ndcg.data.templates.TemplateSpec`): per request shape (``query``, ``document``,
            ``pair``), an ordered list of fixed frame segments and content spans, with the special tokens
            written by name and resolved from the tokenizer, the anchor the model reads its output from, and
            the ``add_special_tokens`` flag per shape (the engine's behaviour for that route). ``None`` fits
            raw text: the budget then reserves only the tokenizer post-processor's tokens (the pooling
            routes' default). Content.
        on_overflow: What an input over the budget does: ``cut`` (the default: the content is cut to the
            budget the template's fixed tokens leave, every cut recorded in the census under
            ``text_budget``), ``chunk`` (the document is split into :attr:`chunk` pieces, each carrying the
            full template, scores pooled back by ``max``), or ``fail`` (the input is refused). Content.
        chunk: The chunk geometry for ``on_overflow: chunk`` (a :class:`~rcp_ndcg.data.preprocess.ChunkPolicy`
            reused, not copied). Content.
        aggregation: How a chunked document's scores pool back onto it: ``max``, its best chunk's -- the same
            rule as ``max_pool_scores_by_document``, recorded on every chunked census row. Content.
        empty_doc: What an empty document becomes: ``send`` (the default: the empty string goes out, as
            today), ``omit_zero`` (it is never sent and scores ``0.0``, the jina-v3 and hosted-API rule), or
            ``send_text`` (a literal placeholder goes out, :attr:`empty_doc_text` names it). Content.
        empty_doc_text: The placeholder text ``empty_doc: send_text`` sends. Content.
        request_shape: How a request crosses the wire: ``text`` (the default: the rendered string),
            ``messages`` (chat parts, the chat-embed form), or ``token_ids`` (pre-tokenised ids, for the
            routes that take them). Declares what the adapter sends; the adapters implement it. Content.
        query_prompt: Text prepended to every query (an asymmetric embedder's instruction prefix). Content.
        doc_prompt: Text prepended to every document. Content.
        normalize: Whether the client L2-normalises the vectors. Content: it changes the vectors (normalising
            twice is harmless, so a server that already normalised is unaffected).
        dimensions: The Matryoshka cut served, when the config sets one. Content.
        batch_size: Items per request. Runtime: how fast, never what.
    """

    #: ``Endpoint``'s roles are inherited; these are this config's own fields.
    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {
        "recipe": FieldRole.CONTENT,
        "tokenizer": FieldRole.RUNTIME,
        "max_tokens": FieldRole.CONTENT,
        "template": FieldRole.CONTENT,
        "on_overflow": FieldRole.CONTENT,
        "chunk": FieldRole.CONTENT,
        "aggregation": FieldRole.CONTENT,
        "empty_doc": FieldRole.CONTENT,
        "empty_doc_text": FieldRole.CONTENT,
        "request_shape": FieldRole.CONTENT,
        "query_prompt": FieldRole.CONTENT,
        "doc_prompt": FieldRole.CONTENT,
        "normalize": FieldRole.CONTENT,
        "dimensions": FieldRole.CONTENT,
        "batch_size": FieldRole.RUNTIME,
    }

    api: str = "openai_embeddings"  # type: ignore[assignment]  # this role's wire adapter, defaulted
    recipe: str | None = Field(default=None, min_length=1)
    tokenizer: str | None = Field(default=None, min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    template: TemplateSpec | None = None
    on_overflow: Literal["cut", "chunk", "fail"] = "cut"
    chunk: ChunkPolicy | None = None
    aggregation: Literal["max"] = "max"
    empty_doc: Literal["omit_zero", "send", "send_text"] = "send"
    empty_doc_text: str | None = None
    request_shape: Literal["text", "messages", "token_ids"] = "text"
    query_prompt: str = ""
    doc_prompt: str = ""
    normalize: bool = True
    dimensions: int | None = Field(default=None, ge=1)
    batch_size: int = Field(default=32, ge=1)

    @model_validator(mode="after")
    def _explicit_budget_and_empty_documents(self) -> EmbeddingEndpoint:
        """A self-hosted role declares its budget (tokenizer and max_tokens); ``send_text`` names its text; a
        chunk geometry belongs to ``on_overflow: chunk`` only."""
        _require_explicit_budget(self)
        _no_inert_overflow_policies(self)
        _chunk_geometry_matches_overflow(self)
        _empty_doc_pairing(self)
        return self


class PoolingEndpoint(EmbeddingEndpoint):
    """A multi-vector (late interaction) endpoint speaking vLLM ``POST {base_url}/pooling`` (task ``token_embed``).

    The result is ragged: one slice of vectors per item, not one vector. Everything else works as
    :class:`EmbeddingEndpoint` (the prompts, the tokenizer's cut, the batch size).

    Attributes:
        api: The wire adapter; ``"vllm_pooling"`` by default.
        embed_dtype: The precision the vectors cross the wire in; ``"float16"`` (the owner's Q11 decision)
            halves the bytes of a ragged buffer, ``"float32"`` is the lossless opt-in. Content: it changes the
            vectors.
    """

    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {
        "embed_dtype": FieldRole.CONTENT,
    }

    api: str = "vllm_pooling"  # type: ignore[assignment]  # this role's wire adapter, defaulted
    embed_dtype: Literal["float16", "float32"] = "float16"


class RerankEndpoint(Endpoint):
    """A reranking endpoint speaking the Cohere-shaped ``POST {base_url}/rerank``.

    One query's whole candidate set goes per request (the engine reuses the query prefix, and a listwise model
    needs them together).

    Attributes:
        api: The wire adapter; ``"rerank"`` by default.
        recipe: As on :class:`EmbeddingEndpoint`: the server-side settings the package cannot read (the
            ``hf_overrides``, the score template), as a free string. Content.
        tokenizer: The model's tokenizer, in whose tokens ``max_tokens`` and ``query_max_tokens`` are counted.
            Runtime by name.
        max_tokens: What the budget counts is the model's whole input sequence as the engine sees it -- the
            rendered template, its special tokens, the instruction and the query-and-document content
            together, in the declared tokenizer's tokens. The content is cut on the client, in a budget
            computed after reserving every fixed template token (the anchors: a pointwise reranker reads its
            score from the last position, so the generation prompt or "yes-no" suffix always survives; when
            the template puts the document first, so does the query), and the template is re-attached after
            the cut. The cut is never left to the engine: an engine-side truncation of the rendered prompt
            drops anchors from one end or the other. The query is cut first, to ``query_max_tokens``; the
            document gets the rest of the budget. ``None`` sends every pair whole. Content.
        query_max_tokens: The query's share of the pair budget (``max_tokens``), in the declared tokenizer's
            tokens; the document gets what remains. It binds when a pair overflows -- an input under budget is
            sent byte-identical to the uncut render. ``None`` (the default) declares no split, and the
            adapter's recipe decides. Content.
        template: The pair template as data (:class:`~rcp_ndcg.data.templates.TemplateSpec`), which orders
            query and document per model (document first for some rerankers, and then the query block is an
            anchor), names the specials, and declares the anchor and the per-shape ``add_special_tokens``.
            Content.
        on_overflow: What a pair over the budget does: ``cut`` (the default), ``chunk`` (the document side is
            split into :attr:`chunk` pieces, each carrying the full template; scores pool back by ``max``)
            or ``fail``. A query is never chunked. Content.
        chunk: The chunk geometry for ``on_overflow: chunk`` (the judge's
            :class:`~rcp_ndcg.data.preprocess.ChunkPolicy` reused, not copied). Content.
        aggregation: How a chunked document's scores pool back onto it: ``max``, its best chunk's -- the same
            rule as ``max_pool_scores_by_document``, recorded on every chunked census row. Content.
        empty_doc: What an empty document becomes: ``send`` (the default: the empty string goes out),
            ``omit_zero`` (never sent, scored ``0.0``) or ``send_text`` (a literal placeholder,
            :attr:`empty_doc_text`). Content.
        empty_doc_text: The placeholder text ``empty_doc: send_text`` sends. Content.
        request_shape: How a request crosses the wire: ``text`` (the default), ``messages`` or ``token_ids``;
            the adapters implement it. Content.
        instruction: How the reranker's instruction reaches the model: ``"fold"`` folds it into the query text
            (``Task: ...\\nQuery: ...``, today's served behaviour), ``"field"`` sends the engine's own
            ``instruction`` request field (vLLM), ``"system"`` sends it as a system message (the shape some
            models take), ``"none"`` sends none. Content.
        use_activation: ``True`` sends through the engine's activation (a probability), ``False`` asks for the
            raw logit, ``None`` sends nothing and the engine's default applies. Content: raw logit or
            probability is a different stored score.
        listwise: Whether the model scores the whole candidate set in one prompt (listwise) rather than point
            per pair. Content.
        batch_size: Documents per request for a pointwise model; refused for a listwise one, which always gets
            the whole candidate set. Runtime.
    """

    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {
        "recipe": FieldRole.CONTENT,
        "tokenizer": FieldRole.RUNTIME,
        "max_tokens": FieldRole.CONTENT,
        "instruction": FieldRole.CONTENT,
        "use_activation": FieldRole.CONTENT,
        "query_max_tokens": FieldRole.CONTENT,
        "template": FieldRole.CONTENT,
        "on_overflow": FieldRole.CONTENT,
        "chunk": FieldRole.CONTENT,
        "aggregation": FieldRole.CONTENT,
        "empty_doc": FieldRole.CONTENT,
        "empty_doc_text": FieldRole.CONTENT,
        "request_shape": FieldRole.CONTENT,
        "listwise": FieldRole.CONTENT,
        "batch_size": FieldRole.RUNTIME,
    }

    api: str = "rerank"  # type: ignore[assignment]  # this role's wire adapter, defaulted
    recipe: str | None = Field(default=None, min_length=1)
    tokenizer: str | None = Field(default=None, min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    instruction: Literal["none", "field", "fold", "system"] = "fold"
    use_activation: bool | None = None
    query_max_tokens: int | None = Field(default=None, ge=1)
    template: TemplateSpec | None = None
    on_overflow: Literal["cut", "chunk", "fail"] = "cut"
    chunk: ChunkPolicy | None = None
    aggregation: Literal["max"] = "max"
    empty_doc: Literal["omit_zero", "send", "send_text"] = "send"
    empty_doc_text: str | None = None
    request_shape: Literal["text", "messages", "token_ids"] = "text"
    listwise: bool = False
    batch_size: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _explicit_budget_and_declared_shares(self) -> RerankEndpoint:
        """A self-hosted role declares its budget; a query share at or over the budget would leave the
        document nothing to read; ``send_text`` names its text; the chunk geometry matches the overflow."""
        _require_explicit_budget(self)
        _no_inert_overflow_policies(self)
        _chunk_geometry_matches_overflow(self)
        if (
            self.query_max_tokens is not None
            and self.max_tokens is not None
            and self.query_max_tokens >= self.max_tokens
        ):
            raise ValueError(
                f"query_max_tokens ({self.query_max_tokens}) must be smaller than max_tokens ({self.max_tokens}): "
                "the document's share of the pair budget would be zero or negative"
            )
        _empty_doc_pairing(self)
        return self

    @model_validator(mode="after")
    def _no_batch_size_for_a_listwise_model(self) -> RerankEndpoint:
        """A listwise model always scores the whole candidate set in one prompt; a ``batch_size`` would change
        which documents share a prompt, and with it the scores -- so it is refused, never ignored."""
        if self.listwise and self.batch_size is not None:
            raise ValueError(
                "batch_size is refused for a listwise reranker: it always scores the whole candidate set in "
                "one prompt, and splitting it would change the scores"
            )
        return self


__all__ = ["SELF_HOSTED_APIS", "EmbeddingEndpoint", "PoolingEndpoint", "RerankEndpoint"]
