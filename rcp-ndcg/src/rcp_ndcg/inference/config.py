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

from pydantic import Field, field_validator, model_validator

from rcp_ndcg.data.mrl import MrlKind, MrlProjection
from rcp_ndcg.data.resolution import ImagePolicy, ImageProcessor, VideoPolicy
from rcp_ndcg.data.templates import RequestShape, TemplateSpec
from rcp_ndcg.data.text_policy import ChunkPolicy
from rcp_ndcg.errors import ConfigError
from rcp_ndcg.inference.endpoint import Endpoint
from rcp_ndcg.support.identity import FieldRole

MediaSide = Literal["query", "document"]
"""A side of the retrieval pair that may or may not carry media."""

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
            raise ConfigError(
                "on_overflow 'chunk' needs a chunk geometry: "
                "{on_overflow: chunk, chunk: {max_tokens: ..., overlap_tokens: ...}}",
                hint="declare the chunk geometry (chunk: {max_tokens, overlap_tokens}), or drop it and use "
                "on_overflow: cut",
            )
        raise ConfigError(
            f"on_overflow {config.on_overflow!r} declares a chunk geometry, which applies to on_overflow 'chunk' only",
            hint="drop the chunk field, or set on_overflow: chunk",
        )


def _empty_doc_pairing(config: EmbeddingEndpoint | RerankEndpoint) -> None:
    """``send_text`` names its placeholder text, and nothing else carries one."""
    if config.empty_doc == "send_text" and config.empty_doc_text is None:
        raise ConfigError(
            "empty_doc 'send_text' needs empty_doc_text: the literal placeholder text the empty document is sent as",
            hint="set empty_doc_text to the placeholder, or use empty_doc: send",
        )
    if config.empty_doc != "send_text" and config.empty_doc_text is not None:
        raise ConfigError(
            f"empty_doc_text applies to empty_doc 'send_text' only, not {config.empty_doc!r}",
            hint="drop empty_doc_text, or set empty_doc: send_text",
        )


def _one_home_for_a_prompt_prefix(config: EmbeddingEndpoint) -> None:
    """A prompt prefix has one home (2d, rec-qwen3-embedding-0.6b): the client prepends ``query_prompt``/
    ``doc_prompt`` before the template renders, so declaring both with a template that already renders the
    prefix as a fixed segment doubles it. Refused for that case, naming the template segment to use instead;
    a content-only template is admitted -- its fixed segments are none, so the prompt is the only home (the
    messages route drops fixed segments anyway, and a text route renders none) -- and the fields stay for
    template-less configs (hosted profiles)."""
    if config.template is None:
        return
    declared: list[str] = []
    if config.query_prompt:
        declared.append("query_prompt")
    if config.doc_prompt:
        declared.append("doc_prompt")
    if not declared:
        return
    shapes: dict[str, RequestShape] = {"query_prompt": "query", "doc_prompt": "document"}
    declared_shapes = set(config.template.shapes())
    doubled = [
        name
        for name in declared
        if shapes[name] in declared_shapes
        and any(segment.fixed is not None for segment in config.template.segments(shapes[name]))
    ]
    if not doubled:
        return
    segment = "/".join(shapes[name] for name in doubled)
    raise ConfigError(
        f"{type(config).__name__} declares {' and '.join(doubled)} beside a template whose {segment!r} "
        "shape already renders a fixed segment: the client prepends the prefix before the template renders, "
        "so declaring both doubles the prefix",
        hint=f"write the prefix as a fixed segment of the template's {segment!r} shape instead "
        "(Segment(fixed=...)), and drop " + " and ".join(doubled) + " (the fields stay for template-less "
        "configs, e.g. hosted profiles)",
    )


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
    if getattr(config, "document_max_tokens", None) is not None:
        inert.append("document_max_tokens")
    if getattr(config, "document_skip_token_ids", ()):
        inert.append("document_skip_token_ids")
    if config.chunk is not None:
        inert.append("chunk")
    if config.template is not None:
        inert.append("template")
    if inert:
        raise ConfigError(
            f"{type(config).__name__} declares no tokenizer, so its content is sent uncut (a hosted vendor "
            f"profile) and {inert} would be inert",
            hint="declare tokenizer (the profile then cuts like a self-hosted one), or drop the inert fields",
        )


def _use_activation_is_explicit_on_a_served_wire(config: RerankEndpoint) -> None:
    """``use_activation: None`` sends nothing and the engine's default applies --
    and two engines with different defaults would then share an identity, because ``identity_payload`` omits
    ``None``. A served rerank config (``api: rerank``) sets it explicitly (the hint names both values); a
    hosted profile keeps ``None``: its scale is the vendor's own and fixed."""
    if config.use_activation is None and config.api in SELF_HOSTED_APIS:
        raise ConfigError(
            f"{type(config).__name__} with api {config.api!r} must set use_activation explicitly: None sends "
            "nothing and the engine's default applies, so two engines with different defaults would share "
            "an identity",
            hint="set use_activation: true (the engine's activation runs: the score is a probability) or "
            "use_activation: false (the raw logit is stored) -- the choice is content and enters the identity; "
            "a hosted profile (api: cohere, api: voyage) leaves it unset, its scale is fixed",
        )


def _media_sides_and_the_media_fields(config: _MediaEndpoint) -> None:
    """Media declared where no side may carry it would be silently inert (2b); refused, never ignored."""
    media_declared = (
        config.image_policy is not None
        or config.video_policy is not None
        or bool(config.max_images)
        or bool(config.max_videos)
    )
    if media_declared and not config.media_sides:
        raise ConfigError(
            "media_sides is empty, so no side may carry media, and the declared media fields "
            "(image_policy, video_policy, max_images, max_videos) would be inert",
            hint="declare a side in media_sides, or drop the media fields",
        )


class _MediaEndpoint(Endpoint):
    """The media fields every retrieval role shares: what it declares about the media it sends.

    The judge declares the same fields (:class:`~rcp_ndcg.judging.JudgeConfig`); these reuse its policy types
    (``rcp_ndcg.data.resolution``, no copies), so one preparation path --
    :func:`~rcp_ndcg.data.prepare.prepare_request` -- sizes an encoder's pages exactly as it sizes the
    judge's, and the token counts the role's text budget subtracts are the judge's.
    """

    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {
        "image_processor": FieldRole.CONTENT,
        "image_policy": FieldRole.CONTENT,
        "video_policy": FieldRole.CONTENT,
        "media_sides": FieldRole.CONTENT,
        "max_images": FieldRole.RUNTIME,
        "max_videos": FieldRole.RUNTIME,
    }

    image_processor: ImageProcessor | None = None
    """The served model's image processor family (``qwen2_vl``, ``qwen2_5_vl``, ``qwen3_vl``;
    :data:`~rcp_ndcg.data.resolution.PROCESSORS`). The client resizes every image and video frame exactly as
    that processor would, within the image policy's budget, so the engine needs no media flags. ``None``
    (the default): the family is unknown, and images are sent unchanged, as stored. Content: it changes the
    input the model sees."""

    image_policy: ImagePolicy | None = None
    """The pixel budget every image and video frame is resized to -- the same
    :class:`~rcp_ndcg.data.resolution.ImagePolicy` the judge's ``preprocessing.image`` is, under this
    role's ``image_processor`` -- or ``None`` (the default) for a text-only role. Content: the budget decides
    the pixels (and with them the token count) the model sees."""

    video_policy: VideoPolicy | None = None
    """Which frames of a video the role sends and how they travel -- the judge's
    :class:`~rcp_ndcg.data.resolution.VideoPolicy`, or ``None`` (the default) for a text-only role. Content:
    the frame count and the wire change the input the model sees."""

    max_images: int = Field(default=0, ge=0)
    """Images one request may carry; 0 (the default) means the model reads none. There is no "unlimited":
    a role that sends images declares its limit, which the server's per-request media limit
    (``--limit-mm-per-prompt``) must allow. Runtime: a gate on what is sent, like the judge's."""

    max_videos: int = Field(default=0, ge=0)
    """Video containers one request may carry; 0 (the default) means the model reads none. There is no
    "unlimited". Runtime: like :attr:`max_images`."""

    media_sides: tuple[MediaSide, ...] = ("query", "document")
    """Which sides of the retrieval pair may carry media (2b, G3: the topk reference rejects image
    queries -- images and video are documents-only there). The clients refuse media on a side this field
    does not name, with a typed error naming the field, before the media is fetched or counted. The
    default allows both sides, today's behaviour. Content: it decides what the model reads."""


class EmbeddingEndpoint(_MediaEndpoint):
    """A dense-embedding endpoint speaking OpenAI ``POST {base_url}/embeddings``.

    The package owns every content decision itself: it applies the prompts in the text, sends ``dimensions``
    only when set, and L2-normalises the result. A config that sets ``max_tokens`` (required, with the
    ``tokenizer``, on a self-hosted role) is fitted by the client through the one text-budget mechanism:
    only content spans cut at token boundaries of the declared ``tokenizer``, the template re-attached,
    every cut recorded, the media tokens reserved whole and never cut. The media fields declare what the
    role sends; the client prepares every request through
    :func:`~rcp_ndcg.data.prepare.prepare_request` -- the same preparation path the judge uses, and the
    role's startup probe runs the engine media check when an ``image_processor`` is declared.


    Attributes:
        api: The wire adapter; ``"openai_embeddings"`` by default (a hosted profile overrides it in its own
            config).
        recipe: The server-side settings the package cannot read (the pooling, the template, the overrides), as
            a free string chosen from the engine's docs; ``None`` records none. Content: two recipes never
            share a cache.
        tokenizer: The model's tokenizer, in whose tokens ``max_tokens`` is counted: a Hugging Face repository
            id with an optional ``@revision``, or a local path to a ``tokenizer.json``. Runtime by its name;
            the file's SHA-256 enters the identity (``Endpoint.identity_extra()``), as the judge's already
            does.
        max_tokens: What the budget counts is the model's whole input sequence as the engine sees it -- the
            rendered template, its special tokens, the instruction (the prompts) and the content together,
            in the declared tokenizer's tokens. The content is cut on the client, in a budget computed after
            reserving every fixed template token (the anchors a model reads its output from: for a last-token
            pooler, the trailing end-of-turn marker), and the template is re-attached after the cut, so the
            anchors always survive. The cut is never left to the engine: an engine-side truncation of the
            rendered prompt drops anchors from one end or the other. ``None`` sends every item whole -- which
            a self-hosted role config refuses (declare the budget); a hosted vendor profile with no
            tokenizer sends content uncut. Content: it caps the ``document`` shape (an embedder's document
            side); the ``query`` shape is capped by :attr:`query_max_tokens` when that is declared, by this
            budget when it is not.
        query_max_tokens: The ``query`` shape's budget, in the declared tokenizer's tokens (per-shape
            budgets: a late-interaction or asymmetric embedder caps queries and documents differently --
            topk-embed-v1-small reads 1024 tokens of query, 8192 of document). It is the query shape's
            WHOLE budget there: the fixed frame is reserved out of it exactly as :attr:`max_tokens`
            reserves the document shape's. Must not exceed :attr:`max_tokens` -- the model's whole input
            budget, which no shape's render can be sent over. ``None`` (the default): both shapes are
            capped by :attr:`max_tokens`. On a :class:`RerankEndpoint` the field keeps its pair-share
            meaning instead. Content.
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
            full template, scores pooled back by ``max`` -- a rerank-only mode, refused by the vector
            clients: an embedding has no score to pool), or ``fail`` (the input is refused). Content.
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
            routes that take them). Declares what the adapter sends; the client refuses a shape its wire
            adapter does not implement. Content.
        add_generation_prompt: ``true`` sends ``add_generation_prompt: true`` with every ``messages`` request,
            so the engine's chat template renders its generation prompt (the assistant header) after the
            user turn -- the frame of a checkpoint whose declared template ends with it (Qwen3-VL-Embedding).
            vLLM's chat routes default it to false (vllm/entrypoints/pooling/base/protocol.py:230-237 at
            v0.31.0), so an undeclared flag renders no header. Only on ``request_shape: messages`` (no other
            route renders a chat template); ``false`` is that default and declares nothing (stored as
            ``None``, no re-key). Content.
        query_prompt: Text prepended to every query (an asymmetric embedder's instruction prefix). Content.
        doc_prompt: Text prepended to every document. Content.
        normalize: Whether the client L2-normalises the vectors. Content: it changes the vectors (normalising
            twice is harmless, so a server that already normalised is unaffected).
        dimensions: The Matryoshka cut served by the engine, when the config sets one (the dense
            ``/embeddings`` route). Content. Only on ``mrl_kind: truncation`` (the engine slices the raw
            output before its own normalisation -- the card's order) and only for a ``k`` in
            :attr:`mrl_dims` or :attr:`mrl_range`; refused beside :attr:`mrl_dim` (one cut, one home).
        mrl_kind: What kind of Matryoshka head the checkpoint has, from its model card: ``truncation`` (a
            Matryoshka-trained checkpoint: cut the full-width output to ``k`` and renormalise),
            ``projection`` (the smaller sizes come from the checkpoint's own learned matrices, applied
            client-side) or ``none``; ``None`` (the default) declares no head and is omitted from every
            identity. Content: it decides what a selected ``k`` computes. A declared kind needs
            :attr:`mrl_dims` or :attr:`mrl_range`.
        mrl_dims: The card-supported set of output dimensions, once, in the recipe. Content: it bounds
            every selection (``dimensions`` and ``mrl_dim`` must be members; nothing is selected unless
            it is declared) and keys the ex-post sweep's per-k artifacts. One of :attr:`mrl_dims` and
            :attr:`mrl_range` is declared when :attr:`mrl_kind` is declared; a projection kind needs the
            discrete set (the learned chain's tensor names are target widths).
        mrl_range: The card's output-dimension RANGE, ``[min, max]``, for a card that declares prose
            ("output dimensions ranging from 32 to 1024") rather than a table: every ``k`` in the closed
            interval is selectable and the client enforces the floor (the engine checks only
            ``1 <= k <= width`` when a checkpoint sets ``is_matryoshka`` without a set). Content. One of
            :attr:`mrl_dims` and :attr:`mrl_range`, not both.
        mrl_projection: Where a ``projection`` kind's learned matrices live (:class:`~rcp_ndcg.data.mrl.MrlProjection`:
            a safetensors source whose tensor names are their target widths; the declared
            :attr:`mrl_dims` are the projected sizes).
            Content. Required for ``mrl_kind: projection``, refused for the other kinds.
        mrl_dim: The Matryoshka output size served CLIENT-side, when the config selects one: the one MRL
            head home (:mod:`rcp_ndcg.data.mrl`) applies the declared kind to the full-width reply --
            truncation cuts and renormalises, projection applies the checkpoint's learned matrix for
            ``k`` -- and every row the head changed carries an ``mrl_cut`` ``ProcessingRecord``. Content:
            it changes the vectors. Only for a ``k`` in :attr:`mrl_dims` or :attr:`mrl_range` and only when
            :attr:`mrl_kind` is declared; refused beside :attr:`dimensions`. On
            :class:`PoolingEndpoint` it must be below :attr:`PoolingEndpoint.dim`.
        batch_size: Items per request. Runtime: how fast, never what.
    """

    #: ``Endpoint``'s roles are inherited; these are this config's own fields.
    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {
        "recipe": FieldRole.CONTENT,
        "tokenizer": FieldRole.RUNTIME,
        "max_tokens": FieldRole.CONTENT,
        "query_max_tokens": FieldRole.CONTENT,
        "template": FieldRole.CONTENT,
        "on_overflow": FieldRole.CONTENT,
        "chunk": FieldRole.CONTENT,
        "aggregation": FieldRole.CONTENT,
        "empty_doc": FieldRole.CONTENT,
        "empty_doc_text": FieldRole.CONTENT,
        "request_shape": FieldRole.CONTENT,
        "add_generation_prompt": FieldRole.CONTENT,
        "query_prompt": FieldRole.CONTENT,
        "doc_prompt": FieldRole.CONTENT,
        "normalize": FieldRole.CONTENT,
        "dimensions": FieldRole.CONTENT,
        "mrl_kind": FieldRole.CONTENT,
        "mrl_dims": FieldRole.CONTENT,
        "mrl_range": FieldRole.CONTENT,
        "mrl_projection": FieldRole.CONTENT,
        "mrl_dim": FieldRole.CONTENT,
        "batch_size": FieldRole.RUNTIME,
    }

    #: Whether this endpoint's wire carries the engine-side Matryoshka ``dimensions`` cut: the dense
    #: ``/embeddings`` route does; the pooling route refuses it (``PoolingEndpoint`` overrides), where the
    #: client-side ``mrl_dim`` is the one cut.
    _ENGINE_SIDE_DIMENSIONS: ClassVar[bool] = True

    api: str = "openai_embeddings"  # type: ignore[assignment]  # this role's wire adapter, defaulted
    recipe: str | None = Field(default=None, min_length=1)
    tokenizer: str | None = Field(default=None, min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    query_max_tokens: int | None = Field(default=None, ge=1)
    template: TemplateSpec | None = None
    on_overflow: Literal["cut", "chunk", "fail"] = "cut"
    chunk: ChunkPolicy | None = None
    aggregation: Literal["max"] = "max"
    empty_doc: Literal["omit_zero", "send", "send_text"] = "send"
    empty_doc_text: str | None = None
    request_shape: Literal["text", "messages", "token_ids"] = "text"
    add_generation_prompt: Literal[True] | None = None
    query_prompt: str = ""
    doc_prompt: str = ""
    normalize: bool = True
    dimensions: int | None = Field(default=None, ge=1)
    mrl_kind: MrlKind | None = None
    mrl_dims: tuple[int, ...] | None = None
    mrl_range: tuple[int, int] | None = None
    mrl_projection: MrlProjection | None = None
    mrl_dim: int | None = Field(default=None, ge=1)
    batch_size: int = Field(default=32, ge=1)

    @field_validator("add_generation_prompt", mode="before")
    @classmethod
    def _false_declares_no_generation_prompt(cls, value: object) -> object:
        """``false`` is the chat routes' own default: stored as ``None``, so it never re-keys an identity."""
        return None if value is False else value

    @model_validator(mode="after")
    def _explicit_budget_and_empty_documents(self) -> EmbeddingEndpoint:
        """A self-hosted role declares its budget (tokenizer and max_tokens); ``send_text`` names its text; a
        chunk geometry belongs to ``on_overflow: chunk`` only; a query budget above the model's whole input
        budget cannot fit the served context."""
        _require_explicit_budget(self)
        _no_inert_overflow_policies(self)
        _chunk_geometry_matches_overflow(self)
        _empty_doc_pairing(self)
        _media_sides_and_the_media_fields(self)
        _one_home_for_a_prompt_prefix(self)
        if self.add_generation_prompt and self.request_shape != "messages":
            raise ConfigError(
                f"add_generation_prompt frames a chat render, and request_shape {self.request_shape!r} renders "
                "no chat template: the flag would be sent nowhere",
                hint="declare request_shape: messages (the engine's chat template frames the content), or drop "
                "add_generation_prompt",
            )
        if (
            self.query_max_tokens is not None
            and self.max_tokens is not None
            and self.query_max_tokens > self.max_tokens
        ):
            raise ConfigError(
                f"query_max_tokens ({self.query_max_tokens}) must not exceed max_tokens ({self.max_tokens}): "
                "the query shape's budget would be over the model's whole input budget",
                hint="set query_max_tokens at or below max_tokens",
            )
        return self

    @model_validator(mode="after")
    def _mrl_declarations(self) -> EmbeddingEndpoint:
        """The declared Matryoshka kind, set and selection, refused at load, never defaulted.

        Every refusal names the field and the fix: a set that is not a set of unique positive dimensions;
        a kind other than ``none`` without a set; a set, projection or selection without a kind; a
        projection kind without its source (and its tensors for every declared ``k``); a ``dimensions`` or
        ``mrl_dim`` outside the set; ``dimensions`` beside ``mrl_dim``; and ``dimensions`` on a kind other
        than ``truncation``. The engine-side ``dimensions`` is checked only where the wire carries it
        (:attr:`_ENGINE_SIDE_DIMENSIONS`): the pooling route refuses the field itself.
        """
        dims = self.mrl_dims
        if dims is not None:
            invalid = sorted(dim for dim in dims if dim < 1)
            if not dims or invalid or len(set(dims)) != len(dims):
                raise ConfigError(
                    f"mrl_dims {tuple(dims)} must be a non-empty set of unique positive output dimensions"
                    + (f" (invalid: {invalid})" if invalid else ""),
                    hint="list the model card's Matryoshka dimensions once each, e.g. [64, 128, 256]",
                )
        mrl_range = self.mrl_range
        if mrl_range is not None:
            low, high = mrl_range
            if low < 1 or high < low:
                raise ConfigError(
                    f"mrl_range {tuple(mrl_range)} must be a positive [min, max] with min <= max",
                    hint="declare the card's range, e.g. [32, 1024] (the card's prose range)",
                )
        if dims is not None and mrl_range is not None:
            raise ConfigError(
                "mrl_dims and mrl_range are both declared: one set, one declaration",
                hint="keep the discrete mrl_dims (the card's table) or the mrl_range (the card's prose), not both",
            )

        def selectable(k: int) -> bool:
            """Whether ``k`` is in the declared set or closed range."""
            if dims is not None:
                return k in dims
            assert mrl_range is not None
            return mrl_range[0] <= k <= mrl_range[1]

        def declared_text() -> str:
            """The declaration a refusal names."""
            if dims is not None:
                return f"mrl_dims {tuple(dims)}"
            assert mrl_range is not None
            return f"mrl_range [{mrl_range[0]}, {mrl_range[1]}]"

        def selection_hint(k: int) -> str:
            """The fix a selection refusal names."""
            if dims is not None:
                return f"select one of mrl_dims {tuple(dims)}, or add {k} to mrl_dims when the card supports it"
            assert mrl_range is not None
            return (
                f"select a k in mrl_range [{mrl_range[0]}, {mrl_range[1]}], or widen mrl_range when the card "
                "supports it"
            )

        kind = self.mrl_kind or "none"
        if kind == "none":
            if dims is not None:
                raise ConfigError(
                    "mrl_dims declares a Matryoshka set, but mrl_kind is 'none': nothing could select it",
                    hint="set mrl_kind: truncation (a Matryoshka-trained checkpoint) or mrl_kind: projection "
                    "(learned matrices), or drop mrl_dims",
                )
            if mrl_range is not None:
                raise ConfigError(
                    "mrl_range declares a Matryoshka range, but mrl_kind is 'none': nothing could select it",
                    hint="set mrl_kind: truncation (a Matryoshka-trained checkpoint), or drop mrl_range",
                )
            if self.mrl_projection is not None:
                raise ConfigError(
                    "mrl_projection names learned matrices, but mrl_kind is 'none': the head would never run",
                    hint="set mrl_kind: projection, or drop mrl_projection",
                )
            if self.mrl_dim is not None:
                raise ConfigError(
                    "mrl_dim selects a Matryoshka output, but mrl_kind is 'none': the checkpoint's card "
                    "declares no Matryoshka head",
                    hint="declare mrl_kind: truncation with mrl_dims or mrl_range (the card's set) or "
                    "mrl_kind: projection with mrl_projection, or drop mrl_dim",
                )
            if self.dimensions is not None and self._ENGINE_SIDE_DIMENSIONS:
                raise ConfigError(
                    "dimensions sends the engine-side Matryoshka cut, but mrl_kind is 'none': the "
                    "checkpoint's card declares no Matryoshka head",
                    hint="declare mrl_kind: truncation with mrl_dims or mrl_range (the card's set), or drop dimensions",
                )
            return self
        if dims is None and mrl_range is None:
            raise ConfigError(
                f"mrl_kind {self.mrl_kind!r} declares a Matryoshka head, but neither mrl_dims nor mrl_range "
                "is declared: there is no set to select k from",
                hint="declare mrl_dims (the card's discrete table) or mrl_range (the card's prose range), or "
                "drop mrl_kind",
            )
        if kind == "projection":
            if self.mrl_projection is None:
                raise ConfigError(
                    "mrl_kind 'projection' needs mrl_projection: the checkpoint's smaller sizes are learned "
                    "matrices, not truncation slices",
                    hint="declare mrl_projection (the safetensors source), or use "
                    "mrl_kind: truncation for a Matryoshka-trained checkpoint",
                )
            if mrl_range is not None:
                raise ConfigError(
                    "mrl_kind 'projection' needs the discrete mrl_dims: the learned chain's tensor names are "
                    "target widths, and a range names no chain",
                    hint="declare mrl_dims (the card's projected sizes) with mrl_projection instead of "
                    "mrl_range, or use mrl_kind: truncation for a Matryoshka-trained checkpoint",
                )
            if self.dimensions is not None:
                raise ConfigError(
                    "dimensions sends the engine-side Matryoshka cut, and mrl_kind 'projection' is applied "
                    "client-side: the engine would serve a slice of a vector whose smaller sizes come from "
                    "learned matrices",
                    hint="drop dimensions (the projection head is client-side), or set mrl_kind: truncation",
                )
            assert dims is not None
            for k in dims:
                self.mrl_projection.chain_for(k, dims)  # refuses a declared k the source cannot reach
        elif self.mrl_projection is not None:
            raise ConfigError(
                f"mrl_projection applies to mrl_kind 'projection' only, not {kind!r}",
                hint="drop mrl_projection, or set mrl_kind: projection",
            )
        if self.dimensions is not None and self.mrl_dim is not None:
            raise ConfigError(
                f"dimensions ({self.dimensions}) and mrl_dim ({self.mrl_dim}) are declared together: one "
                "Matryoshka cut, one home",
                hint="keep dimensions (the engine cuts, the card's own order) or mrl_dim (the client cuts "
                "and renormalises), not both",
            )
        if self.mrl_dim is not None and not selectable(self.mrl_dim):
            raise ConfigError(
                f"mrl_dim {self.mrl_dim} is not in the declared {declared_text()}: the run would select an "
                "output dimension the model's card does not declare",
                hint=selection_hint(self.mrl_dim),
            )
        if self.dimensions is not None and self._ENGINE_SIDE_DIMENSIONS and not selectable(self.dimensions):
            raise ConfigError(
                f"dimensions {self.dimensions} is not in the declared {declared_text()}: the engine would "
                "serve a cut the card does not declare",
                hint=selection_hint(self.dimensions),
            )
        return self


class PoolingEndpoint(EmbeddingEndpoint):
    """A multi-vector (late interaction) endpoint speaking vLLM ``POST {base_url}/pooling`` (task ``token_embed``).

    The result is ragged: one slice of vectors per item, not one vector. Everything else works as
    :class:`EmbeddingEndpoint` (the prompts, the declared text budget, the client's fit of every item) --
    except that ``dimensions`` is never sent: vLLM's ``/pooling`` refuses it ("dimensions is currently not
    supported"), and the client refuses an unset ``dim`` at construction (the base64 frame carries no shape).

    Attributes:
        api: The wire adapter; ``"vllm_pooling"`` by default.
        embed_dtype: The precision the vectors cross the wire in; ``"float16"`` (the owner's Q11 decision)
            halves the bytes of a ragged buffer, ``"float32"`` is the lossless opt-in. Content: it changes the
            vectors. MaxSim computes in float32 either way.
        dim: The width of one token vector -- the checkpoint's late-interaction dimension (ColBERT-style
            checkpoints project to a fixed width, e.g. 128). Content: it is the served checkpoint's own
            output width. Needed to decode the base64 frame of ``/pooling``, which is flat and carries no
            shape (``vllm/utils/serial_utils.py::tensor2binary`` flattens); the response's token counts are
            checked against it, so a mistyped width fails loudly instead of silently mis-shaping every
            vector. The self-describing float and bytes frames decode without it, and the ``bytes`` encoding
            makes it unnecessary (its metadata carries each item's ``shape``).
        document_skip_token_ids: The token ids whose DOCUMENT vectors the model scores nothing by (2, the
            topk hand-off): the client drops the vector at every position whose token id is listed, before
            MaxSim (topk-embed-v1-small drops 41 ids -- standalone punctuation and specials; queries keep
            all their vectors). The positions are the ids the client sent: it tokenises the fitted document
            text with the declared tokenizer, and checks the returned vector count against them (a mismatch is a typed
            error, never a silent misalignment). The skip rule at image positions: a MEDIA
            document's positions are the server's chat-template render, which the client cannot tokenise --
            the image positions are exempt (the vision tokens are what the model reads for the media), the
            client keeps every returned vector of a media document, and the deviation is recorded on the
            row's processing record (``skip_unapplied``) -- never silently unskipped. Needs the declared
            tokenizer; a hosted profile without one cannot apply it (refused as inert). Content.
        mrl_dim: The Matryoshka output size served (2g, plug-pplx), below :attr:`dim` when set: applied
            CLIENT-side as cut-then-renormalise (the card's order -- slice the model's vectors to it, then
            L2-normalise the cut), because ``/pooling`` refuses per-request ``dimensions``. ``None`` (the
            default) serves the checkpoint's own :attr:`dim`. Content.
        outputs: What one input yields (2g, plug-pplx): ``"per_token"`` (the default) is the token_embed
            contract -- one vector per prompt token, which the reply's ``usage`` cross-checks;
            ``"per_chunk"`` is a per-chunk multi-output model -- several outputs per input, one slice of
            chunk vectors per input, so the usage cross-check cannot apply and is skipped. Content.
    """

    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {
        "embed_dtype": FieldRole.CONTENT,
        "dim": FieldRole.CONTENT,
        "document_skip_token_ids": FieldRole.CONTENT,
        "outputs": FieldRole.CONTENT,
    }

    #: ``/pooling`` refuses the per-request ``dimensions`` parameter: the client-side ``mrl_dim`` (or the
    #: engine's serve-time ``pooler_config.dimensions``) is the one cut there.
    _ENGINE_SIDE_DIMENSIONS: ClassVar[bool] = False

    api: str = "vllm_pooling"  # type: ignore[assignment]  # this role's wire adapter, defaulted
    embed_dtype: Literal["float16", "float32"] = "float16"
    dim: int | None = Field(default=None, ge=1)
    document_skip_token_ids: tuple[int, ...] = ()
    outputs: Literal["per_token", "per_chunk"] = "per_token"

    @model_validator(mode="after")
    def _no_generation_prompt_on_pooling(self) -> PoolingEndpoint:
        """``add_generation_prompt`` is inherited, but the ``/pooling`` media lowering sends no such field:
        a declared flag would be inert -- refused, never ignored."""
        if self.add_generation_prompt:
            raise ConfigError(
                "PoolingEndpoint.add_generation_prompt would be inert: the /pooling messages lowering sends no "
                "add_generation_prompt field",
                hint="drop add_generation_prompt (it frames the openai_embeddings messages route)",
            )
        return self

    @model_validator(mode="after")
    def _mrl_dim_below_the_checkpoint_width(self) -> PoolingEndpoint:
        """An MRL cut at or above the checkpoint's own width would cut nothing -- a mistyped knob that
        silently changes nothing."""
        if self.mrl_dim is not None and self.dim is not None and self.mrl_dim >= self.dim:
            raise ConfigError(
                f"mrl_dim ({self.mrl_dim}) must be below dim ({self.dim}): the MRL output size cuts the "
                "checkpoint's token vectors, so declaring it at or over the width cuts nothing",
                hint="set mrl_dim below dim, or drop mrl_dim (the checkpoint's full width is served)",
            )
        return self

    @model_validator(mode="after")
    def _no_inert_dimensions(self) -> PoolingEndpoint:
        """``dimensions`` is inherited but never sent: ``/pooling`` refuses the parameter and the client
        slices nothing, so a declared cut would silently yield full-width vectors while re-keying every
        identity over byte-identical vectors -- refused, never ignored (the openai_embeddings wire carries
        the Matryoshka cut)."""
        if self.dimensions is not None:
            raise ConfigError(
                "PoolingEndpoint.dimensions would be silently ignored: /pooling refuses the dimensions "
                "parameter and the client slices nothing, so the config would record a Matryoshka cut that "
                "never happens",
                hint="drop dimensions (the served checkpoint serves its own width; a Matryoshka cut is an "
                "openai_embeddings concern), or speak an embeddings wire",
            )
        return self


class RerankEndpoint(_MediaEndpoint):
    """A reranking endpoint speaking the Cohere-shaped ``POST {base_url}/rerank``.

    One query's whole candidate set goes per request (the engine reuses the query prefix, and a listwise model
    needs them together).

    Attributes:
        api: The wire adapter; ``"rerank"`` by default.
        recipe: As on :class:`EmbeddingEndpoint`: the server-side settings the package cannot read (the
            ``hf_overrides``, the score template), as a free string. Content.
        tokenizer: The model's tokenizer, in whose tokens ``max_tokens`` and ``query_max_tokens`` are counted.
            Runtime by name; the file's SHA-256 enters the identity (``Endpoint.identity_extra()``), as the
            judge's already does.
        max_tokens: What the budget counts is the model's whole input sequence as the engine sees it -- the
            rendered template, its special tokens, the instruction and the query-and-document content
            together, in the declared tokenizer's tokens. The content is cut on the client, in a budget
            computed after reserving every fixed template token (the anchors: a pointwise reranker reads its
            score from the last position, so the generation prompt or "yes-no" suffix always survives; when
            the template puts the document first, so does the query), and the template is re-attached after
            the cut. The cut is never left to the engine: an engine-side truncation of the rendered prompt
            drops anchors from one end or the other. The query is cut first, to ``query_max_tokens``; the
            document gets the rest of the budget. ``None`` sends every pair whole -- which a self-hosted role
            config refuses (declare the budget); a hosted vendor profile with no tokenizer sends pairs uncut.
            Content.
        query_max_tokens: The query's share of the pair budget (``max_tokens``), in the declared tokenizer's
            tokens; the document gets what remains. On the served rerank wire one query rides per request, so
            the client settles the shared query span once per call: whenever the query exceeds its share it
            ships at it (recorded once in the census under the doc id ``<query>``), and every document span
            is verified against the span that ships -- so a pair is never shipped over the budget. ``None``
            (the default) declares no split, and the adapter's recipe decides; a query that alone fills the
            budget is then refused rather than cut undeclared. Content.
        document_max_tokens: The document's own cap, in the declared tokenizer's content tokens, beside the pair
            budget -- for a checkpoint that cuts each document itself (jina-reranker-v3 reads 2048 document
            tokens and 512 query tokens: ``document_max_tokens: 2048`` beside ``query_max_tokens: 512``). The
            client cuts every document over it to it, also in a pair the budget would take whole, on the
            content span only (the template re-attached, the anchors kept), records the cut under the
            document's position (``cause: document_share``) and then fits the pair to :attr:`max_tokens`.
            ``None`` (the default) declares no cap. It must be below :attr:`max_tokens` (at or over it the
            pair budget always binds first) and is refused beside ``on_overflow: chunk``. Content.
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
        empty_query: What an empty query does (2f, qwen3-vl-reranker): ``refuse`` (the default) refuses the
            request with a typed :class:`~rcp_ndcg.errors.DataError` naming the query id -- the reference
            wrapper refuses an empty query, and silently scoring one against every candidate would rank by
            nothing; ``send`` sends the empty string, today's behaviour. Content.
        request_shape: How a request crosses the wire: ``text`` (the default and the only shape the rerank
            wires implement today -- any other value is refused, the field exists so a rerank config stays
            shape-shaped with its siblings). Content.
        instruction: How the reranker's instruction reaches the model: ``"fold"`` folds it into the query text
            (``Task: ...\\nQuery: ...``, today's served behaviour), ``"field"`` sends the engine's own
            ``instruction`` request field (vLLM), ``"none"`` sends none. ``"system"`` is refused at the
            config: no shipped rerank wire has a system-message slot, and a mode the wire cannot carry would
            silently drop the instruction. Content.
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
        "document_max_tokens": FieldRole.CONTENT,
        "template": FieldRole.CONTENT,
        "on_overflow": FieldRole.CONTENT,
        "chunk": FieldRole.CONTENT,
        "aggregation": FieldRole.CONTENT,
        "empty_doc": FieldRole.CONTENT,
        "empty_doc_text": FieldRole.CONTENT,
        "empty_query": FieldRole.CONTENT,
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
    document_max_tokens: int | None = Field(default=None, ge=1)
    template: TemplateSpec | None = None
    on_overflow: Literal["cut", "chunk", "fail"] = "cut"
    chunk: ChunkPolicy | None = None
    aggregation: Literal["max"] = "max"
    empty_doc: Literal["omit_zero", "send", "send_text"] = "send"
    empty_doc_text: str | None = None
    empty_query: Literal["refuse", "send"] = "refuse"
    request_shape: Literal["text", "messages", "token_ids"] = "text"
    listwise: bool = False
    batch_size: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _explicit_budget_and_declared_shares(self) -> RerankEndpoint:
        """A self-hosted role declares its budget; a query share at or over the budget would leave the
        document nothing to read; a served wire sets ``use_activation`` explicitly; ``send_text`` names its
        text; the chunk geometry matches the overflow."""
        if self.instruction == "system":
            raise ConfigError(
                "RerankEndpoint.instruction 'system' would send the instruction as a system message, and no "
                "shipped rerank wire has a system-message slot (the Cohere-shaped body takes a query and "
                "documents only): the instruction would silently never reach the model",
                hint="use instruction: fold (the instruction folded into the query text, the default), "
                "instruction: field (the engine's own request field, served vLLM only) or instruction: none",
            )
        _require_explicit_budget(self)
        _no_inert_overflow_policies(self)
        _use_activation_is_explicit_on_a_served_wire(self)
        _chunk_geometry_matches_overflow(self)
        _media_sides_and_the_media_fields(self)
        if self.request_shape != "text":
            raise ConfigError(
                f"request_shape {self.request_shape!r} is declared, but the rerank wires send rendered text "
                "(only the embedding and pooling roles implement the messages and token_ids routes)",
                hint="drop request_shape (the default) until the rerank wires land those routes",
            )
        if (
            self.query_max_tokens is not None
            and self.max_tokens is not None
            and self.query_max_tokens >= self.max_tokens
        ):
            raise ConfigError(
                f"query_max_tokens ({self.query_max_tokens}) must be smaller than max_tokens ({self.max_tokens}): "
                "the document's share of the pair budget would be zero or negative",
                hint="set query_max_tokens below max_tokens (the document keeps the rest of the budget)",
            )
        if self.document_max_tokens is not None:
            if self.max_tokens is not None and self.document_max_tokens >= self.max_tokens:
                raise ConfigError(
                    f"document_max_tokens ({self.document_max_tokens}) must be smaller than max_tokens "
                    f"({self.max_tokens}): at or over it the pair budget always binds first, so the cap never would",
                    hint="set document_max_tokens to the checkpoint's own document cap, below max_tokens, or drop it",
                )
            if self.on_overflow == "chunk":
                raise ConfigError(
                    "document_max_tokens cuts every document to its cap, and on_overflow 'chunk' splits an "
                    "over-budget document into chunks instead: the two would decide the same document two ways",
                    hint="declare one: document_max_tokens (the checkpoint's own cut) or on_overflow: chunk",
                )
        _empty_doc_pairing(self)
        return self

    @model_validator(mode="after")
    def _no_batch_size_for_a_listwise_model(self) -> RerankEndpoint:
        """A listwise model always scores the whole candidate set in one prompt; a ``batch_size`` would change
        which documents share a prompt, and with it the scores -- so it is refused, never ignored."""
        if self.listwise and self.batch_size is not None:
            raise ConfigError(
                "batch_size is refused for a listwise reranker: it always scores the whole candidate set in "
                "one prompt, and splitting it would change the scores",
                hint="drop batch_size (a listwise model gets the whole candidate set), or serve a pointwise checkpoint",
            )
        return self


__all__ = ["SELF_HOSTED_APIS", "EmbeddingEndpoint", "PoolingEndpoint", "RerankEndpoint"]
