"""The role endpoint configs: an :class:`~rcp_ndcg.inference.endpoint.Endpoint` per role, with its wire default.

Each role sets the fields its wire protocol needs; every field is declared CONTENT or RUNTIME, so
:func:`rcp_ndcg.support.identity.check_declarations` passes and the roles' configs can feed identities. These
configs are not wired into :mod:`rcp_ndcg.retrieval.config` yet (the retrieval-config wiring does
that); they are the frozen shapes the later lanes build against.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field, model_validator

from rcp_ndcg.inference.endpoint import Endpoint
from rcp_ndcg.support.identity import FieldRole


class EmbeddingEndpoint(Endpoint):
    """A dense-embedding endpoint speaking OpenAI ``POST {base_url}/embeddings``.

    The package owns every content decision itself: it applies the prompts in the text, sends ``dimensions``
    only when set, and L2-normalises the result. Cutting text to ``max_tokens`` at token boundaries of the
    declared ``tokenizer`` is the text-budget mechanism's job; until that mechanism is wired into the client,
    a config that sets ``max_tokens`` is refused, never silently ignored.

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
            rendered prompt drops anchors from one end or the other. ``None`` sends every item whole.
            Content. Refused until the text-budget mechanism wires the client-side cut
            (:class:`~rcp_ndcg.inference.clients.EmbeddingClient` raises a ``ConfigError``).
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
    query_prompt: str = ""
    doc_prompt: str = ""
    normalize: bool = True
    dimensions: int | None = Field(default=None, ge=1)
    batch_size: int = Field(default=32, ge=1)


class PoolingEndpoint(EmbeddingEndpoint):
    """A multi-vector (late interaction) endpoint speaking vLLM ``POST {base_url}/pooling`` (task ``token_embed``).

    The result is ragged: one slice of vectors per item, not one vector. Everything else works as
    :class:`EmbeddingEndpoint` (the prompts, the batch size) -- except that ``max_tokens`` is refused until the
    text-budget mechanism is wired (the pooling client raises :class:`~rcp_ndcg.errors.ConfigError`), and
    ``dimensions`` is never sent: vLLM's ``/pooling`` refuses it ("dimensions is currently not supported").

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
    """

    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {
        "embed_dtype": FieldRole.CONTENT,
        "dim": FieldRole.CONTENT,
    }

    api: str = "vllm_pooling"  # type: ignore[assignment]  # this role's wire adapter, defaulted
    embed_dtype: Literal["float16", "float32"] = "float16"
    dim: int | None = Field(default=None, ge=1)


class RerankEndpoint(Endpoint):
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
            document gets the rest of the budget. ``None`` sends every pair whole. Content. Refused until
            the text-budget mechanism wires the client-side cut, rather than silently ignoring a budget.
        query_max_tokens: The query's share of the pair budget (``max_tokens``), in the declared tokenizer's
            tokens; the document gets what remains. ``None`` (the default) declares no split, and the
            adapter's recipe decides. Content.
        instruction: How the reranker's instruction reaches the model: ``"fold"`` folds it into the query text
            (``Task: ...\\nQuery: ...``, today's served behaviour), ``"field"`` sends the engine's own
            ``instruction`` request field (vLLM), ``"none"`` sends none. Content.
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
        "listwise": FieldRole.CONTENT,
        "batch_size": FieldRole.RUNTIME,
    }

    api: str = "rerank"  # type: ignore[assignment]  # this role's wire adapter, defaulted
    recipe: str | None = Field(default=None, min_length=1)
    tokenizer: str | None = Field(default=None, min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    instruction: Literal["none", "field", "fold"] = "fold"
    use_activation: bool | None = None
    query_max_tokens: int | None = Field(default=None, ge=1)
    listwise: bool = False
    batch_size: int | None = Field(default=None, ge=1)

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


__all__ = ["EmbeddingEndpoint", "PoolingEndpoint", "RerankEndpoint"]
