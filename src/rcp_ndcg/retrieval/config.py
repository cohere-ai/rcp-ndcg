"""What a retriever or reranker is, separated from where it runs.

A retriever is a union discriminated on ``kind``: :class:`BM25Config`, :class:`DenseConfig` or
:class:`LateInteractionConfig`. Dense and late-interaction retrievers carry an encoder, and a reranker is a model;
both are unions discriminated on ``provider``, each variant carrying only the fields its provider uses:

======================  ==========================================================  =========  ========
``provider``            where the model runs                                        encoder    reranker
======================  ==========================================================  =========  ========
``local``               in this process, weights loaded from the Hub (``[local]``)   yes        yes
``openai_compatible``   a served endpoint: vLLM, a gateway, or OpenAI's own API      yes        yes
``cohere``              Cohere's public API (Embed v4, Rerank 4 Pro and Fast)        yes        yes
``voyage``              Voyage AI's public API                                       yes        yes
``gemini``              Google's Gemini embedding API                                yes        no
======================  ==========================================================  =========  ========

The served and hosted variants are :class:`~rcp_ndcg.inference.endpoint.Endpoint` s, so they share its fields
(``base_url``, ``model``, ``revision``, ``api_key_env``, the timeouts and retries). A field a provider cannot use
is refused, never ignored.

Every config declares ``IDENTITY_ROLES``: what the model computes (the model, its revision, its pooling and prompts)
enters an index's identity; where and how fast it is asked does not.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rcp_ndcg.inference.endpoint import Endpoint
from rcp_ndcg.support.identity import FieldRole

_CONTENT, _RUNTIME = FieldRole.CONTENT, FieldRole.RUNTIME


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class Local(BaseModel):
    """Weights this package loads in process (the ``[local]`` extra).

    As a reranker: Qwen3-Reranker, ZeRank, ctxl-rerank or Jina reranker v3, the implementation picked from the model
    id.

    Attributes:
        model: The Hub repo id or a local path.
        revision: The Hub revision (commit, tag or branch) loaded; ``None`` is the default branch.
        batch_size: Texts per forward pass; ``None`` for the model's default. ZeRank and Jina batch on their own
            and refuse it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {
        "provider": _CONTENT,
        "model": _CONTENT,
        "revision": _CONTENT,
        "batch_size": _RUNTIME,
    }

    provider: Literal["local"] = "local"
    model: str = Field(min_length=1)
    revision: str | None = None
    batch_size: int | None = Field(default=None, gt=0)


class _Hosted(Endpoint):
    """An endpoint the client reaches over HTTP, with its request size.

    Requests go one at a time with the endpoint's timeout and retries; ``batch_size`` is the documents per rerank
    request or the texts per embedding request (``None`` for the provider's default).
    """

    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {"batch_size": _RUNTIME}

    timeout_s: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=8, ge=0)
    batch_size: int | None = Field(default=None, gt=0)
    base_url: str | None = None  # type: ignore[assignment]  # one URL: the hosted APIs take no replica list yet

    @model_validator(mode="after")
    def _one_request_at_a_time(self) -> _Hosted:
        if "concurrency" in self.model_fields_set and not getattr(self, "_CONCURRENT", False):
            raise ValueError(f"{self.provider} requests are sent one at a time: drop concurrency")  # type: ignore[attr-defined]
        return self


class OpenAICompatible(_Hosted):
    """A served model: vLLM (``vllm serve``), a gateway in front of it, or OpenAI's own API.

    As a reranker, ``POST <base_url>/rerank`` (``vllm serve --runner pooling``), one query's candidates per request,
    ``concurrency`` requests in flight; a late-interaction checkpoint scores MaxSim on the server. ``base_url`` is
    required.
    """

    _CONCURRENT: ClassVar[bool] = True

    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {"provider": _CONTENT}

    provider: Literal["openai_compatible"] = "openai_compatible"
    concurrency: int = Field(default=8, ge=1)
    timeout_s: float = Field(default=600.0, gt=0)
    max_retries: int = Field(default=4, ge=0)


class Cohere(_Hosted):
    """Cohere's public API: Embed v4 (``embed-v4.0``, which also embeds page images) and Rerank 4 Pro and Fast
    (``rerank-v4.0-pro``, ``rerank-v4.0-fast``). The key is read from ``api_key_env``, else ``CO_API_KEY`` or
    ``COHERE_API_KEY``. A rerank request carries up to ``batch_size`` documents (default 100, the API's search
    unit)."""

    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {"provider": _CONTENT}

    provider: Literal["cohere"] = "cohere"


class Voyage(_Hosted):
    """Voyage AI's public API (``voyage-3-large``, ``rerank-2.5``, ``rerank-2.5-lite``, ...). The key is read from
    ``api_key_env``, else ``VOYAGE_API_KEY``. A rerank request carries up to ``batch_size`` documents (default 20)."""

    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {"provider": _CONTENT}

    provider: Literal["voyage"] = "voyage"


class Gemini(_Hosted):
    """Google's Gemini embedding API (``gemini-embedding-001``). The key is read from ``api_key_env``, else
    ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``."""

    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {"provider": _CONTENT}

    provider: Literal["gemini"] = "gemini"


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

_ENCODER_ROLES: dict[str, FieldRole] = {"pooling": _CONTENT, "query_prompt": _CONTENT, "doc_prompt": _CONTENT}


class LocalEncoder(Local):
    """An embedding model run in this process.

    Attributes:
        engine: ``hf`` (transformers, last-token pooling) or ``vllm`` (an in-process vLLM engine, the ``[vllm]``
            extra; for corpus-scale late-interaction indexes, whose vectors are too many to send over HTTP).
        pooling: ``last``, ``mean`` or ``cls`` for one vector per text, ``token`` for one per token (late
            interaction). ``hf`` pools the last token; ``vllm`` serves ``token`` as its ``token_embed`` task and the
            others as ``embed`` (the model's own pooler).
        query_prompt: Text prepended to every query (``vllm``).
        doc_prompt: Text prepended to every document, as the model's recipe defines it (Octen: ``"- "``).
    """

    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {"engine": _CONTENT, **_ENCODER_ROLES}

    engine: Literal["hf", "vllm"] = "hf"
    pooling: Literal["mean", "cls", "last", "token"]
    query_prompt: str | None = None
    doc_prompt: str | None = None

    @model_validator(mode="after")
    def _engine_can_do_it(self) -> LocalEncoder:
        if self.engine == "hf" and self.pooling != "last":
            raise ValueError("the hf engine pools the last token: set pooling: last, or engine: vllm")
        if self.engine == "hf" and self.query_prompt is not None:
            raise ValueError("the hf engine has no query prompt; use doc_prompt, or engine: vllm")
        return self


class OpenAICompatibleEncoder(OpenAICompatible):
    """A served embedding model.

    One vector per text through ``POST <base_url>/embeddings`` (vLLM's OpenAI-compatible route, or OpenAI's API at
    ``https://api.openai.com/v1`` with ``api_key_env: OPENAI_API_KEY``); with ``pooling: token``, one vector per
    token through vLLM's ``/pooling`` route (late interaction).

    Attributes:
        pooling: ``token`` for late interaction; ``None`` (the server's pooler) for one vector per text.
        query_prompt: Text prepended to every query.
        doc_prompt: Text prepended to every document.
    """

    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = dict(_ENCODER_ROLES)
    _CONCURRENT: ClassVar[bool] = False

    base_url: str = Field(min_length=1)  # type: ignore[assignment]
    pooling: Literal["token"] | None = None
    query_prompt: str | None = None
    doc_prompt: str | None = None


EncoderConfig = Annotated[
    LocalEncoder | OpenAICompatibleEncoder | Cohere | Voyage | Gemini, Field(discriminator="provider")
]
"""An embedding model: :class:`LocalEncoder`, :class:`OpenAICompatibleEncoder`, :class:`Cohere`, :class:`Voyage` or
:class:`Gemini`, by ``provider``."""


class OpenAICompatibleReranker(OpenAICompatible):
    """A served ``/rerank`` endpoint (``vllm serve <model> --runner pooling``); ``base_url`` is the server root."""

    base_url: str = Field(min_length=1)  # type: ignore[assignment]

    @model_validator(mode="after")
    def _one_query_per_request(self) -> OpenAICompatibleReranker:
        if self.batch_size is not None:
            raise ValueError("a served reranker takes one query's candidates per request: drop batch_size")
        return self


RerankerConfig = Annotated[Local | OpenAICompatibleReranker | Cohere | Voyage, Field(discriminator="provider")]
"""A reranker: :class:`Local`, :class:`OpenAICompatibleReranker`, :class:`Cohere` or :class:`Voyage`, by
``provider``."""


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------


class BM25Config(BaseModel):
    """Okapi BM25 with ``bm25s``.

    Attributes:
        stemmer: A Snowball stemmer language of PyStemmer (``english``, ``german``, ``french``, ...) that corpus and
            queries are stemmed with, or ``None`` for no stemming.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {"kind": _CONTENT, "stemmer": _CONTENT}

    kind: Literal["bm25"] = "bm25"
    stemmer: str | None = None


class DenseConfig(BaseModel):
    """One vector per text, searched by inner product."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {"kind": _CONTENT, "encoder": _CONTENT}

    kind: Literal["dense"] = "dense"
    encoder: EncoderConfig

    @model_validator(mode="after")
    def _single_vector(self) -> DenseConfig:
        if getattr(self.encoder, "pooling", None) == "token":
            raise ValueError("pooling: token is late interaction: use kind: late_interaction")
        return self


class LateInteractionConfig(BaseModel):
    """One vector per token, searched by MaxSim; the encoder is a vLLM model with ``pooling: token``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {"kind": _CONTENT, "encoder": _CONTENT}

    kind: Literal["late_interaction"] = "late_interaction"
    encoder: LocalEncoder | OpenAICompatibleEncoder = Field(discriminator="provider")

    @model_validator(mode="after")
    def _multi_vector(self) -> LateInteractionConfig:
        if self.encoder.pooling != "token":
            raise ValueError("late interaction needs an encoder with pooling: token")
        if isinstance(self.encoder, LocalEncoder) and self.encoder.engine != "vllm":
            raise ValueError("a local late-interaction encoder runs on engine: vllm")
        return self


RetrieverConfig = Annotated[BM25Config | DenseConfig | LateInteractionConfig, Field(discriminator="kind")]
"""A retriever: :class:`BM25Config`, :class:`DenseConfig` or :class:`LateInteractionConfig`, by ``kind``."""


__all__ = [
    "BM25Config",
    "Cohere",
    "DenseConfig",
    "EncoderConfig",
    "Gemini",
    "LateInteractionConfig",
    "Local",
    "LocalEncoder",
    "OpenAICompatible",
    "OpenAICompatibleEncoder",
    "OpenAICompatibleReranker",
    "RerankerConfig",
    "RetrieverConfig",
    "Voyage",
]
