"""What a retriever or reranker is, separated from where it runs.

A retriever is a union discriminated on ``kind``: :class:`BM25Config`, :class:`DenseConfig` or
:class:`LateInteractionConfig`. Dense and late-interaction retrievers carry an encoder, and a reranker is a model;
all three are :mod:`rcp_ndcg.inference` role configs selected by ``api`` (RFC-0001 section 7.2), each variant
carrying only the fields its wire uses:

=====================  ==========================================================  =========  ========
``api``                where the model runs                                         encoder    reranker
=====================  ============================================================  =========  ========
``openai_embeddings``  a served endpoint (vLLM, SGLang, TEI, Infinity), one vector per text   yes   no
``vllm_pooling``       a served multi-vector engine (late interaction)               yes        no
``rerank``             a served ``/rerank`` endpoint (one query's candidates per request)   no   yes
``cohere``             Cohere's public API (Embed v4; Rerank 4 Pro and Fast)         yes        yes
``voyage``             Voyage AI's public API                                        yes        yes
``gemini``             Google's Gemini embedding API                                 yes        no
=====================  ============================================================  =========  ========

The hosted APIs (``cohere``, ``voyage``, ``gemini``) are reached at their public roots -- their wire adapters
declare them -- so a hosted config omits ``base_url`` (or names a proxy with it); a served one names the engine's
URL (``None`` only when the run's job starts the engine, whose URLs then reach the step at runtime). Every
variant is an :class:`~rcp_ndcg.inference.endpoint.Endpoint`, so all of them share its fields (``base_url``,
``model``, ``revision``, ``api_key_env``, ``headers_env``, ``concurrency``, the timeouts and retries, the outage
wait), and every role client -- :class:`~rcp_ndcg.inference.clients.EmbeddingClient`,
:class:`~rcp_ndcg.inference.clients.PoolingClient`, :class:`~rcp_ndcg.inference.clients.RerankClient` -- runs
them over the shared transport. A field a variant cannot use is refused, never ignored; an old config shape
(``provider:``) is refused with a hint that shows the new one.

Every config declares ``IDENTITY_ROLES``: what the model computes (the model, its revision, its recipe, its
prompts and budgets, the tokenizer's SHA-256 through ``Endpoint.identity_extra()``) enters an index's or a
step's identity; where and how fast it is asked does not.
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from rcp_ndcg.inference.config import EmbeddingEndpoint, PoolingEndpoint, RerankEndpoint
from rcp_ndcg.support.identity import FieldRole

_CONTENT, _RUNTIME = FieldRole.CONTENT, FieldRole.RUNTIME

_OLD_SHAPE_HINT = (
    "retrieval configs select their wire with api, not provider; an encoder is `api: openai_embeddings` "
    "(served, with base_url) or `api: cohere | voyage | gemini` (hosted, base_url omitted), and a "
    "late-interaction encoder is `api: vllm_pooling`; a reranker is `api: rerank` (served, with base_url) or "
    "`api: cohere | voyage` (hosted). In-process models (provider: local) are served now: start the engine "
    "and point api at it"
)
"""The hint an old config shape carries: what replaced ``provider``, in the shape a config file spells."""


class _ApiSelected(BaseModel):
    """The base the retrieval role variants share: frozen, closed, and refusing the pre-``api`` shapes.

    Runs before field validation, so ``provider: local`` and friends answer with the migration hint instead of
    a bare "unknown key" or "api: Field required".

    Raises:
        ValueError: the data is a mapping that names a removed field (``provider``, ``engine``) or an encoder's
            ``pooling`` without an ``api``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _refuse_old_shapes(cls, data: Any) -> Any:
        if isinstance(data, dict):
            removed = sorted(key for key in ("provider", "engine") if key in data)
            if removed:
                given = ", ".join(f"{key}: {data[key]!r}" for key in removed)
                raise ValueError(f"{cls.__name__} is selected by api, not by {' or '.join(removed)} ({given})")
            if "pooling" in data and "api" not in data:
                raise ValueError(f"{cls.__name__} takes no pooling: late interaction is its own retriever kind")
        return data


# ---------------------------------------------------------------------------
# Encoders, by api
# ---------------------------------------------------------------------------


class ServedEmbedding(_ApiSelected, EmbeddingEndpoint):
    """A served embedding model: vLLM (``vllm serve``), SGLang, TEI, Infinity, or OpenAI's own API, through
    ``POST <base_url>/embeddings``.

    Attributes:
        base_url: The endpoint, e.g. ``http://localhost:8000/v1``; ``None`` only when the run's job starts the
            encoder's engine (``serve.encoder``): the engine's URLs then reach the step at runtime, through
            ``RCP_NDCG_ENGINES``, and setting both is refused rather than silently overridden.
    """

    api: Literal["openai_embeddings"] = "openai_embeddings"  # type: ignore[assignment]


class CohereEmbedding(_ApiSelected, EmbeddingEndpoint):
    """Cohere's public API: Embed v4 (``embed-v4.0``, which also embeds page images). The key is read from
    ``api_key_env``, else ``CO_API_KEY`` or ``COHERE_API_KEY``; ``base_url`` is omitted (the profile's public
    URL) unless it names a proxy."""

    api: Literal["cohere"] = "cohere"  # type: ignore[assignment]


class VoyageEmbedding(_ApiSelected, EmbeddingEndpoint):
    """Voyage AI's public API (``voyage-3-large``, ...). The key is read from ``api_key_env``, else
    ``VOYAGE_API_KEY``; ``base_url`` is omitted unless it names a proxy."""

    api: Literal["voyage"] = "voyage"  # type: ignore[assignment]


class GeminiEmbedding(_ApiSelected, EmbeddingEndpoint):
    """Google's Gemini embedding API (``gemini-embedding-001``). The key is read from ``api_key_env``, else
    ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``; ``base_url`` is omitted unless it names a proxy."""

    api: Literal["gemini"] = "gemini"  # type: ignore[assignment]


EncoderConfig = ServedEmbedding | CohereEmbedding | VoyageEmbedding | GeminiEmbedding
"""An embedding model: the served :class:`ServedEmbedding` or the hosted :class:`CohereEmbedding`,
:class:`VoyageEmbedding` and :class:`GeminiEmbedding`, by ``api``. A plain union rather than a discriminated
one, so an old shape (``provider:``) reaches the members' refusals and gets the migration hint instead of a
bare "cannot extract tag"."""


class ServedPooling(_ApiSelected, PoolingEndpoint):
    """A served multi-vector (late interaction) model, through vLLM's ``POST <base_url>/pooling`` with
    ``task: token_embed``.

    ``base_url`` is ``None`` only when the run's job starts the encoder's engine (``serve.encoder``): the
    engine's URLs then reach the step at runtime, through ``RCP_NDCG_ENGINES``, and setting both is refused
    rather than silently overridden.
    """

    api: Literal["vllm_pooling"] = "vllm_pooling"  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Rerankers, by api
# ---------------------------------------------------------------------------


class ServedReranker(_ApiSelected, RerankEndpoint):
    """A served ``/rerank`` endpoint (``vllm serve <model> --runner pooling``), one query's whole candidate set
    per request; ``base_url`` is the server root.

    ``base_url`` is ``None`` only when the run's job starts the reranker's engine (``serve.reranker``): the
    engine's URL then reaches the step at runtime, through ``RCP_NDCG_ENGINES``, and setting both is refused
    rather than silently overridden.
    """

    api: Literal["rerank"] = "rerank"  # type: ignore[assignment]


class _HostedReranker(_ApiSelected, RerankEndpoint):
    """A hosted rerank API's config: the fields its wire has no room for are refused."""

    @model_validator(mode="after")
    def _no_engine_fields_on_the_hosted_wire(self) -> _HostedReranker:
        """The hosted APIs take no engine ``instruction`` field and no activation switch.

        Raises:
            ValueError: ``instruction: field`` or ``use_activation`` is set (neither exists on the hosted wire).
        """
        if self.instruction == "field" or self.use_activation is not None:
            raise ValueError(
                "the hosted rerank APIs take no instruction field and no activation switch: "
                "drop use_activation, and use instruction: fold | none"
            )
        return self


class CohereReranker(_HostedReranker):
    """Cohere's public API (``rerank-v4.0-pro``, ``rerank-v4.0-fast``). The key is read from ``api_key_env``,
    else ``CO_API_KEY`` or ``COHERE_API_KEY``; a request carries up to ``batch_size`` documents (default 100,
    the API's search unit)."""

    api: Literal["cohere"] = "cohere"  # type: ignore[assignment]


class VoyageReranker(_HostedReranker):
    """Voyage AI's public API (``rerank-2.5``, ``rerank-2.5-lite``, ...). The key is read from ``api_key_env``,
    else ``VOYAGE_API_KEY``; a request carries up to ``batch_size`` documents (default 20), and the requests of
    one query are spaced half a second apart (the profile's pause; set ``concurrency`` low -- Voyage enforces
    strict rate limits)."""

    api: Literal["voyage"] = "voyage"  # type: ignore[assignment]


RerankerConfig = ServedReranker | CohereReranker | VoyageReranker
"""A reranker: the served :class:`ServedReranker` or the hosted :class:`CohereReranker` and
:class:`VoyageReranker`, by ``api``."""


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


class DenseConfig(_ApiSelected):
    """One vector per text, searched by inner product."""

    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {"kind": _CONTENT, "encoder": _CONTENT}

    kind: Literal["dense"] = "dense"
    encoder: EncoderConfig


class LateInteractionConfig(_ApiSelected):
    """One vector per token, searched by MaxSim; the encoder is a served ``vllm_pooling`` endpoint."""

    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {"kind": _CONTENT, "encoder": _CONTENT}

    kind: Literal["late_interaction"] = "late_interaction"
    encoder: ServedPooling


RetrieverConfig = Annotated[BM25Config | DenseConfig | LateInteractionConfig, Field(discriminator="kind")]
"""A retriever: :class:`BM25Config`, :class:`DenseConfig` or :class:`LateInteractionConfig`, by ``kind``."""


_RETRIEVER = TypeAdapter(RetrieverConfig)
_RERANKER = TypeAdapter(RerankerConfig)
"""The unions as validators, so a config can be read from data with the old shapes refused by name."""


def validate_retriever(data: Any) -> RetrieverConfig:
    """A retriever config from parsed YAML data, with an old shape refused by name.

    Args:
        data: The parsed YAML (a mapping with ``kind``).

    Returns:
        The :data:`RetrieverConfig`.

    Raises:
        ConfigError: the data is not a retriever config: an old shape (``provider:``, ``engine:``), an unknown
            ``api``, or a field its variant does not take. The hint shows the new shape.
    """
    from rcp_ndcg.support.config import config_error

    try:
        return _RETRIEVER.validate_python(data)
    except Exception as exc:  # noqa: BLE001 -- config_error maps the pydantic ValidationError
        raise config_error(exc, model=_RETRIEVER, hint=_OLD_SHAPE_HINT) from exc


def validate_reranker(data: Any) -> RerankerConfig:
    """A reranker config from parsed YAML data, with an old shape refused by name.

    Args:
        data: The parsed YAML (a mapping with ``api``).

    Returns:
        The :data:`RerankerConfig`.

    Raises:
        ConfigError: the data is not a reranker config: an old shape (``provider:``, ``engine:``), an unknown
            ``api``, or a field its variant does not take. The hint shows the new shape.
    """
    from rcp_ndcg.support.config import config_error

    try:
        return _RERANKER.validate_python(data)
    except Exception as exc:  # noqa: BLE001 -- config_error maps the pydantic ValidationError
        raise config_error(exc, model=_RERANKER, hint=_OLD_SHAPE_HINT) from exc


__all__ = [
    "BM25Config",
    "CohereEmbedding",
    "CohereReranker",
    "DenseConfig",
    "EncoderConfig",
    "GeminiEmbedding",
    "LateInteractionConfig",
    "RerankerConfig",
    "RetrieverConfig",
    "ServedEmbedding",
    "ServedPooling",
    "ServedReranker",
    "VoyageEmbedding",
    "VoyageReranker",
    "validate_reranker",
    "validate_retriever",
]
