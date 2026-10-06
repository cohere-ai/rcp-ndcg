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

The shipped names are not the whole surface: an ``api`` that names any other registered adapter of the role -- a
third party's from the ``rcp_ndcg.adapters`` entry-point group, or a test's -- selects the role's generic
endpoint config (:class:`PluginEmbedding`, :class:`PluginPooling`, :class:`PluginReranker`), after the registry
has confirmed the name (an unregistered or wrong-role name is refused with the registry's hint). The adapter
name is ``CONTENT``: a step identity keys on it, as the judge's does for its third-party adapters.

Every config declares ``IDENTITY_ROLES``: what the model computes (the model, its revision, its recipe, its
prompts and budgets, the tokenizer's SHA-256 through ``Endpoint.identity_extra()``) enters an index's or a
step's identity; where and how fast it is asked does not.
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, TypeAdapter, model_validator

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

_SHIPPED_EMBED_APIS = frozenset({"openai_embeddings", "cohere", "voyage", "gemini"})
_SHIPPED_POOLING_APIS = frozenset({"vllm_pooling"})
_SHIPPED_RERANK_APIS = frozenset({"rerank", "cohere", "voyage"})
"""The ``api`` values that select a shipped config class; every other registered adapter of the role selects
the role's generic endpoint (:class:`PluginEmbedding`, :class:`PluginPooling`, :class:`PluginReranker`)."""


def _resolve_plugin_api(data: Any, shipped: frozenset[str], role: str) -> Any:
    """Confirm a non-shipped ``api`` names a registered adapter of *role* (C2), and pass the data on.

    The shipped names select their own config classes in the union that follows; any other ``api`` is the
    third-party seam, and must resolve in the role's registry before a config is built -- an unregistered name
    or one registered for another role is refused with the registry's hint, where the config is read.

    Args:
        data: The parsed config (a mapping with ``api``, at the union's entry).
        shipped: The role's shipped ``api`` names.
        role: The adapter role whose registry resolves the name.

    Returns:
        ``data`` unchanged, for the union to validate as the role's generic endpoint.

    Raises:
        ConfigError: ``api`` names no adapter of the role (the registry's hint lists that role's names, and
            names the role when the name is registered elsewhere).
    """
    if isinstance(data, dict):
        api = data.get("api")
        if isinstance(api, str) and api not in shipped:
            from rcp_ndcg.inference.adapters.base import get_adapter

            get_adapter(api, role=role)  # type: ignore[arg-type]
    return data


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


class PluginEmbedding(_ApiSelected, EmbeddingEndpoint):
    """A third-party embed wire: ``api`` names a registered adapter of the embed role that is not a shipped
    one, and the role's generic config carries it (the shipped names select their own classes).

    The registry check happens where the config is read (:func:`_resolve_embed_api`); this class is the shape a
    registered plugin name builds, every other field inherited from
    :class:`~rcp_ndcg.inference.config.EmbeddingEndpoint`. A shipped ``api`` is refused here, and ``api`` has
    no default: a plugin config names its adapter, or it does not build.
    """

    api: str = Field(  # type: ignore[assignment]  # required: a plugin is its api, never a default
        description="The registered adapter name of the third-party wire (required: a plugin is its api)",
    )

    @model_validator(mode="before")
    @classmethod
    def _shipped_apis_have_their_own_class(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("api") in _SHIPPED_EMBED_APIS:
            raise ValueError(f"{data['api']!r} is a shipped wire and selects its own config class")
        return data


def _resolve_embed_api(data: Any) -> Any:
    """Confirm a non-shipped ``api`` names a registered embed adapter (the registry's hint when it does not)."""
    return _resolve_plugin_api(data, _SHIPPED_EMBED_APIS, "embed")


def _resolve_pooling_api(data: Any) -> Any:
    """Confirm a non-shipped multi-vector ``api`` against the ``multi_vector`` role's registry."""
    return _resolve_plugin_api(data, _SHIPPED_POOLING_APIS, "multi_vector")


def _resolve_rerank_api(data: Any) -> Any:
    """Confirm a non-shipped rerank ``api`` against the ``rerank`` role's registry."""
    return _resolve_plugin_api(data, _SHIPPED_RERANK_APIS, "rerank")


EncoderConfig = Annotated[
    ServedEmbedding | CohereEmbedding | VoyageEmbedding | GeminiEmbedding | PluginEmbedding,
    BeforeValidator(_resolve_embed_api),
]
"""An embedding model: the served :class:`ServedEmbedding`, the hosted :class:`CohereEmbedding`,
:class:`VoyageEmbedding` and :class:`GeminiEmbedding`, or -- for any other registered embed-role adapter -- the
generic :class:`PluginEmbedding`, by ``api``. A plain union rather than a discriminated one, so an old shape
(``provider:``) reaches the members' refusals and gets the migration hint instead of a bare "cannot extract
tag"; the :func:`BeforeValidator` resolves a non-shipped ``api`` against the embed role's registry before the
union runs."""


class ServedPooling(_ApiSelected, PoolingEndpoint):
    """A served multi-vector (late interaction) model, through vLLM's ``POST <base_url>/pooling`` with
    ``task: token_embed``.

    ``base_url`` is ``None`` only when the run's job starts the encoder's engine (``serve.encoder``): the
    engine's URLs then reach the step at runtime, through ``RCP_NDCG_ENGINES``, and setting both is refused
    rather than silently overridden.
    """

    api: Literal["vllm_pooling"] = "vllm_pooling"  # type: ignore[assignment]


class PluginPooling(_ApiSelected, PoolingEndpoint):
    """A third-party multi-vector wire: ``api`` names a registered adapter of the ``multi_vector`` role that is
    not a shipped one, and the role's generic config carries it. See :class:`PluginEmbedding`; a shipped
    ``api`` is refused here, so ``vllm_pooling`` always selects :class:`ServedPooling`.
    """

    api: str = Field(  # type: ignore[assignment]  # required: a plugin is its api, never a default
        description="The registered adapter name of the third-party wire (required: a plugin is its api)",
    )

    @model_validator(mode="before")
    @classmethod
    def _shipped_apis_have_their_own_class(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("api") in _SHIPPED_POOLING_APIS:
            raise ValueError(f"{data['api']!r} is a shipped wire and selects its own config class")
        return data


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


class PluginReranker(_ApiSelected, RerankEndpoint):
    """A third-party rerank wire: ``api`` names a registered adapter of the rerank role that is not a shipped
    one, and the role's generic config carries it. See :class:`PluginEmbedding`; a shipped ``api`` is refused
    here, so ``rerank``, ``cohere`` and ``voyage`` always select their own classes.
    """

    api: str = Field(  # type: ignore[assignment]  # required: a plugin is its api, never a default
        description="The registered adapter name of the third-party wire (required: a plugin is its api)",
    )

    @model_validator(mode="before")
    @classmethod
    def _shipped_apis_have_their_own_class(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("api") in _SHIPPED_RERANK_APIS:
            raise ValueError(f"{data['api']!r} is a shipped wire and selects its own config class")
        return data


RerankerConfig = Annotated[
    ServedReranker | CohereReranker | VoyageReranker | PluginReranker, BeforeValidator(_resolve_rerank_api)
]
"""A reranker: the served :class:`ServedReranker`, the hosted :class:`CohereReranker` and
:class:`VoyageReranker`, or -- for any other registered rerank-role adapter -- the generic
:class:`PluginReranker`, by ``api``; the :func:`BeforeValidator` resolves a non-shipped ``api`` against the
rerank role's registry before the union runs."""


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
    encoder: Annotated[ServedPooling | PluginPooling, BeforeValidator(_resolve_pooling_api)]


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
    from rcp_ndcg.errors import ConfigError
    from rcp_ndcg.support.config import config_error

    try:
        return _RETRIEVER.validate_python(data)
    except ConfigError:
        raise  # a typed refusal (a plugin api the registry does not know) already carries its own hint
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
    from rcp_ndcg.errors import ConfigError
    from rcp_ndcg.support.config import config_error

    try:
        return _RERANKER.validate_python(data)
    except ConfigError:
        raise  # a typed refusal (a plugin api the registry does not know) already carries its own hint
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
    "PluginEmbedding",
    "PluginPooling",
    "PluginReranker",
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
