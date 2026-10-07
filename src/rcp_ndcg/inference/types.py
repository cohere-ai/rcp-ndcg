"""The wire types of the inference layer: what an adapter asks the transport to send, and what comes back.

Per role, the request and result types are the vocabulary the adapters and the role clients build on: the
judge's prompts and answers, the encoders' content batches and vector buffers, one rerank
query with its documents and the aligned scores. Nothing here knows HTTP: :class:`Call` and :class:`Reply` are
the whole contract between an adapter and the transport.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import DTypeLike
from pydantic import BaseModel, ConfigDict, Field, field_validator
from rcp_ndcg_core.content import Content

from rcp_ndcg.errors import DataError

_EMPTY_HEADERS: Mapping[str, str] = MappingProxyType({})


@dataclass(frozen=True)
class Call:
    """One HTTP request an adapter asks the transport to send.

    Attributes:
        method: ``"GET"`` (a probe) or ``"POST"`` (everything else).
        path: The path appended to the replica's base URL (``"/chat/completions"``, ``"/embeddings"``).
        json: The request body, JSON-serialisable; ``None`` sends no body.
        headers: Extra headers beyond the endpoint's credentials and ``headers_env``.
    """

    method: Literal["GET", "POST"]
    path: str
    json: Any = None
    headers: Mapping[str, str] = field(default=_EMPTY_HEADERS)

    def __post_init__(self) -> None:
        if self.method not in ("GET", "POST"):
            raise DataError(
                f"Call.method must be 'GET' or 'POST', got {self.method!r}",
                hint="an adapter builds GET probes and POST calls only",
            )


@dataclass(frozen=True)
class Reply:
    """One HTTP response, one per :class:`Call`, in the order the calls were sent.

    Attributes:
        status: The HTTP status code.
        body: The decoded JSON body, or the raw bytes for a binary encoding (a base64 or bytes vector frame).
        headers: The response headers (the transport reads ``Retry-After``; an adapter may read others).
        url: The replica base URL that answered, set by the transport (a role client needs it to record a
            per-replica fact such as a completion's ``system_fingerprint``); ``None`` when the reply was not
            sent by a transport (a test builds it by hand).
    """

    status: int
    body: Any
    headers: Mapping[str, str]
    url: str | None = None


@dataclass(frozen=True)
class TokenCount:
    """The tokens one reply reports, when its API reports them.

    Attributes:
        input_tokens: Prompt tokens, or ``None`` when the API does not say.
        output_tokens: Completion tokens, or ``None`` when the API does not say.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None


class Usage(BaseModel):
    """Requests and tokens accumulated by a transport or a role client; merge instances with
    :meth:`merged_with`.

    One type for every role: a role client's ``usage`` property is this shape, the run manifest serialises it
    per step and per run, and the transport accumulates it. Tokens and requests only -- the wire reports
    nothing else (an endpoint's cached-input detail is not tracked and stays 0).

    Attributes:
        requests: Requests sent, as the transport counts them: one logical request however many retries it
            took; a reply the status map returns (even one the adapter refuses) counts as one.
        failed_requests: Requests the transport raised on, sent or not; a parked-out request and one the
            rejection rule refuses are the outage's, not a request's. A role client folds its own refusals
            (a reply its adapter raised on) in on top.
        input_tokens: Prompt tokens summed over the requests that reported them.
        output_tokens: Completion tokens summed over the requests that reported them.
        cached_input_tokens: The endpoint's cached-input detail, when it reports one; the shared transport's
            accounting does not track it and it stays 0.
    """

    model_config = ConfigDict(frozen=True)

    requests: int = 0
    failed_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    def merged_with(self, other: Usage) -> Usage:
        """The element-wise sum of two usages."""
        return Usage(
            requests=self.requests + other.requests,
            failed_requests=self.failed_requests + other.failed_requests,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
        )


def safe_url(url: str) -> str:
    """The form of *url* that may reach a log, an error or a record: userinfo and query stripped.

    A user may embed credentials in a URL (a documented httpx idiom, ``http://user:key@host``) -- and a
    query string can carry a key too. Those two never reach a log line, an exception message or a run
    manifest, beside the code's "keys are never logged" claim; the request itself still uses the full URL.
    """
    scheme, separator, rest = url.partition("://")
    if not separator:
        return url  # not an authority-bearing URL: nothing to strip
    authority, _, path_and_query = rest.partition("/")
    at = authority.rfind("@")
    if at >= 0:
        authority = authority[at + 1 :]
    path, _, _query = path_and_query.partition("?")
    return f"{scheme}://{authority}/{path}" if path else f"{scheme}://{authority}"


class EngineInfo(BaseModel):
    """What one replica says about itself, read best effort: runtime information, never part of an identity.

    Read from the standard ``GET <base_url>/models`` (the entry of the served model) and the response headers,
    and from the ``system_fingerprint`` of the replica's first completion; nothing engine-specific is asked.

    Attributes:
        url: The replica's base URL, with any userinfo and query stripped (a key in a URL never reaches the
            record beside the code's never-logged claim).
        model: The served model id the endpoint lists (the judge's ``model`` when listed, else the first).
        owned_by: The entry's ``owned_by``; open-source engines put their own name there.
        max_model_len: The served context in tokens, when the endpoint reports it.
        headers: The response's ``server`` header and any header naming a version.
        system_fingerprint: The ``system_fingerprint`` of the first completion, when the endpoint sends one (some
            engines put their version in it).
        error: Why the endpoint could not be read; the other fields are then empty.
    """

    model_config = ConfigDict(extra="forbid")

    url: str
    model: str | None = None
    owned_by: str | None = None
    max_model_len: int | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    system_fingerprint: str | None = None
    error: str | None = None

    @field_validator("url")
    @classmethod
    def _no_userinfo_in_the_record(cls, value: str) -> str:
        """The record never carries userinfo or a query: a key embedded in the URL must not be persisted."""
        return safe_url(value)


# ---------------------------------------------------------------------------
# The judge: prompts and answers (moved from rcp_ndcg.llm.client, unchanged)
# ---------------------------------------------------------------------------


class CompletionInput(BaseModel):
    """One prompt: the rendered user text, optional media parts and an optional answer format.

    ``user_prompt`` is the text rendering (what parsers and token estimates read);
    ``user_content`` carries the interleaved parts when the prompt holds media;
    ``response_format`` is sent as the request's OpenAI-standard ``response_format``
    (a ``json_schema`` one when the answer is constrained), and left out when ``None``.
    """

    model_config = ConfigDict(frozen=True)

    user_prompt: str
    user_content: Content | None = None
    response_format: dict[str, Any] | None = None

    @property
    def has_media(self) -> bool:
        return self.user_content is not None and self.user_content.has_media


class Completion(BaseModel):
    """One answer, with the tokens it used when the endpoint reported them (the judging pass stores a Judgement).

    ``finish_reason`` is the endpoint's own word (``"stop"``, ``"length"``, and whatever else a server
    reports, such as ``"abort"``), or ``None`` when it reports none.
    """

    response: str
    reasoning: str | None = None
    finish_reason: str | None = "stop"
    input_tokens: int | None = None
    output_tokens: int | None = None


# ---------------------------------------------------------------------------
# The encoders: content batches in, vectors out
# ---------------------------------------------------------------------------


class EncodeRole(StrEnum):
    """Which side of the retrieval pair a batch is.

    Asymmetric embedders prepend different instructions per side, and getting it
    wrong costs recall without raising -- so it is a required argument rather
    than a default.
    """

    QUERY = "query"
    DOCUMENT = "document"


@dataclass(frozen=True)
class Embeddings:
    """Vectors for a batch of items, single- or multi-vector.

    One type for both because the difference is a layout detail, not a different
    kind of answer.  Multi-vector output is stored ragged -- all vectors
    concatenated, plus an offsets array -- rather than padded to a rectangle: a
    page can be 40 patches or 4000, and padding to the maximum wastes most of
    the buffer while making a padded row indistinguishable from a real one.

    Attributes:
        vectors: ``(N, D)`` when single-vector; ``(total_vectors, D)`` when
            ragged, in item order. Values are float32, except that a ragged
            multi-vector buffer may keep the transfer dtype it crossed the wire
            in (``float16``: half the bytes of a corpus index; :func:`l2_normalize`
            and MaxSim compute in float32 either way).
        offsets: ``None`` when single-vector; otherwise ``(N + 1,)``, so item
            ``i`` owns ``vectors[offsets[i]:offsets[i + 1]]``.
    """

    vectors: np.ndarray
    offsets: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise DataError(
                f"Embeddings.vectors must be 2-D (got shape {self.vectors.shape}). "
                "Multi-vector output is stored flat with `offsets`, not as a 3-D array."
            )
        if self.offsets is None:
            return
        if self.offsets.ndim != 1 or len(self.offsets) < 1:
            raise DataError(f"offsets must be a 1-D array of length num_items + 1, got shape {self.offsets.shape}")
        if int(self.offsets[0]) != 0:
            raise DataError(f"offsets must start at 0, got {int(self.offsets[0])}")
        if int(self.offsets[-1]) != len(self.vectors):
            raise DataError(
                f"offsets end at {int(self.offsets[-1])} but there are {len(self.vectors)} vectors; "
                "every vector must belong to exactly one item"
            )
        if np.any(np.diff(self.offsets) < 0):
            raise DataError("offsets must be non-decreasing")

    # -- shape -------------------------------------------------------------
    @property
    def is_multi_vector(self) -> bool:
        return self.offsets is not None

    @property
    def num_items(self) -> int:
        return len(self.vectors) if self.offsets is None else len(self.offsets) - 1

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def as_matrix(self) -> np.ndarray:
        """The ``(N, D)`` matrix, for the single-vector consumers (the top-k search).

        Raises on multi-vector rather than pooling silently: collapsing a
        late-interaction model to one vector per document is a different
        retrieval method with different numbers, and it should be asked for.
        """
        if self.offsets is not None:
            raise DataError(
                f"as_matrix() on multi-vector embeddings ({self.num_items} items, "
                f"{len(self.vectors)} vectors). Use MaxSim scoring, or pool explicitly first."
            )
        return self.vectors

    # -- construction ------------------------------------------------------
    @classmethod
    def single(cls, vectors: np.ndarray) -> Embeddings:
        """One vector per item, as a contiguous float32 ``(N, D)`` matrix."""
        return cls(vectors=np.ascontiguousarray(vectors, dtype=np.float32))

    @classmethod
    def ragged(cls, per_item: Sequence[np.ndarray], *, dtype: DTypeLike = np.float32) -> Embeddings:
        """Build from one ``(T_i, D)`` array per item (width 0 when every item is empty).

        Args:
            per_item: One ``(T_i, D)`` array per item.
            dtype: The stored precision: float32, or the wire transfer dtype
                (``float16``) to keep a multi-vector buffer in the precision it
                crossed the wire in; every item is cast to it.
        """
        lengths = [len(item) for item in per_item]
        width = next((int(item.shape[1]) for item in per_item if len(item)), 0)
        stacked = (
            np.concatenate([np.asarray(item, dtype=dtype) for item in per_item if len(item)], axis=0)
            if any(lengths)
            else np.zeros((0, width), dtype=dtype)
        )
        offsets = np.zeros(len(per_item) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        return cls(vectors=np.ascontiguousarray(stacked, dtype=dtype), offsets=offsets)

    @classmethod
    def empty(cls, dim: int, *, multi_vector: bool = False, dtype: DTypeLike = np.float32) -> Embeddings:
        """The zero-item value, so an empty shard needs no special-casing."""
        vectors = np.zeros((0, dim), dtype=dtype)
        return cls(vectors=vectors, offsets=np.zeros(1, dtype=np.int64) if multi_vector else None)

    # -- transforms --------------------------------------------------------
    def l2_normalized(self) -> Embeddings:
        """Unit-norm every vector, so an inner product is a cosine.

        Applies per *vector*, not per item, which is what MaxSim needs too.
        The computation runs in float32 and the buffer keeps its dtype.
        """
        return Embeddings(vectors=l2_normalize(self.vectors), offsets=self.offsets)

    def concat(self, other: Embeddings) -> Embeddings:
        """Append *other*'s items after this one's."""
        if self.is_multi_vector != other.is_multi_vector:
            raise DataError(
                "cannot concatenate single-vector and multi-vector embeddings",
                hint="one buffer per layout: pool the multi-vector side explicitly first",
            )
        if self.num_items and other.num_items and self.dim != other.dim:
            raise DataError(
                f"dimension mismatch: {self.dim} vs {other.dim}",
                hint="one endpoint's embeddings share a dimension; these buffers came from different "
                "widths (a zero-width buffer is an all-omitted batch: empty_doc omit_zero with nothing "
                "sent)",
            )
        vectors = np.concatenate([self.vectors, other.vectors], axis=0) if other.num_items else self.vectors
        if self.offsets is None or other.offsets is None:
            return Embeddings(vectors=vectors)
        shifted = other.offsets[1:] + int(self.offsets[-1])
        return Embeddings(vectors=vectors, offsets=np.concatenate([self.offsets, shifted]))


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Every row of ``vectors`` scaled to unit L2 norm (a zero row stays zero).

    Computed in float32 (a float16 sum of squares loses most of its three
    decimal digits), then cast back to the input dtype: float32 in, float32
    out; a float16 buffer stays float16 so a multi-vector index keeps its
    transfer precision end to end.
    """
    source = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(source, axis=1, keepdims=True)
    normalized = source / np.maximum(norms, 1e-12)
    if np.asarray(vectors).dtype == np.float16:
        return normalized.astype(np.float16)
    return normalized


@dataclass(frozen=True)
class EmbedRequest:
    """One dense-embedding call: the items to embed, which side they are, and the cut dimension.

    Attributes:
        contents: The queries or documents as content parts, in order.
        role: Which side of the retrieval pair this batch is; an asymmetric embedder needs it.
        dimensions: The Matryoshka cut the endpoint serves, when the config sets one.
        request_shape: How the batch crosses the wire, copied from the endpoint config (2e, 3):
            ``"text"`` (the default) sends the rendered strings, ``"messages"`` the chat-style embeddings
            input (content parts, image and video parts included), ``"token_ids"`` the pre-tokenised ids.
        token_ids: For ``request_shape: token_ids``, each item's token ids as the client fitted it -- the
            payload the adapter sends. Empty otherwise.
    """

    contents: tuple[Content, ...]
    role: EncodeRole
    dimensions: int | None = None
    request_shape: Literal["text", "messages", "token_ids"] = "text"
    token_ids: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class PoolRequest:
    """One multi-vector (late interaction) pooling call: the items, their side, and the wire precision.

    The result is :class:`Embeddings` in ragged layout -- one slice of vectors per item, not one vector.

    Attributes:
        contents: The queries or documents as content parts, in order.
        role: Which side of the retrieval pair this batch is.
        embed_dtype: The precision the vectors cross the wire in, copied from the endpoint config
            (``"float16"`` by default, ``"float32"`` opt-in); sent as the request's ``embed_dtype`` and the
            dtype the reply's base64 frames decode in.
        dim: The width of one token vector, declared so the flat base64 frame of ``/pooling`` (which carries
            no shape) can be reshaped to ``(tokens, dim)``; ``None`` when undeclared, which only the
            self-describing float frames can decode. Filled from the endpoint config; the reply's token
            counts cross-check it.
        outputs: What one input yields, copied from the endpoint config (2g): ``"per_token"`` (the default)
            is the token_embed contract -- one vector per prompt token, which the reply's own ``usage``
            cross-checks; ``"per_chunk"`` is a per-chunk multi-output model -- several outputs per input, so
            the usage cross-check cannot apply and the adapter skips it.
        request_shape: How the batch crosses the wire, copied from the endpoint config (3): ``"text"`` (the
            default) sends the rendered strings; ``"token_ids"`` sends the pre-tokenised ids (vLLM's
            ``/pooling`` accepts token-id prompts); a media item always travels as ``messages`` (the only
            shape the server's chat template applies to the placeholders).
        token_ids: For ``request_shape: token_ids``, each text item's ids as the client fitted it. Empty
            otherwise.
    """

    contents: tuple[Content, ...]
    role: EncodeRole
    embed_dtype: Literal["float16", "float32"] = "float16"
    dim: int | None = None
    outputs: Literal["per_token", "per_chunk"] = "per_token"
    request_shape: Literal["text", "messages", "token_ids"] = "text"
    token_ids: tuple[tuple[int, ...], ...] = ()


# ---------------------------------------------------------------------------
# The rerankers: one query with its documents, and the aligned scores
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RerankRequest:
    """One rerank call: a query, its candidate documents, and an optional instruction.

    Attributes:
        query: The query as a content part.
        documents: The candidates, in the order the caller gave them; the result's scores align to them.
        instruction: The task instruction the reranker reads (a served reranker's own wording); ``None`` sends
            none.
    """

    query: Content
    documents: tuple[Content, ...]
    instruction: str | None = None


@dataclass(frozen=True)
class RerankResult:
    """The scores of one :class:`RerankRequest`, aligned to its ``documents`` by position.

    Attributes:
        scores: One relevance score per document, in the request's document order.
    """

    scores: tuple[float, ...]

    @classmethod
    def aligned(cls, request: RerankRequest, scores: Sequence[float]) -> RerankResult:
        """The result of ``request``, refusing a score count that does not match its documents.

        Args:
            request: The request the answers answer.
            scores: One relevance score per document of ``request``, in order.

        Returns:
            The aligned result.

        Raises:
            DataError: ``scores`` holds a different number of entries than ``request.documents`` has
                documents.
        """
        if len(scores) != len(request.documents):
            raise DataError(
                f"the reranker returned {len(scores)} score(s) for {len(request.documents)} document(s); "
                "scores must align to the request's documents"
            )
        return cls(scores=tuple(scores))


__all__ = [
    "Call",
    "Completion",
    "CompletionInput",
    "EmbedRequest",
    "Embeddings",
    "EncodeRole",
    "EngineInfo",
    "PoolRequest",
    "RerankRequest",
    "RerankResult",
    "Reply",
    "TokenCount",
    "Usage",
    "l2_normalize",
    "safe_url",
]
