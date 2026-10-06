"""The dense-embedding wire adapters: the OpenAI ``POST {base_url}/embeddings`` shape and the hosted profiles
of the same role.

Every adapter here speaks one role: an :class:`~rcp_ndcg.inference.types.EmbedRequest` in (the items, which side
of the retrieval pair they are, the ``dimensions`` cut), :class:`~rcp_ndcg.inference.types.Embeddings` out
(one float32 vector per item, raw -- the client normalises). The adapters are stateless and registered under
their ``(role, name)``; a config of the embed role selects one with ``api: <name>``:

* ``openai_embeddings`` -- the shape every self-hosted engine (vLLM, SGLang, TEI, Infinity) and the OpenAI API
  serve, with ``dimensions`` when the config sets one and ``encoding_format: float``;
* ``cohere``, ``voyage``, ``gemini`` -- the hosted APIs as profiles (the same names the rerank role registers
  for its own wire; the registry is scoped by role, so the roles' namespaces are separate): their request and
  response shapes, their ``input_type`` / ``taskType`` mapping and their published batch caps, with the
  profile's public base URL used when the config sets no ``base_url``.

The adapters make no content decisions: prompts, normalisation and batching are the role client's
(:class:`rcp_ndcg.inference.clients.EmbeddingClient`), and credentials are resolved there too (the profile only
names the header its key goes in). What an adapter owns is the wire: the request body, the reply parsing, the
endpoint's refusal shapes -- a text-only refusal, an over-length HTTP 400 and a batch-cap HTTP 413 mapped onto
typed errors -- and the token usage its API reports.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
from rcp_ndcg_core.content import Content, ImagePart, VideoPart

from rcp_ndcg.errors import CapabilityError, ProviderError, RequestRejectedError
from rcp_ndcg.inference.adapters.base import AdapterBase, AdapterRole, register_adapter
from rcp_ndcg.inference.types import Call, Embeddings, EmbedRequest, Reply, TokenCount

#: The substring of vLLM's (and SGLang's) over-length answer: an HTTP 400 naming the model's context window.
_OVERLENGTH_MARKER = "maximum context length"


def _texts(contents: Sequence[Content], *, adapter: str) -> list[str]:
    """The text of ``contents``, refusing media: these adapters are text-only for now.

    Raises:
        CapabilityError: An item carries an image or video part; the message names the media type. Media
            embedding is wired later, through the media-preparation mechanism.
    """
    for index, content in enumerate(contents):
        for part in content.parts:
            if isinstance(part, ImagePart):
                raise CapabilityError(
                    f"the {adapter} adapter takes text only, but item {index} carries an image part",
                    hint="embed a text rendering of the media; image embedding is wired with the "
                    "media-preparation mechanism",
                )
            if isinstance(part, VideoPart):
                raise CapabilityError(
                    f"the {adapter} adapter takes text only, but item {index} carries a video part",
                    hint="embed a text rendering of the media; video embedding is wired with the "
                    "media-preparation mechanism",
                )
    return [content.text for content in contents]


def _body_text(body: Any) -> str:
    """The error text of a reply body, however the endpoint nested it (a string, or strings inside JSON)."""
    if isinstance(body, str):
        return body
    if isinstance(body, dict):
        return " ".join(_body_text(value) for value in body.values())
    if isinstance(body, list):
        return " ".join(_body_text(item) for item in body)
    return str(body)


def _one_vector(raw: Any, *, adapter: str, where: str) -> Any:
    """One embedding as a 1-D finite float32 vector: a JSON list of floats, or a base64 float32 string.

    Anything else the endpoint could answer -- an empty or partial entry, a scalar, a non-finite value, a
    base64 frame that is not whole float32 words -- is a refused answer with a typed error, never a silent
    NaN or an untyped exception.
    """
    if isinstance(raw, str):
        try:
            decoded = np.frombuffer(base64.b64decode(raw), dtype="<f4")
        except (binascii.Error, ValueError) as exc:
            raise RequestRejectedError(
                f"{adapter} answered {where} as a base64 string that is not little-endian float32: {exc}"
            ) from exc
        vector: Any = decoded
    elif raw is None:
        raise RequestRejectedError(f"{adapter} answered an entry without {where}")
    else:
        try:
            vector = np.asarray(raw, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise RequestRejectedError(
                f"{adapter} answered {where} with a value that is not a list of numbers: {exc}"
            ) from exc
    if vector.ndim != 1 or vector.size == 0 or not bool(np.isfinite(vector).all()):
        shape = "scalar" if vector.ndim == 0 else f"{vector.ndim}-D of {vector.size}"
        raise RequestRejectedError(
            f"{adapter} answered an unusable {where} ({shape}, empty or non-finite); one embedding is a "
            "1-D list of finite floats"
        )
    return vector


def _data_vectors(body: dict[str, Any], *, adapter: str) -> list[Any]:
    """The vectors of the OpenAI-shaped reply ``{'data': [{'index', 'embedding'}, ...]}``, in ``index`` order.

    The endpoint may answer the entries out of order; a request's vectors must align to its items, so the
    ``index`` field (present on every entry of the OpenAI, vLLM, SGLang and TEI replies) sorts them back. A
    reply that names an index on only some entries, or names anything but exactly one int ``0..n-1`` per
    entry, is refused; a reply with no index at all is read in reply order. A silent misalignment would hand
    a request's item the wrong vector.
    """
    data = body.get("data")
    if not isinstance(data, list):
        raise RequestRejectedError(
            f"{adapter} answered without a 'data' list; the OpenAI embeddings shape is "
            "{'data': [{'index', 'embedding'}, ...]}"
        )
    present = [isinstance(item, dict) and "index" in item for item in data]
    if all(present):
        values = [item["index"] for item in data]
        whole = all(isinstance(value, int) and not isinstance(value, bool) for value in values)
        if not whole or sorted(values) != list(range(len(data))):
            raise RequestRejectedError(
                f"{adapter} answered indices {[str(value) for value in values]}; exactly one int "
                f"0..{len(data) - 1} index per entry, in any order"
            )
        data = sorted(data, key=lambda item: item["index"])
    elif any(present):
        raise RequestRejectedError(f"{adapter} answered an 'index' on only some of its 'data' entries")
    return [
        _one_vector(item.get("embedding") if isinstance(item, dict) else None, adapter=adapter, where="embedding")
        for item in data
    ]


class _EmbedAdapter(AdapterBase):
    """Everything the four embedding adapters share: the status map, the alignment check, the stacking.

    A subclass sets ``name`` and its profile's facts (the credential and capability facts are
    :class:`~rcp_ndcg.inference.adapters.base.AdapterBase`'s declared contract) and implements two hooks:
    ``_body``/``_path`` for the request and ``_parse`` for the reply. Instances are stateless.
    """

    role: ClassVar[AdapterRole] = "embed"

    # -- the wire -----------------------------------------------------------
    def calls(self, request: Any, *, model: str) -> list[Call]:
        """The one ``POST`` ``request`` becomes: its texts as ``input``, the model, the cut dimension.

        Args:
            request: The embedding request: the items (already prompt-prefixed by the client), their side of
                the retrieval pair, and the ``dimensions`` cut when the config sets one.
            model: The served model name, sent as the request's ``model`` (and inside the Gemini path).

        Returns:
            One call: this adapter sends a whole request's items in one HTTP request.

        Raises:
            CapabilityError: An item carries an image or a video part (these adapters are text-only), or the
                request names a ``dimensions`` cut this route has no parameter for.
        """
        if request.dimensions is not None and not self.SUPPORTS_DIMENSIONS:
            raise CapabilityError(
                f"the {self.name} embedding API takes no dimensions parameter; the cut would be silently ignored",
                hint="drop dimensions, or use api: openai_embeddings for a Matryoshka cut",
            )
        texts = _texts(request.contents, adapter=self.name)
        return [Call("POST", self._path(model), self._body(texts, request, model))]

    def interpret(self, request: Any, replies: Sequence[Reply]) -> Embeddings:
        """The request's vectors, one per item in the request's order, stacked as float32.

        Args:
            request: The request the replies answer (its item count checks the alignment).
            replies: One reply per call of :meth:`calls`, in order; this adapter sends one call per request.

        Returns:
            One float32 vector per item (raw: the client normalises).

        Raises:
            CapabilityError: The endpoint cannot take requests of this shape at all: a batch over its cap
                (HTTP 413) or an item over its context (vLLM's "maximum context length" HTTP 400).
            RequestRejectedError: The endpoint refused or mis-answered this one request: another HTTP 400 or
                422, a reply whose vectors do not align to the request's items.
            ProviderError: A status no embedding API answers with (the transport's own map covers the rest).
        """
        vectors: list[Any] = []
        for reply in replies:
            vectors.extend(self._vectors(reply))
        if len(vectors) != len(request.contents):
            raise RequestRejectedError(
                f"{self.name} returned {len(vectors)} vector(s) for {len(request.contents)} item(s); "
                "vectors must align to the request's items"
            )
        if not vectors:
            return Embeddings.empty(0)
        widths = {int(vector.shape[0]) for vector in vectors}
        if len(widths) > 1:
            raise RequestRejectedError(
                f"{self.name} returned vectors of differing dimension ({sorted(widths)}); "
                "one request's vectors share a dimension"
            )
        stacked = np.stack(vectors)
        if request.dimensions is not None and stacked.shape[1] != request.dimensions:
            raise RequestRejectedError(
                f"{self.name} answered {stacked.shape[1]}-wide vectors for a dimensions={request.dimensions} "
                "cut; the endpoint or a gateway in front of it ignored the cut"
            )
        return Embeddings.single(stacked)

    def usage(self, reply: Reply) -> TokenCount | None:
        """The tokens the reply reports (``usage.prompt_tokens``, the OpenAI shape), or ``None``.

        Input tokens only: embeddings answer no completions, so there are no output tokens. A malformed
        report reads as no report, never a crash: usage is accounting, the vectors are the answer.
        """
        body = reply.body
        if not isinstance(body, dict):
            return None
        usage = body.get("usage")
        if not isinstance(usage, dict):
            return None
        prompt = usage.get("prompt_tokens")
        if prompt is None:
            return None
        try:
            return TokenCount(input_tokens=int(prompt))
        except (TypeError, ValueError):
            return None

    # -- hooks --------------------------------------------------------------
    def _path(self, model: str) -> str:
        """The request path, appended to the endpoint's base URL."""
        raise NotImplementedError

    def _body(self, texts: list[str], request: EmbedRequest, model: str) -> dict[str, Any]:
        """The JSON request body for ``texts``."""
        raise NotImplementedError

    def _parse(self, body: dict[str, Any]) -> list[Any]:
        """The reply's vectors, in the endpoint's answer order (the adapters re-sort by ``index``)."""
        raise NotImplementedError

    # -- the status map -----------------------------------------------------
    def _vectors(self, reply: Reply) -> list[Any]:
        """The vectors of one 2xx reply; every other status is mapped onto a typed error."""
        if not 200 <= reply.status < 300:
            self._refusal(reply)
        body = reply.body
        if not isinstance(body, dict):
            raise RequestRejectedError(
                f"{self.name} answered HTTP {reply.status} with a {type(body).__name__} body; expected a JSON object"
            )
        return self._parse(body)

    def _refusal(self, reply: Reply) -> None:
        """Map a non-2xx reply onto the typed error that says what to change."""
        text = _body_text(reply.body)
        if reply.status == 413:
            raise CapabilityError(
                f"{self.name} refused the batch with HTTP 413: it caps how many texts fit in one request "
                "(the config's batch_size)",
                hint="lower batch_size (the endpoint's cap; e.g. TEI's --max-client-batch-size)",
            )
        if reply.status in (400, 422) and _OVERLENGTH_MARKER in text.lower():
            raise CapabilityError(
                f"{self.name} refused the request with HTTP {reply.status}: an item is longer than the "
                f"served context ({text[:300]})",
                hint="set max_tokens (with the config's tokenizer) to cut each text on the client before "
                "sending, or lower batch_size so no item exceeds the context",
            )
        if reply.status in (400, 422):
            raise RequestRejectedError(
                f"{self.name} refused the request with HTTP {reply.status}: {text[:300]}",
            )
        raise ProviderError(f"{self.name} answered HTTP {reply.status}: {text[:300]}", retryable=False)


@register_adapter
class OpenAIEmbeddings(_EmbedAdapter):
    """``POST {base_url}/embeddings``, the shape vLLM, SGLang, TEI, Infinity and the OpenAI API serve.

    The body is ``{"model", "input": [texts], "encoding_format": "float"}`` plus ``dimensions`` only when the
    config sets one; the reply is read from ``data[].embedding`` in ``data[].index`` order, as float lists or
    base64 float32 strings. The 128-texts-per-request cap of the hosted OpenAI API is enforced client-side; a
    served engine answers an over-count batch with its own refusal (TEI's HTTP 413), which maps to a
    :class:`~rcp_ndcg.errors.CapabilityError` naming ``batch_size``.
    """

    name: ClassVar[str] = "openai_embeddings"

    #: The hosted OpenAI API's published cap (128 texts per request); a served engine's own cap answers
    #: HTTP 413 and is mapped like any other.
    MAX_BATCH: ClassVar[int | None] = 128
    DEFAULT_BASE_URL: ClassVar[str | None] = "https://api.openai.com/v1"
    HOSTED: ClassVar[bool] = False  # the served engines speak this shape; the key stays optional
    API_KEY_ENV: ClassVar[tuple[str, ...]] = ("OPENAI_API_KEY",)
    KEY_REQUIRED: ClassVar[bool] = False
    AUTH_HEADER: ClassVar[str | None] = None
    ENCODING_FORMAT: ClassVar[str | None] = "float"

    def _path(self, model: str) -> str:
        return "/embeddings"

    def _body(self, texts: list[str], request: EmbedRequest, model: str) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "input": texts}
        if self.ENCODING_FORMAT is not None:
            body["encoding_format"] = self.ENCODING_FORMAT
        if request.dimensions is not None:
            body["dimensions"] = request.dimensions
        return body

    def _parse(self, body: dict[str, Any]) -> list[Any]:
        return _data_vectors(body, adapter=self.name)


@register_adapter
class CohereEmbeddings(_EmbedAdapter):
    """``POST {base_url}/embed``, the Cohere v2 shape: ``texts`` plus ``input_type`` per side.

    ``input_type`` is ``search_query`` for queries and ``search_document`` for documents; the reply is read
    from ``embeddings.float`` in request order.
    """

    name: ClassVar[str] = "cohere"

    MAX_BATCH: ClassVar[int | None] = 96
    DEFAULT_BASE_URL: ClassVar[str | None] = "https://api.cohere.com/v2"
    HOSTED: ClassVar[bool] = True
    API_KEY_ENV: ClassVar[tuple[str, ...]] = ("CO_API_KEY", "COHERE_API_KEY")
    KEY_REQUIRED: ClassVar[bool] = True
    AUTH_HEADER: ClassVar[str | None] = None
    SUPPORTS_DIMENSIONS: ClassVar[bool] = False

    def _path(self, model: str) -> str:
        return "/embed"

    def _body(self, texts: list[str], request: EmbedRequest, model: str) -> dict[str, Any]:
        return {
            "model": model,
            "texts": texts,
            "input_type": "search_query" if request.role.value == "query" else "search_document",
            "embedding_types": ["float"],
        }

    def _parse(self, body: dict[str, Any]) -> list[Any]:
        embeddings = body.get("embeddings")
        if not isinstance(embeddings, dict) or not isinstance(embeddings.get("float"), list):
            raise RequestRejectedError(
                f"{self.name} answered without an 'embeddings.float' list; the Cohere v2 shape is "
                "{'embeddings': {'float': [[...], ...]}}",
            )
        return [_one_vector(raw, adapter=self.name, where="an 'embeddings.float' entry") for raw in embeddings["float"]]

    def usage(self, reply: Reply) -> TokenCount | None:
        """The billed input tokens (``meta.billed_units.input_tokens``), or ``None`` when the reply names none
        usable."""
        body = reply.body
        if not isinstance(body, dict):
            return None
        meta = body.get("meta")
        units = meta.get("billed_units") if isinstance(meta, dict) else None
        prompt = units.get("input_tokens") if isinstance(units, dict) else None
        if prompt is None:
            return None
        try:
            return TokenCount(input_tokens=int(prompt))
        except (TypeError, ValueError):
            return None


@register_adapter
class VoyageEmbeddings(_EmbedAdapter):
    """``POST {base_url}/embeddings``, the Voyage shape: the OpenAI body with an ``input_type`` field.

    ``input_type`` is ``query`` or ``document``; the reply is the OpenAI ``data[].embedding`` shape, read in
    ``data[].index`` order (float lists or base64 float32).
    """

    name: ClassVar[str] = "voyage"

    MAX_BATCH: ClassVar[int | None] = 128
    DEFAULT_BASE_URL: ClassVar[str | None] = "https://api.voyageai.com/v1"
    HOSTED: ClassVar[bool] = True
    API_KEY_ENV: ClassVar[tuple[str, ...]] = ("VOYAGE_API_KEY",)
    KEY_REQUIRED: ClassVar[bool] = True
    AUTH_HEADER: ClassVar[str | None] = None
    SUPPORTS_DIMENSIONS: ClassVar[bool] = False

    def _path(self, model: str) -> str:
        return "/embeddings"

    def _body(self, texts: list[str], request: EmbedRequest, model: str) -> dict[str, Any]:
        return {
            "model": model,
            "input": texts,
            "input_type": request.role.value,
        }

    def _parse(self, body: dict[str, Any]) -> list[Any]:
        return _data_vectors(body, adapter=self.name)


@register_adapter
class GeminiEmbeddings(_EmbedAdapter):
    """``POST {base_url}/models/{model}:batchEmbedContents``, the Gemini batch shape.

    One ``requests`` entry per text, each with the model prefixed as ``models/<model>`` and a ``taskType``
    (``RETRIEVAL_QUERY`` / ``RETRIEVAL_DOCUMENT``); the key goes in ``x-goog-api-key`` and the reply is read
    from ``embeddings[].values`` in request order.
    """

    name: ClassVar[str] = "gemini"

    MAX_BATCH: ClassVar[int | None] = 100
    DEFAULT_BASE_URL: ClassVar[str | None] = "https://generativelanguage.googleapis.com/v1beta"
    HOSTED: ClassVar[bool] = True
    API_KEY_ENV: ClassVar[tuple[str, ...]] = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    KEY_REQUIRED: ClassVar[bool] = True
    AUTH_HEADER: ClassVar[str | None] = "x-goog-api-key"
    SUPPORTS_DIMENSIONS: ClassVar[bool] = False

    def _path(self, model: str) -> str:
        return f"/models/{model}:batchEmbedContents"

    def _body(self, texts: list[str], request: EmbedRequest, model: str) -> dict[str, Any]:
        task = "RETRIEVAL_QUERY" if request.role.value == "query" else "RETRIEVAL_DOCUMENT"
        return {
            "requests": [
                {"model": f"models/{model}", "content": {"parts": [{"text": text}]}, "taskType": task} for text in texts
            ]
        }

    def _parse(self, body: dict[str, Any]) -> list[Any]:
        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list):
            raise RequestRejectedError(
                f"{self.name} answered without an 'embeddings' list; the Gemini shape is "
                "{'embeddings': [{'values': [...]}, ...]}"
            )
        return [
            _one_vector(item.get("values") if isinstance(item, dict) else None, adapter=self.name, where="values")
            for item in embeddings
        ]


__all__ = [
    "CohereEmbeddings",
    "GeminiEmbeddings",
    "OpenAIEmbeddings",
    "VoyageEmbeddings",
]
