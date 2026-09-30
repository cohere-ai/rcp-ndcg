"""The client of a hosted embedding API: what :class:`~rcp_ndcg.retrieval.encoders.HostedApiEncoder` embeds through.

No GPU is needed: documents are embedded by a vendor endpoint and searched with the exact numpy top-k in
:mod:`rcp_ndcg.retrieval.topk`. Four vendors ship in-tree (OpenAI, Cohere, Voyage, Gemini), each reached over
plain HTTP rather than its SDK: the request shapes are three lines apiece, and four more SDK dependencies would
undo the point of a dependency-light path. An encoder config selects one with ``provider: <vendor>``
(``experiments/paper/retrieval/cohere_embed_v4.yaml`` in the repository).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from rcp_ndcg_core.content import Content

from rcp_ndcg.errors import CapabilityError
from rcp_ndcg.retrieval.encoder import l2_normalize
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)

Vendor = Literal["openai", "cohere", "voyage", "gemini"]
InputType = Literal["query", "document"]


@dataclass(frozen=True)
class VendorSpec:
    """Everything vendor-specific about one embedding API."""

    name: Vendor
    base_url: str
    api_key_env: tuple[str, ...]
    max_batch: int
    """Texts per request.  Vendor-published limits, not tuning knobs."""
    supports_media: bool = False
    """Whether this endpoint embeds images alongside text.

    Cohere's v2 ``/embed`` takes interleaved ``inputs``; the others take strings
    on the models configured here.  Declared rather than attempted, so a page
    corpus pointed at a text-only endpoint fails before spending the budget."""
    media_batch: int = 8
    """Inputs per request when any of them carry an image.

    Far below ``max_batch``: a page image is ~10^6 base64 characters, and the
    vendor's request-size limit binds long before its item count does."""


VENDORS: dict[str, VendorSpec] = {
    "openai": VendorSpec(
        name="openai", base_url="https://api.openai.com/v1", api_key_env=("OPENAI_API_KEY",), max_batch=128
    ),
    "cohere": VendorSpec(
        name="cohere",
        base_url="https://api.cohere.com/v2",
        api_key_env=("CO_API_KEY", "COHERE_API_KEY"),
        max_batch=96,
        supports_media=True,
    ),
    "voyage": VendorSpec(
        name="voyage", base_url="https://api.voyageai.com/v1", api_key_env=("VOYAGE_API_KEY",), max_batch=128
    ),
    "gemini": VendorSpec(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key_env=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        max_batch=100,
    ),
}

TIMEOUT_S = 60.0
"""Seconds per request."""

MAX_RETRIES = 5
"""Attempts after the first, on a rate limit, a server error or a transport error."""


class EmbeddingAPIClient:
    """Batched embedding calls against one vendor, retried with backoff (:func:`rcp_ndcg.retrieval._http.post_json`).

    Args:
        vendor: One of :data:`VENDORS`; ``openai`` is any OpenAI-compatible ``/embeddings`` route.
        model: The vendor's model id.
        api_key: The key; the vendor's environment variables (:attr:`VendorSpec.api_key_env`) when ``None``.
        base_url: Endpoint override, for a proxy, a gateway or a served model.
        batch_size: Texts per request, capped at the vendor's :attr:`VendorSpec.max_batch` (the default).
        key_required: ``False`` for a server that takes no key (a local vLLM): without one, no key is sent.
        timeout_s: Seconds to wait for a response.
        connect_timeout_s: Seconds to wait for the connection.
        max_retries: Attempts after the first, on a rate limit, a server error or a transport error.
    """

    def __init__(
        self,
        vendor: Vendor | str,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int | None = None,
        key_required: bool = True,
        timeout_s: float = TIMEOUT_S,
        connect_timeout_s: float = 5.0,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        if vendor not in VENDORS:
            raise ValueError(f"unknown embedding vendor {vendor!r}; known: {sorted(VENDORS)}")
        self.spec = VENDORS[vendor]
        self.model = model
        self.base_url = (base_url or self.spec.base_url).rstrip("/")
        self.batch_size = min(batch_size or self.spec.max_batch, self.spec.max_batch)
        self.key_required = key_required
        self.timeout_s = timeout_s
        self.connect_timeout_s = connect_timeout_s
        self.max_retries = max_retries
        self._api_key = api_key

    def _key(self) -> str | None:
        from rcp_ndcg.retrieval._http import api_key

        if not self.key_required and not self._api_key:
            return None
        return api_key(self._api_key, self.spec.api_key_env, what=f"the {self.spec.name} embedding API")

    # ------------------------------------------------------------------
    # Request / response shapes
    # ------------------------------------------------------------------

    def _request(self, texts: list[str], input_type: InputType) -> tuple[str, dict[str, str], dict[str, Any]]:
        from rcp_ndcg.retrieval._http import bearer

        key = self._key()
        if self.spec.name == "openai":
            payload: dict[str, Any] = {"model": self.model, "input": texts}
            return f"{self.base_url}/embeddings", bearer(key), payload

        if self.spec.name == "cohere":
            payload = {
                "model": self.model,
                "texts": texts,
                "input_type": "search_query" if input_type == "query" else "search_document",
                "embedding_types": ["float"],
            }
            return f"{self.base_url}/embed", {"Authorization": f"Bearer {key}"}, payload

        if self.spec.name == "voyage":
            payload = {"model": self.model, "input": texts, "input_type": input_type}
            return f"{self.base_url}/embeddings", {"Authorization": f"Bearer {key}"}, payload

        task = "RETRIEVAL_QUERY" if input_type == "query" else "RETRIEVAL_DOCUMENT"
        requests = [
            {"model": f"models/{self.model}", "content": {"parts": [{"text": text}]}, "taskType": task}
            for text in texts
        ]
        return (
            f"{self.base_url}/models/{self.model}:batchEmbedContents",
            {"x-goog-api-key": key or ""},
            {"requests": requests},
        )

    def _multimodal_request(
        self,
        contents: Sequence[Content],
        input_type: InputType,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """The interleaved-``inputs`` shape (the OpenAI content-parts shape), for vendors that take images.

        Images go inline as base64 data URLs rather than as public URLs: the
        corpus lives in a private bucket, and handing the vendor a signed URL
        would make embedding depend on a link that expires.
        """
        from rcp_ndcg.data.media import content_parts_payload

        if not self.spec.supports_media:
            raise CapabilityError(
                f"the {self.spec.name} embedding API takes no images",
                hint="use provider: cohere for image content, or embed a text rendering",
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "inputs": [{"content": content_parts_payload(content)} for content in contents],
            "input_type": "search_query" if input_type == "query" else "search_document",
            "embedding_types": ["float"],
        }
        return f"{self.base_url}/embed", {"Authorization": f"Bearer {self._key()}"}, payload

    def _parse(self, body: dict[str, Any]) -> list[list[float]]:
        if self.spec.name in ("openai", "voyage"):
            return [item["embedding"] for item in body["data"]]
        if self.spec.name == "cohere":
            return body["embeddings"]["float"]
        return [item["values"] for item in body["embeddings"]]

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, texts: Sequence[str], input_type: InputType = "document", *, progress: bool = False) -> np.ndarray:
        """Embed *texts*, returning an L2-normalised ``(N, dim)`` float32 matrix (inner product = cosine)."""
        return self._embed_all(list(texts), input_type, batch_size=self.batch_size, progress=progress)

    def embed_contents(
        self, contents: Sequence[Content], input_type: InputType = "document", *, progress: bool = False
    ) -> np.ndarray:
        """Embed interleaved text/image content, returning an L2-normalised ``(N, dim)`` float32 matrix.

        Text-only content still goes down the plain ``texts`` path, so an
        ordinary corpus produces byte-identical requests to :meth:`embed`.
        """
        if not any(content.has_media for content in contents):
            return self.embed([content.text for content in contents], input_type, progress=progress)
        batch_size = min(self.batch_size, self.spec.media_batch)
        return self._embed_all(list(contents), input_type, batch_size=batch_size, progress=progress)

    def _embed_all(
        self, items: list[str] | list[Content], input_type: InputType, *, batch_size: int, progress: bool
    ) -> np.ndarray:
        if not items:
            return np.zeros((0, 0), dtype=np.float32)
        vectors: list[list[float]] = []
        for offset in range(0, len(items), batch_size):
            batch = items[offset : offset + batch_size]
            vectors.extend(self._embed_batch(batch, input_type))
            if progress and len(items) > batch_size:
                logger.info(f"[embed-api] {min(offset + len(batch), len(items)):,}/{len(items):,}")
        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    def _embed_batch(self, items: list[str] | list[Content], input_type: InputType) -> list[list[float]]:
        from rcp_ndcg.retrieval._http import post_json

        if items and isinstance(items[0], Content):
            url, headers, payload = self._multimodal_request(cast("list[Content]", items), input_type)
        else:
            url, headers, payload = self._request(cast("list[str]", items), input_type)
        body = post_json(
            url,
            payload,
            headers=headers,
            what=f"{self.spec.name} embedding",
            timeout_s=self.timeout_s,
            connect_timeout_s=self.connect_timeout_s,
            max_retries=self.max_retries,
        )
        return self._parse(body)


__all__ = ["MAX_RETRIES", "TIMEOUT_S", "VENDORS", "EmbeddingAPIClient", "VendorSpec"]
