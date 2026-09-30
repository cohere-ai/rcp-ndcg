"""HTTP client for a vLLM pooling server: ``/pooling`` and ``/rerank``.

Two endpoints, chosen by what each is actually good at:

* **``/rerank``** is the right transport for scoring. The server holds the model,
  computes MaxSim on the worker when it is a late-interaction checkpoint, and
  returns one float per document. That is a few hundred bytes for a 150-document query.
* **``/pooling``** returns embeddings. It is here for queries and for small
  corpora, *not* for building a corpus index out of a late-interaction model: a
  page is ~1030 vectors at 128 dims, so a 40k-page corpus is ~21 GB of JSON-encoded
  floats over a socket. That is what the offline engine in
  :mod:`rcp_ndcg.retrieval.encoders.vllm_encoder` is for.

The requests are three lines apiece, so this is plain ``httpx`` rather than a
client library -- consistent with how the vendor embedding APIs are reached in
:mod:`rcp_ndcg.retrieval.api_dense`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from rcp_ndcg_core.content import Content

from rcp_ndcg.errors import ProviderError
from rcp_ndcg.retrieval.encoder import Embeddings
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT_S = 600.0
"""Long by default. A rerank call over 150 page images is a real GPU workload,
and a client that gives up at 60s turns a slow batch into a failed run."""


MAX_RETRIES = 4
"""Attempts after the first, on a busy engine queue, a restarting server or a transport error."""


class VllmPoolingClient:
    """Calls one vLLM pooling server.

    Args:
        api_base: Server root, with or without a trailing ``/v1``. Both are
            accepted because the pooling routes live at the root while the chat
            routes live under ``/v1``, and configs in the wild carry either.
        model: Model id to send. The server's own id when ``None``, which it will
            reject if it serves something else -- a useful check rather than an
            inconvenience.
        api_key: Bearer token, if the server was started with one.
        timeout_s: Seconds to wait for a response.
        connect_timeout_s: Seconds to wait for the connection.
        max_retries: Attempts after the first, on a busy queue, a restarting server or a transport error.
    """

    def __init__(
        self,
        api_base: str,
        *,
        model: str | None = None,
        api_key: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        connect_timeout_s: float = 5.0,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self.api_base = api_base.rstrip("/").removesuffix("/v1")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.connect_timeout_s = connect_timeout_s
        self.max_retries = max_retries

    # -- endpoints ---------------------------------------------------------
    def pooling(self, contents: Sequence[Content]) -> Embeddings:
        """Embed *contents*, one request per item when any of them carries media.

        Text-only items batch into a single ``input`` list. Media items go one per
        request through ``messages``, because that is the only shape in which the
        server applies the model's chat template to the image placeholders -- and
        a batch of page images is a large enough request body on its own.
        """
        if not contents:
            return Embeddings.empty(0)

        if not any(content.has_media for content in contents):
            payload = {"input": [content.text for content in contents]}
            return self._as_embeddings(self._post("/pooling", payload))

        from rcp_ndcg.data.media import content_parts_payload

        per_item: list[Embeddings] = []
        for content in contents:
            payload = {"messages": [{"role": "user", "content": content_parts_payload(content)}]}
            per_item.append(self._as_embeddings(self._post("/pooling", payload)))
        return _concat_all(per_item)

    def rerank(
        self,
        query: Content | str,
        documents: Sequence[Content | str],
        *,
        max_tokens_per_query: int | None = None,
        truncate_prompt_tokens: int | None = None,
    ) -> list[float]:
        """Relevance scores for *documents* against *query*, in input order.

        The response is sorted by score and carries an ``index`` per result; this
        undoes that. Callers align scores with their own ``doc_ids`` positionally,
        so returning them ranked would silently permute the association.

        Args:
            query: The query, as text or content parts.
            documents: Candidates, as text or content parts.
            max_tokens_per_query: Server-side query truncation.
            truncate_prompt_tokens: Cap on the combined query + document length.
        """
        if not documents:
            return []

        payload: dict[str, Any] = {
            "query": _score_input(query),
            "documents": [_score_input(document) for document in documents],
            "top_n": len(documents),  # every document gets a score
        }
        if max_tokens_per_query:
            payload["max_tokens_per_query"] = max_tokens_per_query
        if truncate_prompt_tokens:
            payload["truncate_prompt_tokens"] = truncate_prompt_tokens

        body = self._post("/rerank", payload)
        results = body.get("results")
        if results is None:
            raise ProviderError(f"/rerank response has no 'results': {str(body)[:300]}")

        scores = [float("nan")] * len(documents)
        for result in results:
            index = int(result["index"])
            if not 0 <= index < len(documents):
                raise ProviderError(f"/rerank returned index {index} for {len(documents)} documents")
            scores[index] = float(result["relevance_score"])
        missing = [index for index, score in enumerate(scores) if score != score]
        if missing:
            raise ProviderError(
                f"/rerank scored {len(documents) - len(missing)} of {len(documents)} documents; "
                f"{len(missing)} have no score (first: index {missing[0]}). "
                "Check the server log for dropped requests."
            )
        return scores

    # -- transport ---------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, route: str, payload: dict[str, Any]) -> dict[str, Any]:
        from rcp_ndcg.retrieval._http import post_json

        body = {**payload, "model": self.model} if self.model else payload
        return post_json(
            f"{self.api_base}{route}",
            body,
            headers=self._headers(),
            what=f"vLLM {route}",
            timeout_s=self.timeout_s,
            connect_timeout_s=self.connect_timeout_s,
            max_retries=self.max_retries,
        )

    # -- response shapes ---------------------------------------------------
    @staticmethod
    def _as_embeddings(body: dict[str, Any]) -> Embeddings:
        """``/pooling``'s ``data`` into :class:`Embeddings`.

        The per-item payload is ``(D,)`` for a pooled task and ``(T_i, D)`` for
        ``token_embed``, and the layout of the answer is what says which one it
        was -- so it is read from the response rather than from configuration that
        could disagree with the server.
        """
        items = body.get("data")
        if items is None:
            raise ProviderError(f"/pooling response has no 'data': {str(body)[:300]}")
        arrays = [np.asarray(item["data"], dtype=np.float32) for item in sorted(items, key=_response_index)]
        if not arrays:
            return Embeddings.empty(0)
        if arrays[0].ndim == 1:
            return Embeddings.single(np.stack(arrays))
        return Embeddings.ragged(arrays)


def _response_index(item: dict[str, Any]) -> int:
    """Sort key restoring request order; absent on some builds, hence the default."""
    return int(item.get("index", 0))


def _score_input(item: Content | str) -> Any:
    """One side of a scoring request in vLLM's ``ScoreInput`` shape.

    A plain string when there is no media, so a text-only rerank produces exactly
    the request a text-only server expects; ``{"content": [parts]}`` otherwise.
    """
    if isinstance(item, str):
        return item
    if not item.has_media:
        return item.text

    from rcp_ndcg.data.media import content_parts_payload

    return {"content": content_parts_payload(item)}


def _concat_all(chunks: Sequence[Embeddings]) -> Embeddings:
    result = chunks[0]
    for chunk in chunks[1:]:
        result = result.concat(chunk)
    return result


__all__ = ["DEFAULT_TIMEOUT_S", "MAX_RETRIES", "VllmPoolingClient"]
