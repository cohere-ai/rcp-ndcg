"""The embedding role client: every content decision between an :class:`~rcp_ndcg.inference.config.EmbeddingEndpoint`
and the wire.

The client owns what the model reads and how the requests go out -- the client base
(:class:`rcp_ndcg.inference.clients._base.RoleClient`) and the adapters/transport own everything around that:

* **prompts per side** -- ``query_prompt`` / ``doc_prompt`` prepended to every item of that side;
* **the text budget** -- when the config declares one (``max_tokens``), every request is fitted through
  :func:`rcp_ndcg.data.preprocess.fit`: the side's shape (``query`` or ``document``), only content spans
  cut, the template re-attached with every anchor a model reads its output from, every cut recorded in the
  census under ``text_budget``. A hosted profile that declares only ``max_tokens`` (no tokenizer) sends
  content uncut and records the vendor's documented limit. ``on_overflow: chunk`` is refused: chunked
  documents pool *scores* by maximum, and an embedding has no score to pool -- use ``cut`` (the default)
  or chunk at the corpus layer;
* **the Matryoshka cut** -- ``dimensions`` sent to the adapter only when the config sets one;
* **normalisation** -- the vectors L2-normalised when ``normalize`` (the default), so an inner product is a
  cosine (normalising twice is harmless);
* **batching and concurrency** -- items sliced into ``batch_size``-sized requests, at most ``concurrency``
  requests in flight under one :class:`asyncio.TaskGroup` (a failing request cancels its siblings, R7),
  reassembled in the input's order.

Credentials are the transport's (R6): the adapter profile's key variables, their required-ness and the
header travel to the transport as an :class:`~rcp_ndcg.inference.transport.AuthProfile`; no client-side key
handling remains. Media is refused before it is fetched (these adapters are text-only; the refusal is the
base's, in front of the one preparation path).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import numpy as np
from rcp_ndcg_core.content import Content

from rcp_ndcg.data.prepare import MediaCensus
from rcp_ndcg.data.preprocess import TextTruncationCensus
from rcp_ndcg.data.templates import RequestShape
from rcp_ndcg.errors import ConfigError, RequestRejectedError
from rcp_ndcg.inference.adapters import embeddings as _shipped_adapters  # noqa: F401  # registers them
from rcp_ndcg.inference.adapters.base import get_adapter
from rcp_ndcg.inference.clients._base import PreparedItems, RoleClient
from rcp_ndcg.inference.config import EmbeddingEndpoint
from rcp_ndcg.inference.transport import Sender
from rcp_ndcg.inference.types import Embeddings, EmbedRequest, EncodeRole, l2_normalize


def _profile(adapter: type[Any], attribute: str, default: Any) -> Any:
    """A profile attribute an adapter may leave out (``MAX_BATCH``, ``DEFAULT_BASE_URL``, ...)."""
    return getattr(adapter, attribute, default)


def _check_batch_size(adapter: type[Any], size: int) -> None:
    """Refuse a request size above the profile's published cap, instead of silently capping it.

    The cap is a HOSTED profile's own fact: a served engine (vLLM, SGLang, TEI, Infinity -- the
    ``openai_embeddings`` shape) answers an over-count batch with its own refusal, which the adapter maps to
    a typed :class:`~rcp_ndcg.errors.CapabilityError` naming ``batch_size`` -- a stale client-side cap must
    not refuse a batch the engine would serve.
    """
    max_batch = _profile(adapter, "MAX_BATCH", None)
    if max_batch is not None and getattr(adapter, "HOSTED", False) and size > max_batch:
        raise ConfigError(
            f"the {getattr(adapter, 'name', '?')} embedding API takes at most {max_batch} texts per request; "
            f"batch_size is {size}",
            hint=f"set batch_size to {max_batch} or less, or leave it unset",
        )


class EmbeddingClient(RoleClient):
    """The content decisions of one embedding endpoint, on the :class:`~rcp_ndcg.inference.transport.Sender` seam.

    Args:
        config: The endpoint: the wire adapter (``api``), the model, the per-side prompts, ``normalize``,
            ``dimensions``, ``batch_size``, ``concurrency``, the text budget (``tokenizer`` + ``max_tokens``)
            and the credentials.
        sender: What sends the calls; a :class:`~rcp_ndcg.inference.transport.Transport` built on ``config``
            when ``None`` (it resolves the key in the profile's header). Any other sender must provide the
            sync bridge (``run``).
        census: Where the text-budget cuts are recorded; ``None`` gives the client a fresh in-memory census
            (:attr:`census`).

    Raises:
        ConfigError: ``api`` names no registered adapter of the embed role (the hint lists that role's
            names), ``batch_size`` exceeds the profile's cap, ``dimensions`` is set on a profile that takes
            none, ``request_shape`` names a shape the wire does not implement, ``token_ids`` is declared
            without a tokenizer, or ``on_overflow: chunk`` is declared (vector roles do not pool chunks).
    """

    ROLE = "embed"

    #: The embed role's shipped adapters are text-only by default: media rides the ``messages`` route only
    #: (2e), and only when the wire implements it (the adapter's ``REQUEST_SHAPES``).
    MEDIA_ON_WIRE = False

    def __init__(
        self,
        config: EmbeddingEndpoint,
        *,
        sender: Sender | None = None,
        census: TextTruncationCensus | None = None,
        media_census: MediaCensus | None = None,
    ) -> None:
        adapter_cls = get_adapter(config.api, role=self.ROLE)
        if config.dimensions is not None and not _profile(adapter_cls, "SUPPORTS_DIMENSIONS", True):
            raise ConfigError(
                f"the {config.api} embedding API takes no dimensions parameter; the cut would be silently ignored",
                hint="drop dimensions, or use api: openai_embeddings for a Matryoshka cut",
            )
        _check_batch_size(adapter_cls, config.batch_size)
        if config.on_overflow == "chunk":
            raise ConfigError(
                "on_overflow 'chunk' pools scores by max, and an embedding has no score to pool: the declared "
                "aggregation would be reinterpreted, so it is refused instead",
                hint="use on_overflow: cut (the content is cut to the budget), or chunk the corpus at load "
                "(the retrieval index keeps one slice per chunk)",
            )
        supported = getattr(adapter_cls, "REQUEST_SHAPES", frozenset({"text"}))
        if config.request_shape not in supported:
            raise ConfigError(
                f"request_shape {config.request_shape!r} is declared, but the "
                f"{getattr(adapter_cls, 'name', config.api)} wire implements {sorted(supported)}",
                hint="declare a request shape the wire implements (the default is text)",
            )
        if config.request_shape == "token_ids" and config.tokenizer is None:
            raise ConfigError(
                "request_shape 'token_ids' needs a tokenizer: the ids are the client's tokenisation of the "
                "fitted text, and a hosted profile without one cannot tokenise",
                hint="declare the tokenizer (with max_tokens), or drop request_shape (the default sends text)",
            )
        super().__init__(config, sender=sender, census=census, media_census=media_census)
        self._adapter: Any = self._adapter_cls(self.endpoint)

    def _media_is_on_wire(self) -> bool:
        """Whether this client's wire carries media (2e): the ``messages`` route lowers image and video
        parts; the text and token-ids routes carry none (the adapters refuse them)."""
        return self.config.request_shape == "messages"

    # -- the public calls ---------------------------------------------------
    def encode(self, contents: Sequence[Content], role: EncodeRole, *, batch_size: int | None = None) -> Embeddings:
        """Embed ``contents`` as ``role``, synchronously, through the sender's sync bridge.

        Args:
            contents: The queries or documents, in order.
            role: Which side of the retrieval pair these are (the prompts differ per side).
            batch_size: The request size for this call; the config's ``batch_size`` when ``None``. At most the
                profile's cap.

        Returns:
            One float32 vector per content, in the input's order, L2-normalised when ``normalize``.
        """
        return self._run(self.aencode(contents, role, batch_size=batch_size))

    async def aencode(
        self, contents: Sequence[Content], role: EncodeRole, *, batch_size: int | None = None
    ) -> Embeddings:
        """Embed ``contents`` as ``role``, asynchronously: the batch requests in flight at once, in order.

        Same arguments and result as :meth:`encode`; the sync method runs this on the sender's event loop.
        The requests run in one :class:`asyncio.TaskGroup`: a failing request cancels its siblings and no
        task is left pending.
        """
        prepared = self._prepare(contents, role)
        size = self._request_size(batch_size)
        if not prepared.items:
            if not contents:
                return Embeddings.empty(0)
            # Every input was omitted (empty_doc: omit_zero): no request goes out, and the result is one
            # zero vector per input -- the score an omitted document contributes.
            return Embeddings.single(np.zeros((len(contents), 0), dtype=np.float32))

        token_ids = self._token_ids_of(prepared.items, role)
        requests = [
            EmbedRequest(
                contents=tuple(prepared.items[offset : offset + size]),
                role=role,
                dimensions=self.config.dimensions,
                request_shape=self.config.request_shape,
                token_ids=token_ids[offset : offset + size],
            )
            for offset in range(0, len(prepared.items), size)
        ]
        calls = [list(self._adapter.calls(request, model=self.config.model)) for request in requests]
        for batch_calls in calls:
            self._gate_media_calls(batch_calls)

        gate = asyncio.Semaphore(self.config.concurrency)

        async def one(index: int) -> Embeddings:
            async with gate:
                replies = await self._sender.send(calls[index])
            self._record_usage(replies)
            return self._adapter.interpret(requests[index], replies)

        parts = await RoleClient.gather([one(index) for index in range(len(requests))])
        widths = sorted({part.dim for part in parts})
        if len(widths) > 1:
            raise RequestRejectedError(
                f"{self.config.api} answered vectors of differing dimension ({widths}) across batches; "
                "one endpoint's embeddings share a dimension"
            )
        width = widths[0] if widths else 0
        # One matrix row per input: the sent items' vectors in their order, zeros where omit_zero omitted.
        matrix = np.zeros((len(contents), width), dtype=np.float32)
        sent = np.concatenate([part.as_matrix() for part in parts]) if parts else np.zeros((0, width))
        for out_row, position in enumerate(prepared.positions):
            matrix[position] = sent[out_row]
        if self.config.normalize:
            matrix = l2_normalize(matrix)
        return Embeddings.single(matrix)

    # -- the content decisions ---------------------------------------------
    def _prepare(self, contents: Sequence[Content], role: EncodeRole) -> PreparedItems:
        """The content decisions, in one place: the per-side prompt, the media preparation, then the budget.

        Args:
            contents: The items as given.
            role: Which side of the retrieval pair they are (``query_prompt`` vs ``doc_prompt``; the fit's
                request shape follows it).

        Returns:
            The items to send (each with the side's prompt and -- when the config declares a budget -- the
            content fitted to it: only content spans cut, the template re-attached, cuts recorded), each
            with its original position, and the positions ``empty_doc: omit_zero`` never sends (they score
            0.0). Media rides the ``messages`` route (2e): there each item's media is prepared with the
            request (``prepare_request``), its tokens counted and reserved whole beside the fitted text;
            on the text and token-ids routes media is refused before it is fetched.
        """
        prompt = self.config.query_prompt if role is EncodeRole.QUERY else self.config.doc_prompt
        prepared = tuple(content.with_text_prefix(prompt) for content in contents)
        self._refuse_media_off_its_side(role.value, prepared)
        if self._media_is_on_wire():
            # Media on the messages wire (2e): one preparation path, each item's media sized exactly as the
            # judge's, its tokens reserved whole beside the item's text (the embeddings budget is per item:
            # each input must fit the served context, the batch is how fast).
            request = self._prepare_request(list(prepared))
            fitted_pairs = [
                self._fit_media_for_request([content], doc_ids=[str(index)])
                for index, content in enumerate(request.contents)
            ]
            prepared = tuple(pair[0][0] for pair in fitted_pairs)
            media_tokens = [pair[1] for pair in fitted_pairs]
        else:
            media_tokens = [0] * len(prepared)
        if self._budget is None:
            kept, omitted = list(prepared), []
            return PreparedItems(
                items=tuple(kept),
                positions=tuple(range(len(contents))),
                omitted=tuple(omitted),
            )
        if self._budget is not None:
            result = self._fit(
                [content.text for content in prepared],
                "query" if role is EncodeRole.QUERY else "document",
                media_tokens=media_tokens,
            )
            prepared = tuple(
                self._with_text(content, text) for content, text in zip(prepared, result.texts, strict=True)
            )
        kept, omitted = self._apply_empty_documents(prepared)
        positions = [index for index in range(len(prepared)) if index not in set(omitted)]
        return PreparedItems(
            items=tuple(kept),
            positions=tuple(positions),
            omitted=tuple(omitted),
        )

    def _token_ids_of(self, items: Sequence[Content], role: EncodeRole) -> tuple[tuple[int, ...], ...]:
        """The token ids of each sent text, as the engine reads it, for ``request_shape: token_ids`` (3):
        the client's tokenisation of the fitted render, under the shape's ``add_special_tokens`` flag --
        the same count the fit verified."""
        if self.config.request_shape != "token_ids":
            return ()
        assert self._tokenizer is not None  # refused at construction without one
        shape: RequestShape = "query" if role is EncodeRole.QUERY else "document"
        flag = self.config.template.adds_special_tokens(shape) if self.config.template is not None else True
        return tuple(tuple(self._tokenizer.ids(item.text, add_special_tokens=flag)) for item in items)

    def _request_size(self, batch_size: int | None) -> int:
        """The request size of one call: ``batch_size``, else the config's; below 1 or above the profile's
        cap is refused (typed, R12)."""
        size = self.config.batch_size if batch_size is None else batch_size
        if size < 1:
            raise ConfigError(f"batch_size must be at least 1, got {size}")
        _check_batch_size(self._adapter_cls, size)
        return size

    async def probe(self) -> Any:
        """The role's startup probe: the transport's replica probe. The embed role sends no media probe
        request, so no engine media check runs here (the ``text`` and ``token_ids`` routes refuse media
        before it is fetched; the ``messages`` route counts its media on the declared policy)."""
        return await self._sender.probe()


__all__ = ["EmbeddingClient"]
