"""The pooling role client: late-interaction (multi-vector) encoding with every content decision applied.

A :class:`PoolingClient` turns :class:`~rcp_ndcg_core.content.Content` into the ragged
:class:`~rcp_ndcg.inference.types.Embeddings` of a multi-vector endpoint, one
:class:`~rcp_ndcg.inference.types.PoolRequest` per ``batch_size`` items sent through its
:class:`~rcp_ndcg.inference.transport.Sender`:

* **Prompts per role** -- the config's ``query_prompt``/``doc_prompt`` is prepended to the side it names;
* **the text budget** -- when the config declares one (``max_tokens``), every item's text is fitted through
  :func:`rcp_ndcg.data.preprocess.fit` (the side's shape): only content spans cut, the template re-attached
  with the anchor kept, every cut recorded in the census under ``text_budget``. A hosted profile with only
  ``max_tokens`` sends content uncut (the vendor path). ``on_overflow: chunk`` is refused: chunked documents
  pool *scores* by maximum, and token vectors are not scores -- a late-interaction document's chunking
  happens at the corpus layer (the retrieval index keeps one slice per chunk). Media items are prepared
  with the request (``prepare_request``), their tokens counted and reserved whole beside the fitted text;
* **Wire precision** -- the config's ``embed_dtype`` (``float16`` by the owner's decision, ``float32``
  opt-in) travels on every request and survives to the result: the ragged buffer keeps its transfer dtype
  end to end, so an index built from float16 vectors stores float16 (2 bytes per token vector, against 4
  for float32). MaxSim computes in float32 either way.
* **Normalisation** -- when ``normalize`` is set (the default), every token vector is L2-normalised in
  float32 and stored back in the transfer dtype (the engine's own token_embed pooling already normalises;
  normalising twice is harmless).
* **Concurrency** -- at most ``concurrency`` batch requests in flight under one :class:`asyncio.TaskGroup`
  (a failing request cancels its siblings, R7), reassembled in input order.

The sync :meth:`encode` runs the async path on the sender's sync bridge (the base's one rule: a
:class:`~rcp_ndcg.inference.transport.Transport`'s ``run``, or the sender's own).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import numpy as np
from rcp_ndcg_core.content import Content, ImagePart, TextPart

from rcp_ndcg.data.mrl import MrlHead
from rcp_ndcg.data.postprocess import kept_vector_count, skip_keep_mask
from rcp_ndcg.data.prepare import MediaCensus
from rcp_ndcg.data.resolution import (
    VISION_WRAPPER_TOKENS,
    ImagePolicy,
    VideoPolicy,
    content_media_tokens,
    sample_video_part,
)
from rcp_ndcg.data.templates import RequestShape
from rcp_ndcg.data.text_budget import FitResult, ProcessingRecord, TextTruncationCensus
from rcp_ndcg.errors import CapabilityError, ConfigError, ProviderError
from rcp_ndcg.inference.adapters.base import Adapter, get_adapter
from rcp_ndcg.inference.clients._base import PreparedItems, RoleClient
from rcp_ndcg.inference.config import PoolingEndpoint
from rcp_ndcg.inference.transport import Sender
from rcp_ndcg.inference.types import Call, Embeddings, EncodeRole, PoolRequest, Reply, TokenCount


class PoolingClient(RoleClient):
    """A served late-interaction encoder: content in, ragged token vectors out.

    The role client of a :class:`~rcp_ndcg.inference.config.PoolingEndpoint`: it applies the role's prompt
    and the text budget (:meth:`_prepare`), splits the batch, sends ``batch_size``-sized pooling requests
    ``concurrency`` at a time through the wire adapter the config's ``api`` names, and reassembles the
    ragged vectors in input order.

    Args:
        config: The pooling endpoint: where the model is served, its wire adapter (``vllm_pooling``), the
            prompts, ``embed_dtype``, ``dim``, the text budget and the batching.
        sender: What sends the calls. ``None`` builds a :class:`~rcp_ndcg.inference.transport.Transport`
            for the endpoint; anything else must provide the sync bridge (``run``).
        census: Where the text-budget cuts are recorded; ``None`` gives the client a fresh in-memory census.

    Raises:
        ConfigError: ``dim`` is not set (the base64 frame of ``/pooling`` is flat and carries no shape; a
            refusal at construction keeps the GPU idle-time free, R13), ``on_overflow: chunk`` is declared
            (vector roles do not pool chunks), ``request_shape`` names a shape the wire does not implement,
            ``token_ids`` is declared without a tokenizer, ``batch_size < 1``, or ``api`` names no adapter
            of the multi_vector role.
    """

    ROLE = "multi_vector"

    def __init__(
        self,
        config: PoolingEndpoint,
        *,
        sender: Sender | None = None,
        census: TextTruncationCensus | None = None,
        media_census: MediaCensus | None = None,
    ) -> None:
        if config.dim is None:
            raise ConfigError(
                "the pooling endpoint's encoding needs dim: the base64 frame of /pooling is flat and carries "
                "no shape, so the config's dim rebuilds (tokens, dim) client-side",
                hint="set dim to the checkpoint's token-vector width (a ColBERT-style checkpoint projects to "
                "a fixed width, e.g. 128); the bytes encoding carries its shape, but the adapter asks for "
                "base64",
            )
        if config.on_overflow == "chunk":
            raise ConfigError(
                "on_overflow 'chunk' pools scores by max, and token vectors have none to pool: the declared "
                "aggregation would be reinterpreted, so it is refused instead",
                hint="use on_overflow: cut (the content is cut to the budget), or chunk the corpus at load "
                "(the retrieval index keeps one slice per chunk)",
            )
        supported = getattr(get_adapter(config.api, role=self.ROLE), "REQUEST_SHAPES", frozenset({"text"}))
        if config.request_shape not in supported:
            raise ConfigError(
                f"request_shape {config.request_shape!r} is declared, but the {config.api} wire implements "
                f"{sorted(supported)}",
                hint="declare a request shape the wire implements (the default is text)",
            )
        if config.request_shape == "messages":
            raise ConfigError(
                "request_shape 'messages' is declared, but the pooling wire lowers chat parts for its media "
                "items itself; a text batch would silently travel as the rendered strings while the config's "
                "identity declared the chat form",
                hint="drop request_shape (the default): the pooling wire applies the chat form to media "
                "items on its own, and token_ids sends the fitted ids for the text batches",
            )
        if config.request_shape == "token_ids" and config.tokenizer is None:
            raise ConfigError(
                "request_shape 'token_ids' needs a tokenizer: the ids are the client's tokenisation of the "
                "fitted text, and a hosted profile without one cannot tokenise",
                hint="declare the tokenizer (with max_tokens), or drop request_shape (the default sends text)",
            )
        super().__init__(config, sender=sender, census=census, media_census=media_census)
        self._adapter: Adapter[PoolRequest, Embeddings] = self._adapter_cls(self.endpoint)
        self._mrl = MrlHead(
            kind=config.mrl_kind or "none",
            dims=config.mrl_dims or (),
            mrl_range=config.mrl_range,
            projection=config.mrl_projection,
        )

    # -- encoding ----------------------------------------------------------
    def encode(
        self,
        contents: Sequence[Content],
        role: EncodeRole,
        *,
        batch_size: int | None = None,
        instruction: str | None = None,
    ) -> Embeddings:
        """Ragged token vectors for ``contents``, in order (the synchronous form).

        Runs :meth:`aencode` to completion on the sender's sync bridge (a transport's ``run`` keeps the
        pool on one loop; an injected sender's own ``run`` is used).

        Args:
            contents: The queries or documents as content parts, in order.
            role: Which side of the retrieval pair the batch is; the prompts depend on it.
            batch_size: Items per pooling request; the config's ``batch_size`` when ``None``.
            instruction: The side's task instruction, when the caller has one (the config's ``instruction``
                mode places it, as the embedding role's).

        Returns:
            Ragged embeddings in the transfer dtype (one slice of vectors per item), or single-vector
            embeddings when the served task pooled instead and the reply reported no usage.
        """
        return self._run(self.aencode(contents, role, batch_size=batch_size, instruction=instruction))

    async def aencode(
        self,
        contents: Sequence[Content],
        role: EncodeRole,
        *,
        batch_size: int | None = None,
        instruction: str | None = None,
    ) -> Embeddings:
        """Ragged token vectors for ``contents``, in order (the asynchronous form).

        Args:
            contents: The queries or documents as content parts, in order.
            role: Which side of the retrieval pair the batch is; the prompts depend on it.
            batch_size: Items per pooling request; the config's ``batch_size`` when ``None``.
            instruction: The side's task instruction, when the caller has one.

        Returns:
            Ragged embeddings in the transfer dtype, in input order. An empty batch is the zero-item value
            and sends nothing. A served task that pooled instead of token-embedding shows up as one vector
            per item, which the adapter refuses when the reply reports usage.
        """
        prepared = self._prepare(contents, role, instruction=instruction)
        if not prepared.items:
            if not contents:
                return Embeddings.empty(0, multi_vector=True, dtype=self.config.embed_dtype)
            # Every input was omitted (empty_doc: omit_zero): no request goes out; the result is one empty
            # slice per input -- the MaxSim score an omitted document contributes.
            return Embeddings.ragged(
                [np.zeros((0, 0), dtype=self.config.embed_dtype)] * len(contents),
                dtype=self.config.embed_dtype,
            )
        size = self._request_size(batch_size)
        batches = [prepared.items[start : start + size] for start in range(0, len(prepared.items), size)]
        id_batches = [prepared.token_ids[start : start + size] for start in range(0, len(prepared.items), size)]
        position_batches = [prepared.positions[start : start + size] for start in range(0, len(prepared.items), size)]
        gate = asyncio.Semaphore(self.config.concurrency)

        async def one(
            batch: list[Content], batch_ids: tuple[tuple[int, ...], ...], batch_positions: tuple[int, ...]
        ) -> Embeddings:
            async with gate:
                return await self._encode_batch(batch, role, batch_ids, batch_positions)

        chunks = await RoleClient.gather(
            [
                one(list(batch), batch_ids, batch_positions)
                for batch, batch_ids, batch_positions in zip(batches, id_batches, position_batches, strict=True)
            ]
        )
        if not prepared.omitted:
            return _concat_all(chunks)
        # Some inputs were omitted (empty_doc: omit_zero): rebuild the buffer with an empty slice (a zero
        # MaxSim score) at each omission, the gathered slices in their places, in input order.
        per_item: list[np.ndarray] = []
        for chunk in chunks:
            offsets = chunk.offsets
            assert offsets is not None
            per_item.extend(chunk.vectors[a:b] for a, b in zip(offsets[:-1], offsets[1:], strict=True))
        sent = iter(per_item)
        slices: list[np.ndarray] = []
        for index in range(len(contents)):
            if index in set(prepared.omitted):
                slices.append(np.zeros((0, 0), dtype=self.config.embed_dtype))
            else:
                slices.append(next(sent))
        return Embeddings.ragged(slices, dtype=self.config.embed_dtype)

    def _prepare(
        self, contents: Sequence[Content], role: EncodeRole, *, instruction: str | None = None
    ) -> PreparedItems:
        """The contents as they are sent, through the pipeline (:data:`STAGES`, one order for every role):
        the role's prompt prepended, the task instruction where the config places it, the media prepared,
        then the budget.

        The content decisions stay the ones a late-interaction encoder needs -- the role's prompt, the one
        media preparation call (:meth:`RoleClient._prepare_request`, which records the kept media), the
        budget's media fit per wire request with every drop recorded (:meth:`RoleClient._fit_media_for_request`,
        slicing the one preparation), and the text fit: only the text's content span is cut (the template
        re-attached, every cut recorded), and media tokens are reserved whole and never cut. The client cuts
        nothing else: a model-side change without a config field is a silent change to the vectors.
        """
        shape: RequestShape = "query" if role is EncodeRole.QUERY else "document"
        prepared = self._prepare_rows(contents, side=role.value, shape=shape, instruction=instruction)
        # The tracked token ids are a property of the sent texts (the fit verified their count), so they are
        # built here, after the pipeline, from what the lower stage put on the item; a text item carrying a
        # media-allowlist id is refused here, before anything is sent (the allowlist is its own gate).
        texts = [content.text for content in prepared.items]
        ids = self._sent_ids(texts, role)
        self._refuse_allowlist_collisions(prepared.items, ids)
        return PreparedItems(
            items=prepared.items,
            positions=prepared.positions,
            omitted=prepared.omitted,
            token_ids=ids,
        )

    def _refuse_allowlist_collisions(self, items: Sequence[Content], ids: tuple[tuple[int, ...], ...]) -> None:
        """Refuse a TEXT item whose sent ids carry a media allowlist id, before anything is sent.

        The media allowlist is its own gate engine-side: a row carrying one of its ids is treated as a media
        document and keeps only those positions. A text render that carries the literal special token would
        therefore lose every other vector, so the collision is refused by name here (the ids are checked
        where the client tracks them: a document under the text rule or the allowlist, or either side under
        ``request_shape: token_ids``). A media item's own ids are its caption's and are exempt -- the
        allowlist is what a media render is supposed to carry. A role that tracks no ids (the query side
        under a document-side rule) has nothing to check: an empty id set is skipped, never zipped.

        Raises:
            CapabilityError: a text item's sent ids intersect ``media_keep_token_ids``.
        """
        keep = set(self.config.media_keep_token_ids)
        if not keep or not ids:
            return
        for index, (content, row) in enumerate(zip(items, ids, strict=True)):
            if content.has_media or not row:
                continue
            collision = sorted(keep.intersection(row))
            if collision:
                raise CapabilityError(
                    f"item {index}'s sent text tokenises to the media allowlist's id(s) {collision} "
                    "(media_keep_token_ids): the served plugin treats a row carrying one of them as a media "
                    "document and would keep only those positions, not the text's vectors",
                    hint="remove the literal special token from the text, or drop media_keep_token_ids",
                )

    def _stage_lower(
        self,
        contents: Sequence[Content],
        *,
        result: FitResult | None,
        shape: RequestShape,
    ) -> tuple[list[Content], tuple[tuple[int, ...], ...]]:
        """The pooling role's ``lower`` stage: a text item's wire carries the rendered string (the text and
        ``token_ids`` routes -- the pooling wire refuses a declared ``messages`` shape), while a media item
        rides the ``messages`` route and the engine's chat template frames the content once -- its text part
        carries the fitted content span, exactly like the embed role's messages route (framing it here too
        would put the declared frame inside the engine's render twice). The pool tracks no ids at this
        stage (the ids are built from the sent texts in :meth:`_prepare`, when the role tracks them)."""
        if result is None:
            return super()._stage_lower(contents, result=result, shape=shape)
        items: list[Content] = []
        for content, span, render in zip(contents, result.contents, result.texts, strict=True):
            sent = str(span) if content.has_media else str(render)
            items.append(self._with_text(content, sent))
        return items, ()

    def _sent_ids(self, texts: Sequence[str], role: EncodeRole) -> tuple[tuple[int, ...], ...]:
        """The token ids of each sent text, as the engine reads it, when the role tracks them (2, 3): the
        document side under declared ``document_skip_token_ids``, or both sides under ``request_shape:
        token_ids``. The client tokenises the fitted render with the shape's ``add_special_tokens`` flag --
        the same count the fit verified -- so the ids are what the engine reads; a reply whose vector count
        disagrees is a typed error. A MEDIA item's ids are its sent text's (a caption): the messages route's
        render is the engine's chat-template render, which the client cannot tokenise, so those ids feed the
        declared kept count under ``document_skip_engine_side`` (the caption's positions) and are otherwise
        unused -- the skip rule at image positions keeps every vector of one, on record, unless a media
        allowlist is declared (then the engine keeps only its positions)."""
        wants_ids = (
            role is EncodeRole.DOCUMENT
            and (bool(self.config.document_skip_token_ids) or bool(self.config.media_keep_token_ids))
        ) or self.config.request_shape == "token_ids"
        if not wants_ids:
            return ()
        assert self._tokenizer is not None, "the config refuses tracked ids without a tokenizer (inert without one)"
        shape: RequestShape = "query" if role is EncodeRole.QUERY else "document"
        flag = self.config.template.adds_special_tokens(shape) if self.config.template is not None else True
        return tuple(tuple(self._tokenizer.ids(text, add_special_tokens=flag)) for text in texts)

    def _probe_calls(self, content: Content) -> Sequence[Call]:
        """The pooling calls one prepared probe item is sent as."""
        request = PoolRequest(
            contents=(content,),
            role=EncodeRole.DOCUMENT,
            embed_dtype=self.config.embed_dtype,
            dim=self.config.dim,
            system_head=self._media_system_head("document"),
        )
        return self._adapter.calls(request, model=self.config.model)

    def _probe_baseline_calls(self, content: Content) -> Sequence[Call] | None:
        """The probe request without its media, in the same ``messages`` shape the media request takes (the
        pooling wire routes media through the chat template; the baseline must ride it too, or the delta
        would carry the template and the declared system head). A wire adapter that offers no baseline form
        records the check ``not_checked``."""
        baseline = getattr(self._adapter, "media_probe_baseline", None)
        if baseline is None:
            return None
        request = PoolRequest(
            contents=(content,),
            role=EncodeRole.DOCUMENT,
            embed_dtype=self.config.embed_dtype,
            dim=self.config.dim,
            system_head=self._media_system_head("document"),
        )
        return [baseline(request, model=self.config.model)]

    def _media_system_head(self, shape: RequestShape) -> str | None:
        """The media side's leading fixed template segments when the config sends them as a system message
        (``media_head_as_system``): the trained role prefix a pass-through engine chat template would
        otherwise drop from an image document. The head is resolved from the template's own segments
        (specials by name), stops at the first content span, and is ``None`` when the config does not
        declare the mechanism."""
        if not getattr(self.config, "media_head_as_system", False):
            return None
        template = self.config.template
        assert template is not None, "the config refuses media_head_as_system without a template"
        assert self._tokenizer is not None, "a template implies a tokenizer (the config refuses one without it)"
        head: list[str] = []
        for segment in template.segments(shape):
            if segment.content is not None:
                break
            head.append(segment.render(self._tokenizer))
        return "".join(head) or None

    async def probe(self) -> Any:
        """The role's startup probe: the transport's replica probe, plus -- when the config declares an
        ``image_processor`` -- the engine media check (one prepared probe image beside its no-media baseline,
        the engine's media delta compared with the counted media tokens; never silent)."""
        probe: Any = await self._sender.probe()
        await self.check_engine_media()
        return probe

    def _probe_usage(self, reply: Reply) -> TokenCount | None:
        """The reply's prompt-token report (the pooling adapter's, ``None`` when it reported none)."""
        return self._adapter.usage(reply)

    async def _encode_batch(
        self,
        contents: Sequence[Content],
        role: EncodeRole,
        batch_ids: tuple[tuple[int, ...], ...] = (),
        batch_positions: tuple[int, ...] = (),
    ) -> Embeddings:
        """One batch: a pooling request through the adapter and the sender, checked for alignment.

        ``batch_ids`` carries each document's sent token ids (tracked when the config declares
        ``document_skip_token_ids`` or ``media_keep_token_ids``): the returned vectors are checked against
        them (a mismatch is a typed error, never a silent misalignment) and the skip positions' vectors are
        dropped before MaxSim. ``batch_positions`` carries each document's ORIGINAL input position (the
        census rows and the processing records name it), for the skip's per-row record of a media item.

        A batch under a document-side rule that MIXES text-only and media documents is refused before
        anything is sent: one media item routes the whole batch through the ``messages`` wire, where the
        text items' replies are the engine's chat-template render -- the client cannot align its sent ids or
        its declared counts there, so no rule could be applied to them honestly. Pure-text batches (the text
        rule by id) and pure-media batches (the media allowlist, or every vector kept on record) are the two
        honest shapes.

        Raises:
            CapabilityError: the batch mixes text-only and media documents under ``document_skip_token_ids``
                or ``media_keep_token_ids``.
        """
        if (self.config.document_skip_token_ids or self.config.media_keep_token_ids) and role is EncodeRole.DOCUMENT:
            has_media = any(content.has_media for content in contents)
            has_text_items = any(not content.has_media for content in contents)
            if has_media and has_text_items:
                declared = "document_skip_token_ids" if self.config.document_skip_token_ids else "media_keep_token_ids"
                raise CapabilityError(
                    f"{self.config.model} declares {declared}, and this batch mixes text-only and media "
                    "documents: one media item routes the whole batch through the messages wire, whose "
                    "positions are the engine's chat-template render -- the text documents' sent ids (and "
                    "the declared kept counts) cannot be found there",
                    hint="encode the text documents with the declared rule and the media documents "
                    "separately (lower batch_size, or split the call; a media document alone keeps the "
                    "rule's positions, on record)",
                )
        request = PoolRequest(
            contents=tuple(contents),
            role=role,
            embed_dtype=self.config.embed_dtype,
            dim=self.config.dim,
            outputs=self.config.outputs,
            request_shape=self.config.request_shape,
            token_ids=batch_ids,
            kept_counts=self._declared_kept_counts(contents, batch_ids) if role is EncodeRole.DOCUMENT else (),
            system_head=self._media_system_head("query" if role is EncodeRole.QUERY else "document")
            if any(content.has_media for content in contents)
            else None,
        )
        calls = self._adapter.calls(request, model=self.config.model)
        self._gate_media_calls(calls)
        replies = await self._sender.send(calls)
        self._record_usage(replies)
        embeddings = self._adapter.interpret(request, replies)
        if embeddings.num_items != len(contents):
            raise ProviderError(
                f"the pooling endpoint returned {embeddings.num_items} item(s) for {len(contents)} input(s); "
                "refusing to return misaligned vectors"
            )
        if (self.config.document_skip_token_ids or self.config.media_keep_token_ids) and role is EncodeRole.DOCUMENT:
            embeddings = self._apply_document_skips(contents, embeddings, batch_ids, batch_positions)
        if self.config.mrl_dim is not None:
            # The declared normalisation of the FULL-WIDTH reply runs first, then the head: the head
            # renormalises its own output (and the learned projection is linear), so the cut's direction
            # is the card's, and this order is what makes the ex-post sweep over a full-width store
            # bit-identical to a direct run. A k equal to the observed width is the identity selection:
            # no head, no record.
            if self.config.normalize:
                embeddings = embeddings.l2_normalized()
            if self.config.mrl_dim != embeddings.dim:
                embeddings = self._apply_mrl_cut(embeddings, role, batch_positions)
        elif self.config.normalize:
            embeddings = embeddings.l2_normalized()
        return embeddings

    def _apply_mrl_cut(
        self, embeddings: Embeddings, role: EncodeRole, batch_positions: tuple[int, ...] = ()
    ) -> Embeddings:
        """The Matryoshka head, through its one home (:mod:`rcp_ndcg.data.mrl`): the declared kind
        applied to the model's full-width vectors -- the truncation cut (slice then renormalise, the
        card's order) or the checkpoint's learned projection. The caller normalises the full-width reply
        first when ``normalize``; renormalising before or after the cut gives the same direction (the
        head renormalises the cut, and the projection is linear), and the order makes the ex-post sweep
        over a full-width store bit-identical to a direct run. ``/pooling`` refuses per-request
        ``dimensions``, so the cut is the client's. The config refuses an ``mrl_dim`` wider than ``dim``,
        and the adapter refuses a reply whose width differs from ``dim``, so the head sees full-width
        vectors; every item it changed carries an ``mrl_cut`` :class:`ProcessingRecord`. The caller does
        not reach here when ``mrl_dim`` equals the observed width (the identity selection applies no head
        and writes no record).
        """
        assert self.config.mrl_dim is not None
        full_width = embeddings.dim
        vectors = self._mrl.apply(embeddings.vectors, self.config.mrl_dim)
        offsets = embeddings.offsets
        shape: RequestShape = "query" if role is EncodeRole.QUERY else "document"
        if offsets is not None:
            for index in range(embeddings.num_items):
                if int(offsets[index]) == int(offsets[index + 1]):
                    continue  # an empty slice was not changed by the head: no record
                input_id = str(batch_positions[index]) if index < len(batch_positions) else str(index)
                self.processing.append(
                    ProcessingRecord(
                        corpus=self.ROLE,
                        input_id=input_id,
                        shape=shape,
                        mechanisms=("mrl_cut",),
                        mrl_kind=self.config.mrl_kind,
                        mrl_dim=self.config.mrl_dim,
                        full_width=full_width,
                    )
                )
        return Embeddings(vectors=vectors, offsets=offsets)

    def _declared_kept_counts(
        self, contents: Sequence[Content], batch_ids: tuple[tuple[int, ...], ...]
    ) -> tuple[int, ...]:
        """The declared count of kept vectors per item, for a batch the served plugin applies a rule to.

        Each rule's declared home is the plugin (``document_skip_engine_side`` for the text skip rule,
        ``media_keep_token_ids`` for the media allowlist), so the wire carries only the kept vectors and the
        client cannot count the reply by the positions it sent -- it declares the kept count per item
        instead, and the adapter refuses a reply that disagrees. The counts are per BATCH TYPE (a batch
        mixing text and media documents is refused earlier):

        * a TEXT batch (the engine-side skip rule): each item's sent render's ids outside the rule
          (:func:`kept_vector_count`);
        * a MEDIA batch under the allowlist: the media block's patch run -- the allowlist keeps the render's
          media positions only, so the vision wrapper (and the head and a caption, which the block's count
          never included) is not kept;
        * a MEDIA batch under the engine-side skip rule (no allowlist): the render the engine reads -- the
          leading fixed head the client sends as a system message (``media_head_as_system``), the item's
          sent text (a caption), and the prepared media block's counted tokens. The block's own positions
          are the processor's structural tokens, which the declared rule never names, so they are kept
          whole.

        A batch whose rule the engine does not apply returns no counts: the usage cross-check stands.
        """
        if any(content.has_media for content in contents):
            if self.config.media_keep_token_ids:
                return tuple(self._media_allowlist_kept_count(content) for content in contents)
            if not self.config.document_skip_engine_side:
                return ()
            return tuple(
                self._media_skip_kept_count(content, ids) for content, ids in zip(contents, batch_ids, strict=True)
            )
        if not self.config.document_skip_engine_side:
            return ()
        skip = self.config.document_skip_token_ids
        return tuple(kept_vector_count(ids, skip) for ids in batch_ids)

    def _media_skip_kept_count(self, content: Content, ids: tuple[int, ...]) -> int:
        """The declared kept count of one media document under the engine-side TEXT skip rule: the render the
        engine reads -- the sent head's kept ids, the item's sent text (a caption) and the prepared media
        block's counted tokens (the vision wrapper plus the patch run)."""
        assert self._tokenizer is not None, "the config refuses the rule without a tokenizer"
        head = self._media_system_head("document")
        flag = self.config.template.adds_special_tokens("document") if self.config.template is not None else True
        head_ids = tuple(self._tokenizer.ids(head, add_special_tokens=flag)) if head else ()
        caption = tuple(ids) if content.text else ()
        media_tokens = self._media_counts_of([content])[0].tokens
        return kept_vector_count((*head_ids, *caption), self.config.document_skip_token_ids, media_tokens=media_tokens)

    def _media_allowlist_kept_count(self, content: Content) -> int:
        """The declared kept count of one media document under the allowlist: the media's patch run.

        The allowlist is the checkpoint's own ``keep_only_token_ids`` (topk-embed-v1's image-patch token),
        so only the media's patch positions are kept: each image -- and each sampled video frame --
        contributes its merged patches, not its vision wrapper, and a video CONTAINER's own accounting (its
        timestamp lines and vision blocks) cannot be derived from the block's count: it is refused, never
        guessed. The head and a caption are outside the allowlist and are not kept.

        Raises:
            CapabilityError: a video container is part of the content (its kept positions are not derivable
                from the counted media block).
        """
        image, video = self._media_policies()
        policy = image if image is not None else ImagePolicy.native()
        kept = 0
        for part in content.parts:
            if isinstance(part, TextPart):
                continue
            if isinstance(part, ImagePart):
                kept += max(self._part_media_tokens(part, policy, video) - VISION_WRAPPER_TOKENS, 0)
                continue
            shown = sample_video_part(part, video)
            if not shown.frames:
                raise CapabilityError(
                    f"{self.config.model} declares media_keep_token_ids, but this media item is a video "
                    "container: its prompt renders timestamp lines and vision blocks, and the allowlist's "
                    "kept positions cannot be derived from the counted block",
                    hint="send the clip as frames (wire: frames), whose per-frame patches the count names, "
                    "or drop media_keep_token_ids",
                )
            kept += max(self._part_media_tokens(shown, policy, video) - VISION_WRAPPER_TOKENS * len(shown.frames), 0)
        return kept

    def _part_media_tokens(self, part: Any, image: ImagePolicy, video: VideoPolicy | None) -> int:
        """One media part's counted tokens (the client's own policy and tokenizer, as the block's count)."""
        return content_media_tokens(Content.from_parts([part]), image, video, tokenizer=self._tokenizer).tokens

    def _apply_document_skips(
        self,
        contents: Sequence[Content],
        embeddings: Embeddings,
        batch_ids: tuple[tuple[int, ...], ...],
        batch_positions: tuple[int, ...] = (),
    ) -> Embeddings:
        """The document vectors without the rules' excluded positions (2, the topk hand-off).

        The skip rule at image positions: a TEXT document's positions are the ids the client sent (the
        engine's tokenisation of the fitted render), and the vectors at the skip ids are dropped -- a reply
        whose per-item vector count disagrees with the sent ids is a typed :class:`ProviderError`, never a
        silent misalignment. A MEDIA document's positions are the server's chat-template render, which the
        client cannot tokenise -- the image positions are exempt from the skip (the vision tokens are what
        the model reads for the media) and the text positions cannot be located within the render, so the
        client keeps every returned vector for a media item and records the deviation on the row's
        :class:`~rcp_ndcg.data.text_budget.ProcessingRecord` (``skip_unapplied``): never silently unskipped.

        When the served plugin applies a rule engine-side -- the text rule (``document_skip_engine_side``)
        or the media allowlist (``media_keep_token_ids``) -- the reply IS the kept set for that batch's
        items, so the client slices nothing and records no deviation; the declared kept counts
        (:meth:`_declared_kept_counts`) travelled on the request, and the adapter refused a reply that
        disagrees with them (a mismatch is typed, never silent).

        Args:
            contents: The batch's documents as sent.
            embeddings: The reply's ragged vectors, aligned to ``contents``.
            batch_ids: Each document's sent token ids (a media item's are its caption's: nothing is
                attributable to the render).
            batch_positions: Each document's ORIGINAL input index, for the record's ``input_id``.

        Returns:
            The ragged embeddings with the excluded positions' vectors dropped (media items whole, or the
            engine's already-kept vectors when the plugin applies a rule).

        Raises:
            CapabilityError: ids were not tracked for a text item under the skip list.
            ProviderError: a returned vector count disagrees with the sent ids, or the served task is not
                ``token_embed``.
        """
        if embeddings.offsets is None:
            raise ProviderError(
                "the pooling endpoint answered one pooled vector per input, and the config declares "
                "document_skip_token_ids: a token_embed answer has one vector per prompt token, so the "
                "served task is not token_embed",
                hint="check the served pooler task against the endpoint config",
            )
        skip = self.config.document_skip_token_ids
        engine_side = self.config.document_skip_engine_side
        media_engine_side = bool(self.config.media_keep_token_ids)
        offsets = embeddings.offsets
        slices: list[np.ndarray] = []
        for index, ids in enumerate(batch_ids):
            vectors = np.asarray(embeddings.vectors[offsets[index] : offsets[index + 1]])
            if contents[index].has_media:
                # The image positions are exempt from the TEXT skip (never skipped) and the render's text
                # positions cannot be located client-side: keep the item whole, on record -- unless the
                # plugin applied a rule engine-side (the media allowlist, or the text rule on the render),
                # when the reply already carries exactly the kept set (its count was checked).
                slices.append(vectors)
                if not (media_engine_side or engine_side):
                    self._record_skip_unapplied(contents[index], batch_positions, index)
                continue
            if engine_side:
                slices.append(vectors)  # the engine's kept vectors; the adapter checked the count
                continue
            if len(vectors) != len(ids):
                raise ProviderError(
                    f"the pooling endpoint returned {len(vectors)} token vector(s) for document {index} whose "
                    f"sent render the client tokenised to {len(ids)} token id(s); the skip positions would "
                    "not align to the ids the client sent, so none are returned",
                    hint="the reply's prompt tokens must be the client's tokenisation of the sent text: check "
                    "the served route's add_special_tokens against the declared template",
                )
            slices.append(vectors[skip_keep_mask(ids, skip)])
        return Embeddings.ragged(slices, dtype=self.config.embed_dtype)

    def _record_skip_unapplied(self, content: Content, batch_positions: tuple[int, ...], index: int) -> None:
        """Record the ``skip_unapplied`` deviation of one media document: the declared skip was not applied
        and every returned vector was kept. The row's id is its ORIGINAL input index (the caller's batch
        positions; a batch-local one when the caller passed none), so the harness's per-row gating reads it
        beside the preparation records."""
        input_id = str(batch_positions[index]) if index < len(batch_positions) else str(index)
        self.processing.append(
            ProcessingRecord(
                corpus=self.ROLE,
                input_id=input_id,
                shape="document",
                mechanisms=("skip_unapplied",),
            )
        )


def _concat_all(chunks: Sequence[Embeddings]) -> Embeddings:
    """Append every chunk's items in order (one chunk needs no copy)."""
    result = chunks[0]
    for chunk in chunks[1:]:
        result = result.concat(chunk)
    return result


__all__ = ["PoolingClient"]
