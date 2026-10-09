"""The role-client base: the one class every role client derives from.

Three role clients built in parallel answered the same questions three ways -- how credentials are
resolved, how a synchronous caller reaches asyncio, how a client closes, where a hosted profile's base
URL comes from. This base answers each once:

* **the adapter lookup within the client's role** -- :func:`~rcp_ndcg.inference.adapters.base.get_adapter`
  with :attr:`ROLE`, so a config whose ``api`` names another role's adapter is refused here, once;
* **the hosted profile's default base URL** -- the config's ``base_url``, else the adapter's
  ``DEFAULT_BASE_URL``; a config with neither is refused;
* **the transport** -- built from the resolved config (with the adapter profile's
  :class:`~rcp_ndcg.inference.transport.AuthProfile`, so the key resolution lives in the transport)
  unless the caller supplies a :class:`~rcp_ndcg.inference.transport.Sender`;
* **the sync bridge, one rule** -- the sender's ``run``: a :class:`~rcp_ndcg.inference.transport.Transport`
  always has it, and any other sender must provide ``run`` (or the constructor raises a ``ConfigError``).
  Chosen over an ``asyncio.run`` fallback: a fresh loop per call would give a non-transport sender no pool
  reuse and would fail inside a running loop (a notebook), where ``Transport.run`` already knows to switch
  to a background thread;
* **the lifecycle** -- ``close()`` synchronous, ``async aclose()`` awaiting, and both context
  managers (``with`` and ``async with``);
* **the fan-out, one rule** -- :meth:`RoleClient.gather`: :class:`asyncio.TaskGroup` semantics, so a
  failing request cancels its siblings and leaves no task pending;
* **the text budget and the media** -- :class:`~rcp_ndcg.data.preprocess.TextBudget` resolved
  from the role config's fields once, the tokenizer it names loaded once, the shared
  :func:`rcp_ndcg.data.preprocess.fit` called from each client's ``_prepare``, and the media prepared per
  request through :func:`~rcp_ndcg.data.prepare.prepare_request` (tokens reserved whole, never cut).

Typed errors only: a wrong role, a missing bridge, a missing base URL and a ``batch_size < 1`` are all
:class:`~rcp_ndcg.errors.ConfigError`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Sequence
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, Self, TypeVar

from rcp_ndcg_core._records import Query
from rcp_ndcg_core.content import Content, ImagePart, TextPart, VideoPart

from rcp_ndcg.data.prepare import (
    MediaCensus,
    PreparedRequest,
    apply_media_fit,
    fit_media_to_budget,
    media_policies_for,
    prepare_request,
)
from rcp_ndcg.data.resolution import ImagePolicy, MediaTokenCount, VideoPolicy, content_media_tokens
from rcp_ndcg.data.templates import RequestShape
from rcp_ndcg.data.text_budget import (
    ChangeMechanism,
    FitResult,
    ProcessingRecord,
    TextBudget,
    TextBudgetExceededError,
    TextTruncationCensus,
    fit,
    fixed_overhead,
    processing_records,
)
from rcp_ndcg.errors import CapabilityError, ConfigError, DataError
from rcp_ndcg.inference.adapters import embeddings as _shipped_adapters  # noqa: F401  # registers them
from rcp_ndcg.inference.adapters.base import AdapterRole, get_adapter
from rcp_ndcg.inference.endpoint import Endpoint
from rcp_ndcg.inference.transport import AuthProfile, Sender, Transport
from rcp_ndcg.inference.types import Call, Reply, TokenCount, Usage
from rcp_ndcg.support.logging import get_logger

_TRANSPORT_CLASS = Transport
"""The transport class, captured at import: a test may patch the module attribute (an offline wire), and
the base's own transport checks must not break when it does."""

if TYPE_CHECKING:
    from rcp_ndcg.data.tokenizer import TextTokenizer

T = TypeVar("T")
"""The result type of a coroutine the fan-out runs."""


def _check_batch_size(adapter: type[Any], size: int, *, noun: str = "items") -> None:
    """Refuse a request size above the profile's published cap, instead of silently capping it -- the one
    batch-cap rule of every role.

    The cap is a HOSTED profile's own fact: a served engine (vLLM, TEI, Infinity -- the
    ``openai_embeddings`` and ``/pooling`` shapes) answers an over-count batch with its own refusal, which
    the adapter maps to a typed :class:`~rcp_ndcg.errors.CapabilityError` naming ``batch_size`` -- a stale
    client-side cap must not refuse a batch the engine would serve.

    Args:
        adapter: The resolved adapter class (its ``MAX_BATCH`` and ``HOSTED`` facts).
        size: The request size.
        noun: What one batch entry is called in the message (``texts``, ``items``).

    Raises:
        ConfigError: a HOSTED profile's published cap is exceeded.
    """
    max_batch = getattr(adapter, "MAX_BATCH", None)
    if max_batch is not None and getattr(adapter, "HOSTED", False) and size > max_batch:
        raise ConfigError(
            f"the {getattr(adapter, 'name', '?')} API takes at most {max_batch} {noun} per request; "
            f"batch_size is {size}",
            hint=f"set batch_size to {max_batch} or less, or leave it unset",
        )


class PreparedItems(NamedTuple):
    """What a role client's ``_prepare`` decided to send, aligned back to the inputs.

    Attributes:
        items: The contents to send, in fit order (the media prepared, the text fitted).
        positions: For each item, its original input index (``positions[i]`` is where ``items[i]`` came
            from); every input not in :attr:`omitted` appears exactly once.
        omitted: The input indices ``empty_doc: omit_zero`` never sends -- ascending; the caller places the
            missing result (a zero vector, an empty slice, a 0.0 score) at each.
        token_ids: For each item, the token ids of its sent text as the engine reads it (the client's
            tokenisation of the fitted render under the shape's ``add_special_tokens`` flag), when the role
            tracks them -- the pooling role's ``document_skip_token_ids`` needs the positions. Empty when
            not tracked.
    """

    items: tuple[Content, ...]
    positions: tuple[int, ...]
    omitted: tuple[int, ...]
    token_ids: tuple[tuple[int, ...], ...] = ()


#: The one preparation pipeline, in the order every role client applies it. A role's ``_prepare``
#: composes exactly these stages -- the base's runners below, with the role's own hooks in the two
#: role-shaped places (:meth:`RoleClient._stage_normalise`, :meth:`RoleClient._stage_lower`; the
#: rerank's pair budget is its own hook beside the shared fit) -- and nothing re-orders them. The
#: order is behaviour: the empty stage decides on the content AS GIVEN (before the template frames it
#: and before any media is fetched), the media fit can make a document empty (its all-dropped outputs
#: re-enter the same empty policy, so the policy always sees the content as it will be sent), and the
#: budget fit reserves the fixed frame before it cuts a content span. A test pins the order; a stage
#: added out of place is a bug, never a composition.
STAGES: tuple[str, ...] = ("normalise", "empty", "media", "render", "budget", "lower")


class RoleClient[C: Endpoint]:
    """The shared ground of every role client; subclasses set :attr:`ROLE` and own their content decisions.

    Attributes:
        config: The role config as it was given (a hosted profile's ``base_url`` stays ``None``).
        endpoint: The config the transport and the adapter see: the profile's public base URL filled in
            when the config set none.
        census: The :class:`~rcp_ndcg.data.preprocess.TextTruncationCensus` every text-budget cut is
            recorded into -- the one passed in, or a fresh in-memory one.
        media_census: The :class:`~rcp_ndcg.data.prepare.MediaCensus` the prepared media (and the items a
            budget dropped, with ``dropped=True``) are recorded into -- the one passed in, or a fresh
            in-memory one.
        processing: The :class:`~rcp_ndcg.data.preprocess.ProcessingRecord` of every input row the client
            changed before sending it, in the order the requests were prepared (append-only; a row without a
            record was sent as given). The per-row reading of the census rows and the media fit's decisions:
            what a consumer such as the equivalence harness decides gating from.
    """

    #: The adapter role this client speaks: the registry namespace its config's ``api`` resolves in, and
    #: the engine role of the configs it serves (the two vocabularies are mapped in one place, in
    #: :func:`~rcp_ndcg.inference.adapters.base.check_engine_api`).
    ROLE: ClassVar[AdapterRole]

    #: The adapter a config's unset ``api`` resolves to, per client class. The other roles default the field
    #: on the config class itself; the judge's ``api`` is ``None``-means-default (its ``openai_chat`` wire)
    #: so an unset one stays out of the identity payload. ``None``: the config must name the adapter.
    DEFAULT_API: ClassVar[str | None] = None

    #: Whether this role's wires carry media. ``False`` (the embed role: its adapters are text-only)
    #: refuses a media-carrying request before the media is fetched or counted, with the adapter's own
    #: typed refusal.
    MEDIA_ON_WIRE: ClassVar[bool] = True

    #: How one batch entry is named in the batch-cap refusal (:func:`_check_batch_size`).
    BATCH_NOUN: ClassVar[str] = "items"

    #: The role config, as it was given (a hosted profile's ``base_url`` stays ``None``).
    config: C

    #: The wire adapter instance the subclass builds (an embed/pool adapter is stateless; the rerank
    #: family's holds the config it serves).
    _adapter: Any

    def __init__(
        self,
        config: C,
        *,
        sender: Sender | None = None,
        census: TextTruncationCensus | None = None,
        media_census: MediaCensus | None = None,
    ):
        """Build the client for ``config``, sending over ``sender``.

        Args:
            config: The role config; its ``api`` names the wire adapter (resolved within :attr:`ROLE`), and
                a hosted profile without a ``base_url`` takes its adapter's ``DEFAULT_BASE_URL``.
            sender: What sends the calls. ``None`` builds a :class:`~rcp_ndcg.inference.transport.Transport`
                for the resolved config; anything else must be a ``Sender`` with a sync bridge (``run``) --
                a transport's, or the sender's own.
            census: Where the text-budget cuts are recorded; ``None`` gives the client a fresh in-memory
                census (:attr:`census`), whose rows a caller can read or hand a sink to.
            media_census: Where the prepared media (and the items a budget dropped) are recorded; ``None``
                gives the client a fresh in-memory :class:`~rcp_ndcg.data.prepare.MediaCensus`
                (:attr:`media_census`).

        Raises:
            ConfigError: ``api`` names no adapter of this client's role (the hint lists that role's names),
                a hosted profile is left without a base URL and its adapter declares none, or a non-transport
                sender has no sync bridge (``run``).
        """
        api = getattr(config, "api", None) or type(self).DEFAULT_API
        if not isinstance(api, str) or not api:
            raise ConfigError(
                f"{type(config).__name__} declares no wire adapter (api)", hint="api names the wire adapter"
            )
        self._adapter_cls: type[Any] = get_adapter(api, role=self.ROLE)
        self.config = config
        self.endpoint = self._resolved_endpoint(config, self._adapter_cls)
        self.census = census if census is not None else TextTruncationCensus()
        self.media_census = media_census if media_census is not None else MediaCensus()
        self.processing: list[ProcessingRecord] = []
        if sender is None:
            self._sender: Any = Transport(self.endpoint, auth=self._auth_profile())
        elif isinstance(sender, _TRANSPORT_CLASS) or callable(getattr(sender, "run", None)):
            self._sender = sender
        else:
            raise ConfigError(
                "the sender has no sync bridge: a Transport bridges synchronous calls through its run(), and "
                "an injected sender must provide one too",
                hint="pass a Transport (the transport built from the config), or give the sender a "
                "run(coroutine) method",
            )
        self._budget, self._tokenizer = self._resolve_budget()
        self._check_use_activation(config)
        self._point_sender_at_the_profile()

    # -- the shared construction decisions ----------------------------------
    @staticmethod
    def _resolved_endpoint(config: C, adapter: type[Any]) -> C:
        """The config as the transport and adapter receive it: the profile's public URL when ``base_url`` is
        ``None``. The credentials are the transport's business now: ``api_key_env`` is never cleared."""
        if config.base_url is not None:
            return config
        default = getattr(adapter, "DEFAULT_BASE_URL", None)
        if not default:
            raise ConfigError(
                f"the {getattr(adapter, 'name', '?')} endpoint needs base_url",
                hint="a served engine has no public root: set base_url to the engine's URL "
                "(e.g. http://127.0.0.1:8000/v1)",
            )
        return config.model_copy(update={"base_url": default})

    def _check_use_activation(self, config: C) -> None:
        """Keyed on the resolved adapter: a *served* rerank wire (one whose profile is not HOSTED)
        must be told whether its activation runs -- ``None`` would send nothing and let the engine's
        default apply, and two engines with different defaults would then share an identity. Hosted
        profiles keep ``None`` (their scale is fixed). The shipped names are refused at the config, so the
        config-level validator stays; this catch reaches a served third-party adapter too.

        Raises:
            ConfigError: a config with a ``use_activation`` field builds on a served adapter and leaves
                it unset.
        """
        if "use_activation" not in type(config).model_fields:
            return
        if getattr(config, "use_activation", None) is not None:
            return
        hosted = getattr(self._adapter_cls, "HOSTED", False)
        if hosted:
            return
        raise ConfigError(
            f"{type(config).__name__} on the served {self._adapter_cls_name()} wire must set use_activation "
            "explicitly: None sends nothing and the engine's default applies, so two engines with different "
            "defaults would share an identity",
            hint="set use_activation: true (the engine's activation runs: the score is a probability) or "
            "use_activation: false (the raw logit is stored) -- the choice is content and enters the identity",
        )

    def _adapter_cls_name(self) -> str:
        """The wire adapter's registered name, for messages."""
        return str(getattr(self._adapter_cls, "name", self.config.api))

    def _auth_profile(self) -> AuthProfile:
        """The credential facts of this client's config and adapter, for the transport to resolve the key
        from, per replica: the config's ``api_key_env`` names the variable when it is set (required: an
        unset named variable is an error), homed at the config's own URLs -- its replicas, or the profile's
        default host when it names none -- so an injected transport aimed elsewhere never receives it; else
        the adapter profile's variables with its required-ness, homed at the profile's own default host (a
        variable set for one vendor never authenticates a request to a self-hosted engine, a gateway or a
        third party, however the transport was built). The header is always the adapter's."""
        header = getattr(self._adapter_cls, "AUTH_HEADER", None)
        named = getattr(self.config, "api_key_env", None)
        if named is not None:
            return AuthProfile(variables=(named,), required=True, header=header, homes=self.endpoint.urls)
        default = getattr(self._adapter_cls, "DEFAULT_BASE_URL", None)
        return AuthProfile(
            variables=tuple(getattr(self._adapter_cls, "API_KEY_ENV", ())),
            required=bool(getattr(self._adapter_cls, "KEY_REQUIRED", False)),
            header=header,
            homes=(default,) if default is not None else (),
        )

    def _point_sender_at_the_profile(self) -> None:
        """An injected ``Transport`` is pointed at this client's credential facts, so an auth-bearing profile
        behaves the same whichever way the transport was built: the profile carries the config's named
        variable (when the config names one) and the adapter's variables otherwise. The transport's own
        endpoint keeps precedence for its ``base_url`` and its own ``api_key_env`` (the transport's config);
        the profile fills the key in the adapter's header."""
        if isinstance(self._sender, _TRANSPORT_CLASS):
            self._sender.set_auth(self._auth_profile())

    # -- the text budget ----------------------------------------------------
    @property
    def text_budget(self) -> TextBudget | None:
        """The text budget every request of this client is fitted to, as built from its config (``None``: the
        config declares no ``max_tokens``, so nothing is fitted or cut)."""
        return self._budget

    def _resolve_budget(self) -> tuple[TextBudget | None, TextTokenizer | None]:
        """The client's text budget, from the role config's fields, with the tokenizer it names loaded once.

        ``max_tokens`` unset (a hosted profile with no declared limit): no budget, nothing is fitted and
        content is sent as given. Declared: a :class:`~rcp_ndcg.data.preprocess.TextBudget`, whose vendor
        path (no tokenizer) sends content uncut and records the documented limit.

        Raises:
            ConfigError: the config declares a tokenizer the loader cannot load (an unknown repository, a
                missing file) -- at construction, never at the first request.
        """
        max_tokens = getattr(self.config, "max_tokens", None)
        if max_tokens is None:
            return None, None
        tokenizer_name = getattr(self.config, "tokenizer", None)
        tokenizer = None
        if tokenizer_name is not None:
            from rcp_ndcg.data import load_tokenizer

            tokenizer = load_tokenizer(tokenizer_name)
        budget = TextBudget(
            tokenizer=tokenizer_name,
            max_tokens=max_tokens,
            query_max_tokens=getattr(self.config, "query_max_tokens", None),
            document_max_tokens=getattr(self.config, "document_max_tokens", None),
            template=getattr(self.config, "template", None),
            on_overflow=getattr(self.config, "on_overflow", "cut"),
            chunk=getattr(self.config, "chunk", None),
            aggregation=getattr(self.config, "aggregation", "max"),
            # A role that sends its instruction as the engine's own request field (the rerank role's
            # ``instruction: field``) still has the engine place it when the declared template does not:
            # the budget reserves its tokens, so the measured render is never smaller than the engine's.
            instruction_field=getattr(self.config, "instruction", None) == "field",
        )
        return budget, tokenizer

    def _media_policies(self) -> tuple[ImagePolicy | None, VideoPolicy | None]:
        """The effective media policies of this client's config (the one shared rule, no per-client code)."""
        return media_policies_for(self.config)

    def media_sides(self) -> frozenset[str]:
        """The sides this config allows media on: the config's ``media_sides`` field (the default: both).
        An explicitly EMPTY field allows NO side -- the default applies only when the config has no such
        field at all, so ``media_sides: []`` refuses every side exactly as the config documents."""
        sides = getattr(self.config, "media_sides", None)
        if sides is None:
            sides = ("query", "document")
        return frozenset(sides)

    def _refuse_media_off_its_side(self, side: str, contents: Sequence[Content]) -> None:
        """Media on a side the config does not allow (2b, G3) is refused naming the ``media_sides`` field,
        before the media is fetched, sized or counted.

        Raises:
            CapabilityError: an item of ``contents`` carries media and ``side`` is not in the config's
                ``media_sides``.
        """
        allowed = self.media_sides()
        if side in allowed:
            return
        sides = f"the {' and '.join(sorted(allowed))} side(s)" if allowed else "no side"
        for index, content in enumerate(contents):
            if content.has_media:
                raise CapabilityError(
                    f"item {index} of this request's {side} side carries media, but {self.config.model} takes "
                    f"media on {sides} only (media_sides)",
                    hint=f"declare {side!r} in media_sides on the role config, or drop the media from the {side}",
                )

    def _gate_media_calls(self, calls: Sequence[Call]) -> None:
        """The per-request media gates, as the judge's: each wire CALL's image and video parts against the
        config's ``max_images``/``max_videos``, refused before the call is sent. The gate runs over the
        request the engine actually sees -- the pooling wire's one media item per call, the rerank call's
        query plus that chunk's documents.

        Raises:
            CapabilityError: a call carries more images or videos than the config declares the model to
                read, or any at all when it reads none.
        """
        max_images = getattr(self.config, "max_images", 0)
        max_videos = getattr(self.config, "max_videos", 0)

        def count(value: Any) -> tuple[int, int]:
            images = videos = 0
            if isinstance(value, str):
                return 0, 0
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "image_url":
                        images += 1
                    elif key == "video_url":
                        videos += 1
                    else:
                        sub_images, sub_videos = count(item)
                        images, videos = images + sub_images, videos + sub_videos
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        images += 1
                    elif isinstance(item, dict) and item.get("type") == "video_url":
                        videos += 1
                    else:
                        sub_images, sub_videos = count(item)
                        images, videos = images + sub_images, videos + sub_videos
            return images, videos

        for call in calls:
            images, videos = count(call.json)
            for kind, count_, limit in (("images", images, max_images), ("videos", videos, max_videos)):
                if count_ and not limit:
                    raise CapabilityError(
                        f"{self.config.model} is not declared to read {kind} (max_{kind}: 0), but this "
                        f"request carries {count_}. Declare max_{kind} for a checkpoint that reads them, or "
                        "drop the media parts from the corpus.",
                        hint=f"declare max_{kind} on the role config for a checkpoint that reads them",
                    )
                if count_ > limit:
                    raise CapabilityError(
                        f"this request carries {count_} {kind} and the endpoint accepts {limit} per request "
                        f"(max_{kind}). Split the request, or raise the limit on the server and here.",
                        hint=f"lower batch_size so fewer {kind} ride one request, or raise max_{kind} (and "
                        "the server's per-request media limit with it)",
                    )

    def _prepare_request(self, contents: Sequence[Content], *, doc_ids: Sequence[str] | None = None) -> PreparedRequest:
        """The one preparation call for a request's contents: media sized exactly as the judge's, the
        request's media token counts -- and, when ``doc_ids`` names each content, the kept media recorded
        into the media census (the judge's rows, beside which the fit records its drops; the outcome is
        part of the census' dedup key, so a kept row never hides a later drop).

        The media gates are each wire call's (see :meth:`_gate_media_calls`), not this call's.
        """
        if not self._media_is_on_wire() and any(content.has_media for content in contents):
            self._refuse_media_before_preparation(contents)
        image, video = self._media_policies()
        prepared = prepare_request(contents, image, video, tokenizer=self._tokenizer)
        if doc_ids is not None and prepared.media:
            if len(doc_ids) != len(contents):
                raise DataError(
                    f"{len(doc_ids)} doc_id(s) for {len(contents)} content(s); one doc_id per content names "
                    "the census rows of a request's prepared media",
                )
            for one, doc_id in zip(prepared.per_content(), doc_ids, strict=True):
                if one.media:
                    self.media_census.record(corpus=self.ROLE, doc_id=doc_id, media=one.media, dropped=False)
        return prepared

    def _media_is_on_wire(self) -> bool:
        """Whether this client's wire carries media: the class flag (the pool and rerank wires lower media
        parts on every shape). The embed role overrides it -- its ``messages`` route only."""
        return self.MEDIA_ON_WIRE

    def _refuse_media_before_preparation(self, contents: Sequence[Content]) -> None:
        """Refuse media for a text-only role before the media is fetched, sized or counted.

        Raises:
            CapabilityError: an item carries an image or video part; the message names the media type and
                the item (the shipped embed adapters' own refusal, kept one home in front of preparation).
        """
        for index, content in enumerate(contents):
            for part in content.parts:
                if isinstance(part, ImagePart):
                    raise CapabilityError(
                        f"the {getattr(self._adapter_cls, 'name', self.config.api)} adapter takes text only, "
                        f"but item {index} carries an image part",
                        hint="embed a text rendering of the media, or declare the media fields on the role "
                        "config for a wire that reads it",
                    )
                if isinstance(part, VideoPart):
                    raise CapabilityError(
                        f"the {getattr(self._adapter_cls, 'name', self.config.api)} adapter takes text only, "
                        f"but item {index} carries a video part",
                        hint="embed a text rendering of the media; video embedding is wired with the "
                        "media-preparation mechanism",
                    )

    def _fit_media_per_item(
        self,
        request: PreparedRequest,
        *,
        shape: RequestShape,
        doc_ids: Sequence[str],
        changes: dict[str, list[ChangeMechanism]] | None = None,
    ) -> tuple[list[Content], list[int]]:
        """The media fit of a request whose wire carries ONE item per budget (the pooling items, the
        embeddings inputs): each item's media fitted on its own, sliced from the one preparation
        (:meth:`PreparedRequest.per_content`, never a second preparation), against the item shape's budget
        (:meth:`~rcp_ndcg.data.preprocess.TextBudget.shape_max_tokens`) minus that shape's fixed frame
        (:func:`~rcp_ndcg.data.preprocess.fixed_overhead`) -- the threshold the text fit measures, so the
        media fit never keeps what the text fit then refuses.

        Args:
            request: The request's one preparation (:meth:`_prepare_request`).
            shape: The items' request shape (``query`` or ``document``).
            doc_ids: One census doc_id per item, for the drop rows.

        Returns:
            ``(contents, media_tokens)``: per item, the content to send and its media token count after
            the fit. Without a budget or media, the prepared contents and zeros.
        """
        if self._budget is None or not request.media:
            return list(request.contents), [0] * len(request.contents)
        allowance = max(self._budget.shape_max_tokens(shape) - fixed_overhead(self._budget, self._tokenizer, shape), 0)
        fitted = [
            self._fit_media_for_request(
                one.contents, doc_ids=[doc_id], prepared=one, allowance=allowance, changes=changes
            )
            for one, doc_id in zip(request.per_content(), doc_ids, strict=True)
        ]
        return [pair[0][0] for pair in fitted], [pair[1] for pair in fitted]

    def _fit_media_for_request(
        self,
        contents: Sequence[Content],
        *,
        doc_ids: Sequence[str],
        prepared: PreparedRequest | None = None,
        allowance: int | None = None,
        changes: dict[str, list[ChangeMechanism]] | None = None,
    ) -> tuple[list[Content], int]:
        """The media fit for ONE wire request's contents: media never cut, drops recorded.

        The budget's ``max_tokens`` bounds one wire request (the shipped tests cut each request
        individually; ``batch_size`` is how fast, never what), so the fit runs per request -- the pool
        role's per item (its media wire is one item per call), the rerank role's per (query, document)
        pair. When the request's media alone exceed the allowance, the declared overflow policy decides --
        ``cut`` (the default): :func:`~rcp_ndcg.data.prepare.fit_media_to_budget` shrinks to the policy
        minimum, then drops whole items most expensive first, every drop recorded in the media census with
        ``dropped=True`` under the request's ``doc_ids``; ``fail``: the request is refused naming the media
        tokens and the budget; ``chunk``: refused -- media are not chunkable, a vision block is atomic.

        Args:
            contents: The wire request's contents, as prepared.
            doc_ids: One census doc_id per content, for the drop rows.
            prepared: The caller's own preparation of exactly ``contents`` (it prepared the whole request
                once, through :meth:`_prepare_request`, and slices it with :meth:`PreparedRequest.per_content`);
                ``None`` prepares here. A second preparation of already-prepared contents would re-inline
                the bytes and record census rows against ``data:`` URIs, so callers that already prepared
                pass the request in.
            allowance: The token count this wire request's media may cost: the budget minus the fixed
                template overhead and everything else the request reserves (:func:`fixed_overhead`), never
                the bare ``max_tokens`` -- in the dead zone between them the media alone fit the budget but
                the text fit would refuse the request (``the fixed template overhead ... plus the declared
                media ... already fill the budget``). ``None`` (a caller with no overhead to name): the
                budget's ``max_tokens``.
            changes: Where the fit notes, per doc_id, a ``media_resize`` (an item shrunk below its prepared
                size) and a ``media_drop`` (an item dropped) for the row's :class:`ProcessingRecord`.

        Returns:
            ``(contents, tokens)``: the contents to send (the kept media in place, possibly shrunk, drops
            removed, text untouched) and the request's media token count after the fit -- what the caller's
            :func:`~rcp_ndcg.data.preprocess.fit` call subtracts from the budget, never cutting media.
            The media gates are the wire call's (see :meth:`_gate_media_calls`), not this method's.

        Raises:
            TextBudgetExceededError: ``on_overflow: fail`` and media alone fill the allowance.
            ConfigError: ``on_overflow: chunk`` and media alone fill the allowance.
        """
        image, video = self._media_policies()
        if prepared is None:
            prepared = prepare_request(contents, image, video, tokenizer=self._tokenizer)
        media = prepared.media
        tokens = prepared.tokens.tokens
        bound = allowance if allowance is not None else (self._budget.max_tokens if self._budget else None)
        if self._budget is not None and bound is not None and tokens > bound:
            if image is None:
                raise DataError(
                    f"counted media ({tokens} tokens) imply an effective image policy, and this config declares none",
                    hint="declare image_processor with a bounded image_policy (a media token count needs a "
                    "processor family and a pixel budget), or drop the media parts",
                )
            if self._budget.on_overflow == "fail":
                raise TextBudgetExceededError(
                    f"this request's media alone cost {tokens} tokens, over the declared text budget of "
                    f"{self._budget.max_tokens}; the media are never cut, and the policy refuses to shorten "
                    "them",
                    hint="raise max_tokens, or declare a smaller image_policy (a smaller pixel budget is "
                    "fewer vision tokens)",
                )
            if self._budget.on_overflow == "chunk":
                raise ConfigError(
                    f"this request's media alone cost {tokens} tokens, over the declared text budget of "
                    f"{self._budget.max_tokens}, and media overflow cannot be chunked: a vision block is "
                    "atomic -- the engine sees a whole media item or none of it",
                    hint="declare on_overflow: cut (the media fit shrinks to the policy minimum, then drops "
                    "whole items, every drop recorded), or a smaller image_policy",
                )
            fit = fit_media_to_budget(
                media, image=image, video=video, text_budget_tokens=bound, tokenizer=self._tokenizer
            )
            # The dropped items are the original prepared items at their positions; each position's
            # doc_id is its owning content's (doc_ids expanded per media item over the PRE-fit contents,
            # in the part order the fit's decisions index). Built before apply_media_fit, whose contents
            # have the dropped media removed.
            media_doc_ids = [
                doc_id
                for doc_id, content in zip(doc_ids, prepared.contents, strict=True)
                for _ in range(sum(len(part.media_refs()) for part in content.parts))
            ]
            for position in fit.dropped_positions:
                self.media_census.record(
                    corpus=self.ROLE,
                    doc_id=media_doc_ids[position] if position < len(media_doc_ids) else self.ROLE,
                    media=[media[position]],
                    dropped=True,
                )
            if changes is not None:
                for position, decision in enumerate(fit.decisions):
                    owner = media_doc_ids[position] if position < len(media_doc_ids) else self.ROLE
                    if decision is None:
                        changes.setdefault(owner, []).append("media_drop")
                    elif decision != media[position].sent:
                        changes.setdefault(owner, []).append("media_resize")
            contents = list(apply_media_fit(list(prepared.contents), fit))
        else:
            contents = list(prepared.contents)
        counted = self._media_counts_of(contents)
        return contents, sum(count.tokens for count in counted)

    def _media_counts_of(self, contents: Sequence[Content]) -> list[MediaTokenCount]:
        """The exact media token count of each content, as the engine adds it to the prompt.

        The client's own loaded tokenizer is passed so a ``qwen3_vl`` container under the engine's fps rule
        counts its timestamp lines exactly, not at the family's bound: the count the client gates and cuts
        with is the count the engine reports."""
        image, video = self._media_policies()
        return [
            content_media_tokens(
                content, image if image is not None else ImagePolicy.native(), video, tokenizer=self._tokenizer
            )
            for content in contents
        ]

    def _fit(
        self,
        inputs: Sequence[str] | Sequence[tuple[str, str]],
        shape: RequestShape,
        *,
        media_tokens: Sequence[int] | None = None,
        instruction: str | None = None,
        record: bool = True,
        ids: Sequence[str] | None = None,
    ) -> FitResult:
        """One :func:`~rcp_ndcg.data.preprocess.fit` call for this client's budget: the shared mechanism the
        brief wires into every ``_prepare``. ``corpus`` is the client's role name; ids are positional (the
        caller's ``ids`` when given -- the census rows then name the caller's own inputs, so an omitted
        document's position is never taken by a later one). The ``instruction`` reserves the instruction's
        tokens in the fixed overhead where the declared template frames it (the reranker's ``instruction:
        field`` and ``system`` modes -- the instruction is then engine-rendered into the frame, never inside
        the cut spans); ``record=False`` makes a probe call, whose spans are decided without recording census
        rows."""
        budget, tokenizer = self._budget, self._tokenizer
        assert budget is not None  # callers only fit when a budget is declared
        return fit(
            inputs,
            shape,
            budget,
            tokenizer,
            ids=list(ids) if ids is not None else [str(index) for index in range(len(inputs))],
            instruction=instruction,
            media_tokens=media_tokens,
            corpus=self.ROLE,
            census=self.census if record else None,
        )

    def _record_processing(
        self,
        shape: RequestShape,
        *,
        cuts: Sequence[Any] = (),
        changes: dict[str, list[ChangeMechanism]] | None = None,
    ) -> None:
        """Append the :class:`~rcp_ndcg.data.preprocess.ProcessingRecord` of every row one preparation changed
        to :attr:`processing`: its text cuts (the census rows the fit or the settlement wrote) and the other
        mechanisms noted per input id (:func:`~rcp_ndcg.data.preprocess.processing_records`)."""
        self.processing.extend(processing_records(self.ROLE, shape, cuts=cuts, changes=changes))

    @staticmethod
    def _with_text(content: Content, text: str) -> Content:
        """The content with its text parts replaced by ``text``, in place: the text stands where the content's
        first text part stood (its other text parts were joined into it), media parts untouched -- the order
        of an item's parts is information the model reads (a media-first item stays media-first). A content
        without a text part gets the text first. An empty text on a media item is dropped."""
        keep_text = bool(text) or not content.has_media
        if not any(isinstance(part, TextPart) for part in content.parts):
            parts: list[Any] = [TextPart(text=text)] if keep_text else []
            return Content.from_parts(parts + list(content.parts))
        parts = []
        placed = False
        for part in content.parts:
            if not isinstance(part, TextPart):
                parts.append(part)
            elif not placed:
                placed = True
                if keep_text:
                    parts.append(TextPart(text=text))
        return Content.from_parts(parts)

    async def check_engine_media(self) -> None:
        """The startup media probe: one prepared probe image beside its no-media baseline, the engine's
        media DELTA compared with the counted media tokens (never silent).

        Runs when the role declares an ``image_processor``: the probe sends one prepared image through the
        adapter -- and the same request without its media -- and takes the DELTA of the engine's two
        ``usage.prompt_tokens`` reports. The delta cancels everything the two requests share (a server-side
        chat template, the route's special tokens, the probe's text), so it reports the media block alone;
        :func:`~rcp_ndcg.data.resolution.engine_media_check` compares it with the counted media tokens of
        the prepared probe. A mismatch is a typed :class:`~rcp_ndcg.errors.ProviderError` whose message
        names ``image_processor`` and the server's media flags; a reply without usage is recorded in the
        media census as ``not_checked`` -- the check never passes silently. A wire that offers no no-media
        form of its probe request is recorded ``not_checked`` too.

        Raises:
            CapabilityError: the engine refused a probe request, or the config's own media gate refused
                the probe (the request's media against ``max_images``/``max_videos``).
            ProviderError: the engine's media delta disagrees with the counted media tokens.
        """
        image_policy, video_policy = self._media_policies()
        if image_policy is None or getattr(self.config, "image_processor", None) is None:
            return  # a text-only role: nothing to check (the gate refuses media it cannot count)
        from tempfile import NamedTemporaryFile

        from PIL import Image as PILImage

        from rcp_ndcg.data.resolution import engine_media_check
        from rcp_ndcg.errors import ProviderError

        probe_path = ""
        try:
            with NamedTemporaryFile(suffix=".png", delete=False) as probe_file:
                PILImage.new("RGB", (224, 224), (120, 120, 120)).save(probe_file, format="PNG")
                probe_path = probe_file.name
            probe = Content.from_image(f"file://{probe_path}")
            prepared = self._prepare_request([probe])
            probe_content = prepared.contents[0]
            calls = list(self._probe_calls(probe_content))
            self._gate_media_calls(calls)  # the probe's own request follows the config's limits too
            baseline_calls = self._probe_baseline_calls(probe_content)
            if baseline_calls is None:
                self.media_census.record(
                    corpus=self.ROLE,
                    doc_id="engine_media_check:not_checked",
                    media=[prepared.media[0]],
                    dropped=False,
                )
                get_logger(__name__).warning(
                    "engine media check: the %s adapter offers no no-media baseline for its probe request, "
                    "so the engine's media delta cannot be taken (recorded as not_checked); media counting "
                    "proceeds on the declared %s policy",
                    getattr(self._adapter_cls, "name", self.config.api),
                    getattr(self.config, "image_processor", None),
                )
                return
            # The counted number is the prepared probe's media block alone (the engine's delta is exactly
            # that: both probe requests carry the same text and whatever template the wire renders, so
            # those cancel in the difference). The prepared reference carries its sizes, so the count is
            # exact -- no bound is involved.
            counted = self._media_counts_of([probe_content])[0].tokens
            replies = await self._sender.send([*calls, *baseline_calls])
            with_media = self._probe_usage(replies[0])
            without_media = self._probe_usage(replies[-1])
            if (
                with_media is None
                or without_media is None
                or with_media.input_tokens is None
                or without_media.input_tokens is None
            ):
                self.media_census.record(
                    corpus=self.ROLE,
                    doc_id="engine_media_check:not_checked",
                    media=[prepared.media[0]],
                    dropped=False,
                )
                get_logger(__name__).warning(
                    "engine media check: the %s reply reported no usage, so the engine's media delta cannot "
                    "be taken (recorded as not_checked); media counting proceeds on the declared %s policy",
                    getattr(self._adapter_cls, "name", self.config.api),
                    getattr(self.config, "image_processor", None),
                )
                return
            mismatch = engine_media_check(with_media.input_tokens - without_media.input_tokens, counted)
            if mismatch is not None:
                raise ProviderError(
                    mismatch.message,
                    hint="verify the served engine's media handling against the declared image_processor and "
                    "image_policy (no engine-side media flags that resize again), or correct the declaration",
                )
        finally:
            if probe_path:
                Path(probe_path).unlink(missing_ok=True)

    def _probe_baseline_calls(self, content: Content) -> Sequence[Call] | None:
        """The probe request without its media -- the baseline the engine's media DELTA is taken against
        (the same wire shape, the media parts gone). ``None``: this role's adapter offers no baseline form,
        and the media check is recorded ``not_checked``."""
        return None

    def _probe_calls(self, content: Content) -> Sequence[Call]:
        """The calls one prepared probe item is sent as (the role's wire)."""
        raise NotImplementedError

    def _probe_usage(self, reply: Reply) -> TokenCount | None:
        """The reply's prompt-token report (``None``: the engine reported none)."""
        raise NotImplementedError

    def _apply_empty_documents(
        self,
        contents: Sequence[Content],
        *,
        changes: dict[str, list[ChangeMechanism]] | None = None,
        prefix: str = "",
    ) -> tuple[list[Content], list[int]]:
        """The request's empty documents as the config's ``empty_doc`` policy sends them.

        An empty document is one with no text and no media left (a document whose every media item the
        budget dropped is one, exactly like an empty text document), decided on the content -- before the
        template frames it, and under the side's prompt ``prefix`` (the content then carries the prompt alone):
        a framed or prompted empty document is still empty. ``send`` (the default) sends the empty string as
        today; ``send_text`` sends the configured placeholder text (after the prompt, framed like any
        content); ``omit_zero`` never sends the item -- it scores 0.0 -- and the caller places the missing
        result (a zero vector, an empty slice, a 0.0 score) at its position. A request is never sent empty:
        the omitted items leave it, and the caller returns without one when nothing remains.

        ``changes`` (when given) notes ``empty_doc`` under each substituted or omitted input's position, for its
        :class:`~rcp_ndcg.data.preprocess.ProcessingRecord`.

        Returns:
            ``(kept, omitted)``: the contents to send and the indices of the omitted inputs (``omit_zero``).
        """
        policy = getattr(self.config, "empty_doc", "send")
        kept: list[Content] = []
        omitted: list[int] = []
        for index, content in enumerate(contents):
            if content.text != prefix or content.has_media:
                kept.append(content)
                continue
            if policy := getattr(self.config, "empty_doc", "send"):
                if policy == "omit_zero":
                    omitted.append(index)
                    if changes is not None:
                        changes.setdefault(str(index), []).append("empty_doc")
                    continue
                if policy == "send_text":
                    placeholder = getattr(self.config, "empty_doc_text", None) or ""
                    kept.append(self._with_text(content, prefix + placeholder))
                    if changes is not None:
                        changes.setdefault(str(index), []).append("empty_doc")
                    continue
            kept.append(content)  # "send": the empty string goes out, as today
        return kept, omitted

    # -- the one preparation pipeline -----------------------------------------
    # The stages, in order, are the module-level STAGES tuple; these methods are their one homes. A role
    # supplies the hooks (the prompt prefix, the lowering); the runners compose them and never re-order.

    def _side_prefix(self, side: str) -> str:
        """The prompt prefix of one side of the retrieval pair, as the role config declares it ("": the
        role declares no per-side prompt -- a reranker's pair template is the frame's home)."""
        if side == "query":
            return str(getattr(self.config, "query_prompt", "") or "")
        if side == "document":
            return str(getattr(self.config, "doc_prompt", "") or "")
        return ""

    def _stage_normalise(
        self,
        contents: Sequence[Content],
        *,
        side: str,
        prompt: str,
        instruction: str | None = None,
    ) -> list[Content]:
        """The ``normalise`` stage (role hook): the content as the role declares it before anything is
        measured -- the side's prompt prefix, and the task instruction where the role places it.

        The default prefixes ``prompt``, then places the task instruction where the role declares it: a
        template ``instruction`` span carries it (the fit renders it), else the config's ``instruction``
        mode decides -- ``fold`` prefixes the QUERY side with the generic default
        (``Task: <instruction>\\nQuery: <text>``), ``none`` sends none. An endpoint that declares no mode
        (``instruction: None``, the default) refuses a request that carries one, naming the two choices: a
        recipe that never chose a policy must not have its text changed silently. A document side is never
        folded -- the generic default is the query's frame -- and a document-side task instruction without a
        template span is refused too, never silently dropped. A role with its own normalisation overrides
        (the rerank places the instruction for its pairs).

        Args:
            contents: The inputs as given (already materialised to content parts).
            side: Which side of the retrieval pair the batch is (``query`` or ``document``).
            prompt: The side's prompt prefix (:meth:`_side_prefix`).
            instruction: The side's task instruction, when the caller has one.

        Returns:
            One content per input, in order.

        Raises:
            ConfigError: the request carries a task instruction and the endpoint declares no ``instruction``
                mode (the hint names ``fold``/``none``); a document-side task instruction and no template
                span to place it.
        """
        prepared = [content.with_text_prefix(prompt) for content in contents]
        if not instruction:
            return prepared
        mode = getattr(self.config, "instruction", None)
        if mode is None:
            raise ConfigError(
                f"this request carries a task instruction, and {type(self.config).__name__} declares no "
                "instruction policy: the instruction would either change the text of a recipe that never "
                "chose it or vanish",
                hint="declare instruction: fold (the generic Task: <instruction>\\nQuery: <text> prefix) or "
                "instruction: none (the model takes no instruction) on the role config",
            )
        if mode == "none":
            return prepared
        shape: RequestShape = "query" if side == "query" else "document"
        if self._template_places_the_instruction(shape):
            return prepared
        if side != "query":
            raise ConfigError(
                f"this request's {side} side carries a task instruction and the config's template declares no "
                "instruction span for it: the generic default frames the QUERY side only",
                hint="declare an {content: instruction} span in the template's document shape, or declare "
                "instruction: none on the role config (the model takes no instruction), so nothing is dropped",
            )
        return [self._fold_instruction(content, instruction) for content in prepared]

    def _template_places_the_instruction(self, shape: RequestShape) -> bool:
        """Whether the declared template has an ``instruction`` span for ``shape``: the template places the
        instruction itself (the fit fills the span), so the client does not also fold it.

        The rerank role's span is rendered by the ENGINE (the wire carries the cut spans and the engine
        renders its own chat template), so a rerank recipe with a span reads the instruction from the
        request's ``instruction`` field -- :meth:`_sends_the_instruction_field` sends it there. The predicate
        is the template's own (:meth:`~rcp_ndcg.data.templates.TemplateSpec.places`), one home for the
        rerank adapter's wire-fact check too.
        """
        template = getattr(self.config, "template", None)
        return template is not None and template.places(shape, "instruction")

    @staticmethod
    def _fold_instruction(content: Content, instruction: str) -> Content:
        """The generic task-instruction frame (the one home is the core record's own
        :meth:`~rcp_ndcg_core._records.Query.format_content`): ``Task: <instruction>\\nQuery: <text>``."""
        return Query(query_id="", query=content.text, content=content).format_content(task_instruction=instruction)

    def _stage_media(
        self,
        contents: Sequence[Content],
        *,
        shape: RequestShape,
        positions: Sequence[int],
        changes: dict[str, list[ChangeMechanism]],
    ) -> tuple[list[Content], list[int]]:
        """The ``media`` stage (shared default): one preparation of the request's kept contents (nothing is
        fetched for an item the empty stage omitted), then the media fit per wire item, every drop recorded.

        Args:
            contents: The kept contents, in their original order (``positions`` names each one's index in
                the caller's inputs, for the census rows and the records).
            shape: The request shape the items are fitted as.
            positions: Each content's original input index.
            changes: Where the fit notes, per input id, a ``media_resize`` or ``media_drop``.

        Returns:
            ``(contents, media_tokens)``: the fitted contents (in the given order) and, PER GIVEN CONTENT,
            its media token count after the fit. A wire that carries no media returns the contents and
            zeros (:meth:`_media_is_on_wire`).
        """
        if not self._media_is_on_wire() or not any(content.has_media for content in contents):
            return list(contents), [0] * len(contents)
        doc_ids = [str(position) for position in positions]
        request = self._prepare_request(list(contents), doc_ids=doc_ids)
        return self._fit_media_per_item(request, shape=shape, doc_ids=doc_ids, changes=changes)

    def _stage_budget(
        self,
        contents: Sequence[Content],
        *,
        shape: RequestShape,
        media_tokens: Sequence[int],
        positions: Sequence[int],
        instruction: str | None = None,
        changes: dict[str, list[ChangeMechanism]],
    ) -> tuple[list[Content], tuple[Any, ...], FitResult | None]:
        """The ``render`` + ``budget`` stages (shared default): one call of the shared fit -- the fixed
        frame measured and reserved, the content spans cut to what remains, the template re-attached, the
        spans verified against the assembled render, every cut recorded. A role whose budget is pair-shaped
        (the reranker's query share, document cap and pair budget around one shared query) overrides it.

        Returns:
            ``(kept, cuts, result)``: the contents to send (their text spans replaced below, by the lower
            stage), the cut rows the fit recorded, and the fit result (``None`` without a budget: nothing
            was fitted or cut).
        """
        cuts: tuple[Any, ...] = ()
        if self._budget is None or not contents:
            return list(contents), cuts, None
        result = self._fit(
            [content.text for content in contents],
            shape,
            media_tokens=list(media_tokens),
            instruction=instruction,
            ids=[str(position) for position in positions],
        )
        return list(contents), result.cuts, result

    def _stage_lower(
        self,
        contents: Sequence[Content],
        *,
        result: FitResult | None,
        shape: RequestShape,
    ) -> tuple[list[Content], tuple[tuple[int, ...], ...]]:
        """The ``lower`` stage (role hook): the wire form of the fitted contents -- which route takes the
        rendered strings and which takes the content spans, and what token ids the role tracks.

        Args:
            contents: The kept contents (their text still the prompted, fitted content text).
            result: The fit result (its ``texts`` are the rendered strings; ``None`` without a budget).
            shape: The request shape the batch was fitted as.

        Returns:
            ``(items, token_ids)``: the contents to send and the tracked token ids (empty when the role
            tracks none).
        """
        texts = [content.text for content in contents] if result is None else list(result.texts)
        return [self._with_text(content, str(text)) for content, text in zip(contents, texts, strict=True)], ()

    def _prepare_rows(
        self,
        contents: Sequence[Content],
        *,
        side: str,
        shape: RequestShape,
        instruction: str | None = None,
    ) -> PreparedItems:
        """The pipeline: every input row of one side through :data:`STAGES`, in order, and the per-row
        :class:`~rcp_ndcg.data.text_budget.ProcessingRecord` of everything it changed as the pipeline's one
        output. The embed and the pool role run this runner; the rerank's pair-shaped preparation composes
        the same stage methods in the same order (:meth:`~rcp_ndcg.inference.clients.rerank.RerankClient._fit_pair`).

        Args:
            contents: The side's contents as given.
            side: Which side of the retrieval pair the batch is (``query`` or ``document``).
            shape: The request shape the batch is fitted as (follows the side).
            instruction: The task instruction, when the caller has one.

        Returns:
            The prepared items, each with its original position, the positions ``empty_doc: omit_zero``
            never sends, and the role's tracked token ids.
        """
        prompt = self._side_prefix(side)
        # normalise (role): the side's prompt prefix; media on a forbidden side is refused here, before
        # anything is fetched, sized or counted.
        prepared = self._stage_normalise(contents, side=side, prompt=prompt, instruction=instruction)
        self._refuse_media_off_its_side(side, prepared)
        changes: dict[str, list[ChangeMechanism]] = {}  # per input id, for the rows' processing records
        # empty: decided on the content AS GIVEN -- before the template frames it (a framed empty document
        # is a non-empty turn and the policy would never fire) and before any media is fetched. omit_zero
        # never sends an empty item; the caller places the missing result at its position.
        kept, omitted = self._apply_empty_documents(prepared, changes=changes, prefix=prompt)
        # media: one preparation of the kept items, then the media fit. A document whose every media item
        # the budget dropped is empty -- exactly like an empty text document -- so the same empty policy
        # applies to the fitted contents (the media stage's contract; the policy never sees a framed
        # non-empty turn where the sent content is empty).
        kept_positions = [index for index in range(len(prepared)) if index not in set(omitted)]
        fitted, media_tokens = self._stage_media(
            kept,
            shape=shape,
            positions=kept_positions,
            changes=changes,
        )
        # The media stage's output is what is sent (the prepared media in place); the empty policy fires
        # again only when the media fit emptied something: a document whose every media item was dropped is
        # empty, exactly like an empty text document, and the policy decides on the content as it will be
        # sent. The re-entry's indices are the KEPT list's -- mapped back to the caller's input ids before
        # they meet the first pass's omissions or name a record (the two passes decide on different lists;
        # the record ids and the positions are the caller's coordinates, always).
        kept = fitted
        dropped_empty: list[int] = []
        if any("media_drop" in applied for applied in changes.values()):
            re_changes: dict[str, list[ChangeMechanism]] = {}
            kept, dropped_empty_kept = self._apply_empty_documents(fitted, changes=re_changes, prefix=prompt)
            for kept_id, applied in re_changes.items():
                original = str(kept_positions[int(kept_id)])
                merged = changes.setdefault(original, [])
                merged.extend(mechanism for mechanism in applied if mechanism not in merged)
            dropped_empty = [kept_positions[index] for index in dropped_empty_kept]
        dropped = sorted(set(omitted) | set(dropped_empty))
        positions = [index for index in range(len(prepared)) if index not in set(dropped)]
        # render + budget (shared): the fixed frame reserved, the content spans cut to what remains, the
        # template re-attached, every cut recorded.
        kept, cuts, result = self._stage_budget(
            kept,
            shape=shape,
            media_tokens=[media_tokens[kept_positions.index(position)] for position in positions]
            if kept_positions
            else [],
            positions=positions,
            instruction=instruction,
            changes=changes,
        )
        # lower (role): the wire form -- which route takes the rendered strings and which the content
        # spans -- and the role's tracked token ids.
        items, token_ids = self._stage_lower(kept, result=result, shape=shape)
        # The record is the pipeline's one output: emitted here, once per preparation, for every change
        # and only for a change (the census rows the fit or the settlement wrote, and the other mechanisms).
        self._record_processing(shape, cuts=cuts, changes=changes)
        return PreparedItems(
            items=tuple(items), positions=tuple(positions), omitted=tuple(dropped), token_ids=token_ids
        )

    # -- batching -------------------------------------------------------------
    def _request_size(self, batch_size: int | None) -> int:
        """The request size of one call: ``batch_size``, else the config's; below 1, or above a HOSTED
        profile's published cap (:func:`_check_batch_size`), is refused (typed)."""
        size = getattr(self.config, "batch_size", None) if batch_size is None else batch_size
        if not isinstance(size, int) or size < 1:
            raise ConfigError(f"batch_size must be at least 1, got {size}")
        _check_batch_size(self._adapter_cls, size, noun=self.BATCH_NOUN)
        return size

    # -- usage (the judge's per-reply rule) ------------------------------------
    @property
    def usage(self) -> Usage:
        """The sender's accounting (requests, failed requests, tokens): the role client folds every reply's
        token report into it at send time (:meth:`_record_usage`, the rule the judge's replies go through
        too) -- an embed/pool/rerank run records the tokens its replies reported, never zeros."""
        return self._sender.usage

    def _record_usage(self, replies: Sequence[Any]) -> None:
        """Fold each reply's token report into the transport's usage, once per reply -- the one rule every
        role client (the judge's included) sends its replies through: the adapter reads the tokens (the API's
        field names are its business), the transport accumulates them. A sender without accounting (a bare
        fake) adds nothing."""
        add_usage = getattr(self._sender, "add_usage", None)
        if add_usage is None:
            return
        for reply in replies:
            add_usage(self._adapter.usage(reply))

    # -- the fan-out, one rule ------------------------------------------------
    @staticmethod
    async def gather(tasks: list[Coroutine[Any, Any, T]]) -> list[T]:
        """Run the request coroutines concurrently, a failure cancelling the siblings: one
        :class:`asyncio.TaskGroup` per fan-out, so no request outlives its failed sibling, no task is left
        pending, and no callback lands after the failure.

        A group carrying exactly one exception is re-raised as that exception, so callers -- and the typed
        errors -- see the request's own failure, not a wrapper (a group of several failures is raised as the
        group, which names them all).

        Returns:
            The results, in the order the coroutines were given.
        """
        try:
            async with asyncio.TaskGroup() as group:
                created = [group.create_task(task) for task in tasks]
        except BaseExceptionGroup as group:
            if len(group.exceptions) == 1:
                raise group.exceptions[0] from group.__cause__
            raise
        return [task.result() for task in created]

    # -- the lifecycle --------------------------------------------------------
    def _run(self, coroutine: Coroutine[Any, Any, T]) -> T:
        """The sync bridge: the sender's ``run`` (a transport runs its own loop and pool; an injected sender
        its own), guaranteed present by the constructor."""
        return self._sender.run(coroutine)

    def close(self) -> None:
        """Close the sender, when it closes: the transport the client built, or an injected one that defines
        ``close``; safe to call twice."""
        close = getattr(self._sender, "close", None)
        if callable(close):
            close()

    async def aclose(self) -> None:
        """Close the sender asynchronously: awaited on the pool's own loop. A sender with only a sync
        ``close`` falls back to it; safe to call twice."""
        aclose = getattr(self._sender, "aclose", None)
        if aclose is None:
            self.close()
            return
        result = aclose()
        if result is not None and hasattr(result, "__await__"):
            await result

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()


__all__ = ["RoleClient"]
