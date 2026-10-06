"""The role-client base: the one class every role client derives from (R5, R14, R15).

Three role clients built in parallel answered the same questions three ways -- how credentials are
resolved, how a synchronous caller reaches asyncio, how a client closes, where a hosted profile's base
URL comes from. This base answers each once:

* **the adapter lookup within the client's role** -- :func:`~rcp_ndcg.inference.adapters.base.get_adapter`
  with :attr:`ROLE`, so a config whose ``api`` names another role's adapter is refused here, once;
* **the hosted profile's default base URL** -- the config's ``base_url``, else the adapter's
  ``DEFAULT_BASE_URL``; a config with neither is refused;
* **the transport** -- built from the resolved config (with the adapter profile's
  :class:`~rcp_ndcg.inference.transport.AuthProfile`, so the key resolution lives in the transport, R6)
  unless the caller supplies a :class:`~rcp_ndcg.inference.transport.Sender`;
* **the sync bridge, one rule** -- the sender's ``run``: a :class:`~rcp_ndcg.inference.transport.Transport`
  always has it, and any other sender must provide ``run`` (or the constructor raises a ``ConfigError``).
  Chosen over an ``asyncio.run`` fallback: a fresh loop per call would give a non-transport sender no pool
  reuse and would fail inside a running loop (a notebook), where ``Transport.run`` already knows to switch
  to a background thread;
* **the lifecycle** -- ``close()`` synchronous, ``async aclose()`` awaiting (R15), and both context
  managers (``with`` and ``async with``);
* **the fan-out, one rule** (R7) -- :meth:`RoleClient.gather`: :class:`asyncio.TaskGroup` semantics, so a
  failing request cancels its siblings and leaves no task pending;
* **the text budget and the media** (item 4) -- :class:`~rcp_ndcg.data.preprocess.TextBudget` resolved
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

from rcp_ndcg_core.content import Content, ImagePart, TextPart, VideoPart

from rcp_ndcg.data.prepare import (
    MediaCensus,
    PreparedRequest,
    apply_media_fit,
    fit_media_to_budget,
    media_policies_for,
    prepare_request,
)
from rcp_ndcg.data.preprocess import (
    FitResult,
    TextBudget,
    TextBudgetExceededError,
    TextTruncationCensus,
    fit,
)
from rcp_ndcg.data.resolution import ImagePolicy, MediaTokenCount, VideoPolicy, content_media_tokens
from rcp_ndcg.data.templates import RequestShape
from rcp_ndcg.errors import CapabilityError, ConfigError
from rcp_ndcg.inference.adapters import embeddings as _shipped_adapters  # noqa: F401  # registers them
from rcp_ndcg.inference.adapters.base import AdapterRole, get_adapter
from rcp_ndcg.inference.adapters.chat import media_counts
from rcp_ndcg.inference.endpoint import Endpoint
from rcp_ndcg.inference.transport import AuthProfile, Sender, Transport
from rcp_ndcg.inference.types import Call, Reply, TokenCount
from rcp_ndcg.support.logging import get_logger

_TRANSPORT_CLASS = Transport
"""The transport class, captured at import: a test may patch the module attribute (an offline wire), and
the base's own transport checks must not break when it does."""

if TYPE_CHECKING:
    from rcp_ndcg.data.tokenizer import TextTokenizer


"""The transport class, captured at import: a test may patch the module attribute (an offline wire), and
the base's own transport checks must not break when it does."""

T = TypeVar("T")
"""The result type of a coroutine the fan-out runs."""

C = TypeVar("C", bound=Endpoint)
"""The role config a client serves (an :class:`~rcp_ndcg.inference.endpoint.Endpoint` subclass)."""


class PreparedItems(NamedTuple):
    """What a role client's ``_prepare`` decided to send, aligned back to the inputs.

    Attributes:
        items: The contents to send, in fit order (the media prepared, the text fitted).
        positions: For each item, its original input index (``positions[i]`` is where ``items[i]`` came
            from); every input not in :attr:`omitted` appears exactly once.
        omitted: The input indices ``empty_doc: omit_zero`` never sends -- ascending; the caller places the
            missing result (a zero vector, an empty slice, a 0.0 score) at each.
    """

    items: tuple[Content, ...]
    positions: tuple[int, ...]
    omitted: tuple[int, ...]


class RoleClient[C: Endpoint]:
    """The shared ground of every role client; subclasses set :attr:`ROLE` and own their content decisions.

    Attributes:
        config: The role config as it was given (a hosted profile's ``base_url`` stays ``None``).
        endpoint: The config the transport and the adapter see: the profile's public base URL filled in
            when the config set none.
        census: The :class:`~rcp_ndcg.data.preprocess.TextTruncationCensus` every text-budget cut is
            recorded into -- the one passed in, or a fresh in-memory one.
    """

    #: The adapter role this client speaks: the registry namespace its config's ``api`` resolves in, and
    #: the engine role of the configs it serves (F7 keeps the two vocabularies mapped in one place).
    ROLE: ClassVar[AdapterRole]

    #: Whether this role's wires carry media. ``False`` (the embed role: its adapters are text-only)
    #: refuses a media-carrying request before the media is fetched or counted, with the adapter's own
    #: typed refusal.
    MEDIA_ON_WIRE: ClassVar[bool] = True

    #: The role config, as it was given (a hosted profile's ``base_url`` stays ``None``).
    config: C

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
        api = getattr(config, "api", None)
        if not isinstance(api, str) or not api:
            raise ConfigError(
                f"{type(config).__name__} declares no wire adapter (api)", hint="api names the wire adapter"
            )
        self._adapter_cls: type[Any] = get_adapter(api, role=self.ROLE)
        self.config = config
        self.endpoint = self._resolved_endpoint(config, self._adapter_cls)
        self.census = census if census is not None else TextTruncationCensus()
        self.media_census = media_census if media_census is not None else MediaCensus()
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
        ``None``. The credentials are the transport's business now (R6): ``api_key_env`` is never cleared."""
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
        """F10, keyed on the resolved adapter: a *served* rerank wire (one whose profile is not HOSTED)
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
        from (R6): the config's ``api_key_env`` names the variable when it is set (an unset named variable is
        an error, whatever the profile's rule), else the adapter profile's variables with its required-ness;
        the header is always the adapter's."""
        named = getattr(self.config, "api_key_env", None)
        if named is not None:
            return AuthProfile(
                variables=(named,),
                required=True,
                header=getattr(self._adapter_cls, "AUTH_HEADER", None),
            )
        return AuthProfile(
            variables=tuple(getattr(self._adapter_cls, "API_KEY_ENV", ())),
            required=bool(getattr(self._adapter_cls, "KEY_REQUIRED", False)),
            header=getattr(self._adapter_cls, "AUTH_HEADER", None),
        )

    def _point_sender_at_the_profile(self) -> None:
        """An injected ``Transport`` is pointed at this client's credential facts, so an auth-bearing profile
        behaves the same whichever way the transport was built: the profile carries the config's named
        variable (when the config names one) and the adapter's variables otherwise. The transport's own
        endpoint keeps precedence for its ``base_url`` and its own ``api_key_env`` (the transport's config);
        the profile fills the key in the adapter's header."""
        if isinstance(self._sender, _TRANSPORT_CLASS):
            self._sender.set_auth(self._auth_profile())

    # -- the text budget (item 4) -------------------------------------------
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
            template=getattr(self.config, "template", None),
            on_overflow=getattr(self.config, "on_overflow", "cut"),
            chunk=getattr(self.config, "chunk", None),
            aggregation=getattr(self.config, "aggregation", "max"),
        )
        return budget, tokenizer

    def _media_policies(self) -> tuple[ImagePolicy | None, VideoPolicy | None]:
        """The effective media policies of this client's config (the one shared rule, no per-client code)."""
        return media_policies_for(self.config)

    def _prepare_request(self, contents: Sequence[Content]) -> PreparedRequest:
        """The one preparation call for a request's contents: media sized exactly as the judge's, the
        request's media token counts, and the per-request media gates before anything is sent.

        The gates run over one wire request's contents (the unit the judge's own per-request gate uses: the
        pool adapter sends one media item per call, the rerank pair per pair), so a limit the server
        enforces per prompt is checked against what one prompt will carry.

        Raises:
            CapabilityError: the wire request carries more images or videos than the config's
                ``max_images``/``max_videos`` allow (as the judge's per-request gate).
        """
        if not self.MEDIA_ON_WIRE and any(content.has_media for content in contents):
            self._refuse_media_before_preparation(contents)
        image, video = self._media_policies()
        prepared = prepare_request(contents, image, video)
        counts = media_counts(Content.from_parts([part for content in prepared.contents for part in content.parts]))
        for kind, count, limit in (
            ("images", counts.images, getattr(self.config, "max_images", 0)),
            ("videos", counts.videos, getattr(self.config, "max_videos", 0)),
        ):
            if count and not limit:
                raise CapabilityError(
                    f"{self.config.model} is not declared to read {kind} (max_{kind}: 0), but this request "
                    f"carries {count}. Declare max_{kind} for a checkpoint that reads them, or drop the "
                    "media parts from the corpus."
                )
            if count > limit:
                raise CapabilityError(
                    f"this request carries {count} {kind} and the endpoint accepts {limit} per request "
                    f"(max_{kind}). Split the request, or raise the limit on the server and here."
                )
        return prepared

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

    def _media_slices(self, prepared: PreparedRequest) -> list[list[Any]]:
        """The request's prepared media, sliced per content (the same part order ``prepare_request``
        flattened them in)."""
        slices: list[list[Any]] = []
        cursor = 0
        for content in prepared.contents:
            count = sum(len(part.media_refs()) for part in content.parts)
            slices.append(prepared.media[cursor : cursor + count])
            cursor += count
        return slices

    def _fit_media_for_request(
        self, contents: Sequence[Content], *, doc_ids: Sequence[str]
    ) -> tuple[list[Content], int]:
        """The media fit for ONE wire request's contents: media never cut, drops recorded.

        The budget's ``max_tokens`` bounds one wire request (the shipped tests cut each request
        individually; ``batch_size`` is how fast, never what), so the fit runs per request -- the pool
        role's per item (its media wire is one item per call), the rerank role's per (query, document)
        pair. When the request's media alone exceed it, the declared overflow policy decides -- ``cut``
        (the default): :func:`~rcp_ndcg.data.prepare.fit_media_to_budget` shrinks to the policy minimum,
        then drops whole items most expensive first, every drop recorded in the media census with
        ``dropped=True`` under the request's ``doc_ids``; ``fail``: the request is refused naming the media
        tokens and the budget; ``chunk``: refused -- media are not chunkable, a vision block is atomic.

        Returns:
            ``(contents, tokens)``: the contents to send (the kept media in place, possibly shrunk, drops
            removed, text untouched) and the request's media token count after the fit -- what the caller's
            :func:`~rcp_ndcg.data.preprocess.fit` call subtracts from the budget, never cutting media.

        Raises:
            TextBudgetExceededError: ``on_overflow: fail`` and media alone fill the budget.
            ConfigError: ``on_overflow: chunk`` and media alone fill the budget.
        """
        image, video = self._media_policies()
        prepared = prepare_request(contents, image, video)
        media = prepared.media
        tokens = prepared.tokens.tokens
        if self._budget is not None and tokens > self._budget.max_tokens:
            assert image is not None, (
                "counted media imply an effective image policy (content_media_tokens refused one without a family)"
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
            fit = fit_media_to_budget(media, image=image, video=video, text_budget_tokens=self._budget.max_tokens)
            contents = list(apply_media_fit(contents, media, fit))
            media = fit.media
            tokens = fit.tokens
            for item in fit.dropped:
                owner = next(
                    (
                        doc_id
                        for doc_id, slice_ in zip(doc_ids, self._media_slices(prepared), strict=False)
                        if item in slice_
                    ),
                    doc_ids[0] if doc_ids else self.ROLE,
                )
                self.media_census.record(corpus=self.ROLE, doc_id=owner, media=[item], dropped=True)
        else:
            contents = list(contents)
        counted = self._media_counts_of(contents)
        return contents, sum(count.tokens for count in counted)

    def _media_counts_of(self, contents: Sequence[Content]) -> list[MediaTokenCount]:
        """The exact media token count of each content, as the engine adds it to the prompt."""
        image, video = self._media_policies()
        return [
            content_media_tokens(content, image if image is not None else ImagePolicy.native(), video)
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
    ) -> FitResult:
        """One :func:`~rcp_ndcg.data.preprocess.fit` call for this client's budget: the shared mechanism the
        brief wires into every ``_prepare``. ``corpus`` is the client's role name; ids are positional. The
        ``instruction`` reserves the instruction's tokens in the fixed overhead where the declared template
        frames it (the reranker's ``instruction: field`` and ``system`` modes -- the instruction is then
        engine-rendered into the frame, never inside the cut spans); ``record=False`` makes a probe call,
        whose spans are decided without recording census rows."""
        budget, tokenizer = self._budget, self._tokenizer
        assert budget is not None  # callers only fit when a budget is declared
        return fit(
            inputs,
            shape,
            budget,
            tokenizer,
            ids=[str(index) for index in range(len(inputs))],
            instruction=instruction,
            media_tokens=media_tokens,
            corpus=self.ROLE,
            census=self.census if record else None,
        )

    @staticmethod
    def _with_text(content: Content, text: str) -> Content:
        """The content with its text parts replaced by ``text``, media parts untouched."""
        parts: list[Any] = [TextPart(text=text)] if (text or not content.has_media) else []
        parts += [part for part in content.parts if not isinstance(part, TextPart)]
        return Content.from_parts(parts)

    async def check_engine_media(self) -> None:
        """The startup media probe: one prepared probe image to the engine, its prompt tokens compared
        with the counted ones (never silent).

        Runs when the role declares an ``image_processor``: the probe sends one prepared image through the
        adapter, counts the request's prompt tokens exactly (the media block plus the render the engine
        reads), and :func:`~rcp_ndcg.data.resolution.engine_media_check` compares the engine's own
        ``usage.prompt_tokens`` with it -- the counted number covers the same request (the media block
        plus the probe's text tokens). A mismatch is a typed :class:`~rcp_ndcg.errors.ProviderError`
        (the message names ``image_processor`` and the server's media flags); a reply without usage is
        recorded in the media census as ``not_checked`` -- the check never passes silently.

        Raises:
            CapabilityError: the engine refused the probe request.
            ProviderError: the engine's prompt-token count disagrees with the counted one.
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
            # The counted number must cover the same request the engine's report covers (the contract in
            # :func:`~rcp_ndcg.data.resolution.engine_media_check`): the media block plus every text token
            # the probe request carries, in the declared tokenizer's tokens (the render the engine reads
            # is the client's own here -- the declared budget counts the client's render, not a server
            # template the client cannot see).
            assert self._tokenizer is not None, "an image_processor implies a declared budget tokenizer"
            counted = sum(count.tokens for count in self._media_counts_of(prepared.contents)) + sum(
                self._tokenizer.count(content.text) for content in prepared.contents
            )
            calls = list(self._probe_calls(prepared.contents[0]))
            replies = await self._sender.send(calls)
            tokens = self._probe_usage(replies[0])
            if tokens is None or tokens.input_tokens is None:
                self.media_census.record(
                    corpus=self.ROLE,
                    doc_id="engine_media_check:not_checked",
                    media=[prepared.media[0]],
                    dropped=False,
                )
                get_logger(__name__).warning(
                    "engine media check: the %s reply reported no usage, so the counted prompt tokens cannot "
                    "be verified against the engine (recorded as not_checked); media counting proceeds on "
                    "the declared %s policy",
                    getattr(self._adapter_cls, "name", self.config.api),
                    getattr(self.config, "image_processor", None),
                )
                return
            mismatch = engine_media_check(tokens.input_tokens, counted)
            if mismatch is not None:
                raise ProviderError(mismatch.message, hint=mismatch.message)
        finally:
            if probe_path:
                Path(probe_path).unlink(missing_ok=True)

    def _probe_calls(self, content: Content) -> Sequence[Call]:
        """The calls one prepared probe item is sent as (the role's wire)."""
        raise NotImplementedError

    def _probe_usage(self, reply: Reply) -> TokenCount | None:
        """The reply's prompt-token report (``None``: the engine reported none)."""
        raise NotImplementedError

    def _apply_empty_documents(self, contents: Sequence[Content]) -> tuple[list[Content], list[int]]:
        """The request's empty documents as the config's ``empty_doc`` policy sends them.

        An empty document is one with no text and no media left (a document whose every media item the
        budget dropped is one, exactly like an empty text document). ``send`` (the default) sends the empty
        string as today; ``send_text`` sends the configured placeholder text; ``omit_zero`` never sends the
        item -- it scores 0.0 -- and the caller places the missing result (a zero vector, an empty slice, a
        0.0 score) at its position. A request is never sent empty: the omitted items leave it, and the
        caller returns without one when nothing remains.

        Returns:
            ``(kept, omitted)``: the contents to send and the indices of the omitted inputs (``omit_zero``).
        """
        policy = getattr(self.config, "empty_doc", "send")
        kept: list[Content] = []
        omitted: list[int] = []
        for index, content in enumerate(contents):
            if content.text or content.has_media:
                kept.append(content)
                continue
            if policy := getattr(self.config, "empty_doc", "send"):
                if policy == "omit_zero":
                    omitted.append(index)
                    continue
                if policy == "send_text":
                    kept.append(self._with_text(content, getattr(self.config, "empty_doc_text", None) or ""))
                    continue
            kept.append(content)  # "send": the empty string goes out, as today
        return kept, omitted

    # -- the fan-out, one rule (R7) --------------------------------------------
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
        """Close the sender asynchronously (R15): awaited on the pool's own loop. A sender with only a sync
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
