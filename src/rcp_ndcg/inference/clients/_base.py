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
* **the text budget** (item 4) -- :class:`~rcp_ndcg.data.preprocess.TextBudget` resolved from the role
  config's fields once, the tokenizer it names loaded once, the shared :func:`rcp_ndcg.data.preprocess.fit`
  called from each client's ``_prepare``, and a named seam (:meth:`RoleClient._media_tokens`) for the
  media lane.

Typed errors only: a wrong role, a missing bridge, a missing base URL and a ``batch_size < 1`` are all
:class:`~rcp_ndcg.errors.ConfigError`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any, ClassVar, Self, TypeVar

from rcp_ndcg_core.content import Content, TextPart

from rcp_ndcg.data.preprocess import FitResult, TextBudget, TextTruncationCensus, fit
from rcp_ndcg.data.templates import RequestShape
from rcp_ndcg.errors import ConfigError
from rcp_ndcg.inference.adapters import embeddings as _shipped_adapters  # noqa: F401  # registers them
from rcp_ndcg.inference.adapters.base import AdapterRole, get_adapter
from rcp_ndcg.inference.endpoint import Endpoint
from rcp_ndcg.inference.transport import AuthProfile, Sender, Transport

if TYPE_CHECKING:
    from rcp_ndcg.data.tokenizer import TextTokenizer

T = TypeVar("T")
"""The result type of a coroutine the fan-out runs."""

C = TypeVar("C", bound=Endpoint)
"""The role config a client serves (an :class:`~rcp_ndcg.inference.endpoint.Endpoint` subclass)."""


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

    #: The role config, as it was given (a hosted profile's ``base_url`` stays ``None``).
    config: C

    def __init__(self, config: C, *, sender: Sender | None = None, census: TextTruncationCensus | None = None):
        """Build the client for ``config``, sending over ``sender``.

        Args:
            config: The role config; its ``api`` names the wire adapter (resolved within :attr:`ROLE`), and
                a hosted profile without a ``base_url`` takes its adapter's ``DEFAULT_BASE_URL``.
            sender: What sends the calls. ``None`` builds a :class:`~rcp_ndcg.inference.transport.Transport`
                for the resolved config; anything else must be a ``Sender`` with a sync bridge (``run``) --
                a transport's, or the sender's own.
            census: Where the text-budget cuts are recorded; ``None`` gives the client a fresh in-memory
                census (:attr:`census`), whose rows a caller can read or hand a sink to.

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
        if sender is None:
            self._sender: Any = Transport(self.endpoint, auth=self._auth_profile())
        elif isinstance(sender, Transport) or callable(getattr(sender, "run", None)):
            self._sender = sender
        else:
            raise ConfigError(
                "the sender has no sync bridge: a Transport bridges synchronous calls through its run(), and "
                "an injected sender must provide one too",
                hint="pass a Transport (the transport built from the config), or give the sender a "
                "run(coroutine) method",
            )
        self._budget, self._tokenizer = self._resolve_budget()
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
        if isinstance(self._sender, Transport):
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

    def _media_tokens(self, contents: Sequence[Content]) -> list[int]:
        """Per-input media token counts, reserved whole out of the budget and never cut.

        **The media lane's seam** (``budget-media``): that lane wires the role config's media policy through
        here -- each image or video block's token count, from the policy the config declares -- and the
        :func:`~rcp_ndcg.data.preprocess.fit` call reserves them.

        Until it lands, the reservation is zero, and the roles differ: the embedding adapters refuse media
        Until it lands, the reservation is zero, and the roles differ: the embedding adapters refuse media
        outright (a declared budget is never under-counted there); the rerank and pooling wires carry media
        today (content parts, or the pooling ``messages`` shape) **without** reserving its tokens -- a
        declared budget counts text only. Declared interim policy: a media item's cost is outside the
        declared budget until the media lane lands. When that lane wires non-zero counts through here, the
        rerank settlement probe must reserve the documents' maximum media count (per-pair caps then differ
        per document; the probe's cap must match what ships), and the document's media parts ride on every
        chunk of it -- the media lane owns both policies.
        """
        return [0] * len(contents)

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
