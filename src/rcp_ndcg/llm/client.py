"""The judge: OpenAI-compatible endpoints, configured by :class:`JudgeConfig`, called by :class:`JudgeClient`.

Any server that speaks the OpenAI chat-completions protocol can judge: a vLLM or
SGLang server, a gateway in front of several workers, or a hosted API. That URL is
the whole contract with the model: the package never starts an engine or reads its
flags. ``base_url`` is one URL or a list of replica URLs of the same model (a gateway
is a list of one).

The judge is one role of the :mod:`rcp_ndcg.inference` layer: :class:`JudgeClient` builds the request with
its wire adapter (:func:`~rcp_ndcg.inference.adapters.base.get_adapter`, ``openai_chat`` by default) and
sends it through the shared :class:`~rcp_ndcg.inference.transport.Transport`, which owns everything around a
request -- the bounded concurrency, the least-busy replica pick, the bounded retries, the parking on outage,
the credentials and the shared status map (its behaviour, the numbers and the retry delays are documented in
:mod:`rcp_ndcg.inference.transport`; the retry delays are the transport's policy). The client adds what is
the judge's own:

* the wire checks that turn a misconfigured server into a message: a refusal of a window's image or video
  count is a :class:`~rcp_ndcg.errors.CapabilityError` that names the server's per-request media limit, a
  ``decoding: json_schema`` judge whose first answers carry no reasoning logs one warning that the server
  probably runs without the model's reasoning parser, and the client's per-replica probe records
  (:class:`EngineInfo`, best effort) what each replica says about itself for the run manifest and the
  judgement store;
* the usage the judging passes and the CLI report (requests, failed requests, tokens), accumulated over
  every call.

``JudgeConfig.fake(seed)`` builds the offline judge over the fake transport: the
``fake://`` chat completions route answers with the same messages and parsers as a
model (:mod:`rcp_ndcg.llm._fake`), below the real transport.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Self
from urllib.parse import urlsplit

from pydantic import Field, model_validator
from rcp_ndcg_core.schemas import Decoding

from rcp_ndcg.data.preprocess import TextBudget
from rcp_ndcg.data.resolution import ImagePolicy, ImageProcessor, VideoPolicy
from rcp_ndcg.data.tokenizer import TextTokenizer
from rcp_ndcg.errors import (
    BackendUnavailableError,
    CredentialsError,
    RequestRejectedError,
)
from rcp_ndcg.inference.adapters.base import AdapterRole
from rcp_ndcg.inference.adapters.chat import REASONING_KEYS, REASONING_WATCH, OpenAIChat
from rcp_ndcg.inference.clients import RoleClient
from rcp_ndcg.inference.endpoint import Endpoint
from rcp_ndcg.inference.fake import FAKE_SCHEME
from rcp_ndcg.inference.transport import Transport
from rcp_ndcg.inference.types import Call, Completion, CompletionInput, EngineInfo, TokenCount, Usage
from rcp_ndcg.support.identity import FieldRole, identity_payload
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)

#: ``base_url`` of the offline fake judge (``JudgeConfig.fake``); the scheme is the inference layer's, at
#: :data:`rcp_ndcg.inference.fake.FAKE_SCHEME`.
FAKE_URL_SCHEME = FAKE_SCHEME

# The reasoning-channel keys and the reasoning watch are the chat wire's
# (:mod:`rcp_ndcg.inference.adapters.chat`); they are imported at the top, so every path that imports
# them from here keeps working. They stay in this module's ``__all__``.

# The outage and rejection types are the inference layer's: their home is
# ``rcp_ndcg.errors`` and they are imported at the top, so every path that imports them from
# ``rcp_ndcg.llm.client`` keeps working. They stay in this module's ``__all__``.


#: Endpoints whose model names float unless they end in a snapshot date.
_DATED_SNAPSHOT_HOSTS = frozenset({"api.openai.com"})
_DATED_SNAPSHOT = re.compile(r"-\d{4}-\d{2}-\d{2}$")


class JudgeConfig(Endpoint):
    """One judge: an OpenAI-compatible :class:`~rcp_ndcg.inference.endpoint.Endpoint` with sampling settings.

    The endpoint fields (``api``, ``model``, ``revision``, ``api_key_env``, ``headers_env``, ``concurrency``,
    the timeouts, retries and ``wait_on_outage_s``) are :class:`~rcp_ndcg.inference.endpoint.Endpoint`'s;
    ``revision`` is recorded in the judgement family, so two checkpoints served under one name never pool.

    Attributes:
        base_url: The endpoint, e.g. ``http://localhost:8000/v1``, or a list of replica URLs of the same served model
            (each request goes to the live replica with the fewest requests in flight); ``fake://`` for the offline
            judge. Required.
        api: The wire adapter that speaks the endpoint's protocol; ``None`` (the default) is the judge's
            ``openai_chat`` wire. Naming another adapter (a third-party judge wire) enters the identity: it
            decides what is computed. Content.
        temperature: Sampling temperature; ``None`` (the default) sends none, so the server's default applies
            (some reasoning models reject a temperature).
        max_output_tokens: Completion token cap per request, reasoning included (sent as
            ``max_completion_tokens``); ``None`` leaves it to the endpoint.
        extra_body: Further request fields the endpoint understands (e.g. ``reasoning_effort``,
            ``chat_template_kwargs``); merged into the request body's top level.
        context_tokens: The prompt and completion tokens one request may hold (the served context
            window, or a hosted API's input limit when that is lower); sizes the per-window text
            budget, counted with ``tokenizer``. ``None``, or no ``tokenizer``, sends every document whole.
        decoding: ``"json_schema"``: each tournament and rubric request carries the stage's answer schema as
            ``response_format`` (vLLM, SGLang and the OpenAI API constrain the answer to it), and the judgement
            family records it; an endpoint that refuses the schema fails the pass with a
            :class:`~rcp_ndcg.errors.CapabilityError`. ``"free"`` (the default): the judge answers in free text.
        max_images: Images the served model accepts per request; 0 (the default) means it reads none.
        max_videos: Video containers the served model accepts per request; 0 (the default) means it reads none.
            There is no "unlimited": a multimodal judge declares its limits, which the server's per-request
            media limits must allow, and which are the ceiling a window's media is checked against.
        image_processor: The served model's image processor family (``qwen2_vl``, ``qwen2_5_vl``, ``qwen3_vl``;
            :data:`~rcp_ndcg.data.resolution.PROCESSORS`). The client resizes every image and video frame exactly
            as that processor would, within the pass's pixel budget, so the engine needs no media flags. ``None``
            (the default): the family is unknown, and images are sent unchanged, as stored.
        tokenizer: The served model's tokenizer, in whose tokens text limits and the window budget are counted:
            a Hugging Face repository id with an optional ``@revision`` (``Qwen/Qwen3.5-397B-A17B-FP8``), or a
            local path to a ``tokenizer.json`` (:func:`~rcp_ndcg.data.tokenizer.load_tokenizer`; ``[hf]`` extra).
            ``None`` (the default): no text is cut, so only ``on_overflow: keep`` runs, and estimates
            approximate tokens from characters. The file's SHA-256 enters the judgement family.
        allow_floating_model: Accept an undated model alias on the OpenAI API. An alias
            (``gpt-5``) moves between snapshots, so judgements recorded against it are not
            reproducible; by default only a dated snapshot (``gpt-5-2025-08-07``) is accepted.
    """

    #: The endpoint's roles are inherited (``api`` is CONTENT: the wire adapter computes the answers; a judge
    #: that leaves it unset resolves to ``openai_chat``, which stays out of the payload, so identities do not
    #: move). The sampling settings and the per-window text budget change what the judge returns; the gates
    #: and the transport refuse or route requests, never change one. The tokenizer changes what the judge
    #: reads, and enters every identity by its content (the judgement family's SHA-256 of its tokenizer.json),
    #: never by how it is named: its name is runtime.
    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {
        "temperature": FieldRole.CONTENT,
        "max_output_tokens": FieldRole.CONTENT,
        "extra_body": FieldRole.CONTENT,
        "context_tokens": FieldRole.CONTENT,
        "decoding": FieldRole.CONTENT,
        "image_processor": FieldRole.CONTENT,
        "tokenizer": FieldRole.RUNTIME,
        "max_images": FieldRole.RUNTIME,
        "max_videos": FieldRole.RUNTIME,
        "allow_floating_model": FieldRole.RUNTIME,
    }

    base_url: str | list[str]  # type: ignore[assignment]  # a judge is always reached at a URL
    temperature: float | None = None
    max_output_tokens: int | None = Field(default=None, ge=1)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    context_tokens: int | None = Field(default=None, ge=1)
    decoding: Decoding = "free"
    max_images: int = Field(default=0, ge=0)
    max_videos: int = Field(default=0, ge=0)
    image_processor: ImageProcessor | None = None
    tokenizer: str | None = Field(default=None, min_length=1)
    allow_floating_model: bool = False

    # base_url's trailing-slash stripping, the empty-URL refusal and the replica-list rules are
    # :class:`~rcp_ndcg.inference.endpoint.Endpoint`'s (and its ``urls`` property is): a judge is an endpoint,
    # and the copies here had drifted to accept ``base_url: "" -- a config that validates and can never be
    # sent to.

    @model_validator(mode="after")
    def _snapshot_is_pinned(self) -> Self:
        hosted = any(urlsplit(url).hostname in _DATED_SNAPSHOT_HOSTS for url in self.urls)
        if hosted and not self.allow_floating_model and not _DATED_SNAPSHOT.search(self.model):
            raise ValueError(
                f"model {self.model!r} is a floating alias: it moves between snapshots, so judgements "
                "recorded against it are not reproducible. Name a dated snapshot (e.g. gpt-5-2025-08-07), "
                "or set allow_floating_model: true"
            )
        return self

    @classmethod
    def fake(cls, seed: int = 0) -> JudgeConfig:
        """The offline judge (``fake://``, answered by the fake chat route of :mod:`rcp_ndcg.llm._fake`) with
        the given seed.

        Recorded as ``judge_model: "fake"``, so its judgements never pool with a real judge's.
        """
        return cls(base_url=f"{FAKE_URL_SCHEME}seed/{seed}", model="fake", temperature=None)

    @property
    def is_fake(self) -> bool:
        """Whether this is the offline fake judge."""
        return self.urls[0].startswith(FAKE_URL_SCHEME)

    @classmethod
    def load(cls, path: str | Path) -> JudgeConfig:
        """Read a judge config: a YAML path, or the name of a shipped one (:mod:`rcp_ndcg.llm.judges`).

        The YAML may ``extends:`` another config.

        Raises:
            MissingInputError: ``path`` is neither a file nor a shipped name.
        """
        from rcp_ndcg.llm.judges import judge_config_path
        from rcp_ndcg.support.config import load_config

        return cls.model_validate(load_config(judge_config_path(path)))

    def identity(self) -> dict[str, Any]:
        """The CONTENT fields: who judges, and how they are asked to answer."""
        return identity_payload(self)

    def api_key(self) -> str:
        """The API key from :attr:`api_key_env`, or ``"EMPTY"`` when none is configured.

        Raises:
            CredentialsError: ``api_key_env`` names a variable that is not set.
        """
        if self.api_key_env is None:
            return "EMPTY"
        value = os.environ.get(self.api_key_env)
        if not value:
            raise CredentialsError(
                f"the judge needs an API key in ${self.api_key_env}, which is not set",
                hint=f"export {self.api_key_env}=...  (keys are read from the environment, never from configs)",
                details={"variable": self.api_key_env},
            )
        return value


class JudgeClient(RoleClient[JudgeConfig]):
    """The judge client: sends one prompt, returns one :class:`Completion`.

    Build it with :meth:`from_config`. It is the judge role's :class:`~rcp_ndcg.inference.clients.RoleClient`:
    the adapter lookup (its ``api`` resolves within the judge role, ``None`` meaning the ``openai_chat`` wire
    and staying out of the identity), the profile's base URL, the transport/Sender bridge, the auth profile
    and the close/``aclose``/``gather`` lifecycle are the shared client base's; the thin layer here is the
    judge's own content decisions -- the request body, the answer parsing and the wire checks are the
    ``openai_chat`` adapter's (:mod:`rcp_ndcg.inference.adapters.chat`), and the routing, retries, parking,
    credentials and token accounting are the shared :class:`~rcp_ndcg.inference.transport.Transport`'s. Safe
    to share across the coroutines of one event loop (each new loop gets a fresh semaphore and pool);
    ``usage`` accumulates over every call.
    """

    ROLE: ClassVar[AdapterRole] = "judge"

    #: A judge config's ``api`` is ``None``-means-``openai_chat`` (an unset one stays out of the identity).
    DEFAULT_API: ClassVar[str | None] = "openai_chat"

    def __init__(self, config: JudgeConfig, *, httpx_transport: Any = None) -> None:
        """A client of ``config``'s replicas; ``httpx_transport`` replaces the endpoint below the transport
        (tests pass a mock one), which the transport still wraps with the endpoint's timeouts and pool."""
        if config.is_fake:
            from rcp_ndcg.llm import _fake  # noqa: F401  # registers the fake:// chat completions route

        self._httpx_transport = httpx_transport
        self._adapter: OpenAIChat | None = None
        self._wired_for: JudgeConfig | None = None
        self._answered = 0
        self._refused = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._config = config  # the property's backing: super()'s assignment is then a no-op, no rewire
        # The pass's effective image policy (the run's ``preprocessing`` section, resolved under the judge's
        # ``image_processor``): set by the pass (``judging._plan``); the probe's engine media check probes
        # with exactly what the pass sends. A client built without a pass checks nothing.
        self.image_policy: ImagePolicy | None = None
        self._sender = Transport(config, httpx_transport=httpx_transport)
        super().__init__(config, sender=self._sender)

    def _resolve_budget(self) -> tuple[TextBudget | None, TextTokenizer | None]:
        """The judge declares no ``max_tokens`` request budget (its per-window text budget is the pass's,
        counted over ``context_tokens`` in :mod:`rcp_ndcg.llm.judging`), and nothing loads here: the judge
        keeps the other roles' fail-at-construction contract for its wire fields, while its tokenizer is
        judged at the pass (and loaded for the media check at probe time, see :meth:`probe`)."""
        return None, None

    def _media_policies(self) -> tuple[ImagePolicy | None, VideoPolicy | None]:
        """The pass's effective image policy (the judge config declares none: the run's ``preprocessing``
        section does, resolved under the judge's ``image_processor``); the media check probes with exactly
        what the pass sends. ``None`` (a client built without a pass): nothing to check."""
        return getattr(self, "image_policy", None), None

    # The property overrides RoleClient's plain attribute: the swap rewires, as it always did.
    # (basedpyright reports the override at the setter's def line.)
    @property
    def config(self) -> JudgeConfig:  # pyright: ignore[reportIncompatibleVariableOverride, reportAttributeAccessIssue]
        """The judge's config; assigning a new one (a ``model_copy``) rebuilds the adapter and the transport
        from it at the next call, as the offline fakes' tests do."""
        return self._config

    @config.setter
    def config(self, config: JudgeConfig) -> None:  # pyright: ignore[reportIncompatibleVariableOverride]
        if getattr(self, "_config", None) is config:
            return
        self._config = config
        self._rewire()

    def _rewire(self) -> None:
        """Drop the wire built for the previous config, closing its transport's pool and keeping its failures
        counted (a config swap must not reset what the endpoint already refused)."""
        transport = self._sender
        self._adapter = self._wired_for = None
        if transport is not None:
            self._refused += transport.usage.failed_requests
            transport.close()  # the sync twin (R15: an async caller awaits aclose())
        self._sender = Transport(self._config, httpx_transport=self._httpx_transport)
        self._point_sender_at_the_profile()

    @property
    def model(self) -> str:
        """The served model name."""
        return self._config.model

    @classmethod
    def from_config(cls, config: JudgeConfig) -> JudgeClient:
        """The client ``config`` describes: a real client over the shared transport (the offline fake judge for
        ``JudgeConfig.fake()``, answered by the ``fake://`` chat route below the transport)."""
        return cls(config)

    @property
    def engines(self) -> list[EngineInfo]:
        """What each replica reported (:meth:`probe`), with the fingerprint of its first completion."""
        transport = self._sender
        return [] if transport is None else transport.engines

    @property
    def usage(self) -> Usage:
        """Calls and tokens accumulated by this client.

        A request counts as a request when it was answered; one the endpoint refused (the adapter raised on
        its reply) counts as a failed request, as does one the transport raised on (a credentials refusal, or
        a status the shared status map raises on). A request parked out by ``wait_on_outage_s``, and one the
        rejection rule refuses, are the outage's: neither count.
        """
        transport = self._sender
        failed = 0 if transport is None else transport.usage.failed_requests
        return Usage(
            requests=self._answered,
            failed_requests=failed + self._refused,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            # the wire reports tokens and calls only; the endpoint's cached-input detail is not tracked
        )

    # ------------------------------------------------------------------
    # The wire: the adapter above the transport
    # ------------------------------------------------------------------

    def _wire(self) -> tuple[OpenAIChat, Transport]:
        """The adapter and transport of the current config, built once and rebuilt when the config is replaced.

        The adapter resolves through the shared client base's role lookup (the constructor's
        ``self._adapter_cls``); the transport is the sender that base built for the resolved config."""
        if self._adapter is None or self._wired_for is not self._config:
            # The Adapter protocol fixes no constructor: the caller instantiates it with the role config its
            # request fields depend on, and the shipped judge adapter takes the JudgeConfig.
            self._adapter = self._adapter_cls(self._config)  # pyright: ignore[reportCallIssue]
            self._wired_for = self._config
        return self._adapter, self._sender  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    async def probe(self) -> list[EngineInfo]:
        """Ask each replica what it serves (``GET <base_url>/models``), best effort; the result is :attr:`engines`.

        Never raises for an endpoint that cannot be read: it is recorded with its ``error``, and judging goes
        on (the requests themselves park while it is down). When the pass's effective preprocessing declares an
        image policy, the engine media check runs after the probe (one prepared probe image; the engine's own
        prompt-token count compared with the counted one, as every served role's probe does) -- a mismatch is
        the typed :class:`~rcp_ndcg.errors.ProviderError`, never a silent budget on the wrong footing.
        """
        if self._config.is_fake:
            return []
        infos = await self._sender.probe()
        if getattr(self, "image_policy", None) is not None and self._config.image_processor is not None:
            from rcp_ndcg.data.tokenizer import load_tokenizer

            # The media check counts the probe's text in the declared tokenizer's tokens (the shared client
            # base's contract); the judge loads it here, not at construction, so a client without a pass (and
            # the offline fakes' tests' placeholder paths) never touches the tokenizer.
            self._tokenizer = load_tokenizer(self._config.tokenizer) if self._config.tokenizer is not None else None
            if self._tokenizer is None:
                get_logger(__name__).warning(
                    "the judge %s declares an image policy but no tokenizer: the engine media check cannot "
                    "count the probe's prompt tokens, so it is not run; declare judge.tokenizer to check the "
                    "engine's vision pipeline",
                    self._config.model,
                )
            else:
                await self.check_engine_media()
        return infos

    # ------------------------------------------------------------------
    # The media check's wire shape (RoleClient's probe hooks)
    # ------------------------------------------------------------------

    def _probe_calls(self, content: Any) -> Sequence[Call]:
        """The wire calls one prepared probe item is sent as: one chat completion carrying the image."""
        adapter, _ = self._wire()
        return adapter.calls(CompletionInput(user_prompt="probe", user_content=content), model=self.model)

    def _probe_usage(self, reply: Any) -> TokenCount | None:
        """The probe reply's prompt-token report (the judge adapter's, ``None`` when it reported none)."""
        adapter, _ = self._wire()
        return adapter.usage(reply)

    # ------------------------------------------------------------------
    # The call
    # ------------------------------------------------------------------

    async def complete(self, request: CompletionInput) -> Completion:
        """Answer one prompt on the least busy live replica, waiting out an outage as the module docstring says.

        Raises:
            CapabilityError: the prompt carries media the model is not declared to read, or the server refuses
                the window's number of images or videos, or its answer schema.
            BackendUnavailableError: every replica stayed down for longer than ``wait_on_outage_s``.
            CredentialsError: the endpoint refused the credentials (HTTP 401 or 403), or a configured
                environment variable is not set.
            ProviderError: the endpoint has no such route or model (HTTP 404).
            RequestRejectedError: the endpoint refused this request, or answered it with no choices.
            DataError: an image or video reached the request unprepared.
        """
        adapter, transport = self._wire()
        replies = await transport.send(adapter.calls(request, model=self.model))
        try:
            completion = adapter.interpret(request, replies)
        except Exception:
            self._refused += 1
            raise
        for reply in replies:
            transport.add_usage(adapter.usage(reply))
        fingerprint = getattr(adapter, "fingerprint", None)
        if fingerprint is not None and replies[0].url is not None:
            reported = fingerprint(replies[0])
            if reported:
                transport.note_system_fingerprint(replies[0].url, reported)
        self._answered += 1
        self._input_tokens += completion.input_tokens or 0
        self._output_tokens += completion.output_tokens or 0
        return completion


__all__ = [
    "FAKE_URL_SCHEME",
    "REASONING_KEYS",
    "REASONING_WATCH",
    "BackendUnavailableError",
    "CompletionInput",
    "EngineInfo",
    "JudgeClient",
    "JudgeConfig",
    "RequestRejectedError",
    "Usage",
]
