"""The judge: OpenAI-compatible endpoints, configured by :class:`JudgeConfig`, called by :class:`JudgeClient`.

Any server that speaks the OpenAI chat-completions protocol can judge: a vLLM or
SGLang server, a gateway in front of several workers, or a hosted API. That URL is
the whole contract with the model: the package never starts an engine or reads its
flags. ``base_url`` is one URL or a list of replica URLs of the same model (a gateway
is a list of one), and the client adds five things:

* **bounded concurrency** -- at most ``concurrency`` requests in flight, over all replicas;
* **balancing** -- each request goes to the live replica with the fewest requests in
  flight from this client (the client is normally the only sender, so its counts are exact);
* **bounded retries** -- the SDK retries transient failures ``max_retries`` times;
* **parking on outage** -- a replica that stays unavailable (connection error, timeout,
  HTTP 408, 429 or 5xx) is set aside with backoff, and its requests move to another live
  replica. When every replica is down, requests wait and are re-sent with backoff until one
  answers again, or until ``wait_on_outage_s`` passes (:class:`BackendUnavailableError`).
  A run against dead servers therefore parks instead of turning the outage into missing
  judgements; a request that keeps failing on a replica that answers other requests is
  refused (:class:`RequestRejectedError`);
* **usage accounting** -- calls and tokens per call, as the endpoint reports them.

Two runtime checks turn a misconfigured server into a message: a refusal of a window's
image or video count is a :class:`~rcp_ndcg.errors.CapabilityError` that names the
server's per-request media limit, and a ``decoding: json_schema`` judge whose first
answers carry no reasoning logs one warning that the server probably runs without the
model's reasoning parser. :meth:`JudgeClient.probe` records what each replica says about
itself (:class:`EngineInfo`, best effort) for the run manifest and the judgement store.

``JudgeClient.from_config(JudgeConfig.fake(seed))`` returns the offline
:class:`~rcp_ndcg.testing.FakeJudge`, which answers through the same messages
and parsers.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator, model_validator
from rcp_ndcg_core.schemas import Decoding

from rcp_ndcg.data.resolution import ImageProcessor
from rcp_ndcg.errors import (
    BackendUnavailableError,
    CapabilityError,
    CredentialsError,
    ProviderError,
    RequestRejectedError,
)
from rcp_ndcg.inference.endpoint import Endpoint
from rcp_ndcg.inference.fake import FAKE_SCHEME
from rcp_ndcg.inference.types import Completion, CompletionInput, EngineInfo
from rcp_ndcg.support.identity import FieldRole, identity_payload
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)

#: ``base_url`` of the offline fake judge (``JudgeConfig.fake``); the scheme is the inference layer's, at
#: :data:`rcp_ndcg.inference.fake.FAKE_SCHEME`.
FAKE_URL_SCHEME = FAKE_SCHEME

#: HTTP statuses that say "this endpoint cannot serve right now", not "this request is wrong".
_UNAVAILABLE_STATUSES = frozenset({408, 429})

#: Keys the endpoint may use for the reasoning channel.
REASONING_KEYS = ("reasoning_content", "reasoning")

# The outage and rejection types are the inference layer's: their home is
# ``rcp_ndcg.errors`` and they are imported at the top, so every path that imports them from
# ``rcp_ndcg.llm.client`` keeps working. They stay in this module's ``__all__``.


def _status(exc: BaseException) -> tuple[int, str] | None:
    """The HTTP status and message of an error response, from the OpenAI SDK or from httpx; ``None`` otherwise."""
    import httpx
    import openai

    if isinstance(exc, openai.APIStatusError):
        return exc.status_code, str(exc.message)
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code, exc.response.text or str(exc)
    return None


def is_unavailable(exc: BaseException) -> bool:
    """Whether ``exc`` says the endpoint is unavailable rather than the request invalid.

    Connection errors and timeouts, and HTTP 408, 429 and 5xx responses, whether the OpenAI SDK raised them or a
    transport of httpx directly (a client that replaces :meth:`JudgeClient._send`).
    """
    import httpx
    import openai

    if isinstance(exc, openai.APIConnectionError | httpx.TransportError):
        return True
    status = _status(exc)
    return status is not None and (status[0] >= 500 or status[0] in _UNAVAILABLE_STATUSES)


#: Endpoints whose model names float unless they end in a snapshot date.
_DATED_SNAPSHOT_HOSTS = frozenset({"api.openai.com"})
_DATED_SNAPSHOT = re.compile(r"-\d{4}-\d{2}-\d{2}$")


class JudgeConfig(Endpoint):
    """One judge: an OpenAI-compatible :class:`~rcp_ndcg.inference.endpoint.Endpoint` with sampling settings.

    The endpoint fields (``model``, ``revision``, ``api_key_env``, ``headers_env``, ``concurrency``, the
    timeouts, retries and ``wait_on_outage_s``) are :class:`~rcp_ndcg.inference.endpoint.Endpoint`'s;
    ``revision`` is recorded in the judgement family, so two checkpoints served under one name never pool.

    Attributes:
        base_url: The endpoint, e.g. ``http://localhost:8000/v1``, or a list of replica URLs of the same served model
            (each request goes to the live replica with the fewest requests in flight); ``fake://`` for the offline
            judge. Required.
        temperature: Sampling temperature; ``None`` (the default) sends none, so the server's default applies
            (some reasoning models reject a temperature).
        max_output_tokens: Completion token cap per request, reasoning included (sent as
            ``max_completion_tokens``); ``None`` leaves it to the endpoint.
        extra_body: Further request fields the endpoint understands (e.g. ``reasoning_effort``,
            ``chat_template_kwargs``).
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

    #: The endpoint's roles are inherited. The sampling settings and the per-window text budget change what
    #: the judge returns; the gates and the transport refuse or route requests, never change one. The tokenizer
    #: changes what the judge reads, and enters every identity by its content (the judgement family's SHA-256 of
    #: its tokenizer.json), never by how it is named: its name is runtime.
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

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str | list[str]) -> str | list[str]:  # type: ignore[override]
        if isinstance(value, str):
            return value.rstrip("/")
        if not value:
            raise ValueError("base_url: give one URL or a non-empty list of replica URLs")
        urls = [url.rstrip("/") for url in value]
        if len(set(urls)) < len(urls):
            raise ValueError(f"base_url lists a replica twice: {urls}")
        if len(urls) > 1 and any(url.startswith(FAKE_URL_SCHEME) for url in urls):
            raise ValueError("the offline judge (fake://) is one URL, not a replica list")
        return urls

    @property
    def urls(self) -> tuple[str, ...]:
        """The replica URLs: ``base_url`` as a tuple (one element for a single URL or a gateway)."""
        return (self.base_url,) if isinstance(self.base_url, str) else tuple(self.base_url)

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
        """The offline judge (:class:`~rcp_ndcg.testing.FakeJudge`) with the given seed.

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


# The judge's prompt and answer types, and the per-replica probe record, are the inference layer's
# their home is ``rcp_ndcg.inference.types`` and they are imported at the top, so every
# path that imports them from ``rcp_ndcg.llm.client`` keeps working. They stay in this module's ``__all__``.


class Usage(BaseModel):
    """Calls and tokens accumulated by a client."""

    requests: int = 0
    failed_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    def merged_with(self, other: Usage) -> Usage:
        """The element-wise sum."""
        return Usage(
            requests=self.requests + other.requests,
            failed_requests=self.failed_requests + other.failed_requests,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
        )


@dataclass
class _Replica:
    """One replica URL and what the client knows about it."""

    url: str
    sdk: Any = None
    in_flight: int = 0
    sent: int = 0
    successes: int = 0
    down_until: float = 0.0
    """``time.monotonic()`` before which no request is sent to it (0: live)."""
    backoff: float | None = None
    """The next time it is set aside, for how long (``None``: the client's first backoff)."""
    engine: EngineInfo | None = None


#: Answers of a ``json_schema`` judge read before concluding that the server returns no reasoning.
REASONING_WATCH = 8

#: A server's refusal of the number of images or videos in one request (vLLM: "At most 4 image(s) may be provided
#: in one prompt."; SGLang: "Image count 12 exceeds limit 10 per request.").
_MEDIA_LIMIT = re.compile(
    r"(?:at most \d+|too many)\s+(image|video)|\b(image|video)s?\s+count\s+\d+\s+exceeds", re.IGNORECASE
)


class JudgeClient:
    """The judge client: sends one prompt, returns one :class:`Completion`.

    Build it with :meth:`from_config`. Safe to share across the coroutines of one
    event loop; ``usage`` accumulates over every call.
    """

    #: First time a replica is set aside after it failed (seconds); doubles up to the maximum.
    BACKOFF_S: ClassVar[float] = 5.0
    MAX_BACKOFF_S: ClassVar[float] = 60.0

    def __init__(self, config: JudgeConfig, *, http_client: Any = None) -> None:
        """A client of ``config``'s replicas; ``http_client`` replaces the transport (tests pass a mock one)."""
        self.config = config
        self.usage = Usage()
        self._http_client = http_client
        self._pool: Any = None
        self._replicas = [_Replica(url) for url in config.urls]
        self._semaphore: asyncio.Semaphore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_error: BaseException | None = None
        self._reasoning_checked = False
        self._answers_without_reasoning = 0

    @classmethod
    def from_config(cls, config: JudgeConfig) -> JudgeClient:
        """The client ``config`` describes (the offline fake judge for ``JudgeConfig.fake()``)."""
        if config.is_fake:
            from rcp_ndcg.llm._fake import FakeJudge

            return FakeJudge.from_config(config)
        return cls(config)

    @property
    def model(self) -> str:
        """The served model name."""
        return self.config.model

    @property
    def engines(self) -> list[EngineInfo]:
        """What each replica reported (:meth:`probe`), with the fingerprint of its first completion."""
        return [replica.engine for replica in self._replicas if replica.engine is not None]

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _transport(self) -> Any:
        """The HTTP client every replica shares, pooled to the concurrency."""
        if self._http_client is not None:
            return self._http_client
        if self._pool is None:
            import httpx

            # httpx pools 100 connections by default: a larger concurrency would queue on the pool, not the endpoint.
            concurrency = self.config.concurrency
            self._pool = httpx.AsyncClient(
                timeout=self._timeout(),
                limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
            )
        return self._pool

    def _timeout(self) -> Any:
        import httpx

        return httpx.Timeout(self.config.timeout_s, connect=self.config.connect_timeout_s)

    def _sdk(self, replica: _Replica) -> Any:
        if replica.sdk is None:
            from openai import AsyncOpenAI

            # The SDK accepts httpx clients and timeouts at runtime but annotates only its own httpx2 types.
            replica.sdk = AsyncOpenAI(
                base_url=replica.url,
                api_key=self.config.api_key(),
                http_client=self._transport(),  # pyright: ignore[reportArgumentType]
                timeout=self._timeout(),  # pyright: ignore[reportArgumentType]
                max_retries=self.config.max_retries,
            )
        return replica.sdk

    def _gate(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._loop is not loop:
            self._semaphore = asyncio.Semaphore(self.config.concurrency)
            self._loop = loop
            self._pool = None  # an HTTP client is bound to the loop it was first used on
            for replica in self._replicas:
                replica.sdk = None
        return self._semaphore

    def _pick(self) -> _Replica | None:
        """The live replica with the fewest requests in flight (then the fewest sent); ``None`` when all are down."""
        now = time.monotonic()
        live = [replica for replica in self._replicas if replica.down_until <= now]
        return min(live, key=lambda replica: (replica.in_flight, replica.sent)) if live else None

    def _set_aside(self, replica: _Replica, exc: BaseException) -> None:
        """Take ``replica`` out of rotation after it failed, for a backoff that doubles while it keeps failing."""
        self._last_error = exc
        now = time.monotonic()
        if replica.down_until > now:
            return  # already set aside, by a request sent before it went down
        wait = replica.backoff or self.BACKOFF_S
        replica.down_until = now + wait
        replica.backoff = min(wait * 2, self.MAX_BACKOFF_S)
        others = sum(other.down_until <= now for other in self._replicas)
        logger.warning(
            "%s unavailable (%s: %s); not sending to it for %.0fs%s",
            replica.url,
            type(exc).__name__,
            exc,
            wait,
            f", {others} other replica(s) live" if len(self._replicas) > 1 else "",
        )

    async def _park(self, outage_since: float) -> None:
        """Wait while every replica is set aside, until the first comes back or ``wait_on_outage_s`` is spent.

        ``outage_since`` is when this request first found the endpoint unavailable (its first failure, or the
        moment it found every replica set aside), measured once the request held a concurrency slot: time spent
        queued behind other requests is not outage.

        Raises:
            BackendUnavailableError: the endpoint has been unavailable to the request for ``wait_on_outage_s``.
        """
        now = time.monotonic()
        waited = now - outage_since
        limit = self.config.wait_on_outage_s
        last = self._last_error
        where = ", ".join(self.config.urls)
        if limit is not None and waited >= limit:
            raise BackendUnavailableError(
                f"{where} was unavailable for {waited:.1f}s (wait_on_outage_s={limit}); "
                f"last error: {type(last).__name__}: {last}"
            ) from last
        wake = min(replica.down_until for replica in self._replicas) - now
        if limit is not None:
            wake = min(wake, limit - waited)
        logger.warning("%s unavailable; waiting %.0fs before re-sending (waited %.0fs)", where, wake, waited)
        await asyncio.sleep(max(wake, 0.0))

    async def probe(self) -> list[EngineInfo]:
        """Ask each replica what it serves (``GET <base_url>/models``), best effort; the result is :attr:`engines`.

        Never raises: an endpoint that cannot be read is recorded with its ``error``, and judging goes on (the
        requests themselves park while it is down).
        """
        if self.config.is_fake:
            return []
        self._gate()  # binds the HTTP client to this event loop, as a request would
        await asyncio.gather(*(self._probe(replica) for replica in self._replicas))
        return self.engines

    async def _probe(self, replica: _Replica) -> None:
        fingerprint = replica.engine.system_fingerprint if replica.engine is not None else None
        try:
            key = self.config.api_key()
            headers = {"Authorization": f"Bearer {key}"} if key != "EMPTY" else {}
            response = await self._transport().get(f"{replica.url}/models", headers=headers, timeout=self._timeout())
            response.raise_for_status()
            entries = [entry for entry in response.json().get("data") or [] if isinstance(entry, dict)]
        except Exception as exc:  # best effort: what the endpoint says is recorded, never required
            replica.engine = EngineInfo(
                url=replica.url, system_fingerprint=fingerprint, error=f"{type(exc).__name__}: {exc}"
            )
            return
        entry = next((e for e in entries if e.get("id") == self.model), None)
        if entry is None and entries:
            logger.warning(
                "%s serves %s, not the judge's model %r: start the server with --served-model-name %s, or set the "
                "judge's model to the served name",
                replica.url,
                [e.get("id") for e in entries],
                self.model,
                self.model,
            )
        entry = entry or (entries[0] if entries else {})
        length = entry.get("max_model_len")
        replica.engine = EngineInfo(
            url=replica.url,
            model=entry.get("id"),
            owned_by=entry.get("owned_by"),
            max_model_len=length if isinstance(length, int) else None,
            headers={
                name: value
                for name, value in response.headers.items()
                if name.lower() == "server" or "version" in name.lower()
            },
            system_fingerprint=fingerprint,
        )

    def _check_media(self, request: CompletionInput) -> None:
        from rcp_ndcg.llm._payload import media_counts

        counts = media_counts(request.user_content)
        for kind, count, limit in (
            ("images", counts.images, self.config.max_images),
            ("videos", counts.videos, self.config.max_videos),
        ):
            if count and not limit:
                raise CapabilityError(
                    f"{self.model} is not declared to read {kind} (max_{kind}: 0), but this prompt carries "
                    f"{count}. Declare max_{kind} for a checkpoint that reads them, or judge the text side of the "
                    "corpus."
                )
            if count > limit:
                raise CapabilityError(
                    f"this window carries {count} {kind} and the endpoint accepts {limit} per request "
                    f"(max_{kind}). Use a smaller window, or raise the limit on the server and here."
                )

    def _request(self, request: CompletionInput) -> dict[str, Any]:
        from rcp_ndcg.llm._payload import build_messages

        params: dict[str, Any] = {"model": self.model, "messages": build_messages(request)}
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        if self.config.max_output_tokens is not None:
            params["max_completion_tokens"] = self.config.max_output_tokens
        if request.response_format is not None:
            params["response_format"] = request.response_format
        if self.config.extra_body:
            params["extra_body"] = dict(self.config.extra_body)
        return params

    def _count_usage(self, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> None:
        """Add one call's tokens to :attr:`usage`."""
        self.usage = self.usage.merged_with(
            Usage(
                requests=1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
            )
        )

    def _account(self, response: Any, replica: _Replica) -> Completion:
        if not response.choices:
            raise ProviderError(f"{self.model} returned no choices")
        fingerprint = getattr(response, "system_fingerprint", None)
        if isinstance(fingerprint, str) and fingerprint:
            engine = replica.engine or EngineInfo(url=replica.url)
            if engine.system_fingerprint is None:
                replica.engine = engine.model_copy(update={"system_fingerprint": fingerprint})
        choice = response.choices[-1]
        message = choice.message
        reasoning = next(
            (getattr(message, key) for key in REASONING_KEYS if isinstance(getattr(message, key, None), str)), None
        )
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None)
        self._count_usage(int(input_tokens or 0), int(output_tokens or 0), int(cached or 0))
        return Completion(
            response=message.content or "",
            reasoning=reasoning,
            finish_reason=choice.finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _watch_reasoning(self, request: CompletionInput, completion: Completion) -> None:
        """Warn once when the first :data:`REASONING_WATCH` answers under an answer schema carry no reasoning."""
        if self._reasoning_checked or request.response_format is None:
            return
        if completion.reasoning:
            self._reasoning_checked = True
            return
        self._answers_without_reasoning += 1
        if self._answers_without_reasoning >= REASONING_WATCH:
            self._reasoning_checked = True
            logger.warning(
                "the first %d answers of %s under the stage's answer schema (decoding: json_schema) carried no "
                "reasoning: the server probably runs the model without its reasoning parser, so the schema "
                "constrains the reasoning too. Start the server with the model's reasoning parser (see "
                "docs/concepts/serving.md); a model that does not reason, or an API that does not return its "
                "reasoning, can ignore this.",
                REASONING_WATCH,
                self.model,
            )

    # ------------------------------------------------------------------
    # The call
    # ------------------------------------------------------------------

    async def _send(self, request: CompletionInput, replica: _Replica) -> Completion:
        """One request to one replica, without parking. Subclasses (the fake judge) replace this."""
        response = await self._sdk(replica).chat.completions.create(**self._request(request))
        return self._account(response, replica)

    async def complete(self, request: CompletionInput) -> Completion:
        """Answer one prompt on the least busy live replica, waiting out an outage as the module docstring says.

        Raises:
            CapabilityError: the prompt carries media the model is not declared to read, or the server refuses
                the window's number of images or videos, or its answer schema.
            BackendUnavailableError: every replica stayed down for longer than ``wait_on_outage_s``.
            CredentialsError: the endpoint refused the credentials (HTTP 401 or 403).
            ProviderError: the endpoint has no such route or model (HTTP 404).
            RequestRejectedError: the endpoint refused this request, or answered it with no choices.
        """
        if request.has_media:
            self._check_media(request)
        #: Per replica: its successes when this request first failed there.
        failed_at: dict[int, int] = {}
        #: When this request first found the endpoint unavailable; its outage clock (never the queueing time).
        outage_since: float | None = None
        async with self._gate():
            while True:
                replica = self._pick()
                if replica is None:
                    outage_since = time.monotonic() if outage_since is None else outage_since
                    await self._park(outage_since)
                    continue
                replica.in_flight += 1
                replica.sent += 1
                try:
                    completion = await self._send(request, replica)
                except Exception as exc:
                    if not is_unavailable(exc):
                        self.usage = self.usage.merged_with(Usage(failed_requests=1))
                        raise _request_error(exc, schema_sent=request.response_format is not None) from exc
                    index = self._replicas.index(replica)
                    if index in failed_at and replica.successes > failed_at[index]:
                        # Other requests got through on this replica since this one first failed there: the
                        # replica is up, and this failure belongs to this request.
                        raise RequestRejectedError(f"{type(exc).__name__}: {exc}") from exc
                    failed_at.setdefault(index, replica.successes)
                    outage_since = time.monotonic() if outage_since is None else outage_since
                    self._set_aside(replica, exc)
                    continue
                finally:
                    replica.in_flight -= 1
                replica.successes += 1
                replica.down_until, replica.backoff = 0.0, None
                self._watch_reasoning(request, completion)
                return completion


#: Words an endpoint's refusal of a ``response_format`` names.
_SCHEMA_REFUSAL = re.compile(r"response_format|json_schema|guided|structured", re.IGNORECASE)


def _request_error(exc: Exception, *, schema_sent: bool = False) -> Exception:
    """The typed error an endpoint's refusal of one request becomes; any other exception is returned as is.

    A refusal that names the answer schema, of a request that carried one, is a :class:`CapabilityError`: the
    endpoint cannot do ``decoding: json_schema``, and every window would be refused the same way. So is a refusal
    of the window's number of images or videos: the server's per-request media limit is below the judge's. Error
    responses and failures of the OpenAI SDK and of httpx map alike.
    """
    import httpx
    import openai

    status = _status(exc)
    if status is not None:
        code, text = status
        message = f"HTTP {code}: {text}"
        if code in (400, 422):
            media = _MEDIA_LIMIT.search(text)
            if media is not None:
                kind = (media.group(1) or media.group(2)).lower()
                return CapabilityError(
                    f"the judge endpoint refused the number of {kind}s in a window ({message})",
                    hint=f"the server accepts fewer {kind}s per request than the judge's max_{kind}s: raise the "
                    "server's per-request media limit (its limit on images and videos per prompt; "
                    f"docs/concepts/serving.md shows the setting), or lower max_{kind}s and the window size to "
                    "what the server accepts",
                    details={"kind": kind, "status": code},
                )
            if schema_sent and _SCHEMA_REFUSAL.search(text):
                return CapabilityError(
                    f"the judge endpoint refused the answer schema ({message})",
                    hint="serve the model with its reasoning parser and structured outputs enabled, or set "
                    "decoding: free in the judge config",
                )
        if code in (401, 403):
            return CredentialsError(f"the judge endpoint refused the credentials ({message})")
        if code == 404:
            return ProviderError(f"the judge endpoint has no such route or model ({message})", retryable=False)
        return RequestRejectedError(message)
    if isinstance(exc, openai.APIError | httpx.HTTPError) or (
        isinstance(exc, ProviderError) and not isinstance(exc, BackendUnavailableError)
    ):
        return RequestRejectedError(f"{type(exc).__name__}: {exc}")
    return exc


__all__ = [
    "FAKE_URL_SCHEME",
    "REASONING_WATCH",
    "BackendUnavailableError",
    "CompletionInput",
    "EngineInfo",
    "JudgeClient",
    "JudgeConfig",
    "RequestRejectedError",
    "Usage",
    "is_unavailable",
]
