"""The judge's wire: the OpenAI ``POST {base_url}/chat/completions`` shape, its refusals and its usage.

Any server that speaks the OpenAI chat-completions protocol can judge -- a vLLM or SGLang server, a gateway
in front of several workers, or a hosted API -- and this adapter is the whole contract with it: the request
body the judge sends (:func:`build_messages`, the sampling settings, ``extra_body`` merged at the top level
as the OpenAI SDK used to), the answer's text, reasoning channel, ``finish_reason`` and token usage read back
into a :class:`~rcp_ndcg.inference.types.Completion`, and the endpoint's refusals mapped onto typed errors:

* an HTTP 400/422 that says the request carries too many images or videos, and a refusal of the answer
  schema (``response_format``), are :class:`~rcp_ndcg.errors.CapabilityError` -- the endpoint cannot take
  this request at all;
* a refusal of the request itself (any other returned 4xx, or an answer with no choices) is
  :class:`~rcp_ndcg.errors.RequestRejectedError` -- that window's failure, not every request's. 401, 403
  and 404 never reach the adapter: the transport's shared status map raises them first;
* the pre-send gate of the window's media against the config's ``max_images``/``max_videos``, and the
  reasoning watch under an answer schema, run here, once per client instance, exactly as the judge client
  ran them before the shared transport.

The adapter is instantiated per client with the role config whose fields decide the request (a
:class:`~rcp_ndcg.llm.JudgeConfig` or any object with the same settings, :class:`ChatSettings`); a third
party's judge wire is selectable with ``api: <name>`` the same way.
"""

from __future__ import annotations

import base64
import os
import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, NamedTuple, Protocol

from rcp_ndcg_core.content import Content, ImagePart, MediaRef, TextPart, VideoPart

from rcp_ndcg.errors import CapabilityError, DataError, RequestRejectedError
from rcp_ndcg.inference.adapters.base import AdapterRole, register_adapter
from rcp_ndcg.inference.types import Call, Completion, CompletionInput, Reply, TokenCount
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)

#: Keys the endpoint may use for the reasoning channel.
REASONING_KEYS = ("reasoning_content", "reasoning")

#: Answers of a ``json_schema`` judge read before concluding that the server returns no reasoning.
REASONING_WATCH = 8

#: A server's refusal of the number of images or videos in one request (vLLM: "At most 4 image(s) may be provided
#: in one prompt."; SGLang: "Image count 12 exceeds limit 10 per request.").
_MEDIA_LIMIT = re.compile(
    r"(?:at most \d+|too many)\s+(image|video)|\b(image|video)s?\s+count\s+\d+\s+exceeds", re.IGNORECASE
)

#: Words an endpoint's refusal of a ``response_format`` names.
_SCHEMA_REFUSAL = re.compile(r"response_format|json_schema|guided|structured", re.IGNORECASE)


class ChatSettings(Protocol):
    """The judge settings the ``openai_chat`` wire reads.

    A :class:`~rcp_ndcg.llm.JudgeConfig` (or any object with the same attributes) satisfies it: the adapter
    takes the role config at construction, because its content fields decide what the request bodies carry.
    """

    @property
    def model(self) -> str: ...

    @property
    def temperature(self) -> float | None: ...

    @property
    def max_output_tokens(self) -> int | None: ...

    @property
    def extra_body(self) -> dict[str, Any]: ...

    @property
    def max_images(self) -> int: ...

    @property
    def max_videos(self) -> int: ...


# ---------------------------------------------------------------------------
# The request: messages and media (moved from rcp_ndcg.llm._payload, unchanged)
# ---------------------------------------------------------------------------

#: Largest video container inlined into one request, in bytes. A data URI is
#: base64 (4/3 the size) and is re-sent with every window the clip appears in, so a
#: clip over this is refused by name rather than sent as a request the engine or a
#: proxy rejects -- or accepts after a long upload. Override with
#: ``RCP_NDCG_MAX_VIDEO_BYTES``; a corpus of long clips wants shorter clips.
MAX_VIDEO_BYTES = int(os.environ.get("RCP_NDCG_MAX_VIDEO_BYTES", str(64 * 1024 * 1024)))


class MediaCounts(NamedTuple):
    """What one prompt sends: ``image_url`` blocks and ``video_url`` blocks."""

    images: int
    videos: int


def media_counts(content: Content | None) -> MediaCounts:
    """How many image and video blocks *content* lowers to in :func:`build_messages`.

    The count the per-request gates compare against the server's limits; a video
    frame directory counts one image per frame, a container one video.
    """
    if content is None:
        return MediaCounts(0, 0)
    images = videos = 0
    for part in content.parts:
        if isinstance(part, ImagePart):
            images += 1
        elif isinstance(part, VideoPart):
            if part.frames:
                images += len(part.frames)
            else:
                videos += 1
    return MediaCounts(images, videos)


def build_messages(input: CompletionInput) -> list[dict[str, Any]]:
    """The ``messages`` array for one completion request.

    A text-only input sends a string ``content``; an input with media sends a list
    of text, ``image_url`` and ``video_url`` blocks.
    """
    content = input.user_content
    if content is None or not content.has_media:
        return [{"role": "user", "content": input.user_prompt}]
    return [{"role": "user", "content": _blocks(content)}]


def _blocks(content: Content) -> list[dict[str, Any]]:
    from rcp_ndcg.data.media import default_resolver

    resolver = default_resolver()

    blocks: list[dict[str, Any]] = []
    for part in content.parts:
        if isinstance(part, TextPart):
            if part.text:
                blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            blocks.append(_image_block(part.ref))
        elif isinstance(part, VideoPart):
            if part.frames:
                blocks.extend(_image_block(ref) for ref in part.frames)
            else:
                assert part.ref is not None  # VideoPart validates container-or-frames
                blocks.append(_video_block(part.ref, resolver))
    return blocks


def _video_block(ref: MediaRef, resolver: Any) -> dict[str, Any]:
    """One ``video_url`` block carrying the whole container as a data URI."""
    from rcp_ndcg.data.media import VIDEO_MIME_BY_SUFFIX

    mime = ref.mime or VIDEO_MIME_BY_SUFFIX.get(Path(ref.uri).suffix.lower())
    if mime is None or not mime.startswith("video/"):
        raise DataError(
            f"{ref.uri}: cannot tell which video container this is (mime {ref.mime!r}); record `mime` at "
            f"ingest or use one of {sorted(VIDEO_MIME_BY_SUFFIX)}"
        )
    size = ref.num_bytes if ref.num_bytes is not None else resolver.local_path(ref).stat().st_size
    if size > MAX_VIDEO_BYTES:
        raise DataError(
            f"{ref.uri} is {size} bytes, over the {MAX_VIDEO_BYTES}-byte limit for an inlined video "
            "(`RCP_NDCG_MAX_VIDEO_BYTES`). Shorten or re-encode the clip at ingest, or raise the limit "
            "knowingly -- every window re-sends it."
        )
    path = resolver.local_path(ref)
    return {"type": "video_url", "video_url": {"url": _video_data_uri(ref.cache_key, str(path), mime)}}


def _image_block(ref: MediaRef) -> dict[str, Any]:
    """One ``image_url`` block carrying a prepared image."""
    from rcp_ndcg.data.prepare import is_prepared

    if not is_prepared(ref):
        raise DataError(
            f"{ref.uri}: an image reached the request unprepared. Every image and frame is sent as "
            "rcp_ndcg.data.prepare.prepare_content returns it (resized for the judge's processor, inlined)."
        )
    return {"type": "image_url", "image_url": {"url": ref.uri}}


#: Encoded video containers held per worker process. Far fewer than pages: a clip
#: is megabytes where a page is hundreds of KB, so the image cache's 512 entries
#: would be gigabytes per worker. A clip still recurs across the windows of one
#: query, which is what this small cache spans. Override with
#: ``RCP_NDCG_VIDEO_CACHE_SIZE``.
VIDEO_CACHE_SIZE = int(os.environ.get("RCP_NDCG_VIDEO_CACHE_SIZE", "16"))


@lru_cache(maxsize=VIDEO_CACHE_SIZE)
def _video_data_uri(cache_key: str, path: str, mime: str) -> str:
    """The data URI for one cached video container, keyed by :attr:`MediaRef.cache_key` (the content hash when
    there is one)."""
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


def _reply_text(reply: Reply) -> str:
    """The endpoint's error text of a reply, however the body is shaped (a string body, an ``error.message``,
    or the strings of a nested JSON body); the raw bytes decoded when the body is none of those."""
    body = reply.body
    if isinstance(body, str):
        return body
    if isinstance(body, bytes):
        return body.decode("utf-8", errors="replace")
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
        if isinstance(error, str):
            return error
        return _flatten(body) or repr(body)
    return repr(body)


def _flatten(value: object) -> str:
    """The strings inside a decoded JSON body, joined (the endpoint's own words, wherever it nested them)."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(text for text in (_flatten(item) for item in value.values()) if text)
    if isinstance(value, list):
        return " ".join(text for text in (_flatten(item) for item in value) if text)
    return ""


@register_adapter
class OpenAIChat:
    """The judge's wire adapter: one :class:`~rcp_ndcg.inference.types.CompletionInput` in, one
    :class:`~rcp_ndcg.inference.types.Completion` out, over ``POST {base_url}/chat/completions``.

    The adapter holds the judge settings it serves (a :class:`ChatSettings` -- a
    :class:`~rcp_ndcg.llm.JudgeConfig`), because the config's sampling fields decide the request bodies. Its
    state beyond that is the reasoning watch, so one adapter instance serves one client.

    Args:
        config: The role config whose ``api`` selected this adapter; ``temperature``, ``max_output_tokens``,
            ``extra_body``, ``max_images`` and ``max_videos`` decide the requests.
    """

    name: ClassVar[str] = "openai_chat"
    """The adapter's name, the value a judge config's ``api`` field holds (the role's default)."""

    role: ClassVar[AdapterRole] = "judge"
    """The role the adapter serves: one prompt in, one answer out."""

    def __init__(self, config: ChatSettings) -> None:
        self.config = config
        self._reasoning_checked = False
        self._answers_without_reasoning = 0

    # -- the adapter seam ---------------------------------------------------
    def calls(self, request: CompletionInput, *, model: str) -> list[Call]:
        """The request body of one prompt, and the pre-send media gate.

        The body is the OpenAI chat-completions shape the judge always sent: ``messages`` (text as a string,
        prepared media as ``image_url``/``video_url`` blocks), the sampling settings when set
        (``max_output_tokens`` as ``max_completion_tokens``), the window's ``response_format`` when it carries
        one, and ``extra_body``'s fields merged at the top level (as the OpenAI SDK used to).

        Raises:
            CapabilityError: The prompt carries media the config does not declare the model to read, or more
                of one kind than the config allows -- refused before anything is sent.
            DataError: An image or video reached the request unprepared, or a video container is oversized or
                of an unknown kind (:func:`build_messages`).
        """
        self._check_media(request, model=model)
        params: dict[str, Any] = {"model": model, "messages": build_messages(request)}
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        if self.config.max_output_tokens is not None:
            params["max_completion_tokens"] = self.config.max_output_tokens
        if request.response_format is not None:
            params["response_format"] = request.response_format
        if self.config.extra_body:
            params.update(self.config.extra_body)
        return [Call(method="POST", path="/chat/completions", json=params)]

    def interpret(self, request: CompletionInput, replies: Sequence[Reply]) -> Completion:
        """The answer of ``request``: the first choice's message, its reasoning channel and finish reason.

        The media-count refusal texts (HTTP 400/422) and a refusal of the answer schema become
        :class:`~rcp_ndcg.errors.CapabilityError`; any other refusal -- including an answer with no choices --
        a :class:`~rcp_ndcg.errors.RequestRejectedError`. The reasoning watch logs its one warning here, after
        :data:`REASONING_WATCH` answers under a schema carried none.

        Raises:
            CapabilityError: The endpoint cannot take this request at all (a refused media count or schema).
            RequestRejectedError: The endpoint refused this one request (any other returned 4xx, or no
                choices in the answer).
        """
        reply = replies[0]
        if reply.status != 200:
            raise self._refusal(request, reply)
        completion = self._completion(reply)
        self._watch_reasoning(request, completion)
        return completion

    def usage(self, reply: Reply) -> TokenCount | None:
        """The tokens one answer reports (``usage.prompt_tokens``/``completion_tokens``); ``None`` without."""
        usage = reply.body.get("usage") if isinstance(reply.body, dict) else None
        if not isinstance(usage, dict):
            return None
        return TokenCount(
            input_tokens=usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None,
            output_tokens=(usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else None),
        )

    def fingerprint(self, reply: Reply) -> str | None:
        """The ``system_fingerprint`` one answer reports (some engines put their version there), or ``None``."""
        body = reply.body
        if not isinstance(body, dict):
            return None
        fingerprint = body.get("system_fingerprint")
        return fingerprint if isinstance(fingerprint, str) and fingerprint else None

    # -- the request's gates ------------------------------------------------
    def _check_media(self, request: CompletionInput, *, model: str) -> None:
        """Refuse media the config does not declare the model to read, before the request is built.

        Raises:
            CapabilityError: The prompt carries images or videos the config's ``max_images``/``max_videos``
                do not allow.
        """
        if not request.has_media:
            return
        counts = media_counts(request.user_content)
        for kind, count, limit in (
            ("images", counts.images, self.config.max_images),
            ("videos", counts.videos, self.config.max_videos),
        ):
            if count and not limit:
                raise CapabilityError(
                    f"{model} is not declared to read {kind} (max_{kind}: 0), but this prompt carries "
                    f"{count}. Declare max_{kind} for a checkpoint that reads them, or judge the text side of the "
                    "corpus."
                )
            if count > limit:
                raise CapabilityError(
                    f"this window carries {count} {kind} and the endpoint accepts {limit} per request "
                    f"(max_{kind}). Use a smaller window, or raise the limit on the server and here."
                )

    # -- the answer ---------------------------------------------------------
    def _completion(self, reply: Reply) -> Completion:
        """The :class:`Completion` of one 200 answer; a body with no choices is a refused request."""
        body = reply.body
        if not isinstance(body, dict):
            raise RequestRejectedError(f"the answer is not a chat completion ({type(body).__name__} body)")
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RequestRejectedError(f"{self.config.model} returned no choices")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            raise RequestRejectedError(f"{self.config.model} answered a choice without a message")
        reasoning = next((message[key] for key in REASONING_KEYS if isinstance(message.get(key), str)), None)
        tokens = self.usage(reply)
        return Completion(
            response=message.get("content") or "",
            reasoning=reasoning,
            finish_reason=choice.get("finish_reason"),
            input_tokens=tokens.input_tokens if tokens is not None else None,
            output_tokens=tokens.output_tokens if tokens is not None else None,
        )

    def _refusal(self, request: CompletionInput, reply: Reply) -> Exception:
        """The typed error of a reply the transport returned (a 4xx outside the shared status map).

        A refusal that names the answer schema, of a request that carried one, is a :class:`CapabilityError`:
        the endpoint cannot do ``decoding: json_schema``, and every window would be refused the same way. So
        is a refusal of the window's number of images or videos: the server's per-request media limit is below
        the judge's. Every other refusal is this one request's.
        """
        code, text = reply.status, _reply_text(reply)
        message = f"HTTP {code}: {text}"
        if code in (400, 422):
            media = _MEDIA_LIMIT.search(text)
            if media is not None:
                kind = (media.group(1) or media.group(2)).lower()
                return CapabilityError(
                    f"the judge endpoint refused the number of {kind}s in a window ({message})",
                    hint=f"the server accepts fewer {kind}s per request than the judge's max_{kind}s: raise the "
                    "server's per-request media limit (its limit on images and videos per prompt; "
                    f"docs/concepts/judges.md shows the setting), or lower max_{kind}s and the window size to "
                    "what the server accepts",
                    details={"kind": kind, "status": code},
                )
            if request.response_format is not None and _SCHEMA_REFUSAL.search(text):
                return CapabilityError(
                    f"the judge endpoint refused the answer schema ({message})",
                    hint="serve the model with its reasoning parser and structured outputs enabled, or set "
                    "decoding: free in the judge config",
                )
        return RequestRejectedError(message)

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
                "docs/concepts/judges.md); a model that does not reason, or an API that does not return its "
                "reasoning, can ignore this.",
                REASONING_WATCH,
                self.config.model,
            )


__all__ = [
    "MAX_VIDEO_BYTES",
    "REASONING_KEYS",
    "REASONING_WATCH",
    "OpenAIChat",
    "VIDEO_CACHE_SIZE",
    "build_messages",
    "media_counts",
]
