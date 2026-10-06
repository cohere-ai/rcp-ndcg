"""The vLLM pooling adapter: late-interaction (multi-vector) encoding over ``POST {base_url}/pooling``.

Wire contract, verified field by field against the vLLM checkout (``vllm/entrypoints/pooling/`` and
``vllm/utils/serial_utils.py``):

* **Request** (``POST /pooling``, ``vllm/entrypoints/pooling/pooling/api_router.py``). A
  ``PoolingCompletionRequest`` (``vllm/entrypoints/pooling/pooling/protocol.py``) carries ``model``
  (``PoolingBasicRequestMixin``), ``input`` (``CompletionRequestMixin``: ``str | list[str] | list[int] |
  list[list[int]]``), ``task`` (``"token_embed"``, one of the ``PoolingTask`` values in ``vllm/tasks.py``) and,
  from ``EncodingRequestMixin`` (``vllm/entrypoints/pooling/base/protocol.py``), ``encoding_format``
  (``"float" | "base64" | "bytes" | "bytes_only"``, default ``"float"``), ``embed_dtype`` (``"float32" |
  "float16" | "bfloat16" | "fp8_e4m3" | "fp8_e5m2"``) and ``endianness`` (``"native" | "big" | "little"``).
  This adapter sends ``encoding_format: "base64"``, ``embed_dtype`` from the endpoint config and
  ``endianness: "little"`` (explicit, so the frame decodes the same on any server platform). A media item
  travels as a ``PoolingChatRequest`` instead (``messages``), the only shape in which the server applies the
  model's chat template to the image placeholders. ``dimensions`` is never sent: ``/pooling`` refuses it
  ("dimensions is currently not supported", ``pooling/serving.py::_verify_pooling_task``).
* **Response** (``PoolingResponse``, ``vllm/entrypoints/pooling/pooling/protocol.py``):
  ``data[i]`` is ``{index, object, data}`` with ``data`` a nested float list for ``encoding_format: "float"``
  (shape preserved) or a base64 string otherwise. The base64 frame is the ``embed_dtype`` array
  **flattened** (``vllm/utils/serial_utils.py::tensor2binary`` calls ``.flatten()``), so its
  ``(tokens, dim)`` shape is rebuilt client-side from the declared ``dim``: ``/pooling`` reports no shape for
  base64. What pins the rebuild is the response's own ``usage.prompt_tokens`` (``pooling/utils.py::
  get_pooling_usage``): a ``token_embed`` answer has exactly one vector per prompt token (the ``AllPool``
  pooling method, ``model_executor/layers/pooler/tokwise/methods.py``), so the decoded token counts must sum
  to it -- a mistyped ``dim`` is a loud error, never a silently mis-shaped corpus.
* **The ``bytes`` encodings** (``vllm/entrypoints/pooling/utils.py::encode_pooling_bytes``): the reply body is
  the concatenated per-item byte frames and the ``metadata`` response header carries each item's
  ``{index, embed_dtype, endianness, start, end, shape}`` -- framing clear enough to decode here.
  ``bytes_only`` sends no metadata and cannot be split per item, so it is refused naming the lane that will
  pin its framing.

The replies are accepted in three shapes (the layout of the answer is what says which pooling task ran, never
configuration that could disagree with the server): nested float lists decode as they arrive (2-D -> ragged,
1-D -> one vector per item, today's ``VllmPoolingClient._as_embeddings`` rule), base64 strings decode in the
declared dtype and reshape to ``(tokens, dim)``, and a bytes body decodes through the framing metadata. A reply
that reports ``usage`` (every vLLM JSON reply does) is refused when it answered one vector per item: a pooled
task, not the requested ``token_embed``. Every vector is stored in the transfer dtype the config declared
(float16 by default, float32 opt-in), so an index built from float16 vectors stores float16; the bytes per
token vector are 2 for float16 and 4 for float32.
"""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Sequence
from typing import Any, ClassVar, Final

import numpy as np

from rcp_ndcg.errors import CapabilityError, ConfigError, ProviderError, RequestRejectedError
from rcp_ndcg.inference.adapters.base import AdapterRole, register_adapter
from rcp_ndcg.inference.types import Call, Embeddings, PoolRequest, Reply, TokenCount

_TASK: Final = "token_embed"
"""The pooling task a late-interaction checkpoint serves (``vllm/tasks.py``: ``token_embed`` = "late-interaction")."""

_ENCODING_FORMAT: Final = "base64"
"""The encoding this adapter asks for. ``"float"`` frames are accepted too, but JSON floats are what made
corpus indexing over ``/pooling`` expensive in the first place."""

_ENDIANNESS: Final = "little"
"""Sent explicitly, so a frame decodes the same whatever the server's platform calls native."""

_FRAME_DTYPES: Final[dict[str, np.dtype]] = {
    "float16": np.dtype("<f2"),
    "float32": np.dtype("<f4"),
}
"""The little-endian NumPy dtype each ``embed_dtype`` value decodes from (``serial_utils.EMBED_DTYPES`` maps
``float16`` -> ``np.float16`` and ``float32`` -> ``np.float32``; the other values vLLM accepts, ``bfloat16``
and the two ``fp8`` formats, have no NumPy dtype and are not offered by the endpoint config)."""

_OVERLENGTH_PHRASES: Final = (
    "longer than the maximum model length",
    "maximum context length",
)
"""Phrases of the engine's over-length rejection: ``vllm/v1/engine/input_processor.py`` answers HTTP 400 with
"the ... prompt (length X) is longer than the maximum model length of Y"."""

_MAX_MESSAGE_CHARS: Final = 300


def _error_message(reply: Reply) -> str:
    """The server's own words for a refused request, shortened for an exception message."""
    body = reply.body
    if isinstance(body, bytes):
        return body[:_MAX_MESSAGE_CHARS].decode("utf-8", errors="replace")
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return str(error.get("message", error))[:_MAX_MESSAGE_CHARS]
        if error is not None:
            return str(error)[:_MAX_MESSAGE_CHARS]
    return str(body)[:_MAX_MESSAGE_CHARS]


def _response_index(item: dict[str, Any]) -> int:
    """Sort key restoring request order; absent on some builds, hence the default."""
    return int(item.get("index", 0))


def _decoded_tokens(arrays: Sequence[np.ndarray]) -> int:
    """The number of token vectors in decoded items (a 1-D pooled vector counts as one)."""
    return sum(1 if array.ndim == 1 else len(array) for array in arrays)


def _as_token_count(prompt_tokens: Any) -> TokenCount | None:
    """``TokenCount`` for an integer token count, ``None`` when the server reported nothing readable."""
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
        return None
    return TokenCount(input_tokens=prompt_tokens)


@register_adapter
class VllmPooling:
    """Late-interaction encoding over vLLM ``POST {base_url}/pooling`` (``task: token_embed``).

    One :class:`~rcp_ndcg.inference.types.PoolRequest` becomes one call per media item and, when the batch is
    text-only, one call for the whole batch (a text batch is one ``input`` list; once any item carries media,
    every item goes one per request through ``messages``, so the server's chat template applies uniformly).
    :meth:`interpret` rebuilds the ragged :class:`~rcp_ndcg.inference.types.Embeddings`, keeping the vectors in
    the transfer dtype the config declared.
    """

    name: ClassVar[str] = "vllm_pooling"
    role: ClassVar[AdapterRole] = "multi_vector"

    #: The auth facts of a served pooling wire (R6): the engine takes no key (the transport resolves the
    #: config's ``api_key_env``, for a gateway in front of it, into ``Authorization: Bearer``).
    HOSTED: ClassVar[bool] = False
    API_KEY_ENV: ClassVar[tuple[str, ...]] = ()
    KEY_REQUIRED: ClassVar[bool] = False
    AUTH_HEADER: ClassVar[str | None] = None

    #: The request shapes this wire implements (3): text (the rendered string), token_ids (the ids the
    #: client fitted; vLLM's ``/pooling`` takes token-id prompts) and messages (a media item's only route).
    REQUEST_SHAPES: ClassVar[frozenset[str]] = frozenset({"text", "messages", "token_ids"})

    _PATH: Final = "/pooling"

    def calls(self, request: PoolRequest, *, model: str) -> Sequence[Call]:
        """The ``POST /pooling`` calls for ``request``: one for a text-only batch, one per item otherwise.

        Args:
            request: The batch with its side, wire dtype and declared ``dim``.
            model: The served model name, sent as the request's ``model``.

        Returns:
            One call per request the server takes: the whole text batch in one ``input`` list, or each item
            (media batches included) as its own ``messages`` request -- the only shape in which the server
            applies the model's chat template to image placeholders.
        """
        wire = {
            "task": _TASK,
            "encoding_format": _ENCODING_FORMAT,
            "embed_dtype": request.embed_dtype,
            "endianness": _ENDIANNESS,
        }
        if any(content.has_media for content in request.contents):
            from rcp_ndcg.data.media import content_parts_payload

            return [
                Call(
                    method="POST",
                    path=self._PATH,
                    json={
                        **wire,
                        "model": model,
                        "messages": [{"role": "user", "content": content_parts_payload(content)}],
                    },
                )
                for content in request.contents
            ]
        if request.request_shape == "token_ids":
            return [
                Call(
                    method="POST",
                    path=self._PATH,
                    json={
                        **wire,
                        "model": model,
                        "input": [list(ids) for ids in request.token_ids],
                    },
                )
            ]
        return [
            Call(
                method="POST",
                path=self._PATH,
                json={**wire, "model": model, "input": [content.text for content in request.contents]},
            )
        ]

    def interpret(self, request: PoolRequest, replies: Sequence[Reply]) -> Embeddings:
        """The ragged embeddings of ``request``, from the replies of :meth:`calls` (one per call, in order).

        Args:
            request: The request the replies answer.
            replies: One reply per call of :meth:`calls`, in order.

        Returns:
            Ragged embeddings (one slice per item, in request order) in the request's transfer dtype -- or a
            single-vector buffer when the server answered one vector per item and reported no usage (the
            served task was not ``token_embed``; the layout of the answer, not the config, decides). A reply
            that reports ``usage`` -- every vLLM JSON reply does -- is refused when its token counts
            contradict one vector per item, because a ``token_embed`` answer has one vector per prompt token.

        Raises:
            CapabilityError: An input was longer than the engine's context (HTTP 400 naming it).
            RequestRejectedError: The endpoint refused this request for any other reason.
            ProviderError: A reply is not a ``/pooling`` answer (no ``data``), its items do not match the
                request, a frame does not fit the declared ``dim``, or the decoded token counts disagree with
                the reply's own ``usage``.
            ConfigError: A base64 frame arrived with no ``dim`` declared on the request, or the request
                names an ``embed_dtype`` the adapter cannot decode.
            NotImplementedError: A ``bytes_only`` reply arrived: it sends no framing and cannot be split
                (lane L7 pins the framing).
        """
        if not request.contents:
            return Embeddings.empty(0, multi_vector=True, dtype=request.embed_dtype)
        expected = (
            [1] * len(request.contents) if any(c.has_media for c in request.contents) else [len(request.contents)]
        )
        if len(replies) != len(expected):
            raise ProviderError(
                f"the pooling endpoint answered {len(replies)} reply(ies) for {len(expected)} request(s); "
                "one reply per call, in order, is the transport's contract"
            )

        arrays: list[np.ndarray] = []
        for reply, count in zip(replies, expected, strict=True):
            if reply.status != 200:
                raise self._refused(reply)
            arrays.extend(
                self._decode_reply(
                    reply,
                    expected_items=count,
                    embed_dtype=request.embed_dtype,
                    dim=request.dim,
                    outputs=request.outputs,
                )
            )
        if not arrays:
            return Embeddings.empty(0, multi_vector=True, dtype=request.embed_dtype)
        kinds = {array.ndim for array in arrays}
        if kinds == {1}:
            widths = {array.shape[0] for array in arrays}
            if len(widths) > 1:
                raise ProviderError(
                    f"the /pooling answer's one-vector items have different widths ({sorted(widths)}); "
                    "refusing to guess which pooling task ran"
                )
            return Embeddings.single(np.stack(arrays))
        if kinds == {2}:
            return Embeddings.ragged(arrays, dtype=request.embed_dtype)
        raise ProviderError(
            "the /pooling answer mixes one-vector and per-token items; refusing to guess which pooling task ran"
        )

    def usage(self, reply: Reply) -> TokenCount | None:
        """The prompt tokens one reply reports (``usage.prompt_tokens``), or ``None`` when absent or unreadable.

        A JSON reply reports them in its body; a bytes reply reports them in the framing metadata header. The
        accounting is advisory; the load-bearing cross-check of the usage against the decoded vectors lives in
        :meth:`interpret` and refuses a malformed usage instead of skipping it.
        """
        body = reply.body
        if isinstance(body, bytes):
            metadata = reply.headers.get("metadata")
            if metadata is None:
                return None
            try:
                prompt_tokens = (json.loads(metadata).get("usage") or {}).get("prompt_tokens")
            except (json.JSONDecodeError, AttributeError):
                return None
            return _as_token_count(prompt_tokens)
        if not isinstance(body, dict):
            return None
        return _as_token_count((body.get("usage") or {}).get("prompt_tokens"))

    # -- decoding ----------------------------------------------------------
    def _decode_reply(
        self, reply: Reply, *, expected_items: int, embed_dtype: str, dim: int | None, outputs: str = "per_token"
    ) -> list[np.ndarray]:
        """One reply into one array per item, in request order: JSON (float lists or base64) or framed bytes.

        ``outputs: per_chunk`` (a per-chunk multi-output model) skips the usage cross-check: several outputs
        per input do not map to prompt tokens (2g).
        """
        body = reply.body
        if isinstance(body, bytes):
            return self._decode_bytes_reply(reply, expected_items=expected_items, outputs=outputs)
        if not isinstance(body, dict) or not isinstance(body.get("data"), list):
            raise ProviderError(f"/pooling response has no 'data': {str(body)[:_MAX_MESSAGE_CHARS]}")
        if any(not isinstance(item, dict) for item in body["data"]):
            raise ProviderError(
                f"the /pooling reply's data holds entries that are not items: {str(body)[:_MAX_MESSAGE_CHARS]}"
            )
        items = sorted(body["data"], key=_response_index)
        if len(items) != expected_items:
            raise RequestRejectedError(
                f"the pooling endpoint returned {len(items)} item(s) for {expected_items} input(s); "
                "refusing to return misaligned vectors"
            )
        arrays = [self._decode_item(item.get("data"), embed_dtype=embed_dtype, dim=dim) for item in items]
        self._check_usage(body.get("usage"), arrays, outputs=outputs)
        return arrays

    def _decode_item(self, data: Any, *, embed_dtype: str, dim: int | None) -> np.ndarray:
        """One item's payload into a float array: a nested float list (shape as sent) or a flat base64 frame.

        The base64 frame is the ``embed_dtype`` array flattened
        (``vllm/utils/serial_utils.py::tensor2binary``), little-endian (the request sent
        ``endianness: "little"``); it is reshaped to ``(tokens, dim)`` from the declared width and checked to
        hold a whole number of vectors.
        """
        if isinstance(data, list):
            try:
                array = np.asarray(data, dtype=np.float32)
            except ValueError as exc:
                raise ProviderError(f"the /pooling float frame does not decode as an array: {exc}") from exc
            if array.ndim == 2 and dim is not None and int(array.shape[-1]) != dim:
                raise ProviderError(
                    f"the /pooling float frame's width {array.shape[-1]} does not match the declared dim "
                    f"{dim}; the endpoint config's dim disagrees with the served checkpoint"
                )
            return array
        if isinstance(data, str):
            if dim is None:
                raise ConfigError(
                    "the /pooling base64 frame is flat and carries no shape: set dim on the pooling endpoint "
                    "to the checkpoint's token-vector width",
                    hint="the frame needs the width to reshape to (tokens, dim); the float and bytes encodings "
                    "carry the shape themselves",
                )
            try:
                raw = base64.b64decode(data, validate=True)
            except (ValueError, TypeError) as exc:
                raise ProviderError(f"the /pooling base64 frame is not valid base64: {exc}") from exc
            if embed_dtype not in _FRAME_DTYPES:
                raise ConfigError(
                    f"embed_dtype {embed_dtype!r} has no NumPy frame dtype; the pooling endpoint speaks "
                    "float16 and float32"
                )
            try:
                flat = np.frombuffer(raw, dtype=_FRAME_DTYPES[embed_dtype])
            except ValueError as exc:
                raise ProviderError(f"the /pooling base64 frame does not decode in {embed_dtype}: {exc}") from exc
            if flat.size % dim:
                raise ProviderError(
                    f"the /pooling base64 frame holds {flat.size} value(s), not a multiple of the declared dim "
                    f"{dim}: the config's dim disagrees with the served checkpoint or the frame is corrupt"
                )
            return flat.reshape(-1, dim)
        raise ProviderError(f"unsupported /pooling data payload: {type(data).__name__}")

    def _decode_bytes_reply(self, reply: Reply, *, expected_items: int, outputs: str = "per_token") -> list[np.ndarray]:
        """A ``bytes`` reply: per-item frames split by the ``metadata`` header's ``start``/``end``/``shape``.

        The framing (``vllm/entrypoints/pooling/utils.py::encode_pooling_bytes``) is recorded here; the
        ``bytes_only`` variant sends no metadata and is refused, naming the lane that will pin it.
        """
        metadata_header = reply.headers.get("metadata")
        if metadata_header is None:
            raise NotImplementedError(
                "the /pooling 'bytes_only' encoding sends no per-item framing (start, end, shape) and cannot be "
                "split; wire framing over the transport is pinned with the schemas and transport work (lane L7)"
            )
        try:
            metadata = json.loads(metadata_header)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"the /pooling bytes framing metadata is not valid JSON: {exc}") from exc
        items = metadata.get("data") or []
        if any(not isinstance(item, dict) for item in items):
            raise ProviderError(f"the /pooling bytes framing metadata is incomplete: {metadata!r}")
        items = sorted(items, key=_response_index)
        if len(items) != expected_items:
            raise RequestRejectedError(
                f"the pooling endpoint framed {len(items)} item(s) for {expected_items} input(s); "
                "refusing to return misaligned vectors"
            )
        arrays: list[np.ndarray] = []
        for item in items:
            try:
                frame_dtype = _frame_dtype(item.get("embed_dtype"), item.get("endianness"))
                start, end, shape = (
                    int(item["start"]),
                    int(item["end"]),
                    tuple(int(edge) for edge in item["shape"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ProviderError(f"the /pooling bytes framing metadata is incomplete: {item!r}") from exc
            needed = math.prod(shape) * frame_dtype.itemsize
            fits = 0 <= start <= end <= len(reply.body) and all(edge > 0 for edge in shape) and needed == end - start
            if not fits:
                raise ProviderError(
                    f"the /pooling bytes framing metadata does not fit its frame: shape {shape} needs "
                    f"{needed} byte(s) inside a body of {len(reply.body)}, the framing allots {end - start}"
                )
            arrays.append(np.frombuffer(reply.body[start:end], dtype=frame_dtype).reshape(shape))
        self._check_usage(metadata.get("usage"), arrays, outputs=outputs)
        return arrays

    def _check_usage(self, usage: Any, arrays: Sequence[np.ndarray], *, outputs: str = "per_token") -> None:
        """The decoded token counts must sum to the reply's own ``usage.prompt_tokens``.

        A ``token_embed`` answer has one vector per prompt token, so this catches a mistyped ``dim`` (every
        vector would be silently mis-shaped) and a server that answered a pooled task after all. A usage the
        reply cannot honestly report (a string, a dict, a null) is a malformed reply, never a reason to skip
        the check. A recipe that declares ``outputs: per_chunk`` (2g) opts out: several outputs per input
        (one vector per chunk), so the token count cross-check cannot apply.
        """
        if outputs == "per_chunk":
            return
        if not isinstance(usage, dict) or "prompt_tokens" not in usage:
            return
        reported = usage["prompt_tokens"]
        if isinstance(reported, bool) or not isinstance(reported, int):
            raise ProviderError(
                f"the /pooling reply reports a malformed usage ({reported!r}); the token counts cannot be cross-checked"
            )
        decoded = _decoded_tokens(arrays)
        if decoded != reported:
            raise ProviderError(
                f"the /pooling reply reports {reported} prompt token(s) but its vectors decode to {decoded}; "
                "a token_embed answer has one vector per prompt token, so the declared dim disagrees with the "
                "served checkpoint (or the server ignored the token_embed task)",
                hint="check the endpoint config's dim against the checkpoint's late-interaction width",
            )

    def _refused(self, reply: Reply) -> Exception:
        """The error a refused request (anything but HTTP 200) maps onto.

        Over-length inputs are the caller's budget problem and name the fix (HTTP 400; the text-budget
        mechanism applies the cut once it is wired); every other refusal belongs to this one request.
        """
        message = _error_message(reply)
        if reply.status in (400, 422) and any(phrase in message.lower() for phrase in _OVERLENGTH_PHRASES):
            return CapabilityError(
                f"the pooling endpoint rejected an input longer than its context: {message}",
                hint="declare the endpoint config's text budget (tokenizer and max_tokens), so the client cuts "
                "the content before sending; the inputs it shipped exceeded the served context",
            )
        return RequestRejectedError(f"the pooling endpoint refused the request (HTTP {reply.status}): {message}")


def _frame_dtype(name: Any, endianness: Any) -> np.dtype:
    """The NumPy dtype a framed item's ``embed_dtype``/``endianness`` metadata names.

    ``native`` is read as little: the adapter always requests ``endianness: "little"``, and every platform
    vLLM serves is little-endian, so a frame recorded as native decodes the same either way.
    """
    if name not in ("float16", "float32"):
        raise ProviderError(
            f"the /pooling frame names embed_dtype {name!r}, which has no NumPy dtype; the adapter speaks "
            "float16 and float32"
        )
    if endianness == "big":
        return _FRAME_DTYPES[name].newbyteorder()
    return _FRAME_DTYPES[name]


__all__ = ["VllmPooling"]
