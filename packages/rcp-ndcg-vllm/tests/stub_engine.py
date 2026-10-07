"""A stub vLLM engine for CPU tests: stdlib ``http.server`` + numpy, deterministically.

It serves ``/v1/models``, ``/v1/embeddings`` (float and base64), ``/pooling`` (float, base64 and raw-bytes) and
``/rerank`` and ``/score`` from the same SHA-256-seeded numbers as the fixture references
(``tests/fixtures/deterministic.py``), so a clean stub equals its recipe's reference exactly and stage 2 passes;
``--noise`` adds seeded noise above the gates.  vLLM's own flags are accepted and ignored, so
``rcp_ndcg_vllm.jobs.run_wave --vllm-cmd`` can drive it with a recipe's real ``serve_argv``.

Like vLLM v0.31.0 it honours ``truncate_prompt_tokens`` with ``truncation_side`` (an engine-side cut of the prompt,
counted in the stub's whitespace tokens) on every role route and ``use_activation`` on ``/rerank`` (``false``: the
raw logit of the probability; the default comes from ``--pooler-config``'s ``use_activation``, else true).  Two
flags describe the emulated MODEL, so the negative controls can break it the way a real checkpoint breaks:
``--model-pooling NAME`` (the pooling the checkpoint was trained with -- serving another ``seq_pooling_type`` or
``pooling_type`` yields other numbers) and ``--model-needs-template`` (a reranker scored without its served
``--chat-template`` sees the unframed spans and scores differently).

Chat-shaped requests (``messages`` on ``/v1/embeddings`` and ``/pooling``) go through vLLM's chat path: one
conversation (or a list of them, a batch), its parts handed to the chat template as vLLM hands them
(``image_url`` as ``{"type": "image"}``), rendered with the served ``--chat-template`` -- else the emulated
checkpoint's own (``--model-chat-template``), else the text parts joined -- under the request's
``add_generation_prompt`` (false by default, as vLLM's).  Media parts (and ``/rerank``'s ``{"content": [...]}``
sides) are decoded and resized as the engine's processor resizes them: ``smart_resize`` with the emulated
checkpoint's patch factor (``--model-image-factor``) under ``--mm-processor-kwargs``'s pixel pin (nested
``images_kwargs`` or flat), else the checkpoint's own default budget (``--model-image-pixels MIN,MAX``) -- so
an unpinned engine re-resizes an image whose prepared size lies outside that default.  Every chat-shaped
reply carries ``usage.prompt_tokens``: the rendered text's tokens plus each image's merged patch tokens and
its two vision markers; more images than ``--limit-mm-per-prompt`` allows, a video container (the stub
decodes none) or an undecodable image are a 400, as the engine refuses them.

With ``--port 0`` the stub binds an ephemeral port and prints ``RCPS_STUB_PORT=<n>`` on stdout; the wave runner
reads that line instead of guessing a port.  An input longer than ``--max-model-len`` tokens gets vLLM's
over-length 400; an unknown request field gets a 400 validation body, like the error bodies the adapters map.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))

from deterministic import score, token_vectors, tokens, vector  # noqa: E402

_CUT = {"truncate_prompt_tokens", "truncation_side"}
_CHAT = {"messages", "add_special_tokens", "add_generation_prompt"}
_ALLOWED: dict[str, set[str]] = {
    "/embeddings": {"input", "encoding_format", "dimensions", *_CHAT, *_CUT},
    "/pooling": {"input", "task", "encoding_format", "embed_dtype", "endianness", *_CHAT, *_CUT},
    "/rerank": {"query", "documents", "top_n", "instruction", "use_activation", *_CUT},
    "/score": {"queries", "documents", "query", "text_1", "text_2"},
}
"""Keyed on the normalised route: ``do_POST`` strips ``/v1/`` before dispatch."""

_ARGS = argparse.Namespace(
    served_model_name="stub",
    max_model_len=512,
    noise=0.0,
    tokenizer="",
    pooler_config="{}",
    chat_template=None,
    model_pooling=None,
    model_needs_template=False,
    model_chat_template=None,
    mm_processor_kwargs="{}",
    limit_mm_per_prompt="{}",
    model_image_factor=28,
    model_image_pixels="3136,12845056",
)


def _pooler() -> dict[str, Any]:
    try:
        value = json.loads(_ARGS.pooler_config or "{}")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _wrong_pooling() -> bool:
    """Whether the served pooling differs from the emulated checkpoint's (``--model-pooling``)."""
    if not _ARGS.model_pooling:
        return False
    pooler = _pooler()
    served = pooler.get("seq_pooling_type") or pooler.get("pooling_type")
    return served is not None and str(served).upper() != str(_ARGS.model_pooling).upper()


def _cut(text: str, body: dict[str, Any]) -> str:
    """vLLM's engine-side cut: the first (``right``) or last (``left``) ``truncate_prompt_tokens`` tokens of the
    prompt, counted with the engine's tokenizer (``--tokenizer``; whitespace words without one)."""
    limit = body.get("truncate_prompt_tokens")
    if limit is None or int(limit) < 0:
        return text
    limit = int(limit)
    side = body.get("truncation_side", "right")
    if _ARGS.tokenizer:
        offsets = _get_tokenizer_backend().encode(text, add_special_tokens=False).offsets
        if len(offsets) <= limit:
            return text
        return text[: offsets[limit - 1][1]] if side == "right" else text[offsets[-limit][0] :]
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit] if side == "right" else words[-limit:])


_TOKENIZER_BACKEND: Any = None


def _get_tokenizer_backend() -> Any:
    """The engine's own tokenizer (the tokenizers library, the image's transformers), loaded lazily."""
    global _TOKENIZER_BACKEND
    if _TOKENIZER_BACKEND is None:
        from tokenizers import Tokenizer

        _TOKENIZER_BACKEND = Tokenizer.from_file(_ARGS.tokenizer)
    return _TOKENIZER_BACKEND


class _BadRequest(ValueError):
    """A stub-level stand-in for vLLM's 400 validation errors."""


def _inputs(body: dict[str, Any]) -> list[str]:
    """The request's inputs, as a list of strings; other media types are refused like the engine would."""
    value = body.get("input", body.get("text_1"))
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise _BadRequest("the stub serves string inputs only")


def _json_flag(value: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _engine_pixels() -> tuple[int, int]:
    """The pixel budget the engine resizes images to: the served pin (nested ``images_kwargs`` first, as vLLM
    overlays it for the image modality, else flat keys), else the emulated checkpoint's own default."""
    kwargs = _json_flag(_ARGS.mm_processor_kwargs)
    scoped = kwargs.get("images_kwargs") if isinstance(kwargs.get("images_kwargs"), dict) else {}
    default_min, default_max = (int(value) for value in str(_ARGS.model_image_pixels).split(","))
    low = scoped.get("min_pixels", kwargs.get("min_pixels", default_min))
    high = scoped.get("max_pixels", kwargs.get("max_pixels", default_max))
    return int(low), int(high)


def _image(url: str) -> tuple[int, str]:
    """One image part as the engine reads it: its prompt tokens (merged patches plus the two vision markers)
    and its resized geometry ``<width>x<height>``."""
    import io

    from PIL import Image
    from rcp_ndcg.data.resolution import smart_resize

    if not url.startswith("data:") or "," not in url:
        raise _BadRequest("the stub reads inline data: images only")
    try:
        with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as handle:
            handle.load()
            width, height = handle.size
    except Exception as error:  # noqa: BLE001 - an undecodable image is the engine's 400
        raise _BadRequest(f"cannot load the image: {type(error).__name__}") from None
    factor = int(_ARGS.model_image_factor)
    low, high = _engine_pixels()
    try:
        resized_h, resized_w = smart_resize(height, width, factor=factor, min_pixels=low, max_pixels=high)
    except Exception as error:  # noqa: BLE001 - the processor refuses an extreme aspect ratio
        raise _BadRequest(str(error)) from None
    return (resized_h // factor) * (resized_w // factor) + 2, f"{resized_w}x{resized_h}"


def _parts(content: Any) -> list[dict[str, Any]]:
    """A message's or a rerank side's content as a list of parts (a string is one text part)."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, dict) and isinstance(content.get("content"), list):
        content = content["content"]
    return [{"type": "text", "text": part} if isinstance(part, str) else part for part in content or []]


def _media(parts: list[dict[str, Any]]) -> tuple[int, list[str]]:
    """The media tokens and resized geometries of ``parts``, refused as the engine refuses them."""
    tokens_total, shapes = 0, []
    images = [part for part in parts if part.get("type") == "image_url"]
    limit = _json_flag(_ARGS.limit_mm_per_prompt).get("image")
    if limit is not None and len(images) > int(limit):
        raise _BadRequest(f"At most {limit} image(s) may be provided in one prompt, got {len(images)}")
    for part in parts:
        kind = part.get("type")
        if kind == "image_url":
            count, shape = _image(str((part.get("image_url") or {}).get("url", "")))
            tokens_total += count
            shapes.append(shape)
        elif kind == "video_url":
            raise _BadRequest("the stub decodes no video containers")
    return tokens_total, shapes


def _count(text: str, *, add_special_tokens: bool = False) -> int:
    """The engine's token count of a text: its tokenizer when given, the whitespace words otherwise."""
    if _ARGS.tokenizer:
        return len(_get_tokenizer_backend().encode(text, add_special_tokens=add_special_tokens).ids)
    return len(tokens(text))


def _chat_prompts(body: dict[str, Any]) -> list[tuple[str, int, str, int]]:
    """vLLM's chat path over a ``messages`` body: per conversation, the rendered prompt (the template sees each
    media part as its modality), the prompt tokens (text plus media), the model input key the numbers are
    drawn from (the prompt, then every media part's resized geometry in order) and the media tokens alone."""
    from rcp_ndcg_vllm.equivalence.stages import engine_conversation

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise _BadRequest("messages must be a non-empty list")
    conversations = messages if all(isinstance(entry, list) for entry in messages) else [messages]
    path = _ARGS.chat_template or _ARGS.model_chat_template
    out = []
    for conversation in conversations:
        parts = [part for message in conversation for part in _parts(message.get("content"))]
        media_tokens, shapes = _media(parts)
        if path:
            from jinja2.sandbox import ImmutableSandboxedEnvironment

            template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(
                Path(path).read_text(encoding="utf-8")
            )
            prompt = template.render(
                messages=engine_conversation(conversation),
                add_generation_prompt=bool(body.get("add_generation_prompt", False)),
                tools=None,
            )
        else:
            prompt = "\n".join(str(part.get("text", "")) for part in parts if part.get("type") == "text")
        flag = bool(body.get("add_special_tokens", False))
        key = prompt + "".join(f"\x00{shape}" for shape in shapes)
        out.append((prompt, _count(prompt, add_special_tokens=flag) + media_tokens, key, media_tokens))
    return out


class _Handler(BaseHTTPRequestHandler):
    """The routes; every answer is deterministic in the request's texts."""

    server_version = "rcp-ndcg-stub-engine/0.0.1"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
        print(f"[stub-engine] {format % args}", flush=True)

    # -- plumbing -----------------------------------------------------------------------------------------------
    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("server", self.server_version)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_raw(self, payload: bytes) -> None:
        self.send_response(200)
        self.send_header("content-type", "application/octet-stream")
        self.send_header("server", self.server_version)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        payload: Any = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(payload, dict):
            raise _BadRequest("the request body must be a JSON object")
        return payload

    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        route = self.path.split("?")[0]
        if route in ("/v1/models", "/models"):
            self._send_json({"object": "list", "data": [{"id": _ARGS.served_model_name, "object": "model"}]})
        else:
            self._send_json({"error": {"message": "unknown route"}}, status=404)

    def do_POST(self) -> None:  # noqa: N802 - stdlib name
        route = self.path.split("?")[0]
        # The product's adapters use the un-prefixed paths (vLLM's own): normalise.
        if route.startswith("/v1/"):
            route = route[len("/v1") :]
            self.path = route
        try:
            body = self._body()
            if route == "/tokenize":
                self._tokenize(body)
                return
            self._reject_unknown(route, body)
            if route == "/embeddings":
                self._embeddings(body)
            elif route == "/pooling":
                self._pooling(body)
            elif route == "/rerank":
                self._rerank(body)
            elif route == "/score":
                self._score(body)
            else:
                self._send_json({"error": {"message": f"unknown route {route}"}}, status=404)
        except _BadRequest as error:
            self._send_json({"error": {"message": str(error), "type": "invalid_request_error"}}, status=400)
        except Exception as error:  # noqa: BLE001 - the stub logs and returns 500
            import traceback

            self._send_json(
                {"error": {"message": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()[-500:]}},
                status=500,
            )

    def _tokenize(self, body: dict[str, Any]) -> None:
        """The engine's tokenization (R29): the ids the engine itself reads the prompt as."""
        text = body.get("prompt", body.get("text"))
        if not isinstance(text, str):
            raise _BadRequest("tokenize needs a prompt")
        flag = bool(body.get("add_special_tokens", False))
        ids = _get_tokenizer_backend().encode(text, add_special_tokens=flag).ids
        self._send_json({"tokens": ids, "count": len(ids)})

    # -- routes --------------------------------------------------------------------------------------------------
    def _embeddings(self, body: dict[str, Any]) -> None:
        """One dense float32 vector per input text (or per conversation, framed by the chat path), float or
        base64; a chat-shaped request's reply carries its prompt tokens."""
        usage = None
        if "messages" in body:
            chat = _chat_prompts(body)
            inputs = [key for _, _, key, _ in chat]
            self._check_length([prompt for prompt, _, _, _ in chat])
            usage = sum(count for _, count, _, _ in chat)
        else:
            inputs = _inputs(body)
            self._check_length(inputs)
        fmt = body.get("encoding_format", "float")
        data = []
        tag = "embed-wrong-pooling" if _wrong_pooling() else "embed"
        for index, text in enumerate(inputs):
            value = vector(_cut(text, body), tag, noise=_ARGS.noise).astype(np.float32)
            if fmt == "base64":
                data.append({"index": index, "embedding": base64.b64encode(value.tobytes()).decode("ascii")})
            elif fmt == "float":
                data.append({"index": index, "embedding": [float(x) for x in value]})
            else:
                raise _BadRequest(f"encoding_format {fmt!r} is not served by the stub")
        reply: dict[str, Any] = {"object": "list", "data": data}
        if usage is not None:
            reply["usage"] = {"prompt_tokens": usage, "total_tokens": usage}
        self._send_json(reply)

    def _pooling(self, body: dict[str, Any]) -> None:
        """One vector per whitespace token per input text, in float, base64 (with shape) or raw bytes; a
        chat-shaped request yields one vector per prompt token (its media tokens included) and its usage."""
        if "messages" in body:
            self._pooling_chat(body)
            return
        inputs = _inputs(body)
        self._check_length(inputs)
        fmt = body.get("encoding_format", "float")
        dtype = body.get("embed_dtype", "float16")
        code = {"float16": "<f2", "float32": "<f4"}.get(str(dtype))
        if code is None:
            raise _BadRequest(f"embed_dtype {dtype!r} is not served by the stub")
        inputs = [_cut(text, body) for text in inputs]
        if fmt == "bytes":
            concatenated = b"".join(
                token_vectors(text, "tok", noise=_ARGS.noise).astype(code).tobytes() for text in inputs
            )
            self._send_raw(concatenated)
            return
        data = []
        for index, text in enumerate(inputs):
            matrix = token_vectors(text, "tok", noise=_ARGS.noise).astype(code)
            if fmt == "base64":
                data.append(
                    {
                        "index": index,
                        "data": base64.b64encode(matrix.reshape(-1).tobytes()).decode("ascii"),
                        "shape": list(matrix.shape),
                    }
                )
            elif fmt == "float":
                data.append({"index": index, "data": matrix.tolist()})
            else:
                raise _BadRequest(f"encoding_format {fmt!r} is not served by the stub")
        self._send_json({"object": "list", "data": data})

    def _pooling_chat(self, body: dict[str, Any]) -> None:
        """A chat-shaped ``/pooling`` request: per conversation, one vector per prompt token -- the rendered
        text's whitespace tokens, then one per media token -- with ``usage.prompt_tokens`` their count."""
        dtype = body.get("embed_dtype", "float16")
        code = {"float16": "<f2", "float32": "<f4"}.get(str(dtype))
        if code is None:
            raise _BadRequest(f"embed_dtype {dtype!r} is not served by the stub")
        data, total = [], 0
        for index, (prompt, _, key, media_tokens) in enumerate(_chat_prompts(body)):
            rows = [vector(word, "tok", noise=_ARGS.noise) for word in tokens(prompt)]
            rows += [vector(f"{key}\x00media{position}", "tok", noise=_ARGS.noise) for position in range(media_tokens)]
            if not rows:
                raise _BadRequest("the prompt is empty")
            matrix = np.stack(rows).astype(code)
            total += len(rows)
            data.append(
                {
                    "index": index,
                    "data": base64.b64encode(matrix.reshape(-1).tobytes()).decode("ascii"),
                    "shape": list(matrix.shape),
                }
            )
        self._send_json({"object": "list", "data": data, "usage": {"prompt_tokens": total, "total_tokens": total}})

    def _rerank(self, body: dict[str, Any]) -> None:
        """The Cohere shape: results sorted by relevance score, one per document."""
        query = body.get("query")
        documents = body.get("documents")
        if not isinstance(query, (str, dict)) or not isinstance(documents, list):
            raise _BadRequest("rerank needs a query and documents")
        # A side carrying media is ``{"content": [parts]}``: its text parts joined, its media read as the
        # engine reads them (the usage counts them; the score is drawn over the text and the geometry).
        query_text, query_tokens, query_key = self._side(query)
        sides = [self._side(document) for document in documents]
        self._check_length([query_text] + [text for text, _, _ in sides])
        scores = [self._pair_score(query_key, key, body) for _, _, key in sides]
        order = sorted(range(len(documents)), key=lambda index: scores[index], reverse=True)
        results = [
            {"index": index, "document": {"text": sides[index][0]}, "relevance_score": scores[index]}
            for index in order
        ]
        prompt_tokens = sum(
            _count(query_text) + query_tokens + _count(text) + media_tokens for text, media_tokens, _ in sides
        )
        self._send_json(
            {
                "object": "list",
                "results": results,
                "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
            }
        )

    @staticmethod
    def _side(value: Any) -> tuple[str, int, str]:
        """One rerank side: its text, its media tokens and the key its score is drawn over (the text alone for a
        plain string, so a text-only pair scores exactly as before)."""
        if isinstance(value, str):
            return value, 0, value
        parts = _parts(value)
        text = "\n".join(str(part.get("text", "")) for part in parts if part.get("type") == "text")
        media_tokens, shapes = _media(parts)
        return text, media_tokens, text + "".join(f"\x00{shape}" for shape in shapes)

    def _pair_score(self, query: str, document: str, body: dict[str, Any]) -> float:
        """One pair's score as vLLM serves it: the engine-side cut, the template, the pooling, the activation."""
        if body.get("truncate_prompt_tokens") is not None:
            kept = _cut(f"{query} {document}", body)
            document = kept[len(query) + 1 :] if len(kept) > len(query) else ""
        if _ARGS.model_needs_template and not _ARGS.chat_template:
            query, document = "", f"{query} {document}"  # the unframed spans: no template separates them
        if _wrong_pooling():
            query = f"{query}\x00wrong-pooling"
        value = score(query, document, noise=_ARGS.noise)
        activation = body.get("use_activation", _pooler().get("use_activation", True))
        if activation is False:
            clipped = min(max(value, 1e-6), 1 - 1e-6)
            return float(np.log(clipped / (1 - clipped)))
        return value

    def _score(self, body: dict[str, Any]) -> None:
        """One score per (query, document) pair, in request order."""
        queries = body.get("queries", [body.get("query")])
        documents = body.get("documents", body.get("text_2"))
        if not isinstance(queries, list) or not isinstance(documents, list):
            raise _BadRequest("score needs queries and documents")
        data = [
            {"index": i, "score": score(str(query), str(document), noise=_ARGS.noise)}
            for i, (query, document) in enumerate(zip(queries, documents, strict=False))
        ]
        self._send_json({"object": "list", "data": data})

    # -- rules ----------------------------------------------------------------------------------------------------
    def _check_length(self, inputs: list[str]) -> None:
        """The over-length rule: token count above --max-model-len is the recorded HTTP 400."""
        for text in inputs:
            if len(tokens(text)) > _ARGS.max_model_len:
                raise _BadRequest(
                    f"This model's maximum context length is {_ARGS.max_model_len} tokens, "
                    f"however you requested {len(tokens(text))} tokens"
                )

    def _reject_unknown(self, route: str, body: dict[str, Any]) -> None:
        """Refuse request fields the route does not know, like a strict protocol model."""
        allowed = _ALLOWED.get(route)
        if allowed is None:
            return
        unknown = sorted(set(body) - allowed - {"model"})
        if unknown:
            raise _BadRequest(f"unknown field(s) {unknown} for {route}")


def main(argv: list[str] | None = None) -> int:
    """Bind, announce (``RCPS_STUB_PORT=``) and serve until interrupted."""
    global _ARGS
    parser = argparse.ArgumentParser(prog="stub_engine.py", description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--served-model-name", default="stub")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--pooler-config", default="{}")
    parser.add_argument("--chat-template", default=None)
    parser.add_argument("--model-pooling", default=None)
    parser.add_argument("--model-needs-template", action="store_true")
    parser.add_argument("--model-chat-template", default=None)
    parser.add_argument("--mm-processor-kwargs", default="{}")
    parser.add_argument("--limit-mm-per-prompt", default="{}")
    parser.add_argument("--model-image-factor", type=int, default=28)
    parser.add_argument("--model-image-pixels", default="3136,12845056")
    args, _unknown = parser.parse_known_args(argv)
    _ARGS = args
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    port = server.server_address[1]
    print(f"RCPS_STUB_PORT={port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
