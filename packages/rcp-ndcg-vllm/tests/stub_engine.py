"""A stub vLLM engine for CPU tests: stdlib ``http.server`` + numpy, deterministically.

It serves ``/v1/models``, ``/v1/embeddings`` (float and base64), ``/pooling`` (float, base64 and raw-bytes) and
``/rerank`` and ``/score`` from the same SHA-256-seeded numbers as the fixture references
(``tests/fixtures/deterministic.py``), so a clean stub equals its recipe's reference exactly and stage 2 passes;
``--noise`` adds seeded noise above the gates.  vLLM's own flags are accepted and ignored, so
``rcp_ndcg_vllm.jobs.run_wave --vllm-cmd`` can drive it with a recipe's real ``serve_argv``.

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

_ALLOWED: dict[str, set[str]] = {
    "/v1/embeddings": {"input", "encoding_format"},
    "/pooling": {"input", "task", "encoding_format", "embed_dtype", "endianness"},
    "/rerank": {"query", "documents", "top_n", "instruction", "use_activation"},
    "/score": {"queries", "documents", "query", "text_1", "text_2"},
}

_ARGS = argparse.Namespace(served_model_name="stub", max_model_len=512, noise=0.0, tokenizer="")
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
            return
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
        """One dense float32 vector per input text, float or base64."""
        inputs = _inputs(body)
        self._check_length(inputs)
        fmt = body.get("encoding_format", "float")
        data = []
        for index, text in enumerate(inputs):
            value = vector(text, "embed", noise=_ARGS.noise).astype(np.float32)
            if fmt == "base64":
                data.append({"index": index, "embedding": base64.b64encode(value.tobytes()).decode("ascii")})
            elif fmt == "float":
                data.append({"index": index, "embedding": [float(x) for x in value]})
            else:
                raise _BadRequest(f"encoding_format {fmt!r} is not served by the stub")
        self._send_json({"object": "list", "data": data})

    def _pooling(self, body: dict[str, Any]) -> None:
        """One vector per whitespace token per input text, in float, base64 (with shape) or raw bytes."""
        inputs = _inputs(body)
        fmt = body.get("encoding_format", "float")
        dtype = body.get("embed_dtype", "float16")
        code = {"float16": "<f2", "float32": "<f4"}.get(str(dtype))
        if code is None:
            raise _BadRequest(f"embed_dtype {dtype!r} is not served by the stub")
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

    def _rerank(self, body: dict[str, Any]) -> None:
        """The Cohere shape: results sorted by relevance score, one per document."""
        query = body.get("query")
        documents = body.get("documents")
        if not isinstance(query, str) or not isinstance(documents, list):
            raise _BadRequest("rerank needs a query and documents")
        scores = [score(query, str(document), noise=_ARGS.noise) for document in documents]
        order = sorted(range(len(documents)), key=lambda index: scores[index], reverse=True)
        results = [
            {"index": index, "document": {"text": documents[index]}, "relevance_score": scores[index]}
            for index in order
        ]
        self._send_json({"object": "list", "results": results})

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
