"""The engine client: the wire calls the harness and the recorder make, in plain ``httpx``.

Routes and shapes (the engine is a vLLM server):

- ``GET  /v1/models`` — the served model list.
- ``POST /v1/embeddings`` — ``{model, input, encoding_format}``; one dense vector per text (float or base64).
- ``POST /pooling`` — ``{model, input, task: token_embed, encoding_format, embed_dtype}``; one vector per token.
- ``POST /rerank`` — the Cohere shape ``{model, query, documents, top_n}``; ``top_n`` equals the number of
  documents, and the vLLM extensions (``instruction``, ``use_activation``) travel only when the recipe sets them.
- ``POST /score`` — ``{model, queries, documents}``, one score per (query, document) pair in order.

A base URL may carry a ``/v1`` suffix or not; it is normalised to the engine root and every route is joined from
there.  Non-200 answers raise :class:`~rcp_ndcg_vllm.errors.HarnessError` with the status and a truncated body.
"""

from __future__ import annotations

import base64
from typing import Any

import httpx
import numpy as np

from ..errors import HarnessError
from ..recipe import Recipe, effective_embed_dtype

__all__ = ["EngineClient", "base64_to_array", "fold_instruction"]

_TIMEOUT_S = 300.0
_BODY_SNIPPET = 500
_DTYPE_CODES = {"float16": "<f2", "float32": "<f4"}


def fold_instruction(instruction: str | None, query: str) -> str:
    """The fold rule of ``client.instruction: fold``: the instruction, one newline, the query.

    This is the one documented fold format: the recipe's template renders ``{{ query }}`` with the folded text,
    and the reference folds the same way.
    """
    if not instruction:
        return query
    return f"{instruction}\n{query}"


def base64_to_array(payload: str, dtype: str) -> np.ndarray:
    """Decode one base64-encoded vector payload (little-endian ``float16`` or ``float32``) into a flat array."""
    code = _DTYPE_CODES.get(dtype)
    if code is None:
        raise HarnessError(f"unknown embed_dtype {dtype!r}; expected float16 or float32")
    return np.frombuffer(base64.b64decode(payload), dtype=code).astype(np.float32)


class EngineClient:
    """A thin httpx client over one served engine, bound to one recipe; every method returns decoded values only."""

    def __init__(
        self,
        recipe: Recipe,
        base_url: str,
        *,
        served_model_name: str | None = None,
        timeout_s: float = _TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        root = base_url.rstrip("/")
        if root.endswith(("/v1", "/v2")):
            root = root.rsplit("/", 1)[0]
        self.root = root
        self.recipe = recipe
        self.model = served_model_name or recipe.id
        self.embed_dtype = effective_embed_dtype(recipe)
        self._http = httpx.Client(base_url=self.root, timeout=timeout_s, transport=transport)

    def close(self) -> None:
        """Release the connection pool."""
        self._http.close()

    def __enter__(self) -> EngineClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _post(self, route: str, payload: dict[str, Any]) -> httpx.Response:
        try:
            return _checked(self._http.post(route, json={"model": self.model, **payload}), route)
        except httpx.HTTPError as error:
            raise HarnessError(f"the engine at {self.root} failed {route}: {error}") from error

    def models(self) -> dict[str, Any]:
        """``GET /v1/models``: the served model list."""
        try:
            return _checked(self._http.get("/v1/models"), "/v1/models").json()
        except HarnessError:
            raise
        except httpx.HTTPError as error:
            raise HarnessError(f"the engine at {self.root} failed /v1/models: {error}") from error

    def embeddings(self, texts: list[str], *, encoding_format: str = "float") -> list[np.ndarray]:
        """``POST /v1/embeddings``: one dense float32 vector per text."""
        data = self._post("/v1/embeddings", {"input": texts, "encoding_format": encoding_format}).json()["data"]
        vectors: list[np.ndarray] = []
        for item in sorted(data, key=lambda entry: entry["index"]):
            value = item["embedding"]
            if isinstance(value, str):
                vectors.append(base64_to_array(value, "float32"))
            else:
                vectors.append(np.asarray(value, dtype=np.float32))
        return vectors

    def pooling(
        self, texts: list[str], *, task: str = "token_embed", encoding_format: str = "base64"
    ) -> list[tuple[np.ndarray, str, list[int] | None]]:
        """``POST /pooling``: per text, a ``(flat values, dtype, shape)`` triple.

        The values are the raw flat buffer in the recipe's transfer precision; the harness reshapes with the
        response's ``shape`` metadata when present, else against the reference's token count.  ``bytes`` framing is
        recorded, never decoded here: the harness always asks for ``base64``.
        """
        body: dict[str, Any] = {
            "input": texts,
            "task": task,
            "encoding_format": encoding_format,
            "embed_dtype": self.embed_dtype,
        }
        data = self._post("/pooling", body).json()["data"]
        out: list[tuple[np.ndarray, str, list[int] | None]] = []
        for item in sorted(data, key=lambda entry: entry["index"]):
            payload = item.get("data", item.get("embedding"))
            if isinstance(payload, str):
                values = base64_to_array(payload, self.embed_dtype)
            elif isinstance(payload, list):
                values = np.asarray(payload, dtype=np.float32)
            else:
                raise HarnessError(f"/pooling returned an undecodable payload of type {type(payload).__name__}")
            shape = item.get("shape")
            out.append((values, self.embed_dtype, [int(s) for s in shape] if shape else None))
        return out

    def rerank(
        self, query: str, documents: list[str], *, instruction: str | None = None, use_activation: bool | None = None
    ) -> list[float]:
        """``POST /rerank`` (Cohere shape): one relevance score per document, in the documents' order."""
        body: dict[str, Any] = {"query": query, "documents": documents, "top_n": len(documents)}
        if instruction:
            body["instruction"] = instruction
        if use_activation is not None:
            body["use_activation"] = use_activation
        results = self._post("/rerank", body).json()["results"]
        scores = [0.0] * len(documents)
        for entry in results:
            scores[int(entry["index"])] = float(entry["relevance_score"])
        return scores

    def score(self, queries: list[str], documents: list[str]) -> list[float]:
        """``POST /score``: one score per (query, document) pair, in request order."""
        data = self._post("/score", {"queries": queries, "documents": documents}).json()["data"]
        scores = [0.0] * len(data)
        for entry in data:
            scores[int(entry["index"])] = float(entry["score"])
        return scores


def _checked(response: httpx.Response, route: str) -> httpx.Response:
    if response.status_code != 200:
        raise HarnessError(f"{route} returned HTTP {response.status_code}: {response.text[:_BODY_SNIPPET]}")
    return response
