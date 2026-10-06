"""The two test-local fake engines: a rerank fake and a pooling fake for the tests' fixture recipes.

They implement :class:`rcp_ndcg_test.fakes.FakeEngine` -- the same protocol the shipped embed fake uses
-- outside the package, which is the point: a recipe-level fake is a small, local object, and the
registry takes anything that satisfies the protocol. Both are deterministic surrogates (the same draw
function the product's fakes use), not verified emulators.
"""

from __future__ import annotations

import base64
import math
from pathlib import Path
from typing import Any

import numpy as np
from rcp_ndcg_test.fakes import FakeReply

from rcp_ndcg.data.tokenizer import load_tokenizer
from rcp_ndcg.inference.fake import fake_uniform

_TOKENIZER_PATH = str(Path(__file__).resolve().parent / "fixtures" / "tokenizer.json")


class _Deterministic:
    """The shared plumbing of the two test fakes: the tokenizer and the hash-seeded draws."""

    recipe_id = ""
    name = ""
    dim = 8

    def token_count(self, text: str) -> int:
        """The fixture tokenizer's count of ``text`` (at least one token, like the engines report)."""
        if not text:
            return 0
        return self.tokenizer.count(text)

    @property
    def tokenizer(self) -> Any:
        return load_tokenizer(_TOKENIZER_PATH)

    def unit_vector(self, *parts: object) -> list[float]:
        """One deterministic unit vector drawn from ``parts``."""
        raw = [fake_uniform(*parts, index) * 2.0 - 1.0 for index in range(self.dim)]
        norm = math.sqrt(sum(value * value for value in raw)) or 1.0
        return [float(value) for value in np.asarray([value / norm for value in raw], dtype="<f4")]


class FakeRerankEngine(_Deterministic):
    """The rerank fixture fake: ``POST .../rerank`` scores each document by a hash of the pair, best
    first (the wire returns ranked rows; the adapter realigns by index), and ``/tokenize`` answers with
    the fixture tokenizer."""

    recipe_id = "fake-rerank"
    name = "test-rerank-fake"

    def handle(self, method: str, path: str, body: Any) -> FakeReply:
        if method == "GET" and path.endswith("/models"):
            return FakeReply(200, {"object": "list", "data": [{"id": self.recipe_id, "owned_by": self.name}]})
        if method == "POST" and path.endswith("/rerank"):
            return self._rerank(body)
        if method == "POST" and path.endswith("/tokenize"):
            return self._tokenize(body)
        return FakeReply(404, {"error": {"message": f"the fake {self.name!r} has no route for {method} {path}"}})

    def _rerank(self, body: Any) -> FakeReply:
        if (
            not isinstance(body, dict)
            or not isinstance(body.get("query"), str)
            or not isinstance(body.get("documents"), list)
        ):
            return FakeReply(400, {"error": {"message": "the rerank fake takes {'query': str, 'documents': [str]}"}})
        query, documents = body["query"], list(body["documents"])
        scores = [fake_uniform(self.recipe_id, "relevance", query, document) for document in documents]
        rows = sorted(
            ({"index": index, "relevance_score": float(score)} for index, score in enumerate(scores)),
            key=lambda row: -row["relevance_score"],
        )
        top_n = body.get("top_n")
        if isinstance(top_n, int) and 0 <= top_n < len(rows):
            rows = rows[:top_n]
        return FakeReply(200, {"id": "fake", "model": self.recipe_id, "results": rows})

    def _tokenize(self, body: Any) -> FakeReply:
        prompt = body.get("prompt") if isinstance(body, dict) else None
        if not isinstance(prompt, str):
            return FakeReply(400, {"error": {"message": "the tokenize fake takes {'prompt': str}"}})
        ids = self.tokenizer.ids(prompt, add_special_tokens=bool(body.get("add_special_tokens", True)))
        return FakeReply(200, {"tokens": ids, "count": len(ids), "max_model_len": 512})


class FakePoolEngine(_Deterministic):
    """The pooling fixture fake: ``POST .../pooling`` answers ``task: token_embed`` in base64 float16
    (the framing the vllm_pooling adapter sends), one ragged ``(tokens, dim)`` matrix per input."""

    recipe_id = "fake-pool"
    name = "test-pool-fake"

    def handle(self, method: str, path: str, body: Any) -> FakeReply:
        if method == "GET" and path.endswith("/models"):
            return FakeReply(200, {"object": "list", "data": [{"id": self.recipe_id, "owned_by": self.name}]})
        if method == "POST" and path.endswith("/pooling"):
            return self._pooling(body)
        if method == "POST" and path.endswith("/tokenize"):
            return self._tokenize(body)
        return FakeReply(404, {"error": {"message": f"the fake {self.name!r} has no route for {method} {path}"}})

    def _pooling(self, body: Any) -> FakeReply:
        if not isinstance(body, dict) or body.get("task") != "token_embed":
            return FakeReply(400, {"error": {"message": "the pool fake serves task 'token_embed'"}})
        texts = body.get("input")
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            return FakeReply(400, {"error": {"message": "the pool fake takes string inputs"}})
        encoding = body.get("encoding_format", "float")
        dtype = {"float16": "<f2", "float32": "<f4"}[str(body.get("embed_dtype") or "float16")]
        data = []
        for index, text in enumerate(texts):
            count = self.token_count(text)
            matrix = np.asarray(
                [self.unit_vector(self.recipe_id, "token", text, token) for token in range(count)],
                dtype=np.float32,
            ).reshape(count, self.dim)
            embedding: Any = (
                base64.b64encode(matrix.astype(dtype).tobytes()).decode("ascii")
                if encoding == "base64"
                else matrix.tolist()
            )
            data.append(
                {
                    "index": index,
                    "object": "pooling",
                    "data": embedding,
                    "prompt_token_ids": [
                        int(fake_uniform(self.recipe_id, "token_id", text, token) * 100_000) for token in range(count)
                    ],
                }
            )
        return FakeReply(
            200,
            {
                "object": "list",
                "model": self.recipe_id,
                "data": data,
                "usage": {"prompt_tokens": sum(self.token_count(text) for text in texts), "total_tokens": 0},
            },
        )

    def _tokenize(self, body: Any) -> FakeReply:
        prompt = body.get("prompt") if isinstance(body, dict) else None
        if not isinstance(prompt, str):
            return FakeReply(400, {"error": {"message": "the tokenize fake takes {'prompt': str}"}})
        ids = self.tokenizer.ids(prompt, add_special_tokens=bool(body.get("add_special_tokens", True)))
        return FakeReply(200, {"tokens": ids, "count": len(ids), "max_model_len": 512})
