"""Fixture reference for the pointwise reranker: the same deterministic numbers as the stub engine.

Independent of the package and of the recipe: it builds the served prompt by hand from the documented rules
(the template text, the fold rule ``f"{instruction}\\n{query}"``) and tokenises with the deterministic word ids,
so stage 1 fails when the template or the rendering contract drifts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deterministic import DIM, score, token_id, token_vectors, tokens, vector  # noqa: E402

FOLD = "Follow the task."
_TEMPLATE_HEAD = "SYSTEM: Judge whether the Document answers the Query. Answer yes or no.\nUSER:\nQuery: "
_TEMPLATE_TAIL = "\nDocument: {document}\nASSISTANT:"

_loaded: Any = None


def load(device: str) -> Any:
    """Load the reference model (a fixture: nothing to load)."""
    global _loaded
    _loaded = f"fixture-rerank-reference@{device}"
    return _loaded


def fold(instruction: str | None, query: str) -> str:
    """The documented fold rule of the recipe's client.instruction: the instruction, a newline, the query."""
    if not instruction:
        return query
    return f"{instruction}\n{query}"


def prompt(query: str, document: str, instruction: str | None) -> str:
    """The served prompt text for one (query, document) pair, built from the template by hand."""
    return _TEMPLATE_HEAD + fold(instruction, query) + _TEMPLATE_TAIL.format(document=document)


def render(query: str, document: str, instruction: str | None) -> list[int]:
    """Stage 1: the reference token ids of the served prompt for one (query, document) pair."""
    return [token_id(word) for word in tokens(prompt(query, document, instruction))]


def score_query(query: str, documents: list[str], instruction: str | None) -> list[float]:
    """Stage 2: one probability score per document, on the folded query as the client sends it."""
    folded = fold(instruction, query)
    return [score(folded, doc) for doc in documents]


def embed(texts: list[str], role: str) -> list[np.ndarray]:
    """Stage 2 for embedding roles (unused by this fixture; the shape matches the stub's contract)."""
    assert _loaded
    return [vector(text, f"embed-{role}").astype(np.float16) for text in texts]


def token_embed(contents: list[str], role: str) -> list[np.ndarray]:
    """Stage 2 for multi_vector: one float16 vector per whitespace token."""
    del role
    assert _loaded
    return [token_vectors(f"doc: {content}", "tok") for content in contents]


__all__ = ["DIM", "embed", "fold", "load", "prompt", "render", "score_query", "token_embed", "tokens", "vector"]
