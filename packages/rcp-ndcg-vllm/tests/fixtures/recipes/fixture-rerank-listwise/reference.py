"""Fixture reference for the listwise reranker: prompts carry the instruction as a separate field."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deterministic import FixtureTokenizer, score, token_id, tokens, vector  # noqa: E402

INSTRUCTION = "Rank by recency."
_TEMPLATE_HEAD = (
    "SYSTEM: Judge whether the Document answers the Query. Answer yes or no.\n"
    "SYSTEM: Follow this instruction.\n"
    "USER:\n"
    "Query: "
)
_TEMPLATE_TAIL = "\nDocument: {document}\nASSISTANT:"

_loaded: Any = None


def load(device: str) -> Any:
    """Load the reference model (a fixture: nothing to load)."""
    global _loaded
    _loaded = f"fixture-listwise-reference@{device}"
    return _loaded


def prompt(query: str, document: str, instruction: str | None) -> str:
    """The served prompt text: the instruction is part of the prompt (the client sends it as a field)."""
    return _TEMPLATE_HEAD + query + _TEMPLATE_TAIL.format(document=document)


def render(query: str, document: str, instruction: str | None) -> list[int]:
    """Stage 1: the reference token ids of the served prompt for one (query, document) pair."""
    return [token_id(word) for word in tokens(prompt(query, document, instruction))]


def score_query(query: str, documents: list[str], instruction: str | None) -> list[float]:
    """Stage 2: one probability score per document (the instruction does not change the fixture's numbers)."""
    del instruction
    return [score(query, doc) for doc in documents]


def embed(texts: list[str], role: str) -> list[np.ndarray]:
    """Stage 2 for embedding roles (unused by this fixture)."""
    assert _loaded
    return [vector(text, f"embed-{role}").astype(np.float16) for text in texts]


__all__ = ["embed", "load", "prompt", "render", "score_query", "tokens", "vector"]


def tokenizer() -> FixtureTokenizer:
    """The fixture tokenizer for stage 1 (a production reference omits this; the harness loads client.tokenizer)."""
    return FixtureTokenizer()
