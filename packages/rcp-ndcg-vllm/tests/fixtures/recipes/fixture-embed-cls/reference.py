"""Fixture reference for the dense embedder: prompts are composed client-side, the reference mirrors them."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deterministic import DIM, FixtureTokenizer, vector  # noqa: E402

MAX_TOKENS = 512  # the recipe's client.max_tokens; the reference pins the same budget

QUERY_PROMPT = "query: "
DOC_PROMPT = "doc: "
CLS_PREFIX = "<<CLS>>"
"""The template's fixed head: the first-token anchor, resolved from the tokenizer's special map."""
END_SUFFIX = " [END]"
"""The template's fixed tail: the last-token anchor, reserved and re-attached by the cut."""

_loaded: Any = None


def load(device: str) -> Any:
    """Load the reference model (a fixture: nothing to load)."""
    global _loaded
    _loaded = f"fixture-cls-reference@{device}"
    return _loaded


def render(query: str, document: str, instruction: str | None) -> list[int]:
    """Stage 1 for an embedding recipe: the reference ids of the prompted document (the query side is unused)."""
    del query, instruction
    from deterministic import token_id, tokens

    return [token_id(word) for word in tokens(DOC_PROMPT + document + END_SUFFIX)]


def embed(texts: list[str], role: str) -> list[np.ndarray]:
    """Stage 2: one unit vector per text; the reference composes the same prompts the client sends."""
    assert _loaded
    del role
    return [vector(CLS_PREFIX + text, "embed") for text in texts]


__all__ = ["CLS_PREFIX", "DIM", "DOC_PROMPT", "QUERY_PROMPT", "embed", "load", "render"]


def tokenizer() -> FixtureTokenizer:
    """The fixture tokenizer for stage 1 (a production reference omits this; the harness loads client.tokenizer)."""
    return FixtureTokenizer()


def render_shape(shape: str, query: str, document: str, instruction: str | None) -> list[int]:
    """The optional per-shape stage-1 hook: the same assembly for every declared shape."""
    del instruction
    from deterministic import reserve_and_append, token_id, tokens

    shapes = {
        "query": ({token_id("query: ")} and [token_id(word) for word in tokens("query: ")], []),
        "document": ([token_id(word) for word in tokens(DOC_PROMPT)], []),
    }
    if shape == "query":
        prefix_ids, suffix_ids = shapes["query"]
    else:
        prefix_ids, suffix_ids = shapes["document"]
    content_ids = [token_id(word) for word in tokens(query if shape == "query" else document)]
    return reserve_and_append(prefix_ids, content_ids, suffix_ids, MAX_TOKENS)
